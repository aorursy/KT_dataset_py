import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input/"))
from torchtext.data import Field, Dataset, Example
import pandas as pd

class DataFrameDataset(Dataset):
    """Class for using pandas DataFrames as a datasource"""
    def __init__(self, examples, fields, filter_pred=None):
        """
        Create a dataset from a pandas dataframe of examples and Fields
        Arguments:
         examples pd.DataFrame: DataFrame of examples
         fields {str: Field}: The Fields to use in this tuple. The
             string is a field name, and the Field is the associated field.
         filter_pred (callable or None): use only exanples for which
             filter_pred(example) is true, or use all examples if None.
             Default is None
        """
        self.examples = examples.apply(SeriesExample.fromSeries, args=(fields,), axis=1).tolist()
        if filter_pred is not None:
            self.examples = filter(filter_pred, self.examples)
        self.fields = dict(fields)
        # Unpack field tuples
        for n, f in list(self.fields.items()):
            if isinstance(n, tuple):
                self.fields.update(zip(n, f))
                del self.fields[n]


class SeriesExample(Example):
    """Class to convert a pandas Series to an Example"""

    @classmethod
    def fromSeries(cls, data, fields):
        return cls.fromdict(data.to_dict(), fields)

    @classmethod
    def fromdict(cls, data, fields):
        ex = cls()
        for key, field in fields:
            if key not in data:
                raise ValueError("Specified key `{}` was not found in the input data".format(key))
            if field is not None:
                setattr(ex, key, field.preprocess(data[key]))
            else:
                setattr(ex, key, data[key])
        return ex
# prepare data

import numpy as np
import pandas as pd

from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe
from sklearn.model_selection import StratifiedShuffleSplit


field_text = data.Field(lower=True, tokenize='spacy', include_lengths=True, batch_first=True, fix_length=300)
field_label = data.LabelField()
fields = [("text", field_text), ("label", field_label)]
class CustomGloVe(Vectors):
    def __init__(self, **kwargs):
        name = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
        super(CustomGloVe, self).__init__(name, **kwargs)
def data_generator(n_folds=1, valid_size=0.33, seed=999):
    df_train = pd.read_csv("../input/train.csv", index_col='qid')
    df_train_neg = df_train[df_train['target']==1]
    df_train_pos = df_train[df_train['target']==0].sample(n=df_train_neg.shape[0] * 5)
    df_train = pd.concat([df_train_neg,df_train_pos],axis=0)
    print("down sampling shape is {} {}".format(*df_train.shape))
    
    df_train.rename({'question_text':'text','target':'label'}, axis=1, inplace=True)
    df_train["text"] = df_train['text'].str.replace("\n", " ")

    sss = StratifiedShuffleSplit(n_splits=n_folds, test_size=valid_size, random_state=seed)
    for i, (train_indices, valid_indices) in enumerate(sss.split(df_train, df_train.label)):
        _train, _valid = df_train.iloc[train_indices], df_train.iloc[valid_indices]
        train_ds, valid_ds = DataFrameDataset(_train, fields), DataFrameDataset(_valid, fields)
        yield (train_ds, valid_ds)
def load_dataset():
    train_ds, valid_ds = next(data_generator(n_folds=1))
    field_text.build_vocab(train_ds, vectors=CustomGloVe())
    field_label.build_vocab(train_ds)
    word_embeddings = field_text.vocab.vectors

    print ("Length of Text Vocabulary: " + str(len(field_text.vocab)))
    print ("Vector size of Text Vocabulary: ", field_text.vocab.vectors.size())
    print ("Label Length: " + str(len(field_label.vocab)))

    train_iter, valid_iter = data.BucketIterator.splits((train_ds, valid_ds), 
                                                        batch_size=32, 
                                                        sort_key=lambda x: len(x.text), 
                                                        repeat=False, 
                                                        shuffle=True)
    vocab_size = len(field_text.vocab)
    return field_text, vocab_size, word_embeddings, train_iter, valid_iter
# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class RCNN(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):
        super(RCNN, self).__init__()
        """
		Arguments
		---------
		batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
		output_size : 2 = (pos, neg)
		hidden_sie : Size of the hidden_state of the LSTM
		vocab_size : Size of the vocabulary containing unique words
		embedding_length : Embedding dimension of GloVe word embeddings
		weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 
		
		"""

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length

        self.word_embeddings = nn.Embedding(
            vocab_size, embedding_length)  # Initializing the look-up table.
        self.word_embeddings.weight = nn.Parameter(
            weights, requires_grad=False
        )  # Assigning the look-up table to the pre-trained GloVe word embedding.
        self.dropout = 0.8
        self.lstm = nn.LSTM(
            embedding_length,
            hidden_size,
            dropout=self.dropout,
            bidirectional=True)
        self.W2 = nn.Linear(2 * hidden_size + embedding_length, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)

    def forward(self, input_sentence, batch_size=None):
        """ 
		Parameters
		----------
		input_sentence: input_sentence of shape = (batch_size, num_sequences)
		batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)
		
		Returns
		-------
		Output of the linear layer containing logits for positive & negative class which receives its input as the final_hidden_state of the LSTM
		final_output.shape = (batch_size, output_size)
		
		"""
        """
		
		The idea of the paper "Recurrent Convolutional Neural Networks for Text Classification" is that we pass the embedding vector
		of the text sequences through a bidirectional LSTM and then for each sequence, our final embedding vector is the concatenation of 
		its own GloVe embedding and the left and right contextual embedding which in bidirectional LSTM is same as the corresponding hidden
		state. This final embedding is passed through a linear layer which maps this long concatenated encoding vector back to the hidden_size
		vector. After this step, we use a max pooling layer across all sequences of texts. This converts any varying length text into a fixed
		dimension tensor of size (batch_size, hidden_size) and finally we map this to the output layer.
		"""
        # embedded input of shape = (batch_size, num_sequences, embedding_length)
        input = self.word_embeddings(input_sentence)
        # input.size() = (num_sequences, batch_size, embedding_length)
        input = input.permute(1, 0, 2)
        if batch_size is None:
            if torch.cuda.is_available():
                # Initial hidden state of the LSTM
                h_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size).cuda())
                # Initial cell state of the LSTM
                c_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size).cuda())
            else:
                h_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size))
                c_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size))
        else:
            if torch.cuda.is_available():
                h_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())
                c_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())
            else:
                h_0 = Variable(torch.zeros(2, batch_size, self.hidden_size))
                c_0 = Variable(torch.zeros(2, batch_size, self.hidden_size))

        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))

        final_encoding = torch.cat((output, input), 2).permute(1, 0, 2)
        # y.size() = (batch_size, num_sequences, hidden_size)
        y = self.W2(final_encoding)
        # y.size() = (batch_size, hidden_size, num_sequences)
        y = y.permute(0, 2, 1)
        # y.size() = (batch_size, hidden_size, 1)
        y = F.max_pool1d(y, y.size()[2])
        y = y.squeeze(2)
        logits = self.label(y)

        return logits
import os
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
TEXT, vocab_size, word_embeddings, train_iter, valid_iter = load_dataset()
def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)
def train_model(model, train_iter, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    
    if torch.cuda.is_available():
        model.cuda()
        
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        text = batch.text[0]
        target = batch.label
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()
        if (text.size()[0] is not 32):# One of the batch returned by BucketIterator has length different than 32.
            continue
        optim.zero_grad()
        prediction = model(text)
        loss = loss_fn(prediction, target)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/len(batch)
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1
        
        if steps % 100 == 0:
            print (f'Epoch: {epoch+1}, Idx: {idx+1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')
        
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
        
    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)
def eval_model(model, val_iter):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text = batch.text[0]
            if (text.size()[0] is not 32):
                continue
            target = batch.label
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                text = text.cuda()
                target = target.cuda()
            prediction = model(text)
            loss = loss_fn(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects/len(batch)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter)
earning_rate = 2e-5
batch_size = 32
output_size = 2
hidden_size = 256
embedding_length = 300
model = RCNN(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
loss_fn = F.cross_entropy

for epoch in range(10):
    train_loss, train_acc = train_model(model, train_iter, epoch)
    val_loss, val_acc = eval_model(model, valid_iter)
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')
