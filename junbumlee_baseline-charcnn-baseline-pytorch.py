# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/koreangenderbiasdetection/train.gender_bias.binary.csv')

train_df.head()
dev_df = pd.read_csv('/kaggle/input/koreangenderbiasdetection/dev.gender_bias.binary.csv')

dev_df.head()
train_df['label'] = train_df['label'].map(lambda x: 1 if x else 0)

dev_df['label'] = dev_df['label'].map(lambda x: 1 if x else 0)
train_df.head()
train_sentences = [list(d) for d in train_df['comments'].to_list()]

dev_sentences  = [list(d) for d in dev_df['comments'].to_list()]
train_labels = train_df['label'].to_list()

dev_labels = dev_df['label'].to_list()
from collections import Counter



def make_dictionary(sentences, vocabulary_size=None, initial_words=['<UNK>', '<PAD>', '<SOS>', '<EOS>']):

    """sentences : list of list"""

    

    counter = Counter()

    for words in sentences:

        counter.update(words)

    

    if vocabulary_size is None:

        vocabulary_size = len(counter.keys())

        

    vocab_words = counter.most_common(vocabulary_size)

    

    for initial_word in initial_words:

        vocab_words.insert(0, (initial_word, 0))

    

    word2idx = {word:idx for idx, (word, count) in enumerate(vocab_words)}

    idx2word = {idx:word for word, idx in word2idx.items()}

    

    return word2idx, idx2word





def process_sentences(sentences, word2idx, sentence_length=20, padding='<PAD>'):

    """sentences : list of list

    Only paddding. No SOS or EOS

    """

    

    sentences_processed = []

    for sentence in sentences:

        if len(sentence) > sentence_length:

            fixed_sentence = sentence[:sentence_length]

        else:

            fixed_sentence = sentence + [padding]*(sentence_length - len(sentence))

        

        sentence_idx = [word2idx[word] if word in word2idx.keys() else word2idx['<UNK>'] for word in fixed_sentence]

        

        sentences_processed.append(sentence_idx)



    return sentences_processed



def make_mask(sentences, sentence_length):

    

    masks = []

    for sentence in sentences:

        words_count = len(sentence[:sentence_length])

        sentence_mask = np.concatenate([np.ones(words_count-1), np.ones(1), np.zeros(sentence_length-words_count)])

        masks.append(sentence_mask)

    

    mask = np.array(masks)

    return mask
word2idx, idx2word = make_dictionary(train_sentences, initial_words=['<UNK>', '<PAD>'])
len(word2idx)
train_df['comments'].map(len).max() # 최대 길이 약 150이내
SENTENCE_LENGTH = 150



train_sentences_processed = process_sentences(train_sentences, word2idx, sentence_length=SENTENCE_LENGTH)

dev_sentences_processed = process_sentences(dev_sentences, word2idx, sentence_length=SENTENCE_LENGTH)
train_mask = make_mask(train_sentences, sentence_length=SENTENCE_LENGTH)

dev_mask = make_mask(dev_sentences, sentence_length=SENTENCE_LENGTH)
import torch

from torch import nn



from torch.utils.data import Dataset, DataLoader

from torch.autograd import Variable

from torch.optim import Adam
class DatasetLoader(Dataset):

    def __init__(self, sentences_processed, labels):

        assert len(sentences_processed) == len(labels)

        self.sentences_processed = sentences_processed

        self.labels = torch.LongTensor(labels)



    def __getitem__(self, index):

        return torch.LongTensor(self.sentences_processed[index]), self.labels[index]

        

    def __len__(self):

        return len(self.sentences_processed)
train_dataset = DatasetLoader(train_sentences_processed, train_labels)

dev_dataset = DatasetLoader(dev_sentences_processed, dev_labels)
len(train_dataset), len(dev_dataset)
train_dataloader = DataLoader(train_dataset, batch_size=128)

dev_dataloader = DataLoader(dev_dataset, batch_size=256)
import torch

import torch.nn as nn

import torch.nn.functional as F
class CharCNN(nn.Module):

    def __init__(self, vocab_size, embedding_size, output_size, kernel_num, kernel_sizes):

        super().__init__()

        

        self.embedding = nn.Embedding(

            vocab_size, embedding_size, padding_idx=0

        )

        

        self.convs = nn.ModuleList([

            nn.Conv1d(embedding_size, kernel_num, kernel_size=kernel_size) 

            for kernel_size in kernel_sizes

        ])

        self.maxpools = nn.ModuleList([

            nn.MaxPool1d(kernel_size) 

            for kernel_size in kernel_sizes

        ])

        

        self.linear = nn.Linear(1140, output_size)

        self.softmax = nn.LogSoftmax(dim=1)

        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(p=0.5)

        

    def forward(self, x):

        batch_size = x.size(0)

        embedded = self.embedding(x)

        embedded = embedded.transpose(1,2)

        

        pools = []

        for conv, maxpool in zip(self.convs, self.maxpools):

            feature_map = conv(embedded)

            pooled = maxpool(feature_map)

            pools.append(pooled)

            

        conv_concat = torch.cat(pools, dim=-1).view(batch_size, -1)

        conv_concat = self.dropout(conv_concat)

        logits = self.linear(conv_concat)

        return self.softmax(logits)
model = CharCNN(

    vocab_size=len(word2idx), 

    embedding_size=300,

    output_size=2, 

    kernel_num=10,

    kernel_sizes=[3,4,5]

)
if torch.cuda.is_available():

    model.to('cuda')
loss_function = nn.NLLLoss()
optimizer = Adam(model.parameters())
for epoch in range(6):

    losses = []

    for i, (batch_data, batch_label) in enumerate(train_dataloader):

        

        batch_data, batch_label = Variable(batch_data), Variable(batch_label)

        

        if torch.cuda.is_available():

            batch_data = batch_data.to('cuda')

            batch_label = batch_label.to('cuda')

        

        log_probs = model(batch_data)

        loss = loss_function(log_probs, batch_label.transpose(0, -1))

        

        model.zero_grad()

        loss.backward()

        optimizer.step()

        

        losses.append(loss.item())

        

        corrects = log_probs.data.cpu().numpy().argmax(axis=1) == batch_label.data.cpu().numpy()

        train_accuracy = corrects.astype(np.long).mean()

        

        val_losses = []

        val_accuracies = []

        for test_data, test_label in dev_dataloader:

            test_data, test_label = Variable(test_data), Variable(test_label)

            if torch.cuda.is_available():

                test_data = test_data.to('cuda')

                test_label = test_label.to('cuda')

            val_log_probs = model(test_data)

            val_loss = loss_function(val_log_probs, test_label)

            val_corrects = val_log_probs.data.cpu().numpy().argmax(axis=1) == test_label.data.cpu().numpy()

            val_accuracy = val_corrects.astype(np.float64).mean()

            val_losses.append(val_loss.item())

            val_accuracies.append(val_accuracy)

            

        message = "Epoch: {epoch:<5d}  Iteration: {iteration:<5d}  Loss: {loss:<.3}  Val Loss: {val_loss:<.3} Train Accuracy: {train_acc:<.3}  Test Accuracy: {test_acc:<.3}".format(

                    epoch=epoch,

                    iteration= i,

                    loss=sum(losses)/len(losses),

                    val_loss=sum(val_losses)/len(val_losses),

                    train_acc=train_accuracy,

                    test_acc=sum(val_accuracies)/len(val_accuracies)

                        )

        print(message)
model.eval()
test_df = pd.read_csv('/kaggle/input/koreangenderbiasdetection/test.gender_bias.no_label.csv')
test_df.head()
test_sentences = [list(d) for d in test_df['comments'].to_list()]
test_sentences_processed = process_sentences(test_sentences, word2idx, sentence_length=SENTENCE_LENGTH)

test_mask = make_mask(test_sentences, sentence_length=SENTENCE_LENGTH)

test_dataset = DatasetLoader(test_sentences_processed, [0 for i in range(len(test_sentences_processed))])

test_dataloader = DataLoader(test_dataset, batch_size=1)
res = []

for (batch_data, batch_label) in (test_dataloader):

    if torch.cuda.is_available():

        batch_data = batch_data.to('cuda')

        batch_label = batch_label.to('cuda')

    log_probs = model(batch_data)

    r = log_probs.data.cpu().numpy().argmax(axis=1)

    res.append(r[0])
test_df['label'] = res
test_df['label'].value_counts()
test_df['label'] = test_df['label'].map(lambda x: True if x else False)
test_df.head()
test_df.to_csv('/kaggle/working/prediction.csv', index=None, header=True)