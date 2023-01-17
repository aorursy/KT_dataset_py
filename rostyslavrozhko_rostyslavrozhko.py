import pandas as pd
import numpy as np
import torch
import torchtext
import random
from torch import nn
from sklearn.metrics import f1_score
from nltk import word_tokenize
from torch import optim
from tqdm import tqdm
tqdm.pandas()

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("Train shape : ",train.shape)
print("Test shape : ",test.shape)
def build_vocab(sentences, verbose =  True):
    """
    :param sentences: list of list of words
    :return: dictionary of words and their count
    """
    vocab = {}
    for sentence in tqdm(sentences, disable = (not verbose)):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab
sentences = train["question_text"].progress_apply(lambda x: x.split()).values
vocab = build_vocab(sentences)
from gensim.models import KeyedVectors

news_path = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
embeddings_index = KeyedVectors.load_word2vec_format(news_path, binary=True)
import operator 

def check_coverage(vocab,embeddings_index):
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in tqdm(vocab):
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except:

            oov[word] = vocab[word]
            i += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x
def clean_text(x):

    x = str(x)
    for punct in "/-'":
        x = x.replace(punct, ' ')
    for punct in '&':
        x = x.replace(punct, f' {punct} ')
    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        x = x.replace(punct, '')
    return x
train["question_text"] = train["question_text"].progress_apply(lambda x: clean_text(x))
sentences = train["question_text"].apply(lambda x: x.split())
vocab = build_vocab(sentences)
import re

def clean_numbers(x):

    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x
train["question_text"] = train["question_text"].progress_apply(lambda x: clean_numbers(x))
sentences = train["question_text"].progress_apply(lambda x: x.split())
vocab = build_vocab(sentences)
def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re


mispell_dict = {'colour':'color',
                'centre':'center',
                'didnt':'did not',
                'doesnt':'does not',
                'isnt':'is not',
                'shouldnt':'should not',
                'favourite':'favorite',
                'travelling':'traveling',
                'counselling':'counseling',
                'theatre':'theater',
                'cancelled':'canceled',
                'labour':'labor',
                'organisation':'organization',
                'wwii':'world war 2',
                'citicise':'criticize',
                'instagram': 'social medium',
                'whatsapp': 'social medium',
                'snapchat': 'social medium'

                }
mispellings, mispellings_re = _get_mispell(mispell_dict)

def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)
train["question_text"] = train["question_text"].progress_apply(lambda x: replace_typical_misspell(x))
sentences = train["question_text"].progress_apply(lambda x: x.split())
to_remove = ['a','to','of','and']
sentences = [[word for word in sentence if not word in to_remove] for sentence in tqdm(sentences)]
vocab = build_vocab(sentences)
oov = check_coverage(vocab,embeddings_index)
oov[:20]
from sklearn.model_selection import train_test_split
random_state = 43
batch_size = 64

train, test = train_test_split(train, test_size=0.2, random_state=43)


train_iter = torchtext.data.BucketIterator(dataset=train,
                                           batch_size=batch_size,
                                           sort_key=lambda x: x.text.__len__(),
                                           shuffle=True,
                                           sort=False)

val_iter = torchtext.data.BucketIterator(dataset=val,
                                         batch_size=batch_size,
                                         sort_key=lambda x: x.text.__len__(),
                                         train=False,
                                         sort=False)
print(vocab)
def training(epoch, model, loss_func, optimizer, train_iter, val_iter):
    step = 0
    train_record = []
    losses = []
    val_record = []
    
    for e in range(epoch):
        train_iter.init_epoch()
        for train_batch in iter(train_iter):
            step += 1
            model.train()
            x = train_batch.text.cuda()
            y = train_batch.target.type(torch.Tensor).cuda()
            model.zero_grad()
            pred = model.forward(x).view(-1)
            loss = loss_function(pred, y)
            loss_data = loss.cpu().data.numpy()
            train_record.append(loss_data)
            loss.backward()
            optimizer.step()
            if step % 1000 == 0:
                print("Step: {:06}, loss {:.4f}".format(step, loss_data))
            if step % 10000 == 0:
                model.eval()
                model.zero_grad()
                val_loss = []
                for val_batch in iter(val_iter):
                    val_x = val_batch.text.cuda()
                    val_y = val_batch.target.type(torch.Tensor).cuda()
                    val_pred = model.forward(val_x).view(-1)
                    val_loss.append(loss_function(val_pred, val_y).cpu().data.numpy())
                val_record.append({'step': step, 'loss': np.mean(val_loss)})
                print('Epoch x{:02} - step {:06} - train_loss {:.4f} - val_loss {:.4f} '.format(
                            e, step, np.mean(train_record), val_record[-1]['loss']))
                train_record = []
class SimpleModel(nn.Module):
    def __init__(self, pretrained_lm, padding_idx, hidden_dim, static=True):
        super(SimpleModel, self).__init__()
        self.hidden_dim = 32
        self.embedding = nn.Embedding.from_pretrained(pretrained_lm)
        self.embedding.padding_idx = padding_idx
        if static:
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=1)
        self.lstm_to_linear = nn.Linear(self.hidden_dim, 1)
        
    def forward(self, sents):
        x = self.embedding(sents)
        x = torch.transpose(x, dim0=1, dim1=0)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # gather all the hidden states
        x = torch.cat([c_n[i,:, :] for i in range(c_n.shape[0])], dim=1)
        # feed linear layer
        output = self.lstm_to_linear(x)
        return output
model = SimpleModel(vocab,
                    padding_idx=text.vocab.stoi['<pad>'],
                    hidden_dim=128).cuda()
loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=1e-3)

training(model=model,
         epoch=20,
         loss_func=loss_function,
         optimizer=optimizer,
         train_iter=train_iter,
         val_iter=val_iter)

