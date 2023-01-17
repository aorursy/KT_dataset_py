import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import nltk

from collections import Counter

import itertools

import torch



# Any results you write to the current directory are saved as output.
class InputFeatures(object):

    """A single set of features of data."""



    def __init__(self, input_ids, label_id):

        self.input_ids = input_ids

        self.label_id = label_id

#Класс словаря. Метод word2id возвращает номер слова, id2word - наоборот, восстанавливает слово.



class Vocab:

    def __init__(self, itos, unk_index):

        self._itos = itos

        self._stoi = {word:i for i, word in enumerate(itos)}

        self._unk_index = unk_index

        

    def __len__(self):

        return len(self._itos)

    

    def word2id(self, word):

        idx = self._stoi.get(word)

        if idx is not None:

            return idx

        return self._unk_index

    

    def id2word(self, idx):

        return self._itos[idx]

from tqdm import tqdm_notebook

#Интерфейс объекта, преобразующего тексты в последовательности номеров. transform выполняет преобразование при помощи словаря. fit_transform выучивает словарь из текста и возвращает такое же преобразование при помощи свежеполученного словаря.



class TextToIdsTransformer:

    def transform():

        raise NotImplementedError()

        

    def fit_transform():

        raise NotImplementedError()

#Простая реализация данного интерфейса. Разбиение на слова производится с помощью библиотеки NLTK. В словаре содержатся несколько спец. слов. После токенизации, к полученной последовательности слов добавляются слева и справа спец. слова для начала и конца текста.



class SimpleTextTransformer(TextToIdsTransformer):

    def __init__(self, max_vocab_size):

        self.special_words = ['<PAD>', '</UNK>', '<S>', '</S>']

        self.unk_index = 1

        self.pad_index = 0

        self.vocab = None

        self.max_vocab_size = max_vocab_size

        

    def tokenize(self, text):

        return nltk.tokenize.word_tokenize(text.lower())

        

    def build_vocab(self, tokens):

        itos = []

        itos.extend(self.special_words)

        

        token_counts = Counter(tokens)

        for word, _ in token_counts.most_common(self.max_vocab_size - len(self.special_words)):

            itos.append(word)

            

        self.vocab = Vocab(itos, self.unk_index)

    

    def transform(self, texts):

        result = []

        for text in texts:

            tokens = ['<S>'] + self.tokenize(text) + ['</S>']

            ids = [self.vocab.word2id(token) for token in tokens]

            result.append(ids)

        return result

    

    def fit_transform(self, texts):

        result = []

        tokenized_texts = [self.tokenize(text) for text in texts]

        self.build_vocab(itertools.chain(*tokenized_texts))

        for tokens in tokenized_texts:

            tokens = ['<S>'] + tokens + ['</S>']

            ids = [self.vocab.word2id(token) for token in tokens]

            result.append(ids)

        return result

#Строим экземпляр входных данных. Обеспечиваем длину последовательности номеров равной max_seq_len.



def build_features(token_ids, label, max_seq_len, pad_index, label_encoding):

    if len(token_ids) >= max_seq_len:

        ids = token_ids[:max_seq_len]

    else:

        ids = token_ids + [pad_index for _ in range(max_seq_len - len(token_ids))]

    return InputFeatures(ids, label_encoding[label])

        

#Собираем экземпляры в тензоры



def features_to_tensor(list_of_features):

    text_tensor = torch.tensor([example.input_ids for example in list_of_features], dtype=torch.long)

    labels_tensor = torch.tensor([example.label_id for example in list_of_features], dtype=torch.long)

    return text_tensor, labels_tensor

from sklearn import model_selection

imdb_df = pd.read_csv('../input/imdb_master.csv', encoding='latin-1')

dev_df = imdb_df[(imdb_df.type == 'train') & (imdb_df.label != 'unsup')]

test_df = imdb_df[(imdb_df.type == 'test')]

train_df, val_df = model_selection.train_test_split(dev_df, test_size=0.05, stratify=dev_df.label)

max_seq_len=200

classes = {'neg': 0, 'pos' : 1}

text2id = SimpleTextTransformer(10000)



train_ids = text2id.fit_transform(train_df['review'])

val_ids = text2id.transform(val_df['review'])

test_ids = text2id.transform(test_df['review'])

print(train_df.review.iloc[0][:160])

print(train_ids[0][:30])
train_features = [build_features(token_ids, label,max_seq_len, text2id.pad_index, classes) 

                  for token_ids, label in zip(train_ids, train_df['label'])]



val_features = [build_features(token_ids, label,max_seq_len, text2id.pad_index, classes) 

                  for token_ids, label in zip(val_ids, val_df['label'])]



test_features = [build_features(token_ids, label,max_seq_len, text2id.pad_index, classes) 

                  for token_ids, label in zip(test_ids, test_df['label'])]



train_tensor, train_labels = features_to_tensor(train_features)

val_tensor, val_labels = features_to_tensor(val_features)

test_tensor, test_labels = features_to_tensor(test_features)
from torch.utils.data import TensorDataset,DataLoader



train_ds = TensorDataset(train_tensor,train_labels)

val_ds = TensorDataset(val_tensor,val_labels)

test_ds = TensorDataset(test_tensor,test_labels)
train_loader = DataLoader(train_ds,batch_size=128)

val_loader = DataLoader(val_ds, batch_size=128)

test_loader = DataLoader(test_ds, batch_size=128)

vocab_len = len(text2id.vocab)

print(vocab_len)
import torch.nn as nn

import torch.nn.functional as F



class Model(nn.Module):

    def __init__(self):

        super(Model, self).__init__()

        self.emb = nn.Embedding(vocab_len, 100)

        self.properties = nn.Sequential(

            nn.Conv1d(in_channels=100, out_channels=120, kernel_size=3, padding=2),

            nn.ReLU(),

            nn.Conv1d(in_channels=120, out_channels=130, kernel_size=3),

            nn.ReLU(),

            nn.MaxPool1d(5)

        )

        self.estimator = nn.Sequential(

            nn.Linear(5200,1),

            nn.Sigmoid()

        )

        

    def forward(self, x):

        x = self.emb(x)

        x = x.transpose(1,2)

        return  self.estimator(self.properties(x).view(x.size(0), -1))

        

    def train(self,train_loader,val_loader,epoch,waiting,optimizer):

        self.cuda()

        best_val_loss=1000

        crit = nn.BCELoss()

        for i in range(epoch):

            train_loss = 0

            val_loss = 0

            for xx,yy in train_loader:

                xx = xx.cuda()

                yy=yy.cuda()

                optimizer.zero_grad()

                y_pred = self.forward(xx)

                loss = crit(y_pred,yy.float())

                train_loss += loss

                loss.backward()

                optimizer.step()

            train_loss = train_loss/len(train_loader)

            with torch.no_grad():

                for xx,yy in val_loader:

                    xx, yy = xx.cuda(), yy.cuda()

                    y_pred = self.forward(xx)

                    loss = crit(y_pred,yy.float())

                    val_loss += loss

                val_loss = val_loss/len(val_loader)

                

                if best_val_loss>val_loss:

                    torch.save(self.state_dict(), "../best_model.py")

                    best_val_loss = val_loss

                    wait=waiting

                else:

                    wait -=1

                    if wait==0:

                        break

            print("train loss:", float(train_loss), "___best val loss:",float(best_val_loss), "___remaining:", wait)
import gc

gc.collect()

clf = Model()



optimizer = torch.optim.Adam(clf.parameters(), lr=0.001)

clf.train(train_loader,val_loader,20,10,optimizer)
from sklearn.metrics import classification_report

clf.load_state_dict(torch.load("../best_model.py"))

y_true = []

y_pred = []

for xx,yy in test_loader:

    out = clf.forward(xx.cuda())

    for i in out:

        if i<=0.4:

            y_pred.append(0)

        else:

            y_pred.append(1)

    for i in yy:

        y_true.append(int(i))

print(classification_report(y_true,y_pred))