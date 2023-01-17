# загружаем необходимые библиотеки

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mpl

import re

import string

import seaborn as sns

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import nltk

import torchtext

import random

from nltk.stem import WordNetLemmatizer

from gensim.models import Word2Vec

from sklearn.manifold import TSNE

from tqdm import tqdm

from torchtext.data import BucketIterator

from IPython.display import clear_output

from nltk import word_tokenize

from torch.nn.utils.rnn import pad_sequence

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score
data = pd.read_csv('../input/60k-stack-overflow-questions-with-quality-rate/data.csv')

data.head()
data.info()
data['Body'][0]
# удаляем колонки с ненужной информацией

df = data.drop(['Id', 'CreationDate'], axis=1)

df.head()
# объединяем колокнки с текстовыми признаками в одну

df['text']= df['Title'] + ' ' + df['Body'] + ' ' + df['Tags']

df = df.drop(['Title', 'Body', 'Tags'], axis=1)
# заменяем значения в колонке с целевой переменной

df['Y'] = df['Y'].replace(['HQ'], 0)

df['Y'] = df['Y'].replace(['LQ_EDIT'], 1)

df['Y'] = df['Y'].replace(['LQ_CLOSE'], 2)

df.head()
df['text'][0]
df['Y'].value_counts()
# выделим текстовый признак и целевую перевенную в отдельные списки

text, target = list(df['text']), list(df['Y'])
# функция для предобработки текста

def preprocess(doc):

    prepdoc = []

    

    lemmatizer = WordNetLemmatizer()

    

    urlptr = r'((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)'

    alhptr = '[^a-zA-Z0-9]'

    sqcptr = r'(.)\1\1+'

    rplptr = r'\1\1'

    

    for text in doc:

        # приводим весь текст к нижнему регистру

        text = text.lower()

        # заменяем ссылки на 'URL'

        text = re.sub(urlptr, ' URL', text)      

        # убираем все символы, отличные от буквенных или цифровых

        text = re.sub(alhptr, ' ', text)

        # обрезаем последовательности из трёх и более одинаковых букв

        text = re.sub(sqcptr, rplptr, text)

        

        words = ' '

        for word in text.split():

            # проверяем короткие слова и приводим словоформы к лемме (словарной форме)

            if len(word) > 1:

                word = lemmatizer.lemmatize(word)

                words += (word + ' ')

            

        prepdoc.append(words)

        

    return prepdoc
# обработаем список признака 'text' и отобразим часть сообщений

%time preptext = preprocess(text)

preptext[:2]
# подготовим данные для обработки в Word2Vec

df_w2v = [sentence.split() for sentence in preptext]

df_w2v[:2]
%%time

# обучим Word2Vec на основе нашего датасета

w2v = Word2Vec(sentences=df_w2v, size=100, window=5, 

               min_count=5, workers=2, sg=1, iter=50)     
# сохраним и загрузим обратно модель Word2Vec

w2v.save('../working/w2v.model')

w2v = Word2Vec.load('../working/w2v.model')
# посмотрим на похожие слова на основе векторного представления

w2v.wv.most_similar('javascript')[:5]
w2v.wv.most_similar('console')[:5]
# функция для отображения кластеризации схожих слов

def display_closestwords_tsnescatterplot(model, word, size):

    sns.set_style('darkgrid')

    mpl.rcParams.update({'font.size': 15})

    

    arr = np.empty((0, size), dtype='f')

    word_labels = [word]

    

    close_words = model.wv.most_similar([word])

    arr = np.append(arr, model.wv.__getitem__([word]), axis=0)

    

    for wrd_score in close_words:

        wrd_vector = model.wv.__getitem__([wrd_score[0]])

        word_labels.append(wrd_score[0])

        arr = np.append(arr, wrd_vector, axis=0)

        

    tsne = TSNE(n_components=2, random_state=0)

    np.set_printoptions(suppress=True)

    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]

    y_coords = Y[:, 1]

    plt.figure(figsize=(16, 10))

    plt.scatter(x_coords, y_coords)

    

    for label, x, y in zip(word_labels, x_coords, y_coords):

        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

    plt.xlim(x_coords.min()-50, x_coords.max()+50)

    plt.ylim(y_coords.min()-50, y_coords.max()+50)

    plt.show()
display_closestwords_tsnescatterplot(w2v,'git', 100)
display_closestwords_tsnescatterplot(w2v,'access', 100)
# векторное представление отдельного слова

w2v.wv['git']
# получившийся словарь слов

# w2v.wv.vocab
# создадим заново датасет из предобработанной ранее текстовой информации

dict = {'Text': preptext, 'Target': target}    

df_rnn = pd.DataFrame(dict)

df_rnn.head()
# посмотрим на распределение длин вопросов из датасета на гистограмме 

df_rnn['Text'].map(len).hist(bins=100);
# сохраним изначальный датасет в .csv файл

df.to_csv('df')
# инициализиурем объекты для предобработки датасета при дальнейшей загурзки в torch

description = torchtext.data.Field(tokenize=word_tokenize, lower=True, batch_first=True)

y = torchtext.data.Field(sequential=False, is_target=True, use_vocab=False)
%%time

# загрузим данные в torch с помощью TabularDataset

data = torchtext.data.TabularDataset(path='../working/df', format='csv', 

                                     fields={

                                         'text': ('text', description),

                                         'Y': ('target', y)

                                     })
# создадим словарь на основе текстового признака

description.build_vocab(data)
# загрузим обученный Word2Vec

!wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
# распакуем загруженный Word2Vec

!unzip '../working/wiki-news-300d-1M.vec.zip' -d '../working'
# загурзим ветора из обученного Word2Vec

description.vocab.load_vectors(torchtext.vocab.Vectors('../working/wiki-news-300d-1M.vec'))
description.vocab.vectors.shape
# разобьём датасет на тренировочную и тестовую выборку

train, val = data.split(split_ratio=0.8)
# создадим сеть LSTM

class lstm(nn.Module):

    def __init__(self, w2v, padding_inx, dropout, hidden_size):

        super(lstm, self).__init__()

        

        self.embedding = nn.Embedding.from_pretrained(w2v)

        self.embedding.padding_inx = padding_inx



        self.embedding.weight.requires_grad = True



        self.dropout = nn.Dropout(p = dropout)

        self.lstm = nn.LSTM(input_size = self.embedding.embedding_dim,

                            hidden_size = hidden_size,

                            num_layers = 2,

                            dropout = dropout,

                            bidirectional = True)

        self.label = nn.Linear(hidden_size*2*2, 1)



    def forward(self, sentence):

        x = self.embedding(sentence)

        x = torch.transpose(x, dim0 = 1, dim1 = 0)

        out, (hidden, c) = self.lstm(x)

        x = self.dropout(torch.cat([c[i,:,:] for i in range(c.shape[0])], dim=1))

        x = self.label(x)

        return x
# с помощью BucketIterator создадим объекты загрузки данных в сеть

batch_size = 16

train_i = torchtext.data.BucketIterator(dataset=train,

                                        batch_size=batch_size,

                                        shuffle=True,

                                        sort = False,

                                        train =True)





val_i = torchtext.data.BucketIterator(dataset=val,

                                      batch_size=batch_size,

                                      shuffle=True,

                                      sort = False,

                                      train = False)
# определим нашу созданнуть сеть LSTM

model = lstm(description.vocab.vectors, description.vocab.stoi[description.pad_token], 

             dropout=0.2, hidden_size=128).cuda()
# оптимизатор и функция потерь

optimizer = optim.Adam(model.parameters(), lr=1e-3)

loss = nn.BCEWithLogitsLoss()
# функция для обучения сети

def train(epochs, model, eval_time, loss_f, optimizer, train_i, val_i):

    sns.set_style('white')

    mpl.rcParams.update({'font.size': 10})

    

    step = 0

    losses = []

    val_losses = []

    accuracy = []

    val_accuracy = []

    train_i.init_epoch()

    

    for epoch in range(epochs):

        for batch in iter(train_i):

            step += 1

            model.train()

            x = batch.text.cuda()

            y = batch.target.type(torch.Tensor).cuda()

            model.zero_grad()

            preds = model.forward(x).view(-1)

            loss = loss_f(preds, y)

            losses.append(loss.cpu().data.numpy())

            accuracy.append(accuracy_score(batch.target.data.numpy().tolist(), 

                                           np.round(np.array(torch.sigmoid(preds).cpu().data.numpy().tolist()))

                                          ))

            loss.backward()

            optimizer.step()



            if step % eval_time == 0:

                clear_output(True)

                model.eval()

                model.zero_grad()



                for batch in iter(val_i):

                    x = batch.text.cuda()

                    y = batch.target.type(torch.Tensor).cuda()

                    preds = model.forward(x).view(-1)

                    val_losses.append(loss_f(preds, y).cpu().data.numpy())

                    val_accuracy.append(accuracy_score(batch.target.data.numpy().tolist(), 

                                                   np.round(np.array(torch.sigmoid(preds).cpu().data.numpy().tolist()))

                                                      ))

                    

                fig, axs = plt.subplots(2, 2, figsize=(10, 10))

                fig.suptitle('Accuracy & Loss')

                

                axs[0, 0].set_title('train cross-entropy loss')

                axs[0, 1].set_title('test cross-entropy loss')

                axs[1, 0].set_title('train accuracy')

                axs[1, 1].set_title('test accuracy')



                axs[0, 0].plot(losses)

                axs[0, 0].plot(pd.Series(losses).rolling(400).mean().values)

                axs[0, 1].plot(val_losses)

                axs[0, 1].plot(pd.Series(val_losses).rolling(400).mean().values)

                axs[1, 0].plot(accuracy)

                axs[1, 0].plot(pd.Series(accuracy).rolling(400).mean().values)

                axs[1, 1].plot(val_accuracy)

                axs[1, 1].plot(pd.Series(val_accuracy).rolling(400).mean().values)

            

                for ax in axs.flat:

                    ax.set(xlabel='step')



                plt.show()
# переносим модель на GPU

device = torch.device("cuda:0")

model.to(device)
# обучаем и выводим графики с метриками loss и accurancy

train(3, model, 250, loss, optimizer, train_i, val_i)
# оцениваем обученную модель и выводим метрики качества

model.eval()



real = []

preds = []



for batch in iter(val_i):

    x = batch.text.cuda()

    real += batch.target.data.numpy().tolist()

    preds += torch.sigmoid(model.forward(x).view(-1)).cpu().data.numpy().tolist()



print(classification_report(real, np.round(np.array(preds))))