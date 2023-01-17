import csv

from sklearn.metrics import classification_report

from itertools import islice, chain

import pandas as pd

import os

import numpy as np

import nltk

from nltk.corpus import stopwords

import collections

from IPython.display import Image

import random

import zipfile

from nltk.stem import SnowballStemmer

from string import punctuation

import spacy

import os

import seaborn as sns

import matplotlib.pyplot as plt

from gensim.models.doc2vec import TaggedDocument

from gensim.models import Doc2Vec

from tqdm import tqdm

from six.moves import range

from six.moves.urllib.request import urlretrieve

from sklearn import utils

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix

%matplotlib inline
TRAIN_PATH = os.path.join("../input/", "BBC News Train.csv")

TEST_PATH = os.path.join("../input/", "BBC News Test.csv")
bbc_train = pd.read_csv(TRAIN_PATH, encoding='latin-1')

bbc_test = pd.read_csv(TEST_PATH, encoding='latin-1')
bbc_train.head(10)
bbc_test.head(10)
topic = bbc_train['Category'].value_counts()

plt.figure(figsize=(12,4))

sns.barplot(topic.index, topic.values, alpha=0.8)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Topic', fontsize=12)

plt.xticks(rotation=90)

plt.show();
def read_data(filename):

    """

      Extrai artigos até um determinado limite em um arquivo zip como uma lista de palavras

      e pré-processa usando a biblioteca nltk python

      """

        

    data = [[],[]]

    train_data = {}

    for i in range(filename.shape[0]):

        text_string = filename[filename.columns[1]][i]

        text_string = text_string.lower()

        text_string = nltk.word_tokenize(text_string)

        # Atribui a classe aos arquivos

        data[0].append(text_string)

        data[1].append(filename[filename.columns[2]][i])

        """ Atribui o tópico ao documento """

        train_data[filename[filename.columns[2]][i]+'-'+filename[filename.columns[0]][i].astype(str)] = text_string

        print('\tConcluída a leitura de dados para o tópico: ',filename[filename.columns[2]][i]) 

               

    return data, train_data



def read_test_data(filename):

    """

      Extrai artigos até um determinado limite em um arquivo zip como uma lista de palavras

      e pré-processa usando a biblioteca nltk python

      """

        

    test_data = {}

    for i in range(filename.shape[0]):

        text_string = filename[filename.columns[1]][i]

        text_string = text_string.lower()

        text_string = nltk.word_tokenize(text_string)

        # Atribui a classe aos arquivos

        """ Atribui o tópico ao documento """

        test_data[filename[filename.columns[0]][i].astype(str)] = text_string

        print('\tConcluída a leitura de dados para o tópico: ',filename[filename.columns[0]][i].astype(str)) 

               

    return test_data



print('Processando dados de treinamento...\n')

words, train_words = read_data(bbc_train)



print('\nProcessando dados de teste...\n')



test_words = read_test_data(bbc_test)



#test_words = read_test_data(filename)



#list_test_words = list(map(tuple, test_words.items()))

#random.shuffle(list_test_words)

#test_words = dict(list_test_words)

vocabulary_size = 25000

Words = []

def build_dataset(words):

    for word in words[0]:

        Words.extend(word)    

    count = [['UNK', -1]]

    count.extend(collections.Counter(Words).most_common(vocabulary_size - 1))



    # Dicionário

    dictionary = dict()

    for word, _ in count:

        dictionary[word] = len(dictionary)

    

    data = list()

    unk_count = 0

    

    for word in Words:

        if word in dictionary:

            index = dictionary[word]

        else:

            index = 0  # dictionary['UNK']            

            unk_count = unk_count + 1

            

        data.append(word)



    count[0][1] = unk_count

    

    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 

    assert len(dictionary) == vocabulary_size



    return data, count, dictionary, reverse_dictionary



def build_dataset_with_existing_dictionary(words, dictionary):

    '''

    Aqui usamos essa função para converter strings de palavras em IDs com um determinado dicionário

    '''

    data = list()

    for word in words:

        if word in dictionary:

            index = dictionary[word]

        else:

            index = 0  # dictionary['UNK']

        data.append(word)

    return data



# Processando dados de treino

data, count, dictionary, reverse_dictionary = build_dataset(words)



train_data = {}



for k,v in train_words.items():

    print('Construindo o dataset de treino para o documento ', k)

    train_data[k] = build_dataset_with_existing_dictionary(train_words[k],dictionary)



# Processando dados de teste



test_data = {}



for k,v in test_words.items():

    print('Construindo o dataset de teste para o documento ', k)

    test_data[k] = build_dataset_with_existing_dictionary(test_words[k],dictionary)

    

print('\nPalavras mais comuns (+UNK)', count[:5])

print('\nAmostra de dados', data[:10])

print('\nChaves: ', test_data.keys())

print('\nItems: ', test_data.items())



# Removemos para liberar memória no computador. Não precisamos mais desses objetos.

del words  

#del test_words
# Converte de Dicionário para lista

data_train = [ [k,v] for k, v in train_data.items() ]

data_train = np.array(data_train)

data_test = [ [k,v] for k, v in test_data.items() ]

data_test = np.array(data_test)
# função para Identificar as classes dos documentos e preparando os dados para o algoritimo

def prepara_dados(data):

    datax = [[],[]]

    for x in range(data.shape[0]):

        s = data[x][0]

        s = s.split("-")

        datax[0].append(s[0])

        datax[1].append(data[x][1])                

    return datax



data_train = prepara_dados(data_train)

data_test = prepara_dados(data_test)
def label_sentences(corpus, topics):

    """

    A implementação do Doc2Vec da Gensim exige que cada documento / parágrafo tenha um rótulo associado a ele.

    Fiz isso usando o método TaggedDocument, etiquetando com a própria classe do documento.

    """

   

    labeled = []

    tags = np.unique(topics, return_counts=False)

    for i, v in enumerate(corpus):

        label = [s for s in tags if topics[i] in s]

        doc =  " ".join(str(x) for x in v)

        labeled.append(TaggedDocument(doc.split(), label))

    return labeled

X_train = label_sentences(np.array(data_train[1]), data_train[0])

X_test  = label_sentences(np.array(data_test[1]), data_test[0])
len(X_train)
len(X_test)
Image(url = 'Doc2Vec.png')
# Instanciando um modelo Doc2Vec com um vetor de 128 palavras



model_dbow = Doc2Vec(dm=0, vector_size=128, window=10, negative=5, cbow_mean=1, min_count=1, alpha=0.1, min_alpha=0.005)

model_dbow.build_vocab([x for x in tqdm(X_train)])





# Alicando 50 iterações sobre o corpus de treinamento.



for epoch in range(50):

    model_dbow.train(utils.shuffle([x for x in tqdm(X_train)]), total_examples=len(X_train), epochs=10)

    model_dbow.alpha -= 0.002

    model_dbow.min_alpha = model_dbow.alpha
%%time



# Dados de treino



train_targets, train_regressors = zip(

    *[(doc.tags[0], model_dbow.infer_vector(doc.words, alpha=0.1, min_alpha=0.005, epochs=100)) for doc in X_train])
%%time



# Dados de teste



test_targets, test_regressors = zip(

    *[(doc.tags[0], model_dbow.infer_vector(doc.words, alpha=0.1, min_alpha=0.005, epochs=100)) for doc in X_test])
# Aplicando de Regressão Logistica

logreg = LogisticRegression(n_jobs=1, C=1e5)

logreg.fit(train_regressors, train_targets)

y_pred = logreg.predict(test_regressors)

y_pred.shape[0]
len(test_targets)
bbc_test['Category'] = y_pred
bbc_test.drop(['Text'], axis = 1, inplace = True)
# Saving the dataset with the transformations



bbc_test.to_csv('./BBC News Sample Solution.csv', index=False)