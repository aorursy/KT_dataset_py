# Import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import string

from collections import defaultdict

import csv

import re

from tabulate import tabulate

import tensorflow as tf

from tensorflow import keras

from tensorflow_core.python.keras.utils.data_utils import Sequence

from tensorflow.keras.utils import Sequence
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python









# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

sub_sample = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")

train = pd.read_csv("../input/nlp-getting-started/train.csv")
print (train.shape, test.shape, sub_sample.shape)
train.head()
train.tail()
train.duplicated().sum()
# Class balance

# train.target.value_counts()

sns.countplot(y=train.target);
# NA data

train.isnull().sum()
test.isnull().sum()
# Check number of unique keywords, and whether they are the same for train and test sets

print (train.keyword.nunique(), test.keyword.nunique())

print (set(train.keyword.unique()) - set(test.keyword.unique()))
plt.figure(figsize=(9,6))

sns.countplot(y=train.keyword, order = train.keyword.value_counts().iloc[:10].index)

plt.title('Top 10 keywords')

plt.show()
kw_d = train[train.target==1].keyword.value_counts().head(10)

kw_nd = train[train.target==0].keyword.value_counts().head(10)



plt.figure(figsize=(13,5))

plt.subplot(121)

sns.barplot(kw_d, kw_d.index, color='c')

plt.title('Top keywords for disaster tweets')

plt.subplot(122)

sns.barplot(kw_nd, kw_nd.index, color='y')

plt.title('Top keywords for non-disaster tweets')

plt.show()
print (train.location.nunique(), test.location.nunique())
# Most common locations

plt.figure(figsize=(9,6))

sns.countplot(y=train.location, order = train.location.value_counts().iloc[:15].index)

plt.title('Top 15 locations')

plt.show()
import re



test_str = train.loc[417, 'text']



def clean_text(text):

    text = re.sub(r'https?://\S+', '', text) # Remove link

    text = re.sub(r'\n',' ', text) # Remove line breaks

    text = re.sub('\s+', ' ', text).strip() # Remove leading, trailing, and extra spaces

    return text



print("Original text: " + test_str)

print("Cleaned text: " + clean_text(test_str))
def find_hashtags(tweet):

    return " ".join([match.group(0)[1:] for match in re.finditer(r"#\w+", tweet)]) or 'no'



def find_mentions(tweet):

    return " ".join([match.group(0)[1:] for match in re.finditer(r"@\w+", tweet)]) or 'no'



def find_links(tweet):

    return " ".join([match.group(0)[:] for match in re.finditer(r"https?://\S+", tweet)]) or 'no'



def process_text(df):

    

    df['text_clean'] = df['text'].apply(lambda x: clean_text(x))

    df['hashtags'] = df['text'].apply(lambda x: find_hashtags(x))

    df['mentions'] = df['text'].apply(lambda x: find_mentions(x))

    df['links'] = df['text'].apply(lambda x: find_links(x))

    # df['hashtags'].fillna(value='no', inplace=True)

    # df['mentions'].fillna(value='no', inplace=True)

    

    return df

    

train = process_text(train)

test = process_text(test)
train.head()
cuvinte_text = []

for i in range(0,7613):

    cuvinte_text.append(train['text_clean'][i].split())
contor_cuvinte = defaultdict(int)



for doc in cuvinte_text:

    for word in doc:

        contor_cuvinte[word] += 1



PRIMELE_N_CUVINTE = 1000

        

# transformam dictionarul in lista de tupluri ['cuvant1', frecventa1, 'cuvant2': frecventa2]

perechi_cuvinte_frecventa = list(contor_cuvinte.items())



# sortam descrescator lista de tupluri dupa frecventa

perechi_cuvinte_frecventa = sorted(perechi_cuvinte_frecventa, key=lambda kv: kv[1], reverse=True)



# extragem primele 1000 cele mai frecvente cuvinte din toate textele

perechi_cuvinte_frecventa = perechi_cuvinte_frecventa[0:PRIMELE_N_CUVINTE]



print ("Primele 10 cele mai frecvente cuvinte ", perechi_cuvinte_frecventa[0:10])
list_of_selected_words = []

for cuvant, frecventa in perechi_cuvinte_frecventa:

    list_of_selected_words.append(cuvant)

### numaram cuvintele din toate documentele ###
def get_bow(text, lista_de_cuvinte):

    '''

    returneaza BoW corespunzator unui text impartit in cuvinte

    in functie de lista de cuvinte selectate

    '''

    contor = dict()

    cuvinte = set(lista_de_cuvinte)

    for cuvant in cuvinte:

        contor[cuvant] = 0

    for cuvant in text:

        if cuvant in cuvinte:

            contor[cuvant] += 1

    return contor
def get_bow_pe_corpus(corpus, lista):

    '''

    returneaza BoW normalizat

    corespunzator pentru un intreg set de texte

    sub forma de matrice np.array

    '''

    bow = np.zeros((len(corpus), len(lista)))

    for idx, doc in enumerate(corpus):

        bow_dict = get_bow(doc, lista)

        ''' 

            bow e dictionar.

            bow.values() e un obiect de tipul dict_values 

            care contine valorile dictionarului

            trebuie convertit in lista apoi in numpy.array

        '''

        v = np.array(list(bow_dict.values()))

        #v = v / np.sqrt(np.sum(v ** 2))

        bow[idx] = v

    return bow
data_bow = get_bow_pe_corpus(cuvinte_text, list_of_selected_words)

print ("Data bow are shape: ", data_bow.shape)
train.head()
x_val = data_bow[:1000]

x_train = data_bow[1000:]



y_val = train['target'][:1000]

y_train = train['target'][1000:]
z_train = []

for i in range(1000,7613):

    z_train.append(y_train[i])


z_val = []

for i in range(0,1000):

    z_val.append(y_val[i])
z_train_array = np.asarray(z_train)

z_val_array = np.asarray(z_val)
model = keras.Sequential([

    keras.layers.Dense(512, input_shape=(1000,), activation='relu'),

    keras.layers.Dense(256,input_shape=(512,), activation='relu'),

    keras.layers.Dense(16,input_shape=(256,), activation='relu'),

    keras.layers.Dense(1, activation='sigmoid'),

    keras.layers.Dropout(0.1)])
model.compile(optimizer='adam',

              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              metrics=['accuracy'])
fitmodel = model.fit(x_train, z_train_array, epochs = 50, batch_size = 512, validation_data = (x_val, z_val_array), verbose = 1)
model.summary()
cuvinte_text_pred = []

for i in range(0,3263):

    cuvinte_text_pred.append(test['text'][i].split())
test_data = get_bow_pe_corpus(cuvinte_text_pred, list_of_selected_words)
predict = model.predict(test_data)
pred = []

for i in range (0,3263):

    pred.append(predict[i][0])
norm_pred = []

for i in range (0,3263):

    if pred[i] > 0.5:

        norm_pred.append(1)

    else:

        norm_pred.append(0)

submission = pd.DataFrame({

        "id": test["id"],

        "target": norm_pred

    })
submission.to_csv("submission.csv",index=False)