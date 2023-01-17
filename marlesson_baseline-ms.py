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
import numpy as np

import pandas as pd

import seaborn as sns

import warnings

import matplotlib

import matplotlib.pyplot as plt



%matplotlib inline



sns.set(style="ticks")

warnings.filterwarnings("ignore")
# Bibliotecas do keras

from keras.preprocessing import sequence

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer

from keras.models import Sequential

from keras.layers import Dense, LSTM, Flatten

from keras.layers.embeddings import Embedding

from sklearn.model_selection import train_test_split
# Leitura do Dataset

df = pd.read_csv('/kaggle/input/e-commerce-reviews/train.csv')

print(df.shape)

df.head()
df.info()
df = df.dropna()

df.shape
## Definição de alguns parâmetros dos modelos e tokenização



# Tamanho da sequencia

seq_size     = 5



# Máximo de tokens 

max_tokens   = 100



# Tamanho do embedding

embed_dim    = 5
## Utilizaremos apenas o .review_comment_message (input) e o .review_score (target) da nossa rede

# Textos

text         = df['review_comment_message'].values

tokenizer    = Tokenizer(num_words=max_tokens, split=' ')



# Transforma o texto em números

tokenizer.fit_on_texts(text)

X = tokenizer.texts_to_sequences(text)  



# Cria sequencias de tamanho fixo (input: X)

X = pad_sequences(X, maxlen=seq_size)



# Target (review_score)

Y = df['review_score'].values
def base_model():

    model = Sequential()

    

    # Embedding Layer

    model.add(Embedding(max_tokens, embed_dim, 

                        input_length = seq_size))

    # RNN Layer

    model.add(LSTM(seq_size))

    

    # Dense Layer

    model.add(Dense(1))

    

    model.compile(loss = 'mse', optimizer='sgd')

    

    model.summary()

    

    return model



base_model = base_model()
# Separa o dataset em dados de treinamento/teste

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size = 0.20, random_state = 42)



# Treina o modelo

hist = base_model.fit(X_train, Y_train, 

              validation_data =(X_valid, Y_valid),

              batch_size=1000, nb_epoch = 10,  verbose = 1)
plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])
# Leitura do Dataset de validação dos resultados

df_valid = pd.read_csv('/kaggle/input/e-commerce-reviews/test.csv')

print(df_valid.shape)

df_valid.head()
df_valid.shape
def predict(text):

    '''

    Utiliza o modelo treinado para realizar a predição

    '''

    new_text = tokenizer.texts_to_sequences(text)

    new_text = pad_sequences(new_text, maxlen=seq_size)

    pred     = base_model.predict(new_text)#[0]

    return pred
# Como utilizamos o titulo no treinamento, iremos utilizar o titulo na predição também

pred     = predict(df_valid.review_comment_message)

pred     = pred[:,0]

pred[:5] 
# Atualizando a categoria dos artigos no dataset de validação

df_valid['review_score'] = pred

df_valid.head()
df_valid[['review_id', 'review_score']].to_csv('submission_test.csv', index=False)