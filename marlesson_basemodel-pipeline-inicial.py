import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
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

from keras.layers import Dense, LSTM

from keras.layers.embeddings import Embedding

from sklearn.model_selection import train_test_split
# Leitura do Dataset

df = pd.read_csv('../input/df_train.csv')

print(df.shape)

df.head()
## Definição de alguns parâmetros dos modelos e tokenização



# Tamanho da sequencia

seq_size     = 10



# Máximo de tokens 

max_tokens   = 2500



# Tamanho do embedding

embed_dim    = 128
## Utilizaremos apenas o .title (input) e o .category (target) da nossa rede

# Textos

text         = df['title'].values

tokenizer    = Tokenizer(num_words=max_tokens, split=' ')



# Transforma o texto em números

tokenizer.fit_on_texts(text)

X = tokenizer.texts_to_sequences(text)  



# Cria sequencias de tamanho fixo (input: X)

X = pad_sequences(X, maxlen=seq_size)
# Categoriza o target "category" -> [0,..., 1] (output: y)

Y_classes = pd.get_dummies(df['category']).columns

Y         = pd.get_dummies(df['category']).values
(X.shape, Y.shape)
def base_model():

    model = Sequential()

    

    # Embedding Layer

    model.add(Embedding(max_tokens, embed_dim, 

                        input_length = seq_size))

    # RNN Layer

    model.add(LSTM(seq_size))

    

    # Dense Layer

    model.add(Dense(len(Y_classes), activation='softmax'))

    

    model.compile(loss = 'categorical_crossentropy', 

                  optimizer='adam',

                  metrics = ['accuracy'])

    

    model.summary()

    

    return model



base_model = base_model()
# Separa o dataset em dados de treinamento/teste

X_train, X_valid, Y_train, Y_valid = train_test_split(X,Y, 

                                                      test_size = 0.20, 

                                                      random_state = 42)



# Treina o modelo

hist = base_model.fit(X_train, Y_train, 

              validation_data =(X_valid, Y_valid),

              batch_size=300, nb_epoch = 3,  verbose = 1)
# Avaliação do modelo para o dataset de test



val_loss, val_acc = base_model.evaluate(X_valid, Y_valid)



print('A acurácia do modelo está de: '+str(val_acc*100)+'%')
# Leitura do Dataset de validação dos resultados

df_valid = pd.read_csv('../input/df_valid.csv')

print(df_valid.shape)

df_valid.head()
def predict(text):

    '''

    Utiliza o modelo treinado para realizar a predição

    '''

    new_text = tokenizer.texts_to_sequences(text)

    new_text = pad_sequences(new_text, maxlen=seq_size)

    pred     = base_model.predict_classes(new_text)#[0]

    return pred
# Como utilizamos o titulo no treinamento, iremos utilizar o titulo na predição também



pred         = predict(df_valid.title)

pred_classes = [Y_classes[c] for c in pred]

pred_classes[:5]
# Atualizando a categoria dos artigos no dataset de validação

df_valid['category'] = pred_classes

df_valid.head()
def create_submission(df):

    f = open('submission_valid.csv', 'w')

    f.write('id,category\n')

    for i, row in df.iterrows():

        f.write('{},{}\n'.format(i, row.category))

    f.close()

    

# Criando o arquivo submission_valid.csv contendo os dados para cálculo do raning no kaggle

# Esse arquivo deve ser enviado para o kaggle

create_submission(df_valid)    