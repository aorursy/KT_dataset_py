import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input/fasam-nlp-competition-turma-4"))
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

import keras

from keras.models import Model

from keras.layers import *

from keras.optimizers import *

from keras.losses import *

from keras.regularizers import *

from keras.models import Sequential

from keras.callbacks import *

from keras.preprocessing import sequence

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer

from keras.layers.embeddings import Embedding

from sklearn.model_selection import train_test_split
# Leitura do Dataset

df = pd.read_csv('../input/fasam-nlp-competition-turma-4/train.csv')

print(df.shape)

df.head()
# Definição de alguns parâmetros dos modelos e tokenização



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
Y_classes
Y
(X.shape, Y.shape)
def create_model():

    model = Sequential()

    

    # Embedding Layer

    model.add(Embedding(max_tokens, embed_dim, 

                        input_length = seq_size))

    # RNN Layer

    model.add(LSTM(100))

    

    # Dense Layer

    model.add(Dense(len(Y_classes), activation='softmax'))

    

    model.compile(loss = 'categorical_crossentropy', 

                  optimizer='adam',

                  metrics = ['accuracy'])

    

    model.summary()

    

    return model



model = create_model()
# Separa o dataset em dados de treinamento/validação

X_train, X_valid, Y_train, Y_valid = train_test_split(X,Y, 

                                                      test_size = 0.20, 

                                                      random_state = 42,

                                                      stratify=Y)



weights_filepath = "weights.h5"

callbacks = [ModelCheckpoint(weights_filepath, monitor='val_loss', mode='min',

                             verbose=1, save_best_only=True),

             EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)]



# Treina o modelo

hist = model.fit(X_train, Y_train, 

                 validation_data =(X_valid, Y_valid),

                 batch_size=300, nb_epoch = 100,  verbose = 1, callbacks=callbacks)

model.load_weights(weights_filepath)
plt.plot(hist.history['accuracy'])

plt.plot(hist.history['val_accuracy'])

plt.ylabel('acurácia')

plt.xlabel('época')

plt.legend(['treino', 'validação'], loc = 'upper left')

plt.show()



plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.ylabel('loss')

plt.xlabel('época')

plt.legend(['treino', 'validação'], loc = 'upper left')

plt.show()
# Avaliação do modelo para o dataset de validação



val_loss, val_acc = model.evaluate(X_valid, Y_valid)



print('A acurácia do modelo está de: '+str(val_acc*100)+'%')
# Leitura do Dataset de validação dos resultados

test_df = pd.read_csv('../input/fasam-nlp-competition-turma-4/test.csv')

print(test_df.shape)

test_df.head()
def predict(text):

    '''

    Utiliza o modelo treinado para realizar a predição

    '''

    new_text = tokenizer.texts_to_sequences(text)

    new_text = pad_sequences(new_text, maxlen=seq_size)

    pred     = model.predict_classes(new_text)#[0]

    return pred
# Como utilizamos o titulo no treinamento, iremos utilizar o titulo na predição também



pred         = predict(test_df.title)

pred_classes = [Y_classes[c] for c in pred]

pred_classes[:5]
# Atualizando a categoria dos artigos no dataset de validação

test_df['category'] = pred_classes

test_df.head()
# Criando o arquivo submission.csv contendo os dados para cálculo do ranking no kaggle

# Esse arquivo deve ser enviado para o kaggle

test_df[["article_id", "category"]].to_csv("submission.csv", index=False)