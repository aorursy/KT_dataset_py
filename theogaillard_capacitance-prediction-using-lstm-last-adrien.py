import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from numpy import array

from keras.preprocessing.text import one_hot

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers.core import Activation, Dropout, Dense

from keras.layers import Flatten, LSTM

from keras.layers import GlobalMaxPooling1D

from keras.models import Model

from keras.layers.embeddings import Embedding

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer

from keras.layers import Input

from keras.layers.merge import Concatenate

from keras.layers import Bidirectional



import pandas as pd

import numpy as np

import re



import matplotlib.pyplot as plt



from sklearn.metrics import accuracy_score, confusion_matrix
df_capa = pd.read_csv('/kaggle/input/data-capa-total/dataset_total.csv', encoding="windows-1252", sep=';')
df_capa.head()
X1 = list()

X2 = list()

X3 = list()

Y = list()



X1 = df_capa[df_capa.columns[0]] # pour recuperer toutes les données de la premiere colonne (time)

X2 = df_capa[df_capa.columns[1]] # pour recuperer toutes les données de la deuxieme colonne (tension_entree)

X3 = df_capa[df_capa.columns[3]] # pour recuperer toutes les données de la quatrième colonne (valeur_capa)



Y = df_capa[df_capa.columns[2]] # pour recuperer toutes les données de la troisieme colonne (tension_capa), c'est la valeur qu'on cherche à prédire

print(X1,X2,X3,Y)
X = list()

X = np.column_stack((X1, X2, X3)) # on regroupe dans une même liste toutes les données d'entrées



print(X)
X = array(X).reshape(50005, 1, 3) # on reshape pour avoir 10001 lignes d'1 seul vecteur qui contient 3 valeurs



print(X)
from keras.layers import Bidirectional



model = Sequential()

model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(1, 3)))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

print(model.summary())
model.fit(X, Y, epochs=15, validation_split=0.2, batch_size=5)
time_for_predict_tension_capa = 0.6 # on essaye de predict la tension_capa au bout de 0.06s

tension_entree_for_predict_tension_capa = 3 # on essaye de predict la tension_capa avec une tension d'entrée de 1V

valeur_capa_for_predict_tension_capa = 0.001 # on essaye de predict la tension_capa avec une valeur de capa de 0.001
test_input = []



for i in range(0,10001) :

    test_input.append([X1[i], tension_entree_for_predict_tension_capa,valeur_capa_for_predict_tension_capa])

test_input_array = array(test_input)



test_input_array = test_input_array.reshape((10001, 1, 3))

test_output = model.predict(test_input_array, verbose=0)

print(test_output)



output = []

for i in test_output:

    output.append(i[0])
data_test = pd.read_csv('/kaggle/input/verification/tension3capa1.csv', encoding="windows-1252", sep=';')

Y_test = data_test[data_test.columns[3]]

import plotly.express as px

import plotly.graph_objects as go



fig2 = go.Figure()

fig2.add_trace(go.Scatter(x=X1,

                         y=Y_test,

                         mode='lines',

                         name='Real'))
fig2.add_trace(go.Scatter(x=X1,

                         y=output,

                         mode='lines',

                         name='Predict'))
import sklearn.metrics as metrics

metrics.r2_score(output,Y_test)