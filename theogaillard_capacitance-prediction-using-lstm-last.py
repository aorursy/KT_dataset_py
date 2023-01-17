# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
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
df_capa = pd.read_csv('/kaggle/input/simple-dataset/capa1e-3tension1.csv', encoding="windows-1252", sep=';')
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
X = array(X).reshape(10001, 1, 3) # on reshape pour avoir 10001 lignes d'1 seul vecteur qui contient 3 valeurs



print(X)
model = Sequential()

model.add(LSTM(80, activation='relu', input_shape=(1, 3)))

model.add(Dense(10, activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

print(model.summary())
model.fit(X, Y, epochs=15, validation_split=0.2, batch_size=5)
time_for_predict_tension_capa = 0.06 # on essaye de predict la tension_capa au bout de 0.06s

tension_entree_for_predict_tension_capa = 1 # on essaye de predict la tension_capa avec une tension d'entrée de 1V

valeur_capa_for_predict_tension_capa = 0.001 # on essaye de predict la tension_capa avec une valeur de capa de 0.001
test_input = array([time_for_predict_tension_capa,tension_entree_for_predict_tension_capa,valeur_capa_for_predict_tension_capa])

test_input = test_input.reshape((1, 1, 3))



# on fait un predict sur notre valeur de test

test_output = model.predict(test_input, verbose=0)



# on affiche la valeur

print(test_output)
import plotly.express as px

import plotly.graph_objects as go



fig = go.Figure()

fig.add_trace(go.Scatter(x=X1,

                         y=Y,

                         mode='lines',

                         name='True'))
fig.add_trace(go.Scatter(x=[time_for_predict_tension_capa],

                         y=list(test_output[0]),

                         mode='lines+markers',

                         name='Predict'))
X1 = X1[:1500]

X2 = X2[:1500]

X3 = X3[:1500]



Y = Y[:1500]



print(X1,X2,X3,Y)
X = list()

X = np.column_stack((X1, X2, X3))

print(X)
X = array(X).reshape(1500, 1, 3)

print(X)
model = Sequential()

model.add(LSTM(500, activation='relu', return_sequences=True, input_shape=(1, 3)))

model.add(LSTM(200, activation='relu', return_sequences=True))

model.add(LSTM(120, activation='relu', return_sequences=True))

model.add(LSTM(50, activation='relu'))

model.add(Dense(20, activation='relu'))

model.add(Dense(10, activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

print(model.summary())
model.fit(X, Y, epochs=15, validation_split=0.2, batch_size=5)
time_for_predict_tension_capa = 0.16 # on essaye de predict la tension_capa au bout de 0.16s

tension_entree_for_predict_tension_capa = 1 # on essaye de predict la tension_capa avec une tension d'entrée de 1V

valeur_capa_for_predict_tension_capa = 0.001 # on essaye de predict la tension_capa avec une valeur de capa de 0.001
test_input = array([time_for_predict_tension_capa,tension_entree_for_predict_tension_capa,valeur_capa_for_predict_tension_capa])

test_input = test_input.reshape((1, 1, 3))



test_output = model.predict(test_input, verbose=0)

print(test_output)
import plotly.express as px

import plotly.graph_objects as go



fig = go.Figure()

fig.add_trace(go.Scatter(x=X1,

                            y=Y,

                            mode='lines',

                            name='True'))
Y = df_capa[df_capa.columns[2]] # pour recuperer toutes les données de la troisieme colonne (tension_capa), c'est la valeur qu'on cherche à prédire

print(Y[1600])
fig.add_trace(go.Scatter(x=[time_for_predict_tension_capa],

                        y=list([Y[1600]]),

                        mode='lines+markers',

                        name='Real'))
fig.add_trace(go.Scatter(x=[time_for_predict_tension_capa],

                        y=list(test_output[0]),

                        mode='lines+markers',

                        name='Predict'))
df_capa = pd.read_csv('/kaggle/input/data-capa-total/dataset_total.csv', encoding="windows-1252", sep=';')

df_capa.shape
X1 = df_capa[df_capa.columns[0]] #pour la colonne 1 (time)

X2 = df_capa[df_capa.columns[1]] #pour la colonne 2 (Step:1 (tension entrée))

X3 = df_capa[df_capa.columns[3]] #pour la colonne 2 (Step:1 (tension entrée))



Y = df_capa[df_capa.columns[2]] #pour la colonne 3 (Voltage Measurement:1 (tension instantanée capa))

print(X1,X2,X3,Y)
X = np.column_stack((X1, X2, X3))

print(X)
X = array(X).reshape(5, 10001, 3)

print(X)
model = Sequential()

model.add(LSTM(50, activation='relu', input_shape=(10001, 3)))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
history = model.fit(X, Y, epochs=10, validation_split=0.2, verbose=1)
test_input = array([0.2,5,0.001])

test_input = test_input.reshape((1, 1, 3))

test_output = model.predict(test_input, verbose=0)

print(test_output)
import plotly.express as px

import plotly.graph_objects as go



fig = go.Figure()

fig.add_trace(go.Scatter(x=X1[0:10001],

                            y=Y,

                            mode='lines',

                            name='True'))
fig.add_trace(go.Scatter(x=X1[10001:20002],

                            y=Y,

                            mode='lines',

                            name='True'))