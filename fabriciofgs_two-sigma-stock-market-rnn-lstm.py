import pandas as pd
import numpy as np
from kaggle.competitions import twosigmanews
#importando os dados
env = twosigmanews.make_env() # cria o enviroment  - Deve ser executado somente uma vez
type(env)
# Convertendo e separando os dados
(market_train, news_train) = env.get_training_data() # dados do tipo dataFrame
import matplotlib.pylab as plt
fig,axes = plt.subplots(1,1,figsize=(15,10))
axes.set_title("Time Distro")
axes.set_ylabel("# of records")
axes.set_xlabel("date")
axes.plot(market_train.time.dt.date.value_counts().sort_index().index, market_train.time.dt.date.value_counts().sort_index().values)
market_train.head()
market_train.shape
news_train.head(3)
training_set = market_train.iloc[:, 5].values
training_set = training_set.reshape(-1, 1)
fig,axes = plt.subplots(1,1,figsize=(15,10))
axes.set_title("Time Serie")
axes.set_ylabel("Open variation")
axes.set_xlabel("date")
axes.plot(market_train.time.dt.date, training_set)
# Dividindo o data set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(market_train, training_set, test_size = 0.3, random_state = 0)
X_train.shape
y_train.shape
len(X_train)
# Creating a data structure with 10 timesteps and 1 output
X_train_ts = []
y_train_ts = []
for i in range(10, len(X_train)):
    X_train_ts.append(X_train.iloc[i-10:i, 0].values)
    y_train_ts.append(y_train[i, 0])
X_train_ts, y_train_ts = np.array(X_train_ts), np.array(y_train_ts)

# Reshaping
X_train_ts = np.reshape(X_train_ts, (X_train_ts.shape[0], X_train_ts.shape[1], 1))
# Construção da RNN
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import LSTM
from keras.layers import Dropout
# Inicializando o RNN
regressor = Sequential()

# Adicionando a primeira LSTM layer e Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train_ts.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adicionando a segunda LSTM layer e Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#colcando em dimensao 2D
regressor.add(Flatten())

#adicionando a camada de saida
regressor.add(Dense(units = 1))

#compilando a RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
# Fitting a RNN ao Training set
regressor.fit(X_train, y_train, epochs = 20, batch_size = 32)