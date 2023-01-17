from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
base = pd.read_csv('../input/PRSA_data_2010.1.1-2014.12.31.csv')
base.head()
# Deletion of records with unfilled values (missing values)
base = base.dropna()
base = base.drop('No', axis = 1)
base = base.drop('year', axis = 1)
base = base.drop('month', axis = 1)
base = base.drop('day', axis = 1)
base = base.drop('hour', axis = 1)
base = base.drop('cbwd', axis = 1)
# Predictive attributes are all but not index 0
base_treinamento = base.iloc[:, 1:7].values
base.head()
poluicao = base.iloc[:, 0].values
# Application of normalization
normalizador = MinMaxScaler(feature_range = (0, 1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)
# Need to change the format of the variable to apply normalization
poluicao = poluicao.reshape(-1, 1)
poluicao_normalizado = normalizador.fit_transform(poluicao)
#### Creation of the data structure that represents the time series, considering
#### 10 hours (window) earlier to predict the current time
previsores = []
poluicao_real = []
for i in range(10, 41757):
    previsores.append(base_treinamento_normalizada[i-10:i, 0:6])
    poluicao_real.append(poluicao_normalizado[i, 0])
previsores, poluicao_real = np.array(previsores), np.array(poluicao_real)

regressor = Sequential()
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores.shape[1], 6)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1, activation = 'linear'))

regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error', 
                  metrics = ['mean_absolute_error'])
regressor.fit(previsores, poluicao_real, epochs = 100, batch_size = 64)
previsoes = regressor.predict(previsores)
previsoes = normalizador.inverse_transform(previsoes)
print('Previsoes', previsoes.mean())
print('Poluicao', poluicao.mean())
plt.figure(figsize=(16,12))
plt.plot(poluicao, color = 'red', label = 'Real pollution')
plt.plot(previsoes, color = 'blue', label = 'Predictions')
plt.title('Pollution forecast')
plt.xlabel('Hours')
plt.ylabel('Pollution value')
plt.legend()
