import numpy as np
import keras as K
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from keras.utils import to_categorical

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
#Carregamento dos dados 
df_price = pd.read_csv('../input/extractions/coffee_price.csv')
#df_station31 = pd.read_csv('')
#df_station32 = pd.read_csv('')
#df_station36 = pd.read_csv('')
df_price.head()
df_price.describe()
print('Periodo correspondente ao dataset inicial de precificação:',df_price.date.min(),' à ',
df_price.date.max())
df_price.plot()
#adicionar plotagem de numeros de registros por ano para verificar se existe alguma inconsistencia de dados
# adicionar a analise sobre os dados faltantes de finais de semana
#adicionar o script da tratativa para esse caso
# adicionar o script utilizado para a classificação do dataset
df_station31.head()
df_station31.describe()
#adicionar o grafico que exemplifica as lacunas de medições dos periodos
#Explicar o porque escolhemos trabalhar com apenas 3 estações
#adicionar o script com a tratativa do preenchimento das lacunas das bases de dados metereológicos
#plotar o grafico comparativo de antes e depois da tratativa dos dados 
#Carregamento dos dados 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

sc = MinMaxScaler()
oh = OneHotEncoder()

price = pd.read_csv('../input/coffeevsweather/price_status.csv')
station31 = pd.read_csv('../input/coffeevsweather/training_data_beta_31.csv')
station32 = pd.read_csv('../input/coffeevsweather/training_data_beta_32.csv')
station36 = pd.read_csv('../input/coffeevsweather/training_data_beta_36.csv')
def create_dataset(X, Y, look_back):
    dataX, dataY = [], []
    for i in range(len(Y)-look_back-1):
        a = X[i:(i+look_back)]
        dataX.append(a.T)
        b = Y[(i+look_back)]
        dataY.append(b.T)
    return np.array(dataX), np.array(dataY)
### Input Preparations
heading = ['Precipitacao', 'TempMaxima', 'TempMinima', 'Insolacao', 'Evaporacao Piche', 'Temp Comp Media', 'Umidade Relativa Media', 'Velocidade do Vento Media']

df_full = pd.DataFrame([], columns=heading)
for col in heading:
    x = station31[col].values.astype(float)
    df_full[col] = sc.fit_transform(np.expand_dims(x, -1)).T[0]
    x = station32[col].values.astype(float)
    df_full[col+'2'] = sc.fit_transform(np.expand_dims(x, -1)).T[0]
    x = station36[col].values.astype(float)
    df_full[col+'3'] = sc.fit_transform(np.expand_dims(x, -1)).T[0]
    
X_full = df_full.to_numpy()

df_full.head()
price.head()
import seaborn as sns
sns.relplot(x="Status", y="Valor", data=price);
### Output Preparations
Y =sc.fit_transform(np.expand_dims(price['Valor'], -1)).T[0] ### Value
print(Y)
import matplotlib.pyplot as plt
plt.plot(price['Valor'])
plt.ylabel('some numbers')
plt.show()
X_lstm, Y_lstm = create_dataset(X_full, Y, 21)
#np.expand_dims(X_lstm,-1).shape

division = len(df_full) - 100

x_train = X_lstm[0:division]
x_test = X_lstm[division:]

Y_train =  Y_lstm[0:division]
Y_test = Y_lstm[division:]


input_shape = X_lstm[1].shape
inputA = Input(shape = input_shape)

#x = LSTM(2, input_shape = input_shape)(inputA) ### Camada de LSTM, 2 memórias
#x = LSTM(7, input_shape = input_shape)(inputA) ### Camada de LSTM, 7 memórias
x = LSTM(21, input_shape = input_shape)(inputA) ### Camada de LSTM, 15 memórias

############## Output setting
### Boolean Up vs Down
#z = Dense(2, activation = 'sigmoid')(x)        ### Camada densa de classificação
#model = Model(inputs=[inputA], outputs=z)
#model.compile( loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.summary()

### Numeric Value 
z = Dense(1)(x)
model = Model(inputs=[inputA], outputs=z)
model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
model.summary()



batch_size = 91
max_epochs = 50

h = model.fit(x = x_train, y = Y_train, batch_size= batch_size, epochs= max_epochs, verbose=1)

eval_test1 = model.evaluate(x_test, Y_test, verbose=0)
#print("Erro médio do teste: Perda {0:.4f}, acuracia {1:.4f}".format(eval_test1[0], eval_test1[1]*100))
print("Erro médio do teste: Acuracia {0:.4f}".format(eval_test1[1]*100))
