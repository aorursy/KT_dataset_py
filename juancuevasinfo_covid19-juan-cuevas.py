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
import pandas as pd 

import numpy as np

import matplotlib.pylab as plt

%matplotlib inline

plt.rcParams['figure.figsize'] = (16, 9)

plt.style.use('fast')



from keras.models import Sequential

from keras.layers import Dense,Activation,Flatten

from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv('/kaggle/input/confirmado-p-3csv/confirmado_p_3.csv')
df.describe()
df.head()
paises = ("lat318257long1172264", "lat401824long1164142", "lat300572long107874", "lat360611long1038343","lat233417long1034244")

posicion_y = np.arange(len(paises))

unidades = (990, 429, 576, 296,125)

plt.barh(posicion_y, unidades, align = "center")

plt.yticks(posicion_y, paises)

plt.xlabel('cantidad_infectados_covid-19')

plt.title("confirmados_covid_china_mailand")


PASOS=7

 



def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):

    n_vars = 1 if type(data) is list else data.shape[1]

    df = pd.DataFrame(data)

    cols, names = list(), list()

    # input sequence (t-n, ... t-1)

    for i in range(n_in, 0, -1):

        cols.append(df.shift(i))

        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    # forecast seque (t, t+1, ... t+n)

    for i in range(0, n_out):

        cols.append(df.shift(-i))

        if i == 0:

            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]

        else:

            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    

    agg = pd.concat(cols, axis=1)

    agg.columns = names

    

    if dropnan:

        agg.dropna(inplace=True)

    return agg

 

# load dataset

values = df.values

# ensure all data is float

values = values.astype('float32')

# normalize features

scaler = MinMaxScaler(feature_range=(-1, 1))

values=values.reshape(-1, 1) # esto lo hacemos porque tenemos 1 sola dimension

scaled = scaler.fit_transform(values)

# frame as supervised learning

reframed = series_to_supervised(scaled, PASOS, 1)

reframed.head()
values = reframed.values

n_train_days = 315+289 - (30+PASOS)

train = values[:n_train_days, :]

test = values[n_train_days:, :]

# split into input and outputs

x_train, y_train = train[:, :-1], train[:, -1]

x_val, y_val = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]

x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))

x_val = x_val.reshape((x_val.shape[0], 1, x_val.shape[1]))

print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
PASOS=7

 

# convert series to supervised learning

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):

    n_vars = 1 if type(data) is list else data.shape[1]

    df = pd.DataFrame(data)

    cols, names = list(), list()

    # input sequence (t-n, ... t-1)

    for i in range(n_in, 0, -1):

        cols.append(df.shift(i))

        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)

    for i in range(0, n_out):

        cols.append(df.shift(-i))

        if i == 0:

            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]

        else:

            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    # put it all together

    agg = pd.concat(cols, axis=1)

    agg.columns = names

    # drop rows with NaN values

    if dropnan:

        agg.dropna(inplace=True)

    return agg

 

# load dataset

values = df.values

# ensure all data is float

values = values.astype('float32')

# normalize features

scaler = MinMaxScaler(feature_range=(-1, 1))

values=values.reshape(-1, 1) # esto lo hacemos porque tenemos 1 sola dimension

scaled = scaler.fit_transform(values)

# frame as supervised learning

reframed = series_to_supervised(scaled, PASOS, 1)

reframed.head()
def crear_modeloFF():

    model = Sequential() 

    model.add(Dense(PASOS, input_shape=(1,PASOS),activation='tanh'))

    model.add(Flatten())

    model.add(Dense(1, activation='tanh'))

    model.compile(loss='mean_absolute_error',optimizer='Adam',metrics=["mse"])

    model.summary()

    return model
EPOCHS=5000

 

model = crear_modeloFF()

 

history=model.fit(x_train,y_train,epochs=EPOCHS,validation_data=(x_val,y_val),batch_size=PASOS)


results=model.predict(x_val)

plt.scatter(range(len(y_val)),y_val,c='g')

plt.scatter(range(len(results)),results,c='r')

plt.title('validate')

plt.show()
ultimosDias = df["2/29/20"],[ "3/10/20"]

ultimosDias
values=values.reshape(-1, 1) 

scaled = scaler.fit_transform(values)

reframed = series_to_supervised(scaled, PASOS, 1)

reframed.drop(reframed.columns[[7]], axis=1, inplace=True)

reframed.head(7)
values = reframed.values

x_test = values[6:, :]

x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

x_test
def agregarNuevoValor(x_test,nuevoValor):

    for i in range(x_test.shape[2]-1):

        x_test[0][0][i] = x_test[0][0][i+1]

    x_test[0][0][x_test.shape[2]-1]=nuevoValor

    return x_test

 

results=[]

for i in range(7):

    parcial=model.predict(x_test)

    results.append(parcial[0])

    print(x_test)

    x_test=agregarNuevoValor(x_test,parcial[0])
adimen = [x for x in results]    

inverted = scaler.inverse_transform(adimen)

inverted
prediccion1 = pd.DataFrame(inverted)

prediccion1.columns = ['pronostico']

prediccion1.plot()

prediccion1.to_csv('pronostico.csv')