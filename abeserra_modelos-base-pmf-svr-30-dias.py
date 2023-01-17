# This Python 3 environment comes with many helpful analytics libraries installed

import time



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# machine learning imports

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.svm import SVR



np.random.seed(42)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# 1. Cargar datos

df = pd.read_csv(os.path.join(dirname, 'sep_mo19.csv'), sep=';')

df.head()

df.info()
# 2. Quedarse con datos estación que queremos

estacion_cc = df[df['ESTACION'] == 38]
# 3. Filtrar por datos NO2

estacion_cc_NO2 = estacion_cc[(estacion_cc['MAGNITUD'] == 8)]

estacion_cc_NO2.head()
# 4. Añadir nueva columna fecha

estacion_cc_NO2['fecha']=pd.to_datetime(estacion_cc_NO2[['ANO','MES','DIA']].rename(columns = {'ANO': 'YEAR', 'MES': 'MONTH', 'DIA': 'DAY'}))
# 4. Extraer datos de medidas tomadas

data_columns = [ 'H01', 'H02', 'H03', 'H04', 'H05', 'H06', 'H07', 'H08', 'H09', 'H10', 'H11', 'H12', 'H13', 'H14', 'H15', 'H16', 'H17', 'H18', 'H19', 'H20', 'H21', 'H22', 'H23', 'H24'] # feature

veri_columns = [ 'V01', 'V02', 'V03', 'V04', 'V05', 'V06', 'V07', 'V08', 'V09', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24']



hora_column = np.arange(1,25) # feature label
data = pd.DataFrame(columns=['hora', 'dato', 'verificado'])

aux = pd.DataFrame(columns=['hora', 'dato', 'verificado'])

for  (i,d,v) in zip(hora_column, data_columns, veri_columns): 

     aux['dato'] = estacion_cc_NO2 [ data_columns[i-1] ] 

     aux['verificado'] = estacion_cc_NO2 [ veri_columns[i-1] ] 

     aux['hora'] = i 

     aux['fecha'] = estacion_cc_NO2 ['fecha']

     data = data.append(aux, sort = False) 



data.info()
data = data.sort_values(['fecha','hora'])

data.index = pd.timedelta_range(start='0 days', periods=720, freq='H')

data
# 5. MArcar datos no verificados como NaN

data.loc[data['verificado'] != 'V', ['dato']] = np.nan

data
# borramos columna verificado e index que ya no necesitamos

del data['verificado']

data.info()
# Visualizamos datos

from matplotlib import pyplot

data.dato.plot()

pyplot.show()
from sklearn.metrics import mean_squared_error

from math import sqrt

test_prop=0.2

X = data['dato'].values

test_len = round(len(X) * test_prop)

#test_len = 10

train, test = X[-test_len:], X[-test_len:]

# walk-forward validation

history = [x for x in train]

predictions = list()

for i in range(len(test)):

    # make prediction

    predictions.append(history[-1])

    # observation

    history.append(test[i])

# report performance

rmse = sqrt(mean_squared_error(test, predictions))

print('RMSE: %.3f' % rmse)

# line plot of observed vs predicted

pyplot.plot(test, label='Recorded data')

pyplot.plot(predictions, label='Predicted data')

pyplot.legend()

pyplot.show()
# Empezamos borrando los datos no validos

data = data.dropna(subset=['dato'])

len(data)
# Creamos conjunto de entrada desde los datos origen

X = data['dato'].values

#X
def _get_chunk(data, n_prev = 23):  

    """

    data should be pd.DataFrame()

    """



    docX, docY = [], []

    for i in range(len(data)-n_prev):

        docX.append(data[i:i+n_prev])

        docY.append(data[i+n_prev])

    alsX = np.array(docX)

    alsY = np.array(docY)



    return alsX, alsY



def train_test_split(data, test_size=0.1):  

    """

    This just splits data to training and testing parts

    """

    ntrn = round(len(data) * (1 - test_size))



    X_train, y_train = _get_chunk(data[0:ntrn])

    X_test, y_test = _get_chunk(data[ntrn:])



    return (X_train, y_train), (X_test, y_test)
#(X_train, y_train), (X_test, y_test) = train_test_split(X)
# Partimos conjunto de entrada en trozos de 24 puntos: Damos 23 a la red y el 24th tendrá que ser predicho.

input = []

sequence_length=24

for index in range(len(X) - sequence_length):

    input.append(X[index: index + sequence_length])

input = np.array(input)

np.shape(input)
values = input
scaler = MinMaxScaler(feature_range=(0, 1), copy=True)

scaled_features = scaler.fit_transform(values[:,:-1])

scaled_label = scaler.fit_transform(values[:,-1].reshape(-1,1))

values = np.column_stack((scaled_features, scaled_label))
# Create training and test sets: 90-10.

#  Now that the examples are formatted, we need to split them into train and test, input and target. Here we select 10% of the data as test and 90% to train. We also select the last value of each example to be the target, the rest being the sequence of inputs.

row = round(0.9 * values.shape[0])

train = values[:row, :]

#np.random.shuffle(train)

X_train = train[:, :-1]

y_train = train[:, -1]

X_test = values[row:, :-1]

y_test = values[row:, -1]
x = X_train

y = y_train



regr = SVR(C = 2.0, epsilon = 0.1, kernel = 'rbf', gamma = 0.5, 

           tol = 0.001, verbose=False, shrinking=True, max_iter = 10000)



regr.fit(x, y)

data_pred = regr.predict(x)

y_pred = scaler.inverse_transform(data_pred.reshape(-1,1))

y_inv = scaler.inverse_transform(y.reshape(-1,1))



mse = mean_squared_error(y_inv, y_pred)

rmse = np.sqrt(mse)

print('Mean Squared Error: {:.4f}'.format(mse))

print('Root Mean Squared Error: {:.4f}'.format(rmse))



print('Variance score: {:2f}'.format(r2_score(y_inv, y_pred)))
def plot_preds_actual(preds, actual):

    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.plot(actual, label='Actual data')

    plt.plot(preds, label='Predicted data')

    plt.legend()

    plt.show()
plot_preds_actual(y_pred, y_inv)
def run_test_nonlinear_reg(x, y):

    data_pred = regr.predict(x)

    y_pred = scaler.inverse_transform(data_pred.reshape(-1,1))

    y_inv = scaler.inverse_transform(y.reshape(-1,1))



    mse = mean_squared_error(y_inv, y_pred)

    rmse = np.sqrt(mse)

    print('Mean Squared Error: {:.4f}'.format(mse))

    print('Root Mean Squared Error: {:.4f}'.format(rmse))



    #Calculate R^2 (regression score function)

    #print('Variance score: %.2f' % r2_score(y, data_pred))

    print('Variance score: {:2f}'.format(r2_score(y_inv, y_pred)))

    return y_pred, y_inv
y_pred, y_inv = run_test_nonlinear_reg(X_test, y_test)
plot_preds_actual(y_pred, y_inv)