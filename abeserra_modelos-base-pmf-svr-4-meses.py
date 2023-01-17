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
# set input dirname
input_dir = os.path.join('/kaggle/input', 'seleccion-datos-1-estacion-1-contaminante')
# 1. Cargar datos
df = pd.read_csv(os.path.join(input_dir, 'pre_feb-may_2019.csv'), index_col = 0)
df.head()

df.info()
# Visualizamos datos
df.NO2.plot()
plt.show()
window_width = 23
test_prop = 0.1
validation_prop = 0
training_size = round(len(df) * ( 1 - test_prop - validation_prop))
# Creamos conjunto de entrada desde los datos origen
X = df['NO2'].values
np.shape(X)
from sklearn.metrics import mean_squared_error
from math import sqrt
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
plt.plot(test, label='Recorded data')
plt.plot(predictions, label='Predicted data')
plt.legend()
plt.show()
from sklearn.preprocessing import MinMaxScaler
def normalize_data_sklearn(data, training_size):
    '''
    Normalizing the data by mean-centering the data.
    output within the training set have std = 1 and mean = 0
    '''    
    training_data = data[:training_size]
    training_data = np.reshape(training_data, (training_data.shape[0], 1))
    
    scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
    scaler.fit(training_data)
    
    data = scaler.transform(np.reshape(data, (data.shape[0], 1)))
    
    return (data[:,0], scaler)
X_norm, scaler = normalize_data_sklearn(X, training_size)
print('Valores normalización MinMax:', scaler.data_range_, scaler.scale_)
# Escalado por separado de los targets y de los data de entrada?
#scaled_features = scaler.fit_transform(X[:,:-1])
#scaled_label = scaler.fit_transform(X[:,-1].reshape(-1,1))
#X_norm = np.column_stack((scaled_features, scaled_label))
def _get_chunk(data, seq_len = window_width):  
    """
    data should be pd.DataFrame()
    """

    chunk_X, chunk_Y = [], []
    for i in range(len(data)-seq_len):
        chunk_X.append(data[i:i+seq_len])
        chunk_Y.append(data[i+seq_len])
    chunk_X = np.array(chunk_X)
    chunk_Y = np.array(chunk_Y)

    return chunk_X, chunk_Y

def train_test_split(data):  
    """
    This just splits data to training, testing and validation parts
    """
    
    ntrn1 = round(len(data) * (1 - test_prop)) # get size for train set

    train_set = data[:ntrn1]
    test_set = data[ntrn1:]
    
    X_train, y_train = _get_chunk(train_set)
    X_test, y_test = _get_chunk(test_set)

    return (X_train, y_train), (X_test, y_test)
(X_train, y_train), (X_test, y_test) = train_test_split(X_norm)
print(np.shape(X_train))
print(np.shape(X_test))
print(np.shape(y_train))
print(np.shape(y_test))
print(np.shape(X_train))
print(X_train[:2])
#print(np.shape(y_train))
print(y_train[:2])
#print(np.shape(y_test))
print(y_test[:2])
regr = SVR(C = 2.0, epsilon = 0.1, kernel = 'rbf', gamma = 0.5, 
           tol = 0.001, verbose=False, shrinking=True, max_iter = 10000)

regr.fit(X_train, y_train)
def run_test_nonlinear_reg(x, y):
    data_pred = regr.predict(x)
    y_pred = scaler.inverse_transform(data_pred.reshape(-1,1))
    y_orig = scaler.inverse_transform(y.reshape(-1,1))

    mse = mean_squared_error(y_orig, y_pred)
    rmse = np.sqrt(mse)
    print('Mean Squared Error: {:.4f}'.format(mse))
    print('Root Mean Squared Error: {:.4f}'.format(rmse))

    #Calculate R^2 (regression score function)
    #print('Variance score: %.2f' % r2_score(y, data_pred))
    print('Variance score: {:2f}'.format(r2_score(y_orig, y_pred)))
    return y_pred, y_orig
(y_train_pred, y_train_actual) = run_test_nonlinear_reg(X_train, y_train)
def plot_preds_actual(preds, actual):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(actual, label='Actual data')
    plt.plot(preds, label='Predicted data')
    plt.legend()
    plt.show()
plot_preds_actual(y_train_pred, y_train_actual)
y_pred, y_actual = run_test_nonlinear_reg(X_test, y_test)
plot_preds_actual(y_pred, y_actual)