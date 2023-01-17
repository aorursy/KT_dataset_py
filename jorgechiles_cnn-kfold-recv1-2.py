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


import tensorflow

import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation,SimpleRNN,LSTM

from tensorflow.keras.optimizers import SGD

from tensorflow.keras import backend

from tensorflow.keras import metrics

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Conv1D

from tensorflow.keras.layers import MaxPooling1D, Bidirectional,TimeDistributed

from tensorflow.keras.constraints import max_norm,unit_norm

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator



from termcolor import colored



import numpy

from numpy import arange

from numpy import array

from numpy import mean

from numpy import std



from pandas import concat

from pandas import Series

from pandas import DataFrame



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import RandomizedSearchCV
train=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

test=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')



train.tail()
training_data=train.groupby('Date')['ConfirmedCases','Fatalities'].sum().reset_index()



print(training_data)
from matplotlib import pyplot

from math import sqrt

import matplotlib



# Grafica

pyplot.figure(figsize = (18,9))

pyplot.plot(training_data.ConfirmedCases,'o--',c='b', label='Confirmed')

pyplot.plot(training_data.Fatalities,'o--',c='r', label='Fatalities')





pyplot.xlabel('Date',fontsize=18)

pyplot.ylabel('cases',fontsize=18)

pyplot.title('COVID19',fontsize=18)

pyplot.grid(True,linestyle='-.')

leg=pyplot.legend(loc="best",fontsize=18, shadow=True, fancybox=True)

leg.get_frame().set_alpha(0.8)

pyplot.show()
# hietograma

training_data.Fatalities.hist()

pyplot.show()



# Densidad

training_data.Fatalities.plot(kind='density')

pyplot.show()



#boxplot

training_data.Fatalities.plot(kind='box')

pyplot.show()



print(training_data.Fatalities.describe())
# hietograma

training_data.ConfirmedCases.hist()

pyplot.show()



# Densidad

training_data.ConfirmedCases.plot(kind='density')

pyplot.show()



#boxplot

training_data.ConfirmedCases.plot(kind='box')

pyplot.show()



print(training_data.ConfirmedCases.describe())
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

pyplot.figure(figsize = (18,9))

plot_acf(training_data.ConfirmedCases,lags=20)

plot_pacf(training_data.ConfirmedCases,lags=20)

pyplot.show()


pyplot.figure(figsize = (18,9))

plot_acf(training_data.Fatalities,lags=20)

plot_pacf(training_data.Fatalities,lags=20)

pyplot.show()
pyplot.scatter(training_data.ConfirmedCases, training_data.Fatalities)

pyplot.show()
pyplot.matshow(training_data.corr())

pyplot.show()
from statsmodels.tsa.seasonal import seasonal_decompose



result_addConfirm = seasonal_decompose(training_data.ConfirmedCases, model='additive',period=1)



result_addConfirm.plot()

pyplot.show()
from statsmodels.tsa.seasonal import seasonal_decompose



result_addFatal = seasonal_decompose(training_data.Fatalities, model='additive',period=1)



result_addFatal.plot()

pyplot.show()
from statsmodels.tsa.stattools import adfuller, kpss



result = adfuller(training_data.ConfirmedCases, autolag='AIC')

print(f'ADF Statistic: {result[0]}')

print(f'p-value: {result[1]}')

for key, value in result[4].items():

    print('Critial Values:')

    print(f'   {key}, {value}')

    

    

result = kpss(training_data.ConfirmedCases, regression='c')

print('\nKPSS Statistic: %f' % result[0])

print('p-value: %f' % result[1])

for key, value in result[3].items():

    print('Critial Values:')

    print(f'   {key}, {value}')
diff = training_data.ConfirmedCases.diff()

pyplot.plot(diff)

pyplot.show()
diff = training_data.Fatalities.diff()

pyplot.plot(diff)

pyplot.show()
from statsmodels.tsa.seasonal import seasonal_decompose



result_addConfir = seasonal_decompose(training_data.ConfirmedCases, model='additive',period=1)

detrendedConfirmed = training_data.ConfirmedCases.values - result_addConfir.trend

pyplot.plot(detrendedConfirmed )

pyplot.show()
from statsmodels.tsa.seasonal import seasonal_decompose



result_addFatal = seasonal_decompose(training_data.Fatalities.values, model='additive',period=1)

detrendedFatal = training_data.Fatalities.values - result_addFatal.trend

pyplot.plot(detrendedFatal)

pyplot.show()
deseasonalizedConfirmed = training_data.ConfirmedCases.values - result_addConfir.seasonal

pyplot.plot(deseasonalizedConfirmed )

pyplot.show()
deseasonalizedFatal = training_data.Fatalities.values - result_addFatal.seasonal

pyplot.plot(deseasonalizedFatal)

pyplot.show()
BestLag=DataFrame()

lags=list()

autoCorr=list()



for _ in range(20):

    autoCorr.append( training_data.Fatalities.autocorr(lag=(_+1)))

    lags.append((_+1))





BestLag['Lag'],BestLag['AutoCorr']=lags,autoCorr

print(BestLag)

    

Order=BestLag.sort_values('AutoCorr',ascending=False)

print(Order)
test.tail()

test_data=test.groupby('Date').sum().reset_index()







h=len(test_data)
from sklearn.preprocessing import MinMaxScaler

valores=training_data.ConfirmedCases.values

valores=valores.astype('float32')

valores=valores.reshape(len(valores),1)



scaler = MinMaxScaler()

scaled = scaler.fit_transform(valores)

dataset = scaled.reshape(len(scaled),1 )



print(dataset)
early_stops=tensorflow.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', 

                                                  min_delta=0.01, 

                                                  patience=20, 

                                                  verbose=0, 

                                                  mode='min', 

                                                  baseline=None,

                                                  restore_best_weights=True)

        

filepath="LSTMPolvoOct19.best.hdf5"

checkpoint = tensorflow.keras.callbacks.ModelCheckpoint(filepath, 

                                                        monitor= 'val_mean_squared_error' , 

                                                        verbose=0, 

                                                        save_best_only=True,

                                                        mode= 'min' )

        

reduce_lr = tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='val_mean_squared_error', 

                                                         factor=0.1,

                                                         patience=10, 

                                                         min_lr=0.0001)

            

callbacks_list = [early_stops,checkpoint,reduce_lr]
def fit_CNN(filters=100,n_layers=1,n_input=1):

    

    modelCNN = Sequential()



    modelCNN.add(Conv1D(filters=filters, 

                        kernel_size=n_kernel,

                        kernel_initializer='random_uniform', 

                        bias_initializer='zeros',

                        padding="causal",

                        activation='relu',

                        kernel_constraint=unit_norm(),

                        input_shape=(n_input, 1)))

    modelCNN.add(Dropout(0.5))

    

    for _ in range(1, n_layers):

        

    

        modelCNN.add(Conv1D(filters=filters, 

                            kernel_size=n_kernel,

                            kernel_initializer='random_uniform', 

                            bias_initializer='zeros',

                            padding="causal",

                            kernel_constraint=unit_norm(), 

                            activation='relu'))

        modelCNN.add(Dropout(0.5))

        

    modelCNN.add(MaxPooling1D(pool_size=1))

    modelCNN.add(Dropout(0.5))

    modelCNN.add(Flatten())

    modelCNN.add(Dense(1))



    modelCNN.compile(loss='mean_squared_error',

                        optimizer='adam',

                        metrics=['mean_squared_error'])









    

    return modelCNN


n_epochs=[100]

n_kernel=1

n_stride=1

repeats=1





datosPronosticoCNN=DataFrame()

datosPronosticoCNN=DataFrame(columns=['Param','Day','Forec']) 

Pronostico=list()

rep =list()

dia=list()







valores2=training_data.Fatalities.values

valores2=valores2.astype('float32')

valores2=valores2.reshape(len(valores2),1)

scaler2 = MinMaxScaler()

scaled2 = scaler2.fit_transform(valores2)

scaled_values= scaled2.reshape(len(scaled2),1 )



temps = DataFrame(scaled_values) # Datos sin pronosticos

temps=temps.astype('float32')

    

dataframe = concat([temps.shift(Order.Lag.values[0]), temps], axis=1)

dataframe.columns = ['t+%d'%(Order.Lag.values[0]),'t']

dataframe.dropna(inplace=True)

train= dataframe.values



X, y = train[:,0:-1], train[:,-1]

X=X.reshape(X.shape[0],X.shape[1],1)





num_folds=10

scoring = 'neg_mean_squared_error'



kfold = KFold(n_splits=num_folds, random_state=None,shuffle=False)



keras_reg = tensorflow.keras.wrappers.scikit_learn.KerasRegressor(fit_CNN,verbose=0)



param_CNN = dict(n_layers=[1,2],

                 filters=[500,1000,2000],

                 epochs=n_epochs,

                 n_input=[X.shape[1]])



rnd_search_cv = GridSearchCV(keras_reg, param_CNN,scoring,cv=kfold)



print(colored("\nfitting model...",'red'))



ModelCNN=rnd_search_cv.fit(X, y,

      callbacks=[tensorflow.keras.callbacks.EarlyStopping(patience=10)])





print(colored("\nBest score: %f using %s" % (ModelCNN.best_score_, ModelCNN.best_params_),'yellow')) 







for _ in range (h):





    print(colored("\ngetting day: %d prediction..."% ((_+1)),'magenta'))





    X_pred=X[-1].reshape(1,1,1)

    predictions_CNN = ModelCNN.predict(X_pred)

    

    X=numpy.append(X,predictions_CNN)

    X=X.reshape(len(X),1,1)



    prediction_CNN = predictions_CNN.reshape(1, 1)

    Pronosticos = scaler2.inverse_transform(prediction_CNN)





    print(colored("Prediction day %d: %d Fatalcases"% ((_+1), Pronosticos) ,'blue'))



    Pronostico.append(Pronosticos)

    dia.append(_+1)





Pronostico2=array(Pronostico)



# Grafica

pyplot.figure(figsize = (18,9))

pyplot.plot([None for i in training_data.Fatalities.values] + [x for x in Pronostico2[:,0]],'o-',c='orange', label='Pron CNN')

pyplot.plot(training_data.Fatalities.values,'o--',c='b', label='Data Train')

pyplot.xlabel('Days',fontsize=18)

pyplot.ylabel('Cases',fontsize=18)

pyplot.title('COVID Fatalities.values',fontsize=18)

pyplot.grid(True,linestyle='-.')

leg=pyplot.legend(loc="best",fontsize=18, shadow=True, fancybox=True)

leg.get_frame().set_alpha(0.8)

pyplot.show()





for t in range (len(Pronostico)):

    datosPronosticoCNN=datosPronosticoCNN.append({'Param':ModelCNN.best_params_,'Day':dia[t],'Forec':Pronostico[t]},ignore_index=True)







datosPronosticoCNN