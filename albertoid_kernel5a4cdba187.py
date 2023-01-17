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
!pwd
import pandas as pd

import numpy as np

import kaggle

import zipfile

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from tqdm import tqdm

from sklearn.metrics import mean_squared_error as mse



import warnings

warnings.simplefilter('ignore')
solar_train=pd.read_csv('/kaggle/input/datamex0120/solar_train.csv')

solar_test=pd.read_csv('/kaggle/input/datamex0120/solar_test.csv')



solar_train=solar_train.sort_values(by='UNIXTime')

#Al ser raciacion instantánea no requerimos el tiempo, mas que para establecer su dependencia temporal, no para evaluar periodos

solar_train.drop(columns=['Data','Time','TimeSunRise','TimeSunSet'], inplace=True)

solar_train.reset_index(inplace=True)

solar_test.drop(columns=['Data','Time','TimeSunRise','TimeSunSet'], inplace=True)



#Normalizacion del Tiempo

tmin_train=solar_train.UNIXTime.min()

tmin_test=solar_test.UNIXTime.min()

tmin=min(tmin_train,tmin_test)



solar_train.UNIXTime = solar_train.UNIXTime - tmin

solar_test.UNIXTime = solar_test.UNIXTime - tmin



tmax_train=solar_train.UNIXTime.max()

tmax_test=solar_test.UNIXTime.max()

tmax=max(tmax_train,tmax_test)



solar_train.UNIXTime = solar_train.UNIXTime/tmax

solar_test.UNIXTime = solar_test.UNIXTime/tmax





#Eliminacion de outliers

IQR_Speed=solar_train.Speed.quantile(.75)-solar_train.Speed.quantile(.25)

solar_train=solar_train[(solar_train.Speed <solar_train.Speed.mean()+

                         1.5*IQR_Speed)&(solar_train.Speed >solar_train.Speed.mean()-1.5*IQR_Speed)]

IQR_Pressure=solar_train.Pressure.quantile(.75)-solar_train.Pressure.quantile(.25)

solar_train=solar_train[(solar_train.Pressure <solar_train.Pressure.mean()+

                         1.5*IQR_Pressure)&(solar_train.Pressure >solar_train.Pressure.mean()-1.5*IQR_Pressure)]

IQR_Temperature=solar_train.Temperature.quantile(.75)-solar_train.Temperature.quantile(.25)

solar_train=solar_train[(solar_train.Temperature <solar_train.Temperature.mean()+

                         1.5*IQR_Temperature)&(solar_train.Temperature >solar_train.Temperature.mean()-1.5*IQR_Temperature)]



#Optimizacion de tipo de datos

for e in solar_train.select_dtypes('integer').columns:

    solar_train[e]=pd.to_numeric(solar_train[e], downcast='integer')

for e in solar_train.select_dtypes('float').columns:

    solar_train[e]=pd.to_numeric(solar_train[e], downcast='float')

solar_train.info()



#Preparando conjuntos para entrenamiento y prueba



X=solar_train.drop(columns='Radiation', axis=1)

y=solar_train.Radiation



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)





#Despues de la revision de mérodos, el Random Forest fue el mejor, sin modificar sus hiperparámetros

from sklearn.ensemble import RandomForestRegressor as RFR

rf=RFR()

rf.fit(X_train, y_train)

y_predict=rf.predict(X_test)

MSE = mse(y_test,y_predict)

print(MSE)





#Reentrenar con todos los datos

X_train = X

y_train = y





#Modelo Random Forest

from sklearn.ensemble import RandomForestRegressor as RFR

rf=RFR()

rf.fit(X_train, y_train)

train_score=rf.score(X_train, y_train)

print (train_score)



X_predict=solar_test

y_predict=rf.predict(X_predict)

y_predict=pd.DataFrame(y_predict)

y_predict['id']=y_predict.index

y_predict.columns = ['Radiation','id']



y_predict=y_predict[['id','Radiation']]

y_predict.to_csv('submission.csv',index=False)