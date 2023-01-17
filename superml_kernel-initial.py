# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
%matplotlib inline  

import matplotlib.pyplot as plt

import numpy as np



import numpy as np

import pandas as pd

# import model

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

# import module to calculate model perfomance metrics

import seaborn as sns

from sklearn import metrics



input_train_path = '../input/AFM10000.csv'

input_test_path = '../input/test.csv'



col = ['x1', 'x2', 'Altura']



col_x = ['x1', 'x2']



input_train = pd.read_csv(input_train_path, skiprows=1, names=col)

input_test = pd.read_csv(input_test_path,skiprows=1,  names=col_x)





print(input_train.head())

print(input_test.head())


from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error



X = input_train[col_x]

#X = df.iloc[:, :-1].values 

print (input_train.head())



# salida

y = input_train['Altura']



# Dividir X e Y en juegos de entrenamiento y prueba

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)



print('#Training data points: %d' % X_train.shape[0])

print('#Testing data points: %d' % X_test.shape[0])





# Modelo

linreg = LinearRegression()



# ajustar el modelo a los datos de entrenamiento 

linreg.fit(X_train, y_train)



# Los coeficientes

print('\nCoeficientes:', linreg.coef_)



# hacer predicciones en el conjunto de prueba



y_train_pred = linreg.predict(X_train)



y_pred = linreg.predict(X_test)
# calcular el RMSE de nuestras predicciones

print('MSE train: %.2f, test: %.2f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_pred)))





# Explicación de R^2: 1 es predicción perfecta

print('R^2 train: %.2f, test: %.2f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_pred)))
predictions = linreg.predict(input_test)

print(predictions)



Id=np.arange(1,3999)

Id= Id.flatten

my_submission = pd.DataFrame({'Id': Id, 'Altura': predictions})

my_submission.shape



# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)