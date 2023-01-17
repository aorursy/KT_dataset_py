import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from sklearn import svm

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn import preprocessing
data = pd.read_csv("../input/whitewine/winequalitywhite.csv",sep=";")

data.head()
data.tail()
print("Numero de registros:"+str(data.shape[0]))

for column in data.columns.values:

    print(column + "-NAs:"+ str(pd.isnull(data[column]).values.ravel().sum()))
print(data.dtypes)
print("Correlaciones en el dataset:")

data.corr()
plt.matshow(data.corr())
x = data.values #returns a numpy array

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

data_n = pd.DataFrame(x_scaled, columns=data.columns.values)
data_n.head()
data_vars = data.columns.values.tolist()

Y = ['alcohol']

X = [v for v in data_vars if v not in Y]

X_train, X_test, Y_train, Y_test = train_test_split(data_n[X],data_n[Y], test_size=0.30)  
from sklearn.model_selection import train_test_split, GridSearchCV

parameters = [

    {

        'kernel': ['rbf'],

        'gamma' : [1e-4,1e-3,1e-2, 0.1, 0.2, 0.5],

        'C': [1,10,100,1000]

    },

    {

        'kernel':["linear"],

        'C':[1,10,100,1000]

    }

]



clf = GridSearchCV(svm.SVR(),param_grid=parameters,cv=5)

clf.fit(X_train[X],Y_train[Y].values.ravel())
clf.best_params_
svr_rbf = SVR(kernel="rbf",C=100, gamma=0.2)
svr_rbf.fit(X_train,Y_train.values.ravel())
Y_predict = svr_rbf.predict(X_test)
print("R-square:",svr_rbf.score(X_test,Y_test))
data_prediction = pd.DataFrame()

data_prediction['alcohol_prediction'] = Y_predict

data_prediction['alcohol_real_value'] = Y_test.values.ravel()
print(data_prediction.shape)

data_prediction[:20]
from sklearn.metrics import mean_squared_error
mean_squared_error(data_prediction['alcohol_real_value'] , data_prediction['alcohol_prediction'])
data.shape
min_alcohol = data['alcohol'].min()

min_alcohol
max_alcohol = data['alcohol'].max()

max_alcohol
data_prediction['d_alcohol_prediction'] =  np.multiply(data_prediction['alcohol_prediction'],(max_alcohol - min_alcohol))

data_prediction['d_alcohol_prediction'] =  np.add(data_prediction['alcohol_prediction'],min_alcohol) 

data_prediction['d_alcohol_real_value'] =  np.multiply(data_prediction['alcohol_real_value'],(max_alcohol - min_alcohol))

data_prediction['d_alcohol_real_value'] =  np.add(data_prediction['alcohol_real_value'],min_alcohol) 
data_prediction.head()