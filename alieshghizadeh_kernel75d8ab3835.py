import numpy as np 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns; sns.set(style="ticks", color_codes=True)

from sklearn.model_selection import train_test_split

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.feature_selection import RFE

from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

%matplotlib inline
dataset = pd.read_csv("regressor_forTwosteps.csv", names=['regressor','realprice'],sep=',', header=0)

dataset.shape
dataset.head()
nn=pd.read_csv("NN_forTwosteps.csv", names=['nn','realprice'],sep=',', header=0)

nn.shape
nn.head()
knn=pd.read_csv("Knn_forTwosteps.csv", names=['knn','realprice'],sep=',', header=0)

knn.shape
knn.head()
randomforest=pd.read_csv("randomforest_forTwosteps.csv", names=['randomforest','realprice'],sep=',', header=0)

knn.shape
randomforest.head()
svm=pd.read_csv("svm_forTwosteps.csv", names=['svm','realprice'],sep=',', header=0)

knn.shape
svm.head()
dataset['randomforest']=randomforest['randomforest']

dataset['svm']=svm['svm']

dataset['nn']=nn['nn']

dataset['knn']=knn['knn']

dataset.head()
dataset.shape
n=80000

from sklearn.model_selection import train_test_split

y=dataset['realprice']

X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.2)

X_train.shape
def knn():

    knn = KNeighborsRegressor(n_neighbors=10)

    return knn



def extraTreesRegressor():

    clf = ExtraTreesRegressor(n_estimators=100,max_features='auto', verbose=1, n_jobs=1)

    return clf



def randomForestRegressor():

    clf = RandomForestRegressor(n_estimators=100,max_features='log2', verbose=1)

    return clf



def svm():

    clf = SVR(kernel='rbf', gamma='auto')

    return clf



def nn():

    clf = MLPRegressor(hidden_layer_sizes=(10,),  activation='relu', verbose=3)

    return clf



def predict_(m, test_x):

    return pd.Series(m.predict(test_x))



def model_(type):

    if (type==1):

        return extraTreesRegressor();    

    if (type==2):        

        return randomForestRegressor() ;    

    if (type==3):

        return knn() ;    

    if (type==4):

        return svm() ;    

    if (type==5):

        return nn() ;

    

def train_(train_x, train_y,type):

    m = model_(type)

    m.fit(train_x, train_y)

    return m



def train_and_predict(train_x, train_y, test_x,type):

    m = train_(train_x, train_y,type)

    return predict_(m, test_x), m

#
def calculate_error(test_y, predicted):

    return mean_absolute_error(test_y, predicted)
predicted, model = train_and_predict(X_train, y_train, X_test,1)

error = calculate_error(y_test, predicted)

print('regressor MAE', error)
predicted, model = train_and_predict(X_train, y_train, X_test,2)

error = calculate_error(y_test, predicted)

print('Randome forest MAE', error)
predicted, model = train_and_predict(X_train, y_train, X_test,3)

error = calculate_error(y_test, predicted)

print('knn MAE', error)
predicted, model = train_and_predict(X_train, y_train, X_test,5)

error = calculate_error(y_test, predicted)

print('nn MAE', error)
predicted, model = train_and_predict(X_train, y_train, X_test,4)

error = calculate_error(y_test, predicted)

print('svm MAE', error)