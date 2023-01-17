import pandas as pd

import numpy as np

#names = ['SepLen','SepWid','PetLen','PetWid','Class']

data = pd.read_csv('../input/Iris.csv')
data = data.drop('Id',axis=1)

data.describe()
import matplotlib.pyplot as plt

import matplotlib

%matplotlib inline

matplotlib.style.use('ggplot')

from pandas.tools import scatter_matrix



scatter_matrix(data,figsize=(10,10),diagonal='kde')
data.hist(figsize=(10,10))
data.groupby('Class').SepLen.hist(figsize=(10,10))
from sklearn.preprocessing import StandardScaler

array = data.values

X = array[:,0:4]

Y = array[:,4]
scaler = StandardScaler().fit(X)

rescaledX = scaler.transform(X)

np.set_printoptions(precision=2)

#print(rescaledX[0:6,:])
from sklearn.model_selection import KFold,cross_val_score

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#from sklearn.svm import SVC,LinearSVC

#from sklearn.linear_model import SGDClassifier

#from sklearn.neighbors import KNeighborsClassifier

#from sklearn.tree import DecisionTreeClassifier

kflod = KFold(n_splits=10,random_state=7)

scoring='accuracy'
import pickle

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33,random_state = 7)

model = QuadraticDiscriminantAnalysis()

model.fit(X_train,Y_train)

filename = 'IRIS_QDA_MODEL.sav'

pickle.dump(model,open(filename,'wb'))

loaded_model = pickle.load(open(filename,'rb'))

result = loaded_model.score(X_test,Y_test)

print(result.mean()*100)