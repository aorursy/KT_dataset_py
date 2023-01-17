import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Any results you write to the current directory are saved as output.
QCM3 = pd.read_csv('../input/alcohol-qcm-sensor-dataset/QCM3.csv')

QCM6 = pd.read_csv('../input/alcohol-qcm-sensor-dataset/QCM6.csv')

QCM7 = pd.read_csv('../input/alcohol-qcm-sensor-dataset/QCM7.csv')

QCM10 = pd.read_csv('../input/alcohol-qcm-sensor-dataset/QCM10.csv')

QCM12 = pd.read_csv('../input/alcohol-qcm-sensor-dataset/QCM12.csv')
data = pd.concat([QCM3,QCM6,QCM7,QCM10,QCM12])
data.head()
data.tail()
data.isnull().sum()
data.info()
data.isnull().any()
data.describe()
data.index
data.index = [i for i in range(125)]
data.index
print(data[data['1-Octanol']==1.0].shape)

print(data[data['1-Propanol']==1.0].shape)

print(data[data['2-Butanol']==1.0].shape)

print(data[data['2-propanol']==1.0].shape)

print(data[data['1-isobutanol']==1.0].shape)
data.isnull().sum()
one_Octanol = data[data['1-Octanol']==1]

one_Octanol['Result']=1
one_Propanol = data[data['1-Propanol']==1]

one_Propanol['Result']=2
two_Butanol = data[data['2-Butanol']==1]

two_Butanol['Result']=3
two_propanol = data[data['2-propanol']==1]

two_propanol['Result']=4
one_isobutanol = data[data['1-isobutanol']==1]

one_isobutanol['Result']=5
data = pd.concat([one_Octanol,one_Propanol,two_Butanol,two_propanol,one_isobutanol])
data.shape
data.describe()
data.head()
data.isnull().sum()
data['Result'].value_counts()
data = data.drop(['1-Octanol','1-Propanol','2-Butanol','2-propanol','1-isobutanol'],axis=1)
data.head()
data.shape
x= data.iloc[:,:10]

y= data.iloc[:,10:11]
print(x.shape,y.shape)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(x)

x_scaler = scaler.transform(x)
from sklearn.model_selection import train_test_split



x_train,x_test, y_train,y_test = train_test_split(x_scaler,y, test_size=0.25, stratify=y)
print(x_train.shape,x_test.shape)

print(y_train.shape,y_test.shape)
print(x_train[:5])

print(y_train[:5])
print(x_test[:5])

print(y_test[:5])
from sklearn.neighbors import KNeighborsClassifier
neighbors = KNeighborsClassifier()

neighbors.fit(x_train,y_train)
print(neighbors.score(x_test,y_test))
predict = neighbors.predict(x_test)
predict
y_test= np.array(y_test.to_numpy().T.flat)
y_test
submission = pd.DataFrame({'Originol':y_test,'Predict':predict})
submission.head()
print(np.mean(submission['Originol']==submission['Predict']))