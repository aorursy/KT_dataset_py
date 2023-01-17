import pandas as pd

import numpy as np

import seaborn as sns



from pandas import read_csv

from sklearn.model_selection import train_test_split
nRowsRead = None # specify 'None' if want to read whole file

# imdb.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

data = pd.read_csv('/kaggle/input/imdb.csv', engine='python')

data.dataframeName = 'imdb.csv'

nRow, nCol = data.shape

print(f'There are {nRow} rows and {nCol} columns')
data.head()
a=[]

b=[]

c=[]

for i in data['cast']:

  a.append(i.split(",")[0])

  b.append(i.split(",")[1])

  

for i in data['genre']:

  c.append(i.split(",")[0])

data['genre']=c

data['cast2']=a

data['cast3']=b
data.head()
from sklearn.preprocessing import LabelEncoder

label=LabelEncoder()

data['genre']=label.fit_transform(data['genre'])

data['director']=label.fit_transform(data['director'])

data['cast2']=label.fit_transform(data['cast2'])

data['cast3']=label.fit_transform(data['cast3'])



y=data["imdb"]

y=np.array(y)

y.shape

y=y.reshape(-1,1)

x=data.drop(columns=['movieName','imdb','cast'])

x.head()
x=np.array(x)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(n_estimators=1000)

rf.fit(X_train,y_train)
y_pred_rf=rf.predict(X_test)
from sklearn.metrics import r2_score

import math



error =[np.power((b-a),2) for (a,b) in zip(y_pred_rf,y_test)]

error0 = np.sum(error)

error = math.sqrt(error0)

error = (error/len(y_test))*100



print("Test error % = {}".format(error))

accuracy = 100 - error

print("Test Accuracy % = {}".format(accuracy))

r2_score(y_test,y_pred_rf)