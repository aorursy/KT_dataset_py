# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset = pd.read_csv("../input/insurance/insurance.csv")
dataset
dataset.head(20)
dataset.isnull().any()
dataset.info()
print(dataset["sex"].unique())
print(dataset["smoker"].unique())
print(dataset["region"].unique())
#print(dataset["charges"].unique())
#print(dataset["children"].unique())
#print(dataset["age"].unique())

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dataset["sex"]=le.fit_transform(dataset["sex"])
dataset["sex"]
dataset["smoker"]=le.fit_transform(dataset["smoker"])
dataset["smoker"]
dataset["region"]=le.fit_transform(dataset["region"])
dataset["region"]
dataset.head(10)
#southwest 3 southwest 2 northwest 1 northeast 0
x=dataset.iloc[:,0:6].values

x
y=dataset.iloc[:,6:7].values
y
from sklearn.preprocessing import OneHotEncoder
ohe =OneHotEncoder()
z = ohe.fit_transform(x[:,5:6]).toarray()
z
x
x = np.delete(x,5,axis=1)
x
x=np.concatenate((z,x),axis=1)
x
x.shape
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
x_train
x_train.shape
y_train.shape
x_test.shape
y_test.shape
x_train
y_train
x_test
y_test
from sklearn.tree import DecisionTreeRegressor
dtreg = DecisionTreeRegressor(random_state=0)
dtreg.fit(x_train,y_train)
x_test
dpred=dtreg.predict(x_test)
dpred
y_test
#see accuracy
from sklearn.metrics import r2_score 
accuracy = r2_score(y_test,dpred)
accuracy
dataset.head(10)
#random value prediction
rp=dtreg.predict([[0,0,1,0,22,0,25,3,1]])
rp
rp=dtreg.predict([[0,0,1,0,28,1,33.00,3,0]])
rp
rp=dtreg.predict([[0,0,1,0,28,1,33,3,0]])
rp
#ne nw se sw age sex children smoker region charges
from sklearn.ensemble import RandomForestRegressor                #ensemble dataset will be splitted into different datasets 
RFR=RandomForestRegressor(n_estimators=10,random_state=0)          #10 datsets and 10 decision trees will be created
RFR.fit(x_train,y_train)
rpred=RFR.predict(x_test)
rpred
y_test
raccuracy = r2_score(y_test,rpred)
raccuracy
rpred=RFR.predict([[0,0,1,0,28,1,33.00,3,0]])
rpred

rpred=RFR.predict([[0,0,0,1,19,0,27.900,0,1]])
rpred
dataset
dataset.head(20)
rpred=RFR.predict([[0,1,0,0,23,0,26.5,0,0]])
rpred
import matplotlib.pyplot as plt
x_train.shape
plt.scatter(x_train[:,4:5],y_train,color="red")
plt.plot(x_train[:,4:5],dtreg.predict(x_train))
plt.scatter(x_train[:,4:5],y_train,color="red")
plt.plot(x_train[:,4:5],RFR.predict(x_train))
plt.scatter(x_test[:,4:5],y_test,color="red")
plt.plot(x_test[:,4:5],dtreg.predict(x_test))
plt.scatter(x_test[:,4:5],y_test,color="red")
plt.plot(x_test[:,4:5],RFR.predict(x_test))
x_test
y_test
dataset.loc[dataset['charges'] == 7626.993]
rpred=RFR.predict([[0,0,0,1,44,1,27.5,1,0]])
rpred
dpred=dtreg.predict([[0,0,0,1,44,1,27.5,1,0]])
dpred
