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
import numpy as np

import pandas as pd

import keras

np.random.seed(2)
data=pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
data.head()
from sklearn.preprocessing import StandardScaler

data['normal']=StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))

data.drop("Amount",axis=1,inplace=True)
data.head()
data.info()
x=data.iloc[:,data.columns!='Class']

y=data.iloc[:,data.columns=='Class']

x.shape,y.shape
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
from keras.layers import Dense

from keras.models import Sequential

from keras.layers import Dropout
model=Sequential([Dense(units=16,input_dim=30,activation="relu"),

                 Dense(units=24,activation="relu"),

                 Dropout(0.5),

                 Dense(units=20,activation="relu"),

                 Dense(units=24,activation="relu"),

                 Dense(1,activation="sigmoid")])
model.summary()
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

model.fit(x_train,y_train,epochs=5)
scores=model.evaluate(x_test,y_test)
print(scores)
from sklearn.metrics import confusion_matrix

y_pred=model.predict(x_test)

y_test=pd.DataFrame(y_test)

cnf=confusion_matrix(y_test,y_pred.round())

print(cnf)
fraudindices=np.array(data[data.Class==1].index)
print(len(fraudindices))

nonfraudindices=data[data.Class==0].index
fraudnumbers=np.random.choice(nonfraudindices,492,replace=False)

print(len(fraudnumbers))

undersampleindices=np.concatenate([fraudindices,fraudnumbers]) #both are arrays

#finding sample numbers in nonfraudindices and addding same size of both nonfraud and fraud indices 

print(len(undersampleindices))
undersampledata=data.iloc[undersampleindices,:]

#dividing data into required and non reuquired columns of dataframe

x_under=undersampledata.iloc[:,undersampledata.columns!="Class"]

y_under=undersampledata.iloc[:,undersampledata.columns=="Class"]
xtrain,xtest,ytrain,ytest=train_test_split(x_under,y_under,random_state=0,test_size=0.3)

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

model.fit(xtrain,ytrain,epochs=5)
y_pred=model.predict(xtest)

y_expected=pd.DataFrame(ytest)

cnf=confusion_matrix(y_pred.round(),y_expected)

print(cnf)
from imblearn.over_sampling import SMOTE
xresample,yresample=SMOTE().fit_sample(x,y.values.ravel())

#fitting through smte for over sampling
xtrain,xtest,ytrain,ytest=train_test_split(xresample,yresample,random_state=0,test_size=0.3)

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

model.fit(xtrain,ytrain,epochs=5)
y_pred=model.predict(xtest)

y_expected=pd.DataFrame(ytest)

cnf=confusion_matrix(y_pred.round(),y_expected)

print(cnf)