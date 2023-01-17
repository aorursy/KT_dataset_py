# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
os.getcwd()
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/titanic/test.csv')
data2=pd.read_csv('/kaggle/input/titanic/train.csv')
data.head()
data2.head()
data2.describe()
data2.info()
data2.columns
data2=data2.drop(['Name','Embarked','Age','Fare'],axis=1)
data2.columns
data2.isnull().sum()
data2.head()
data2.info()
data2.describe()
data2.isnull().sum()
from sklearn import preprocessing
encode=preprocessing.LabelEncoder()
en=preprocessing.OrdinalEncoder()


data2['Sex']=encode.fit_transform(data2['Sex'])


 
data2.head()
y_train=data2.Survived.values

data2=data2.drop(['Survived','Cabin','Ticket'],axis=1)
data2
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
data2 = sc.fit_transform(data2)

x_train=np.matrix(data2)
x_train.shape,y_train.shape
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(units=4,activation='relu',input_shape=(5,)))
model.add(Dense(units=3,activation='relu'))
model.add(Dense(units=2,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(optimizer="Adam",loss="binary_crossentropy",metrics=['accuracy'])
my_model=model.fit(x_train,y_train,batch_size=32,epochs=200)
ans1=model.predict(x_train)
ans1
my_model.history.keys()
my_model.history['accuracy']
data.head()
data.isnull().sum()
data['Age']=data['Age'].fillna(data['Age'].mean())

data=data.drop(['Name','Embarked','Age','Fare'],axis=1)
data=data.drop(["Ticket",'Cabin'],axis=1)
data.info()

data['Sex']=encode.fit_transform(data['Sex'])

data.head()
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
data = sc.fit_transform(data)
x_test=np.matrix(data)
x_test.shape
ans=model.predict(x_test)

test_data=pd.read_csv('/kaggle/input/titanic/test.csv')
test_data.head()                      
ans[4]
ans[1]

test_data

lis=list()
Survived=1
Not_Survived=0
for i in range(0,418):
    if ans[i]<0.5:
            lis.append(0)
    else:
            lis.append(1)
    
        
submit=pd.DataFrame()
submit['PassengerId']=test_data['PassengerId']
submit.insert(1,'Survived',lis)
submit.head()  

submit
filename="titanic_version_2.csv"
submit.to_csv(filename,index=False)
print('Saved file: ' + filename)


