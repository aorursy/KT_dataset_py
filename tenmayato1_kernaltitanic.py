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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

titanic=pd.read_csv('../input/titanic/train.csv', header = 0, dtype={'Age': np.float64})

titanicT=pd.read_csv('../input/titanic/test.csv', header = 0, dtype={'Age': np.float64})

titanic.describe()
titanic.Survived.value_counts(normalize=True)
datadict=pd.DataFrame(titanic.dtypes)

datadict["MissingVals"]=titanic.isnull().sum()

datadict["UniqueVals"]=titanic.nunique()

datadict["Count"]=titanic.count()

datadict.rename(columns={0:"dtype"})
datadict
titanic.head()
fig, axes = plt.subplots(2, 4, figsize=(16, 10))

sns.countplot(titanic["Survived"],ax=axes[0,0])

sns.countplot("Pclass",data=titanic,ax=axes[0,1])

sns.countplot('Sex',data=titanic,ax=axes[0,2])

sns.countplot('SibSp',data=titanic,ax=axes[0,3])

sns.countplot('Parch',data=titanic,ax=axes[1,0])

sns.countplot('Embarked',data=titanic,ax=axes[1,1])

sns.distplot(titanic['Fare'], kde=True,ax=axes[1,2])

sns.distplot(titanic['Age'].dropna(),kde=True,ax=axes[1,3])
figbi, axesbi =plt.subplots(2,4, figsize= (16,10))

titanic.groupby("Pclass")["Survived"].mean().plot(kind="barh",ax=axesbi[0,0],xlim=[0,1])

titanic.groupby('SibSp')['Survived'].mean().plot(kind='barh',ax=axesbi[0,1],xlim=[0,1])

titanic.groupby('Parch')['Survived'].mean().plot(kind='barh',ax=axesbi[0,2],xlim=[0,1])

titanic.groupby('Sex')['Survived'].mean().plot(kind='barh',ax=axesbi[0,3],xlim=[0,1])

titanic.groupby('Embarked')['Survived'].mean().plot(kind='barh',ax=axesbi[1,0],xlim=[0,1])

sns.boxplot(x="Survived", y="Age", data=titanic, ax=axesbi[1,1])

sns.boxplot(x="Survived",y="Fare", data= titanic, ax=axesbi[1,2])

titanic.groupby("Survived")["Fare"].mean().plot(kind= "barh", ax=axesbi[1,3])
titanic['Title'] = titanic.Name.str.extract('([a-zA-Z]+)*\.',expand=False)

titanicT["Title"]= titanicT.Name.str.extract('([a-zA-Z]+)*\.', expand=False)







titanic['Title'] = titanic['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 

                                                 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

titanicT['Title'] = titanicT['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 

                                                 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



titanic['Title'] = titanic['Title'].replace('Mlle', 'Miss')

titanic['Title'] = titanic['Title'].replace('Ms', 'Miss')

titanic['Title'] = titanic['Title'].replace('Mme', 'Mrs')



titanicT['Title'] = titanicT['Title'].replace('Mlle', 'Miss')

titanicT['Title'] = titanicT['Title'].replace('Ms', 'Miss')

titanicT['Title'] = titanicT['Title'].replace('Mme', 'Mrs')



titanic.Title.unique()
titanicT.Title.unique()


del titanic["Name"]

del titanic["Ticket"]

del titanic["Cabin"]

del titanic["PassengerId"]



del titanicT["Name"]

del titanicT["Ticket"]

del titanicT["Cabin"]

del titanicT["PassengerId"]
meanS=titanic[titanic.Survived==1].Age.mean()

titanic["age"]=np.where(titanic.Age.isnull() & titanic.Survived==1 , meanS, titanic["Age"])



meanN=titanic[titanic.Survived==0].Age.mean()

titanic.age.fillna(meanN,inplace=True)



meanT=titanicT.Age.mean()

titanicT["age"]=np.where(titanicT.Age.isnull(),meanT,titanicT["Age"])



del titanic["Age"]

del titanicT["Age"]



titanic["Age_Bin"]=pd.cut(titanic["age"],bins=[0,12,20,40,120], labels=["A","B","C","D"])

titanicT["Age_Bin"]=pd.cut(titanicT["age"],bins=[0,12,20,40,120], labels=["A","B","C","D"])



del titanic["age"]

del titanicT["age"]

titanic["Fare"]=titanic.Fare.replace(0,np.NaN)

FareS=(titanic[titanic["Survived"]==1].Fare.mean())

FareN=(titanic[titanic["Survived"]==0].Fare.mean())

FareT=(titanicT.Fare.mean())



titanic["fare"]=np.where(titanic.Fare.isnull() & titanic.Survived==1, FareS, titanic["Fare"])

titanic.fare.fillna(FareN, inplace=True)

titanicT["fare"]=np.where(titanicT.Fare.isnull(), FareT, titanicT["Fare"])



titanic["Fare_Bin"]=pd.cut(titanic["fare"],bins=[0,7.91,14.45,31,120,1000], labels=["A","B","C","D","E"])

titanicT['Fare_bin'] = pd.cut(titanicT['fare'], bins=[0,7.91,14.45,31,120,1000], labels=["A","B","C","D","E"])





del titanic["Fare"]

del titanicT["Fare"]

titanicT.head()
titanic=pd.get_dummies(titanic, columns=["Sex","Embarked","Title","Age_Bin","Fare_Bin"],

                                prefix=["Sex","Embarked","Title","Age","Fare"])

titanicT=pd.get_dummies(titanicT, columns=["Sex","Embarked","Title","Age_Bin","Fare_bin"],

                                prefix=["Sex","Embarked","Title","Age","Fare"])
Y_train=titanic["Survived"]

Y_train=Y_train.values
titanic.columns
titanicT.columns
del titanic["Survived"]

titanic_train=titanic.values

titanic_test=titanicT.values
import tensorflow as tf

from keras import backend as K

from keras.models import load_model,Sequential

from keras.layers import Dense,LeakyReLU,BatchNormalization,Dropout
titanic_train.shape
model=Sequential()

model.add(Dense(12,input_shape=(23,)))

model.add(LeakyReLU(alpha=0.01))

model.add(BatchNormalization(axis=1))



model.add(Dense(8))

model.add(LeakyReLU(alpha=0.01))

model.add(BatchNormalization(axis=1))



model.add(Dense(4))

model.add(LeakyReLU(alpha=0.01))

model.add(BatchNormalization(axis=1))



model.add(Dense(2))

model.add(LeakyReLU(alpha=0.01))

model.add(BatchNormalization(axis=1))





model.add(Dense(1,activation="sigmoid"))



model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])



model.fit(x=titanic_train,y=Y_train,batch_size=32,epochs=100)

model.save("titanic")
titanic_test.shape
predictions=model.predict(x=titanic_test,steps=None)

predictions.shape
p_id=pd.read_csv('../input/titanic/test.csv', header = 0, dtype={'Age': np.float64})

p_id=p_id["PassengerId"].values

predictions=predictions.reshape(418,)

predictions=predictions>0.5

predictions=predictions.astype(np.int)

submission=pd.DataFrame({"PassengerId":p_id,"Survived":predictions})
filename = 'Titanic Predictions 1.csv'



x=submission.to_csv(filename,index=False)
predictions