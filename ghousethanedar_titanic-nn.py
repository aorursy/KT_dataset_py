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
train=pd.read_csv('/kaggle/input/titanic/train.csv')

test_main=pd.read_csv('/kaggle/input/titanic/test.csv')
test=test_main
train.head()
test.isnull().sum()
train.isnull().sum()
train['Age']=train['Age'].fillna(train['Age'].median())

train['Fare']=train['Fare'].fillna(train['Fare'].median())



test['Age']=test['Age'].fillna(test['Age'].median())

test['Fare']=test['Fare'].fillna(test['Fare'].median())




train[['Pclass','Survived','SibSp','Parch']]=train[['Pclass','Survived','SibSp','Parch']].astype('str')



test[['Pclass','SibSp','Parch']]=test[['Pclass','SibSp','Parch']].astype('str')







train=train.drop(['Cabin','Name','Embarked','PassengerId','Ticket'],axis=1)





test=test.drop(['Cabin','Name','Embarked','PassengerId','Ticket'],axis=1)

#train=train.dropna()



#test=test.dropna()





train.Sex=train.Sex.replace({'male':1,'female':0})



test.Sex=test.Sex.replace({'male':1,'female':0})



from sklearn.utils import resample









df_Parch_0 = resample(train[train['Parch']=='0'],n_samples=500,replace=True,random_state=1)

df_Parch_1 = resample(train[train['Parch']=='1'],n_samples=500,replace=True,random_state=1)



df_Parch_2 = resample(train[train['Parch']=='2'],n_samples=500,replace=True,random_state=1)



df_Parch_3 = resample(train[train['Parch']=='3'],n_samples=500,replace=True,random_state=1)



df_Parch_4 = resample(train[train['Parch']=='4'],n_samples=500,replace=True,random_state=1)



df_Parch_5 = resample(train[train['Parch']=='5'],n_samples=500,replace=True,random_state=1)

df_Parch_6 = resample(train[train['Parch']=='6'],n_samples=500,replace=True,random_state=1)

#df_survived_1=resample(tita[tita['Survived']=='1'],n_samples=500,replace=True,random_state=1)

#df_survived_1=resample(tita[tita['Survived']=='0'],n_samples=500,replace=True,random_state=1)









train=pd.concat([df_Parch_0,df_Parch_1,df_Parch_2,df_Parch_3,df_Parch_4,df_Parch_5,df_Parch_6])
train.head()
test.head()
x=train.drop('Survived',axis=1)

y=train['Survived']

test=pd.get_dummies(test)
x=pd.get_dummies(x)
x=x.drop(['SibSp_0','Parch_0'],axis=1)



x.shape
test=test.drop(['SibSp_0','Parch_0','Parch_9'],axis=1)



test.shape
from sklearn.preprocessing import StandardScaler
features=['Age','Fare']

sc=StandardScaler()



sc.fit(x[features].values)

x[features]=sc.transform(x[features].values)

test[features]=sc.transform(test[features].values)



x.head()

test.head()
test.shape
x.shape
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=12)
#importing the requried libraries for model designing.

from  keras.models import Sequential

from keras.layers import Dense
# Initialising the NN

model = Sequential()



# layers

model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 18))

model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))

model.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))

model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))



# Compiling the ANN

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



model.summary()
# Train the ANN

history=model.fit(xtrain,ytrain, batch_size = 32, epochs = 200,validation_data=(xtest,ytest))
from matplotlib import pyplot

import seaborn as sns
# plot loss during training

pyplot.subplot(211)

pyplot.title('Loss')

pyplot.plot(history.history['loss'], label='train')

pyplot.plot(history.history['val_loss'], label='test')

pyplot.legend()

# plot accuracy during training

pyplot.subplot(212)

pyplot.title('Accuracy')

pyplot.plot(history.history['acc'], label='train')

pyplot.plot(history.history['val_acc'], label='test')

pyplot.legend()

pyplot.show()
ypred=model.predict(test)
y_final = (ypred > 0.5).astype(int).reshape(test.shape[0])


submission = pd.DataFrame({

        "PassengerId": test_main["PassengerId"],

        "Survived": y_final

    })
submission.head()
submission.to_csv('tita_nn_sub.csv',index=False)