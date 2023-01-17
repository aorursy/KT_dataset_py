# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline



np.random.seed(1)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,Activation

from keras.optimizers import RMSprop,SGD

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.callbacks import ModelCheckpoint

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/titanic/train.csv')
train
train.info()
g = sns.heatmap(train.isnull())
train.drop('Cabin',axis = 1,inplace = True)
train['Age'].isnull().sum()
ax = sns.catplot(x='Pclass',hue = 'Sex',col ='Survived' ,data = train , kind = 'count')
plt.figure(figsize = (18,20))

g = sns.FacetGrid(train,col = 'Survived',height = 5)

g= g.map(plt.hist,'Age',bins = 20)
def input_age(col) :

   age = col[0]

   pclass = col[1]

   if pd.isnull(age):

    if pclass == 1  :

       return 37

    elif pclass == 2  :

       return 29 

    elif pclass == 3  :

       return 24

   else :

    return age
train['Age'] = train[['Age','Pclass']].apply(input_age,axis = 1)
g = sns.heatmap(train.isnull())#No null value
dummy = pd.get_dummies(train['Sex'])
train = pd.concat([train,dummy],axis = 1)
train.drop('Sex',axis = 1,inplace = True)
dummy = pd.get_dummies(train['Pclass'],prefix='Pclass')
train = pd.concat([train,dummy],axis = 1)
dummy = pd.get_dummies(train['Embarked'],prefix='Embarked')

train = pd.concat([train,dummy],axis = 1)
train.drop(['Embarked'],axis = 1,inplace = True)
train.drop('Name',axis = 1,inplace = True)
train.drop(['Ticket'],axis = 1,inplace = True)# null vallue are present using describe()
train.drop(['PassengerId'],axis = 1,inplace = True)
train
train['Fare'].describe()
plt.figure(figsize=(16,8))

g= sns.heatmap(train.corr(),annot=True)
def input_age(col) :

    fare = col

    if fare <= 7.9  :

       return 0

    elif fare > 7.9 and fare <= 14.45  :

       return 1 

    elif fare > 14.45 and fare <= 31.00  :

       return 2

    else :

       return 3
train['Fare'] = train['Fare'].apply(input_age)
def input_age(col) :

    fare = col

    if fare <= 16  :

       return 0

    elif fare > 16 and fare <= 32  :

       return 1 

    elif fare > 32 and fare <= 48  :

       return 2

    else :

       return 3
train['Age'] = train['Age'].apply(input_age)
train.drop(['SibSp','Parch'],axis = 1,inplace = True)
train.drop(['Pclass'],axis = 1,inplace = True)
train
dummy = pd.get_dummies(train['Fare'],prefix='Fare')

train = pd.concat([train,dummy],axis = 1)

train.drop(['Fare'],axis = 1,inplace = True)
dummy = pd.get_dummies(train['Age'],prefix='Age')

train = pd.concat([train,dummy],axis = 1)

train.drop(['Age'],axis = 1,inplace = True)
train
y = train['Survived']
train.drop('Survived',axis = 1,inplace = True)
y_train = to_categorical(y, num_classes = 2)
X_train, X_val, Y_train, Y_val = train_test_split(train, y_train, test_size = 0.25, random_state=1)
model = Sequential()

model.add(Dense(128,input_dim=X_train.shape[1]))

model.add(Activation('relu'))

model.add(Dropout(0.2))

model.add(Dense(512, activation = "relu"))

model.add(Dropout(0.45))

model.add(Dense(2, activation = "softmax"))
optimizer = RMSprop(lr=0.001)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

from keras.callbacks import ModelCheckpoint

filepath="weights.best.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
history = model.fit(X_train, Y_train,epochs =100, 

         validation_data = (X_val, Y_val),callbacks = [checkpoint], verbose = 2)
model.load_weights("weights.best.hdf5")
score = model.evaluate(X_val,Y_val)
score[1]
test = pd.read_csv('/kaggle/input/titanic/test.csv')
test.drop(['Ticket','Cabin','Name','PassengerId','SibSp','Parch'],axis = 1,inplace = True)
dummy = pd.get_dummies(test['Sex'])

dummy1 = pd.get_dummies(test['Pclass'],prefix='Pclass')

dummy2 = pd.get_dummies(test['Embarked'],prefix='Embarked')

test = pd.concat([test,dummy,dummy1,dummy2],axis = 1)
test.drop(['Sex','Embarked'],axis = 1,inplace = True)
test
def input_age(col) :

   age = col[0]

   pclass = col[1]

   if pd.isnull(age):

    if pclass == 1  :

       return 37

    elif pclass == 2  :

       return 29 

    elif pclass == 3  :

       return 24

   else :

    return age
test['Age'] = test[['Age','Pclass']].apply(input_age,axis = 1)
test
def input_age(col) :

    fare = col

    if fare <= 7.9  :

       return 0

    elif fare > 7.9 and fare <= 14.45  :

       return 1 

    elif fare > 14.45 and fare <= 31.00  :

       return 2

    else :

       return 3
test['Fare'] = test['Fare'].apply(input_age)
def input_age(col) :

    fare = col

    if fare <= 16  :

       return 0

    elif fare > 16 and fare <= 32  :

       return 1 

    elif fare > 32 and fare <= 48  :

       return 2

    else :

       return 3
test['Age'] = test['Age'].apply(input_age)
test


dummy1 = pd.get_dummies(test['Fare'],prefix ='Fare' )

dummy2 = pd.get_dummies(test['Age'],prefix='Age')

test = pd.concat([test,dummy1,dummy2],axis = 1)
test
test.drop(['Pclass','Fare','Age'],axis = 1 , inplace = True)
test
result = model.predict(test)

result
result = np.argmax(result,axis = 1)
result = pd.Series(result,name="Survived")

submission = pd.concat([pd.Series(range(892,1310),name = "PassengerId"),result],axis = 1)
submission.to_csv("Titanic.csv",index=False)
submission