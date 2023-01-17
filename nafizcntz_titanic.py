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
import time

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from keras import models

from keras.layers import Dense, Dropout

from sklearn.preprocessing import MinMaxScaler
test = pd.read_csv("/kaggle/input/titanic/test.csv")

train = pd.read_csv("/kaggle/input/titanic/train.csv")
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
train.columns
test.shape[0] + train.shape[0]
train['Survived'].mean()
train.head(3)


test.columns[test.isna().any()]
train.columns[train.isna().any()]
def get_title(name):

    if '.' in name:

        return name.split(',')[1].split('.')[0].strip()

    else:

        return 'Unknown'

    

def replace_titles(x):

    title = x['Title']

    

    if title in ['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir']:

        return 'Mr'

    elif title in ['the Countess', 'Mme', 'Lady']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title == 'Dr':

        if x['Sex'] == 'male':

            return 'Mr'

        else:

            return 'Mrs'

        

    else:

        return title
def prepare_data(train):

    train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)

    

    train['Age'].fillna(train['Age'].median(), inplace=True)

    

    train['Fare'].fillna(train['Fare'].median(), inplace=True)

    

    

    scaler = MinMaxScaler()

    train [['Age','SibSp','Parch','Fare']] = scaler.fit_transform(train[['Age','SibSp','Parch','Fare']])

    

    

    train['Sex'] = train['Sex'].map({'female':0,'male':1})

    

    

    train_class = pd.get_dummies(train['Pclass'], prefix = 'Class')

    train[train_class.columns] = train_class

    

    

    train_emb = pd.get_dummies(train['Embarked'],prefix='Emb')

    train[train_emb.columns] = train_emb

    

    

    train['Title'] = train['Name'].map(lambda x: get_title(x))

    

    

    train['Title'] = train.apply(replace_titles, axis=1)

    

    

    train_title = pd.get_dummies(train['Title'],prefix='Title')

    train[train_title.columns] = train_title

    

    return

    
prepare_data(train)



train.columns
columns = ['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Class_1', 'Class_2',

       'Class_3', 'Emb_C', 'Emb_Q', 'Emb_S', 'Title_Master',

       'Title_Miss', 'Title_Mr', 'Title_Mrs']
X = np.array(train[columns])



y = np.array(train['Survived'])
network = models.Sequential()

network.add(Dense(32, activation='relu', ))

network.add(Dropout(rate=0.2))

network.add(Dense(16, activation='relu'))

network.add(Dropout(rate=0.2))

network.add(Dense(5, activation='relu'))

network.add(Dropout(rate=0.1))

network.add(Dense(1, activation='sigmoid'))
network.compile(optimizer='adam',

                loss='binary_crossentropy',

                metrics=['accuracy'])



history = network.fit(X, y, epochs=50, batch_size=10, verbose=0, validation_split=0.33)



plt.plot(history.history['accuracy'], label = 'eğitim')

plt.plot(history.history['val_accuracy'], label = 'test')

plt.title('Model Doğruluğu')

plt.ylabel('Doğruluk')

plt.xlabel('Epoch')

plt.legend(loc='lower right')

plt.show()
prepare_data(test)

 

X_pred = np.array(test[columns])

y_pred = network.predict(X_pred)

 

y_pred = y_pred.reshape(418)





train_subm = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':y_pred})



train_subm.head()
def binary(x):



    if x > 0.5:

        return 1

    else:

        return 0

    

    

    

train_subm['Survived'] = train_subm['Survived'].apply(binary)

train_subm.head(5)
time_string = time.strftime("%Y%m%d-%H%M%S")

 

# Dosya adını belirle

filename = 'titanic_submission_nafiz' '.csv'

 

# Csv olarak kaydet

train_subm.to_csv(filename,index=False)

 

print('Saved file: ' + filename)