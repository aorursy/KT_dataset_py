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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train= pd.read_csv('/kaggle/input/titanic/train.csv')

test= pd.read_csv('/kaggle/input/titanic/test.csv')
train.info()
test.info()
train = train[train.Embarked.notna()]

test['Fare'].fillna(test.groupby('Pclass')['Fare'].transform('mean'), inplace=True)
#operations common to train and test datasets

datasets = [train,test]



for data in datasets:

    data['male'] = data.Sex.apply(lambda x: 1 if x=='male' else 0)

    data['Age'].fillna(data.groupby('Pclass')['Age'].transform('mean'), inplace=True)

    data['numfam'] = data['SibSp']+data['Parch']+1

    data['boy'] = data.apply(lambda row: 1 if (row['male']==1) & (row['Age']<=10) else 0, axis=1)

    data['EmbarkC'] = data.Embarked.apply(lambda x: 1 if x=='C' else 0)

    data['EmbarkQ'] = data.Embarked.apply(lambda x: 1 if x=='Q' else 0)

    data['bigfam'] = data['numfam'].apply(lambda x: 1 if x>4 else 0)

    data['single'] = data['numfam'].apply(lambda x: 1 if x==1 else 0)

    data['class1'] = data.Embarked.apply(lambda x: 1 if x==1 else 0)

    data['class2'] = data.Embarked.apply(lambda x: 1 if x==2 else 0)
X = train[['class1','class2','male', 'EmbarkC', 'EmbarkQ','boy','bigfam','single']]



y= train['Survived']
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
#building ANN

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation



model = Sequential() 



model.add(Dense(8, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='adam')
#running ANN

model.fit(x=X_train, 

          y=y_train, 

          epochs=30,

          validation_data=(X_test, y_test), verbose=1

          )
model_loss = pd.DataFrame(model.history.history)

model_loss.plot()
y_pred = model.predict_classes(X_test)
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score
print(classification_report(y_test,y_pred))
test_final = test[['class1','class2','male', 'EmbarkC', 'EmbarkQ','boy','bigfam','single']]
fullscaler = StandardScaler()



X_scaled = fullscaler.fit_transform(X)



test_final = fullscaler.transform(test_final)
# getting ready for submission

#running ANN

model.fit(x=X_scaled, 

          y=y, 

          epochs=40,

         verbose=1

          )
y_prediction = model.predict_classes(test_final)
submit_annmodel = pd.DataFrame(data = y_prediction, columns = ['Survived'])
submit_annmodel['PassengerId'] = test['PassengerId']
submit_annmodel.to_csv('/kaggle/working/submit_annmodel.csv', index=False)