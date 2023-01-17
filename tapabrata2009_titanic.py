# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
sns.distplot(train['Age'].dropna(),kde=True,color='darkred',bins=30)
sns.countplot(x='SibSp',data=train)
train['Fare'].hist(color='green',bins=40,figsize=(8,4))
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.info()
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 29

        else:

            return 24

    else:

        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)
train.drop('Cabin',axis=1,inplace=True)

test.drop('Cabin', axis=1, inplace=True)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.head()
sex = pd.get_dummies(train['Sex'],drop_first=True)

embark = pd.get_dummies(train['Embarked'],drop_first=True)

train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

train = pd.concat([train,sex,embark],axis=1)

y = train['Survived'].values
sex = pd.get_dummies(test['Sex'],drop_first=True)

embark = pd.get_dummies(test['Embarked'],drop_first=True)

test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

test = pd.concat([test,sex,embark],axis=1)

test['Fare'][152] = test['Fare'].mean()
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.head()
test.head()
from keras.models import Sequential

from keras.optimizers import Adam

from keras.layers import Dense
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(train.drop(['Survived'], axis=1))

test = sc.transform(test)
print(train.shape)

print(test.shape)

# Initialising the ANN

classifier = Sequential()

# Adding the input layer and the first hidden layer

classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu', input_dim = 9))

# Adding the second hidden layer

classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the model

classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X, y, batch_size = 32, epochs = 50)
# import the machine learning library that holds the randomforest

import sklearn.ensemble as ske



#instantiate and fit our model

results_rf = ske.RandomForestClassifier(n_estimators=15).fit(X, y)



# Score the results

score = results_rf.score(X, y)

print("Mean accuracy of Random Forest Predictions on the data was: {0}".format(score))
sub = results_rf.predict(test)
test = pd.read_csv('../input/test.csv')

submission = pd.DataFrame({'PassengerId': test["PassengerId"], 'Survived': sub})

print(submission.head(10))

submission.to_csv('submission_own.csv')