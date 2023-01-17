import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv('titanic_train.csv')

#train.describe()
train.head()
train.isnull().sum(axis = 0)

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

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
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

train.isnull().sum(axis = 0)
train.drop('Cabin',axis=1,inplace=True)
sex = pd.get_dummies(train['Sex'],drop_first=True)

embark = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)
train.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 

                                        train['Survived'], test_size=0.30, 

                                        random_state=None,shuffle=False)
y_test.shape
print(y_test)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
prediction = logmodel.predict(X_test)

print(prediction)
prediction.shape


from sklearn.metrics import classification_report
classification_report(y_test,prediction)


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,prediction)