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
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train=pd.read_csv('/kaggle/input/titanic/train.csv')

test= pd.read_csv('/kaggle/input/titanic/test.csv')

gender=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
train.head()
test.head()
train.isnull().sum()
test.isnull().sum()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.set_style('whitegrid')

sns.countplot(x='Survived',data=train,palette='winter')
sns.set_style('whitegrid')

sns.countplot(x='Survived',data=train,palette='winter',hue='Sex')
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
plt.figure(figsize=(12,6))

sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)
train['Age'].hist(bins=30,color='darkred',alpha=0.7)
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=train,palette='rainbow')

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
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='winter')
train.drop('Cabin',axis=1,inplace=True)
train.head()
train.describe()
train.info()
train.dropna(inplace=True)
sex = pd.get_dummies(train['Sex'],drop_first=True)

embark = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 

                                                    train['Survived'], test_size=0.30, 

                                                    random_state=101)
from sklearn.linear_model import LogisticRegression
lr= LogisticRegression()

lr.fit(X_train,y_train)
pred = lr.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))