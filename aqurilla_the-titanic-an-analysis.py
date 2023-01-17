import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv('../input/train.csv')
train.head()
train.info()
train.describe()
train.isnull()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train)
sns.countplot(x='Survived',data=train,hue='Sex',palette='RdBu_r')
sns.countplot(x='Survived',data=train,hue='Pclass')
sns.distplot(train['Age'].dropna(),kde=False,bins=30)
# Check out the sibling or spouse column

sns.countplot(x='SibSp',data=train)
sns.set(rc={"figure.figsize": (10, 6)})

sns.distplot(train['Fare'].dropna(),kde=False,bins=50)
#import cufflinks as cf
#cf.go_offline()
#train['Fare'].iplot(kind='hist',bins=100)
plt.figure(figsize=(10,7))

sns.boxplot(x='Pclass',y='Age',data=train)
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if (Pclass == 1):

            return 37

        elif (Pclass == 2):

            return 29

        else:

            return 24

    else:

        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age, axis=1)
train.info()
# We drop the cabin column, since it has too much missing information

train.drop('Cabin', inplace=True, axis=1)

train.drop('Ticket', inplace=True, axis=1)
train.info()
train.dropna(inplace=True)
# Categorical features have to be converted to numerical dummy variables, in order for the machine learning

# algorithm to accept that data

train.head()
train['Sex'] = pd.get_dummies(train['Sex'],drop_first=True)
train.head()

# The 'Sex' column shows 1 for Male and 0 for Female
train.drop('Embarked', inplace=True, axis=1)
train.head()
# We also currently dont use the name

train.drop('Name', inplace=True, axis=1)

train.head()
# We also dont need the passengerID since it plays no role in predicting survivability

train.drop('PassengerId', inplace=True, axis=1)

train.head()
# The Pclass column is also categorical data, and should be replaced with the dummy variables

pclass = pd.get_dummies(train['Pclass'], drop_first=True)

pclass.columns=['Class=2','Class=3']

pclass
train.head()
train=pd.concat([train,pclass],axis=1)

train.head()
train.drop('Pclass',inplace=True,axis=1)

train.head()
train.columns
X = train[['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Class=2','Class=3']]

y = train['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
# Create and train model

from sklearn.linear_model import LogisticRegression
logmod = LogisticRegression()
logmod.fit(X_train,y_train)
predictions = logmod.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(y_test,predictions),annot=True)