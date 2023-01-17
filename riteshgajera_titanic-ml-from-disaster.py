import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
train = pd.read_csv('../input/train.csv')
train.info()
train.head()
# Check Missing Data
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='YlGnBu')
sns.set_style('whitegrid')
sns.countplot(x='Survived', data=train)
sns.countplot(x='Survived', data=train, hue='Sex', palette='RdBu_r')
sns.countplot(x='Survived', data=train, hue='Pclass')
sns.distplot(train['Age'].dropna(), kde=False, bins=30)
train['Age'].plot.hist(bins=30)
sns.countplot(x='SibSp', data=train)
train['Fare'].hist(bins=40, figsize=(10,4))
import cufflinks as cf
cf.go_offline()
train['Fare'].iplot(kind='hist', bins=50)
sns.boxplot(x='Pclass', y='Age', data=train)
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
            
train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='BuPu')
train.drop('Cabin', axis=1, inplace=True)
train.head()
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='Greens')
train.dropna(inplace=True)
sex = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)
train = pd.concat([train,sex,embark], axis=1)
train.head()
train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
train.head()
train.tail()
train.drop(['PassengerId'], axis=1, inplace=True)
train.head()
X = train.drop('Survived', axis=1)
y = train['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))