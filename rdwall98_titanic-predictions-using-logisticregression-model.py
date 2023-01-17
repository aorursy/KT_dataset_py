import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import os
# Checking that you are in the same directory as the required files...
os.path.realpath('.')
# Importing datasets
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
test.head()
# Finding importance of data...
sns.boxplot(x='Survived', y='Pclass', data=train, palette='hls')
sns.boxplot(x='Survived', y='SibSp', data=train, palette='hls')
sns.boxplot(x='Survived', y='Fare', data=train, palette='hls')
sns.boxplot(x='Survived', y='Age', data=train, palette='hls')
train_data = train.drop(['PassengerId','Name','Ticket','Cabin', 'Embarked'], 1)
train_data['Sex'].replace(['female','male'],[0,1],inplace=True)
train_data.head()
# From data-mania.com to imputate the missing values for age
def age_approx(cols):
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
train_data['Age'] = train_data[['Age', 'Pclass']].apply(age_approx, axis=1)
test_data = test.drop(['PassengerId','Name','Ticket','Cabin', 'Embarked'], 1)
test_data['Age'] = test_data[['Age', 'Pclass']].apply(age_approx, axis=1)
test_data['Fare']  = test_data[['Fare', 'Pclass']].apply(age_approx, axis=1)
X = train_data.loc[:,('Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare')].values
y = train_data.loc[:,'Survived'].values
LogReg = LogisticRegression()
LogReg.fit(X, y)
train_data.head()
test_data['Sex'].replace(['female','male'],[0,1],inplace=True)
y_pred = LogReg.predict(test_data)
df = pd.DataFrame({ 'PassengerId': test['PassengerId'].values,
                            'Survived': y_pred})
df.to_csv("submission.csv", index=False)
