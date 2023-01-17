import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
train_titanic= pd.read_csv('../input/titanic/train.csv')

test_titanic= pd.read_csv('../input/titanic/test.csv')
#lets see how many columns and rows do we have here

train_titanic.shape
check_null=test_titanic.isnull()

check_null.sum()
check_null=train_titanic.isnull()

check_null.sum()
sns.heatmap(test_titanic.isnull(),yticklabels=False,cbar=False,cmap='cividis')
sns.heatmap(train_titanic.isnull(),yticklabels=False,cbar=False,cmap='cividis')
Ratio = {"age ratio": [177/891],

        "cabin ratio": [687/891],

         "embarked ratio": [ 2/891]

        }



df_Ratio = pd.DataFrame(Ratio, columns = ["age ratio", "cabin ratio", "embarked ratio"])

print(df_Ratio)
train_titanic.info()
sns.set_style('darkgrid')

sns.countplot(x='Survived',data=train_titanic)
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Sex',data=train_titanic,palette='RdBu_r')
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Pclass',data=train_titanic,palette='rainbow')
train_titanic_drop = train_titanic.drop(['Cabin'], axis = 1)

test_titanic_drop = test_titanic.drop(['Cabin'], axis = 1)
sns.heatmap(train_titanic_drop.isnull(),yticklabels=False,cbar=False,cmap='cividis')



sns.heatmap(test_titanic_drop.isnull(),yticklabels=False,cbar=False,cmap='cividis')
list(train_titanic_drop['Embarked'].unique())

list(test_titanic_drop['Embarked'].unique())
train_new=train_titanic_drop.dropna(subset=['Embarked'])

test_new=test_titanic_drop.dropna(subset=['Embarked'])
from sklearn.preprocessing import LabelEncoder

# Make copy to avoid changing original data 

train_new = train_new.copy()

test_new = test_new.copy()

# Apply label encoder to column with categorical data

label_encoder = LabelEncoder()

train_new['Embarked_Cat'] = label_encoder.fit_transform(train_new['Embarked'])

train_newer = train_new.drop(['Embarked'], axis = 1)

train_newer.head()

test_new['Embarked_Cat'] = label_encoder.fit_transform(test_new['Embarked'])

test_newer = test_new.drop(['Embarked'], axis = 1)

test_newer.head()
# Make copy to avoid changing original data 

train_newer = train_newer.copy()

test_newer = test_newer.copy()

# Apply label encoder to column with categorical data

label_encoder = LabelEncoder()

train_newer['Sex_Cat'] = label_encoder.fit_transform(train_newer['Sex'])

train_final = train_newer.drop(['Sex'], axis = 1)

train_final.head()

test_newer['Sex_Cat'] = label_encoder.fit_transform(test_newer['Sex'])

test_final = test_newer.drop(['Sex'], axis = 1)

test_final.head()
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputer_train= imputer.fit(train_final[['Age']])

imputer_test=imputer.fit(test_final[['Age']])

train_final[['Age']] = imputer.transform(train_final[['Age']])

test_final[['Age']] = imputer.transform(test_final[['Age']])

train_final[['Age']]
sns.heatmap(train_final.isnull(),yticklabels=False,cbar=False,cmap='cividis')
test_final.Embarked_Cat.isnull().sum()
sns.heatmap(test_final.isnull(),yticklabels=False,cbar=False,cmap='cividis')
# Test



x= train_final.loc[:, ['PassengerId', 'Age','SibSp','Fare', 'Parch', 'Pclass', 'Embarked_Cat', 'Sex_Cat']]

y=train_final.Survived



y
correlation = x.corr()

correlation
sns.heatmap(correlation)
correlation.style.background_gradient()
X_train= train_final.loc[:, ['PassengerId', 'SibSp', 'Parch', 'Pclass', 'Embarked_Cat', 'Sex_Cat']]

y_train=train_final['Survived']



combined = [X_train, y_train]

combined
X_test  = test_titanic.loc[:, ['PassengerId', 'SibSp', 'Parch', 'Pclass', 'Embarked_Cat', 'Sex_Cat']]

X_train.shape, y_train.shape, X_test.shape
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train, y_train)
prediction = logmodel.predict(X_test)
prediction
submission = pd.DataFrame({'PassengerId': X_test.PassengerId, 'Survived':prediction})

submission.to_csv('submission.csv',index=False)

submission