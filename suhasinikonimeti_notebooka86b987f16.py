import numpy as np 

import pandas as pd 
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df= pd.read_csv('../input/titanic/train.csv')
sns.pairplot(df)
corr = df.corr()

corr
df.isnull().sum()
# selecting useful features from the original dataframe.

training=df.loc[:, ['PassengerId', 'Survived', 'Pclass', 'Sex',

                 'Age', 'SibSp','Parch', 'Fare', 'Embarked' ]]
# since there are missing values in age the values that are missing is replaced with the median of age.

median= training['Age'].median()

training['Age'].fillna(median, inplace=True)

training['Age'].describe()
groups = [training['Age'].between(0, 3), 

          training['Age'].between(4, 9), 

          training['Age'].between(10, 18),

          training['Age'].between(19, 59),

          training['Age'].between(60, 80)]

values = [1, 2, 3, 4, 5]

training['AgeCategory'] = np.select(groups, values, 0)

training
training= training.drop(['Age'], axis=1)
# sex is given categorical values

d = {'male': 1, 'female': 0}

training['Sex'] = training['Sex'].map(d)

training['Embarked'].value_counts()
# the missing values are filled ith S since S is the most repeated value

training['Embarked'].fillna(value='S', inplace=True)
#embark is given categorical values

d = {'Q': 1, 'C': 2, 'S': 3}

training['Embarked'] = training['Embarked'].map(d)

training.head()
from sklearn.model_selection import train_test_split



predictors = training.drop(['Survived', 'PassengerId'], axis=1)

target = training["Survived"]

x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.25, random_state = 0)
testing= pd.read_csv('../input/titanic/test.csv')


testing=testing.loc[:, ['PassengerId', 'Pclass', 'Sex',

                  'Age', 'SibSp','Parch', 'Fare',

                  'Embarked' ]]
age_median=testing['Age'].median()

testing['Age'].fillna(value= age_median, inplace=True)
groups = [testing['Age'].between(0, 3), 

          testing['Age'].between(4, 9), 

          testing['Age'].between(10, 18),

          testing['Age'].between(19, 59),

          testing['Age'].between(60, 76)]

 



values = [1, 2, 3, 4, 5]



testing['AgeCategory'] = np.select(groups, values, 0)

testing=testing.drop(['Age'], axis=1)
fare_median=testing['Fare'].median()

testing['Fare'].fillna(value= fare_median, inplace=True)
# converting the values from sex variable into 1 an 0

d = {'male': 1, 'female': 0}

testing['Sex'] = testing['Sex'].map(d)

d={'Q':1, 'C':2, 'S':3}

testing['Embarked']= testing['Embarked'].map(d)
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV 

from sklearn.ensemble import GradientBoostingClassifier

param_grid={'n_estimators':[100], 

            'learning_rate': [0.1],

            'max_depth':[4], 

            'min_samples_leaf':[3], 

            'max_features':[1.0] } 

n_jobs=4

estimator = GradientBoostingClassifier() 

classifier = GridSearchCV(estimator=estimator, cv=5, param_grid=param_grid, n_jobs=n_jobs)

gbc = classifier.fit(x_train, y_train)

y_pred = gbc.predict(x_val)

gbc_cv = round(accuracy_score(y_pred, y_val) * 100, 2)

print(gbc_cv)
predictions_gbc= gbc.predict(testing.drop('PassengerId',axis=1))
#saving predictions in CSV

ids = testing["PassengerId"]

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions_gbc})

output.to_csv('kaggle_submit_gbc', index=False)