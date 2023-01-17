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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()
sns.set_style('whitegrid')
sns.countplot(x='Survived', data=train_data)
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)

#visualizing % of Male and female survived
sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Sex', data=train_data)
#cheking for missing values
train_data.isnull().sum()
#cheking for missing values
test_data.isnull().sum()
sns.distplot(train_data['Age'].dropna(), kde=False, color='blue', bins=40)
sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Pclass', data=train_data)
#visualization for relationship between Age and Pclass
plt.figure(figsize=(15,6))
sns.boxplot(x='Pclass', y='Age', data = train_data, palette='winter')
#since the age is corealted with Pclass we have to add mean or average for the age

def  age_impute(cols):
  Age = cols[0]
  Pclass=cols[1]
  if pd.isnull(Age):
    if Pclass==1:
      return 37
    elif Pclass==2:
      return 29
    else:
      return 24
  else:
    return Age
#impute age
train_data['Age'] = train_data[['Age', 'Pclass']].apply(age_impute, axis=1)
test_data['Age'] = train_data[['Age', 'Pclass']].apply(age_impute, axis=1)

train_data.isnull().sum()
#droping cabin coloum in train and test set 
train_data.drop('Cabin',axis=1, inplace=True)
test_data.drop('Cabin',axis=1, inplace=True)

train_data['Embarked'].isnull().sum()
train_data['Embarked']=train_data['Embarked'].replace(np.NaN, train_data['Embarked'].mode())
train_data.isnull().sum()
test_data.isnull().sum()
test_data['Fare'].mean()
test_data['Fare']=test_data['Fare'].replace(np.NaN, train_data['Fare'].mean())
test_data.isnull().sum()
#caring categorical values -----Train set------
sex = pd.get_dummies(train_data['Sex'],drop_first=True)
embarked = pd.get_dummies(train_data['Embarked'],drop_first=True)
print(sex.head())
print(embarked.head())
#droping unwanted columns
train_data.drop(['Sex','Embarked','Name','Ticket'], axis=1, inplace=True)
train_data.head()
#Concat
train_data = pd.concat([train_data,sex,embarked],axis=1)
train_data.head()
#caring categorical values -----Test set------
gender = pd.get_dummies(test_data['Sex'],drop_first=True)
embark = pd.get_dummies(test_data['Embarked'],drop_first=True)
print(gender.head())
print(embark.head())
#droping unwanted columns
test_data.drop(['Sex','Embarked','Name','Ticket'], axis=1, inplace=True)
test_data.head()
#Concat
test_data = pd.concat([test_data,gender,embark],axis=1)
test_data.head()
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(train_data.drop(['Survived','PassengerId'], axis=1), 
                                                    train_data['Survived'], test_size = 0.2, 
                                                    random_state = 0)
#model building
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=500, criterion='entropy')
classifier.fit(X_train,y_train)
predictions = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,predictions)
cm
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
accuracy
from xgboost import XGBClassifier
xgb_classifier = XGBClassifier()
xgb_classifier.fit(X_train,y_train)
xgb_predictions = xgb_classifier.predict(X_test)
#confusion matrix
xgb_cm = confusion_matrix(y_test, xgb_predictions)
xgb_cm
#checking accuracy
xgb_accuracy = accuracy_score(y_test,xgb_predictions)
xgb_accuracy
from sklearn.ensemble import GradientBoostingClassifier
gbc= GradientBoostingClassifier()
gbc.fit(X_train,y_train)
#prediciion
gbc_predictions = gbc.predict(X_test)
#confusion_matrix
gbc_cm = confusion_matrix(y_test, gbc_predictions)
gbc_cm
gbc_accuracy = accuracy_score(y_test, gbc_predictions)
gbc_accuracy
from sklearn.model_selection import cross_val_score
crossval = cross_val_score(estimator = classifier , X= X_train ,y= y_train, cv = 10)
crossval.mean()
#checking accuracy of each model
models={'MODEL':['RANDOM FOREST','XG BOOSTING','GRADIENT BOOSTING'],'ACCURACY':[accuracy,xgb_accuracy,crossval.mean()]}
model_accuracy=pd.DataFrame(models)
model_accuracy
train_data.head()
test_data.head()
passenger_id = test_data['PassengerId']
predict_values = gbc.predict(test_data.drop('PassengerId', axis=1))


output = pd.DataFrame({ 'PassengerId' : passenger_id, 'Survived': predict_values })
output.to_csv('submission.csv', index=False)


predictions