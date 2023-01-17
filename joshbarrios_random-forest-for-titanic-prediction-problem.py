# Import libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt
import seaborn as sb
train = pd.read_csv('/kaggle/input/titanic/train.csv')
print(train.info())

test = pd.read_csv('/kaggle/input/titanic/test.csv')
import pandas_profiling
train.profile_report()
# build and train model for predicting age
from sklearn.ensemble import RandomForestRegressor

# combine original training and testing datasets, drop survival from training
new_train = train.copy()
del new_train['Survived']
all_data = pd.concat([new_train, test])
# Exclude rows where Age is null
all_data = all_data.loc[all_data.Age.notnull()]
# Fill in missing value in Fare with 0
all_data['Fare'].fillna(0, inplace=True)

y = all_data['Age']
x = all_data[['Parch', 'SibSp', 'Fare']]

# Build and train model
rfr = RandomForestRegressor(n_estimators=1500, n_jobs=-1)
rfr.fit(x,y)
    
# function for predicting age with model
def predict_age(df):
    x = df[['Parch', 'SibSp', 'Fare']].loc[df.Age.isnull()]
    df['Age'].loc[df.Age.isnull()] = rfr.predict(x)
    
# Run predict_age on train and test
#x = train[['Parch', 'SibSp', 'Fare']].loc[train.Age.isnull()]
#y = rfr.predict(x).round()
#train.loc['Age']
predict_age(train)
predict_age(test)
def find_deck(df):
    df['Cabin'].loc[df.Cabin.isnull()] = 'N'
    df['Deck'] = [i[0] for i in df.Cabin]
    
find_deck(train)
find_deck(test)
train['Embarked'].fillna('S', inplace = True)
def get_titles(df):
    df['Title'] = [i.split('.')[0] for i in df.Name]
    df["Title"] = [i.split(',')[1].strip() for i in df.Title]
    
get_titles(train)
get_titles(test)
# Check out our new Title column
train.profile_report()
common_titles = ['Mr', 'Miss', 'Mrs', 'Master']
train['Title'].loc[~train['Title'].isin(common_titles)] = 'Other'
test['Title'].loc[~test['Title'].isin(common_titles)] = 'Other'
train['Deck'].loc[train['Deck'].isin(['T'])] = 'A'
def clear_undesirables(df):
    df.drop('PassengerId', axis=1, inplace=True)
    df.drop('Name', axis=1, inplace=True)
    df.drop('Ticket', axis=1, inplace=True)
    df.drop('Cabin', axis=1, inplace=True)
    
clear_undesirables(train)
clear_undesirables(test)
list(train)
train = pd.get_dummies(train, columns=['Pclass','Sex','Embarked','Deck','Title'], drop_first=False)
test = pd.get_dummies(test, columns=['Pclass','Sex','Embarked','Deck','Title'], drop_first=False)
test.head()
list(train)
X = train.drop(['Survived'], axis = 1)
y = train['Survived']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .33, random_state=0)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(max_depth=5, random_state=0)
rfc.fit(X_train, y_train)

test_prediction = rfc.predict(X_test)

# Evaluate the model, print accuracy, precision, recall, and F1 score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
    
print("Random forest classifier")
print("Accuracy Score: ", '{:,.3f}'.format(float(accuracy_score(y_test, test_prediction)) * 100), "%")
print("     Precision: ", '{:,.3f}'.format(float(precision_score(y_test, test_prediction, average='micro')) * 100), "%")
print("        Recall: ", '{:,.3f}'.format(float(recall_score(y_test, test_prediction, average='micro')) * 100), "%")
print("      F1 score: ", '{:,.3f}'.format(float(f1_score(y_test, test_prediction, average='micro')) * 100), "%")
list(X_train)
list(test)
# Fill in missing value in Fare in the test set
test['Fare'].fillna(0, inplace=True)
# Run prediction on test data, create results file

test_prediction = rfc.predict(test)

# Re-open test.csv to grab PassengerId
test_new = pd.read_csv('/kaggle/input/titanic/test.csv')

submission = pd.DataFrame({
        "PassengerId": test_new['PassengerId'],
        "Survived": test_prediction
    })

submission.PassengerId = submission.PassengerId.astype(int)
submission.Survived = submission.Survived.astype(int)

submission.to_csv("RFC_Barrios_submission.csv", index=False)
