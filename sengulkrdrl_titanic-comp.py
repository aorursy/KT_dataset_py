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
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.impute import SimpleImputer

import plotly.express as px
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")
train["Pclass"].unique()
fig = px.sunburst(train, path=['Pclass', 'Survived'], values='PassengerId',color='Pclass',

                  color_continuous_scale='RdBu',

                  color_continuous_midpoint=np.average(train['Pclass']))

                  

fig.show()
def family(dataFrame):

    dataFrame["Family"] = dataFrame["SibSp"] + dataFrame["Parch"]

    dataFrame["Family"] = dataFrame["Family"].apply(lambda x: 1 if x>0 else 0)

    return dataFrame
train = family(train)
train
train =train.drop(["SibSp","Parch"], axis = 1)
train = pd.get_dummies(train, columns=['Pclass', 'Sex', 'Embarked'])

train
train = train.drop(["Name","Cabin", "Ticket"], axis = 1)

train
train = train.fillna(train.Age.median())
train
test.head()
test = family(test)
test =test.drop(["SibSp","Parch"], axis = 1)
test
test = pd.get_dummies(test, columns=['Pclass', 'Sex', 'Embarked'])
test
test = test.drop(["Name","Cabin", "Ticket"], axis = 1)

test
test = test.fillna(test.Age.median())
test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop(columns=["Survived"]), train["Survived"], random_state = 42)  
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error, confusion_matrix, accuracy_score, classification_report



features = ["Family","Pclass_1","Pclass_2","Pclass_3","Sex_female","Sex_male","Embarked_C","Embarked_Q","Embarked_S","Fare", "Age"]



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(f''' MAE : {mean_absolute_error(y_test, predictions)}

 Confusion Matrix: 

 {confusion_matrix(y_test, predictions)}

Accuracy:  {accuracy_score(y_test, predictions)}

Classification report:{ classification_report(y_test, predictions)} ''')
from xgboost import XGBRegressor

my_model_2 = XGBRegressor(n_estimators=1000, learning_rate=0.05)



# Fit the model

my_model_2.fit(X_train, y_train)



# Get predictions

predictions_2 = my_model_2.predict(X_test)



# Calculate MAE

mae_2 = mean_absolute_error(predictions_2, y_test)

print("Mean Absolute Error:" , mae_2)
# make predictions which we will submit. 

test_preds = my_model_2.predict(test)
# Save test predictions to file

output = pd.DataFrame({'PassengerId': test.PassengerId,

                       'Survived': submit})

output.to_csv('submission.csv', index=False)