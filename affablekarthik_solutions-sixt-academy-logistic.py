import pandas as pd
import numpy as np
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
import os
data = pd.read_csv('/kaggle/input/titanic_data.csv')
data.head()
data.shape
data.info()
data.Age.median()
data['Age'].fillna(data.Age.median(), inplace=True)
data.info()
data.head()
data['family_size'] = data['SibSp']+data['Parch']+1
data.head()
cols_to_remove = ['PassengerId','Name','SibSp','Parch','Fare']
data.drop(cols_to_remove, axis=1, inplace=True)
data.head()
data = pd.get_dummies(data, columns=['Sex'],drop_first=True)
data.head()
sns.boxplot(data=data, x='Survived', y='Age',hue='Sex_male');
data.columns
features = ['Pclass', 'Age', 'family_size', 'Sex_male']
target = ['Survived']
x = data[features]
y = data[target]
x.head()
y.head()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1, random_state=96)
print('x_train shape', x_train.shape)
print('y_train shape', y_train.shape)
print('x_test shape', x_test.shape)
print('y_test shape', y_test.shape)
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model.fit(x_train, y_train)
y_pred = log_model.predict(x_test)
y_pred
log_model.predict_proba(x_test)
from sklearn.metrics import accuracy_score, confusion_matrix
confusion_matrix(y_test, y_pred)
pd.DataFrame(confusion_matrix(y_test, y_pred))
pd.DataFrame(confusion_matrix(y_test, y_pred),
             columns= ['Predicted_not_survived', 'predicted_survived'], 
             index = ['Actual_not_survived', 'Actual_survived'])
(58+15)/(58+10+7+15)
accuracy_score(y_test, y_pred)
#class 1
15/(15+7)
#class 0
58/(58+10)
