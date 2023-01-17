import pandas as pd
import numpy as np
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
import os
data = pd.read_csv('/kaggle/input/titanic_data.csv')
data.head(10)
data.shape
data.info()
sns.boxplot(data['Age']);
(20+22+20+24+80)/5
20,20,22,24,80
data.Age.median()
data['Age'].fillna(data.Age.median(),inplace=True)
data.info()
data.head()
data['family_size'] = data['SibSp']+data['Parch']+1
data.head()
cols_to_remove = ['PassengerId','Name','SibSp','Parch','Fare']
data.drop(cols_to_remove, axis=1, inplace=True)
data.head()
data['Sex'].unique()
data_with_dummies = pd.get_dummies(data, columns=['Sex'],drop_first=True)
data_with_dummies.head()
sns.pairplot(data_with_dummies);
data_with_dummies.head()
data_with_dummies.columns
features = ['Pclass', 'Age', 'family_size', 'Sex_male']
target = ['Survived']
x = data_with_dummies[features]
y = data_with_dummies[target]
x.head()
y.head()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=96)
print('x-train shape', x_train.shape)
print('x-test shape', x_test.shape)
print('y-train shape', y_train.shape)
print('y-test shape', y_test.shape)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
y_pred = log_reg.predict(x_test)
y_pred
log_reg.predict_proba(x_test)[:,1]
y_pred.sum()
from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(y_test, y_pred)
pd.DataFrame(confusion_matrix(y_test, y_pred))
pd.DataFrame(confusion_matrix(y_test, y_pred), columns=['predicted_not_survived','predicted_survived'], index=['actual_not_survived','actual_survived'])
(105+41)/(105+18+15+41)
accuracy_score(y_test, y_pred)
41/(41+18)

pd.DataFrame([[90,10],[0,0]], columns=['predicted_negative','predicted_positive'], index=['actual_negative','actual_positive'])
