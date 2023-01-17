# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd
# data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df= pd.read_csv("../input/titanic/train.csv")
df_test= pd.read_csv("../input/titanic/test.csv")
df.head()
df['Died'] = 1 - df['Survived']

df.groupby('Sex').agg('mean')[['Survived', 'Died']].plot(kind='bar', figsize=(10,6), stacked=True);
figure = plt.figure(figsize=(25, 7))
plt.hist([df[df['Survived'] == 1]['Fare'], df[df['Survived'] == 0]['Fare']], 
         stacked=True, color = ['g','r'],
         bins = 50, label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend();
## Finding out the missing values

df.isna().sum().sort_values(ascending = False)
df.corr()
df.drop({'Cabin', 'Age', 'Embarked'}, axis=1, inplace= True)

# from the test data set as well. 

df_test.drop({'Cabin', 'Age', 'Embarked'}, axis=1, inplace= True)
df_test.isna().sum().sort_values(ascending = False)
df_test[df_test.isna().T.any().T]
df_test1= pd.read_csv("../input/titanic/test.csv")

df_test1['Fare'].groupby([df_test1['Pclass'], df_test1['Sex']]).mean()
df_test1['Fare'].groupby([df_test1['Pclass'], df_test1['Sex']]).median()
# Setting up a loop to fill value for that specific row

for i in range(len(df_test['Fare'])):
    if df_test['PassengerId'][i] == 1044:
        df_test['Fare'][i] = 10
               
# Checking it..

df_test.iloc[[152]]['Fare']
print(df.shape)
print(df_test.shape)
df.head()
df_test.head()
id= df_test['PassengerId']

df.drop({'PassengerId', 'Died'}, axis=1, inplace= True)
df_test.drop({'PassengerId'}, axis=1, inplace= True)
print(df.shape)
print(df_test.shape)
y= df['Survived']
df.drop({'Survived'}, axis= 1, inplace= True)
df1= df
df2= df_test

df= pd.get_dummies(df)
df_test= pd.get_dummies(df_test)
for col in df.columns:
  if col not in df_test.columns:
    df.drop({col}, axis= 1, inplace= True)
    
for col in df_test.columns:
  if col not in df.columns:
    df_test.drop({col}, axis= 1, inplace= True)
# Checking out the shapes of both data sets:

print(df.shape)
print(df_test.shape)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# Splitting data for training, validation

X_train, X_test, y_train, y_test= train_test_split(df, y, random_state= 42)

# Plotting the feature importances using the Boosted Gradient Descent
from xgboost import XGBClassifier
from xgboost import plot_importance

# Training the model
model = XGBClassifier()
model_importance = model.fit(X_train, y_train)

# Plotting the Feature importance bar graph
plt.rcParams['figure.figsize'] = [14,12]
sns.set(style = 'darkgrid')
plot_importance(model_importance);
	
print(model)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

from sklearn.metrics import accuracy_score

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

output = pd.DataFrame({'PassengerId': y_test, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")


