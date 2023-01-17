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
#Import Dataset
df_gender = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_list = [df_train,df_test]
df = pd.concat(df_list)
# df[891:len(df)]
df.head()
df.describe(include = 'all')
df.isnull()
# df_train['age'] = df['a'].fillna(df['a'].mean())

df.columns = map(str.lower, df.columns)

df['name'] = df['name'].str.lower()
df['sex'] = df['sex'].str.lower()
df['age'] = df['age'].replace(to_replace = np.nan, value = 30)
# df[0:890].describe()
df['fare'] = df['fare'].replace(to_replace = np.nan, value = 32)
df.describe()
df.isnull().sum(axis = 0)
df.isna().sum(axis = 0)
df = df.drop('name', axis = 1)
features = ["sex",'embarked']
# X = pd.get_dummies(df[features])
a = pd.get_dummies(df[features])
df = pd.concat([df,a], axis = 1)
# df
# df = df.drop('sex', axis = 1)
df
df['cabin'] = df['cabin'].fillna(0)
df['cabin'] = df['cabin'].astype(bool).astype(int)
df1 = df.drop(['ticket','sex', 'embarked'], axis = 1)
df1
train_data = df1.iloc[0:891]
test_data = df1.iloc[891:1309]
import matplotlib.pyplot as plt
import seaborn as sns
train_data.head()
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
# X = train_data.loc[:, train_data.columns != 'survived']
# y = train_data.loc[:, train_data.columns == 'survived']
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['y']==0]))
print("Number of subscription",len(os_data_y[os_data_y['y']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))
# [['pclass','sibsp','age','cabin']]
import statsmodels.api as sm
logit_model=sm.Logit(y,X[['pclass','sibsp','age','cabin']])
result=logit_model.fit()
print(result.summary2())
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression(max_iter = 100000)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(test_data1)
# print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
y_pred
output = pd.DataFrame({'passengerid': test_data.passengerid, 'survived': y_pred})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
# df['survived'].array
# a.to_numpy()
y = train_data['survived'].array
y
test_data1 = test_data.drop(['survived'], axis = 1)
from sklearn.ensemble import RandomForestClassifier

y = train_data["survived"]

features = ["pclass", "sex", "sibsp", "parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'passengerid': test_data.passengerid, 'survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
df.col1 = df.where(df.col1 == 0, 1)