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

%matplotlib inline

import seaborn as sns

import plotly.express as px
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
test_df.shape
train_df.shape
train_df.describe(include = "all")
test_df.describe(include = "all")
train_df.info()
train_df.head()
train_df.isnull().sum()
total = train_df.isnull().sum().sort_values(ascending = False)

percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending = False)

missing_train_data = pd.concat([total,percent], axis =1, keys = ['Total', 'Percent'])

missing_train_data.head()
missing_train_data.plot.bar()
total = test_df.isnull().sum().sort_values(ascending = False)

percent = (test_df.isnull().sum()/test_df.isnull().count()).sort_values(ascending = False)

missing_test_data = pd.concat([total,percent], axis =1, keys = ['Total', 'Percent'])

missing_test_data.head()
missing_test_data.plot.bar()
df_con = pd.DataFrame()

df_dis = pd.DataFrame()
train_df.info()
ax = sns.countplot(x = train_df["Survived"])
df_dis['Survived'] = train_df['Survived']

df_con['Survived'] = train_df['Survived']
ax = sns.countplot(x = train_df["Pclass"])
df_dis['Pclass'] = train_df['Pclass']

df_con['Pclass'] = train_df['Pclass']
train_df.Age.isnull().sum()
ax = sns.distplot(train_df["Age"], bins = 10)
df_dis["Age"] = train_df["Age"]

df_con["Age"] = train_df["Age"]
ax = sns.countplot(x = train_df["Sex"])
df_dis['Sex'] = train_df['Sex']

df_dis['Sex'] = np.where(df_dis['Sex'] == 'female', 1, 0) # change sex to 0 for male and 1 for female



df_con['Sex'] = train_df['Sex']
df_dis["Survived"] = train_df["Survived"]
df_dis.head()
ax = sns.barplot(y="Survived", x = "Sex", data = df_dis)
train_df["SibSp"].value_counts()
df_dis["SibSp"] = train_df["SibSp"]

df_con["SibSp"] = train_df["SibSp"]
ax = sns.barplot(y="Survived", x = "SibSp", data = df_dis)
train_df["Parch"].value_counts()
df_dis['Parch'] = train_df['Parch']

df_con['Parch'] = train_df['Parch']
ax = sns.barplot(y="Survived", x = "Parch", data = df_dis)
df_con.head()
df_dis.head()
train_df["Ticket"].describe(include = "all")
train_df["Fare"].describe(include = "all")
print("Unique Values:")

print(len(train_df["Fare"].unique()))
df_con["Fare"] = train_df["Fare"]

df_dis["Fare"] = pd.cut(train_df["Fare"], bins = 5)

df_dis.Fare.value_counts()
ax = sns.countplot(x = df_dis["Fare"])

plt.xticks(rotation = "90")
train_df["Cabin"].isnull().sum()
len(train_df["Cabin"].unique())
train_df.Embarked.isnull().sum()
train_df.Embarked.unique()
train_df.Embarked.value_counts()
sns.countplot(x= train_df["Embarked"])
df_dis["Embarked"] = train_df["Embarked"]

df_con["Embarked"] = train_df["Embarked"]
df_con.Embarked.shape
df_con.head()
df_con = df_con.dropna(subset = ["Embarked"])

df_dis = df_dis.dropna(subset = ["Embarked"])

print(df_con.Embarked.shape)

print(df_dis.Embarked.shape)
df_con["PassengerId"] = train_df["PassengerId"]
df_dis.head()
df_con.head()
onehot_cols = df_dis.columns.tolist()

onehot_cols.remove("Survived")

enc_df_dis = pd.get_dummies(df_dis, columns = onehot_cols)

enc_df_dis.head()
df_embarked_onehot = pd.get_dummies(df_con['Embarked'], prefix='embarked')

df_sex_onehot = pd.get_dummies(df_con['Sex'], prefix='sex')

df_plcass_onehot = pd.get_dummies(df_con['Pclass'], prefix='pclass')
enc_df_con = pd.concat([df_con, df_embarked_onehot, df_sex_onehot, df_plcass_onehot], axis =1)

enc_df_con = enc_df_con.drop(["Pclass", "Sex", "Embarked"], axis = 1)

enc_df_con.head()
enc_df_con.info()
enc_df_con.shape
train_df.isnull().sum()
train_df['Age'].fillna((train_df['Age'].mean()), inplace=True)
train_df.isnull().sum()
plt.figure(figsize = (20,10))

sns.heatmap(train_df.corr(),annot = True)

plt.show()
plt.figure(figsize = (20,10))

sns.heatmap(enc_df_con.corr(),annot = True)

plt.show()
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()



enc_df_con[['Fare','Age']] = min_max_scaler.fit_transform(enc_df_con[['Fare','Age']])

enc_df_con.sample(5)
enc_df_con.isnull().sum()
enc_df_con['Age'].fillna((enc_df_con['Age'].mean()), inplace=True)
enc_df_con.isnull().sum()
#train_df = train_df[:train_df.shape[0]]

target = enc_df_con['Survived']

enc_df_con.drop(['Survived'], axis=1, inplace=True)

enc_df_con.isnull().sum()
target.shape
test_df.shape
test_df.isnull().sum()
test_df['Age'].fillna((test_df['Age'].mean()), inplace=True)
test_df.isnull().sum()
test_df['Fare'].fillna((test_df['Fare'].mean()), inplace=True)
test_df.isnull().sum()
test_df = test_df.drop(["Cabin", "Name", "Ticket"], axis = 1)
test_df.isnull().sum()
test_df.head()
enc_df_con.head()
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()



test_df[['Fare','Age']] = min_max_scaler.fit_transform(test_df[['Fare','Age']])

test_df.sample(5)
df_con_test = pd.DataFrame()

df_dis_test = pd.DataFrame()





df_dis_test['Pclass'] = test_df['Pclass']

df_con_test['Pclass'] = test_df['Pclass']



df_dis_test["Age"] = test_df["Age"]

df_con_test["Age"] = test_df["Age"]



df_dis_test['Sex'] = test_df['Sex']

df_dis_test['Sex'] = np.where(df_dis_test['Sex'] == 'female', 1, 0) # change sex to 0 for male and 1 for female

df_con_test['Sex'] = test_df['Sex']





df_dis_test["SibSp"] = test_df["SibSp"]

df_con_test["SibSp"] = test_df["SibSp"]





df_dis_test['Parch'] = test_df['Parch']

df_con_test['Parch'] = test_df['Parch']





df_con_test['PassengerId'] = test_df['PassengerId']

passenger_ID = df_con_test["PassengerId"]





df_con_test["Fare"] = test_df["Fare"]

df_dis_test["Fare"] = pd.cut(test_df["Fare"], bins = 5)







df_dis_test["Embarked"] = test_df["Embarked"]

df_con_test["Embarked"] = test_df["Embarked"]





onehot_cols_test = df_dis.columns.tolist()

onehot_cols_test.remove("Survived")

enc_df_dis_test = pd.get_dummies(df_dis, columns = onehot_cols_test)

enc_df_dis_test.head()







df_embarked_onehot_test = pd.get_dummies(df_con_test['Embarked'], prefix='embarked')

df_sex_onehot_test = pd.get_dummies(df_con_test['Sex'], prefix='sex')

df_plcass_onehot_test = pd.get_dummies(df_con_test['Pclass'], prefix='pclass')









enc_df_con_test = pd.concat([df_con_test, df_embarked_onehot_test, df_sex_onehot_test, df_plcass_onehot_test], axis =1)

enc_df_con_test = enc_df_con_test.drop(["Pclass", "Sex", "Embarked"], axis = 1)

enc_df_con_test.head()
enc_df_con_test.shape
enc_df_con.head()
enc_df_con.shape
enc_df_con.isnull().sum()
enc_df_con_test.isnull().sum()
import scipy

from scipy import stats

import statsmodels.api as sm

import statsmodels.api as sm

from sklearn import metrics

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression
X_train, X_validate, Y_train, Y_validate = train_test_split(enc_df_con, target, test_size=0.2, random_state=3)

print(X_train.shape, X_validate.shape)
model = LogisticRegression()

model = model.fit(X_train, Y_train)

y_pred = model.predict(X_validate)
accuracy_score(Y_validate, y_pred)*100
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

print(classification_report(Y_validate, y_pred))
model = LogisticRegression()

model = model.fit(enc_df_con, target)

y_pred_test = model.predict(enc_df_con_test)
final_Submission_df = pd.DataFrame({"PassengerId": passenger_ID,

                                   "Survived": y_pred_test})

final_Submission_df.set_index("PassengerId", inplace = True)
final_Submission_df.head()
final_Submission_df.to_csv("final_submission.csv")