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
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
actual_survival_test = pd.read_csv('../input/titanic/gender_submission.csv')
# from pandas_profiling import ProfileReport
# report_train = ProfileReport(train)

# report_train
print("train{} test{}".format(train.shape,test.shape))
train.info()
train.describe()
train.head()
df1= train
dummy=pd.get_dummies(df1["Sex"])
df1["Female"]=dummy["female"];

df1["Male"]=dummy["male"];
dummy2=pd.get_dummies(df1["Embarked"])
df1["C"]=dummy2["C"]

df1["Q"]=dummy2["Q"]

df1["S"]=dummy2["S"]
df1.isnull().sum()
a=set(df1["Cabin"])

print(a)
df1["Cabin"].fillna(df1["Cabin"].mode()[0],inplace=True)
df1.isnull().sum()
df1['new_col'] = df1['Cabin'].astype(str).str[0]
print(set(df1["new_col"]))
dummy_cabin=pd.get_dummies(df1["new_col"])
dummy_cabin.head()
df1["A"]=dummy_cabin["A"]

df1["B"]=dummy_cabin["B"]

df1["C"]=dummy_cabin["C"]

df1["D"]=dummy_cabin["D"]

df1["E"]=dummy_cabin["E"]

df1["F"]=dummy_cabin["F"]

df1["G"]=dummy_cabin["G"]

# df1["T"]=dummy_cabin["T"]
df1.columns
df1.drop(["Name","Sex","Cabin","Ticket","new_col","Embarked"],axis=1,inplace=True)
df1.head()
df1["Age"].fillna(df1["Age"].mean(),inplace=True)
df1.describe()
df1["Pclass"]=df1["Pclass"].astype(str)
df1.info()
dummy_Pclass= pd.get_dummies(df1["Pclass"])

dummy_Pclass.head()
df1["One"]=dummy_Pclass["1"]

df1["Two"]=dummy_Pclass["2"]

df1["Three"]=dummy_Pclass["3"]
df1.drop("Pclass",axis=1,inplace=True)
df1.head()
import seaborn as sns

import matplotlib.pyplot as plt
corrmat = df1.corr()

f, ax = plt.subplots(figsize=(15, 12))

sns.heatmap(corrmat, cbar=True, annot=True, square=True, vmax=.8);
df1.drop("Female",axis=1,inplace=True)
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import sklearn.model_selection as model_selection
clf = RandomForestClassifier(n_estimators=100)
y_train=df1["Survived"];

X_train=df1.drop("Survived",axis=1)
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.7,test_size=0.3, random_state=101)
clf.fit(X_train,y_train)
pred1=clf.predict(X_train)
metrics.accuracy_score(pred1, y_train)
# pred2=clf.predict(X_test)

# metrics.accuracy_score(pred2,y_test)
from xgboost.sklearn import XGBClassifier
xgb=XGBClassifier()
xgb.fit(X_train,y_train)
pred_xgb_train=xgb.predict(X_train)
metrics.accuracy_score(pred_xgb_train, y_train)
# pred_xgb_test=xgb.predict(X_test)

# metrics.accuracy_score(pred_xgb_test, y_test)
df2=test
dummy_test=pd.get_dummies(df2["Sex"])
df2["Female"]=dummy_test["female"];

df2["Male"]=dummy_test["male"];
dummy_test2=pd.get_dummies(df2["Embarked"])
df2["C"]=dummy_test2["C"]

df2["Q"]=dummy_test2["Q"]

df2["S"]=dummy_test2["S"]
df2.isnull().sum()
df2["Cabin"].fillna(df2["Cabin"].mode()[0],inplace=True)
df2.isnull().sum()
df2['new_col'] = df2['Cabin'].astype(str).str[0]
print(set(df2["new_col"]))
dummy_test_cabin=pd.get_dummies(df2["new_col"])
dummy_test_cabin.head()
df2["A"]=dummy_test_cabin["A"]

df2["B"]=dummy_test_cabin["B"]

df2["C"]=dummy_test_cabin["C"]

df2["D"]=dummy_test_cabin["D"]

df2["E"]=dummy_test_cabin["E"]

df2["F"]=dummy_test_cabin["F"]

df2["G"]=dummy_test_cabin["G"]

#df2["T"]=dummy_test_cabin["T"]

#df2["n"]=dummy_test_cabin["n"]
df2.drop(["Name","Sex","Cabin","Ticket","new_col","Embarked"],axis=1,inplace=True)
df2["Age"].fillna(df2["Age"].mean(),inplace=True)
df2["Fare"].fillna(df2["Fare"].mean(),inplace=True)
df2["Pclass"]=df2["Pclass"].astype(str)
dummy_Pclass_test=pd.get_dummies(df2["Pclass"])
df2["One"]=dummy_Pclass_test["1"]

df2["Two"]=dummy_Pclass_test["2"]

df2["Three"]=dummy_Pclass_test["3"]
df2.drop("Pclass",axis=1,inplace=True)
# corrmat = df2.corr()

# f, ax = plt.subplots(figsize=(15, 12))

# sns.heatmap(corrmat, cbar=True, annot=True, square=True, vmax=.8);
df2.drop("Female",axis=1,inplace=True)
test_x=df2
test_x.info();
# clf.fit(X,y)
pred_test_rf=clf.predict(test_x)
# pred_test_rf;
# actual_survival_test;
metrics.accuracy_score(pred_test_rf,actual_survival_test["Survived"])
pred_test_xgb=xgb.predict(test_x)
metrics.accuracy_score(pred_test_xgb,actual_survival_test["Survived"])
submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':pred_test_xgb})
filename = 'Titanic Predictions XGB.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)