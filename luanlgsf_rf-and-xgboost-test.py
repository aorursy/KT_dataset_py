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

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import re
df = pd.read_csv("../input/titanic/train.csv")
df_test = pd.read_csv('../input/titanic/test.csv')

df.head()
df.shape
100.0 * df.isnull().sum() / len(df)
df.info()
# create feature from name

r_common = re.compile("(Master\s+)|(Mr.?\s+)|(Miss\s+)|(Mrs\s+)|(Ms\s+)|(Mx\s+)|(M\s+)")
r_formal = re.compile("(Sir\s+)|(Gentleman\s+)|(Sire\s+)|(Mistress\s+)|(Madam\s+)|(Ma'am\s+)|(Dame\s+)|(Lord\s+)|(Lady\s+)|(Esq\.\s+)|(Excellency\s+)|(Honour\s+)|(Honourable\s+)")
r_academic = re.compile("(Dr\.\s+)|(Professor\s+)|(QC\s+)|(Counsel\s+)|(CI\s+)|(Eur\sIng\s+)|(Chancellor\s+)|(Principal\s+)|(Principal\s+)|(Dean\s+)|(Rector\s+)|(Executive\s+)")

print(r_formal.search("Sir Arthur"))
print(r_formal.search("Arthur"))

def honorific(name:str):
    if r_common.search(name):
        return 1
    if r_formal.search(name):
        return 2
    if r_academic.search(name):
        return 3
    else:
        return 0

df_clean = df.copy()
    
df_clean["honorific"] = df_clean["Name"].apply(honorific)
df_test["honorific"] = df_test["Name"].apply(honorific)


df_clean.head()
df_clean.groupby("honorific").agg("count")
# binarizar o gênero e o local de embarque

print(df_clean["Sex"].unique())
print(df_clean["Embarked"].unique())



def bin_embarked(name:str):
    if name == "S":
        return 0
    elif name == "C":
        return 1
    elif name == "Q":
        return 2
    else:
        return -1

    
df_clean["Sex"] = df_clean["Sex"].apply(lambda x: 1 if x == "female" else 0)
df_clean["Embarked"] = df_clean["Embarked"].apply(bin_embarked)

df_test["Sex"] = df_test["Sex"].apply(lambda x: 1 if x == "female" else 0)
df_test["Embarked"] = df_test["Embarked"].apply(bin_embarked)
    
df_clean.head()
len(df_clean["Ticket"].unique()) / df_clean.shape[0]
df_clean = df_clean.drop(columns=["PassengerId", "Name", "Cabin", "Ticket"], axis="columns")
df_test = df_test.drop(columns=["Name", "Cabin", "Ticket"], axis="columns")

df_clean.head()
df_clean.info()
# fill null values
df_clean["Age"] =  df["Age"].fillna(-10)
df_test["Age"] =  df_test["Age"].fillna(-10)
df_test["Fare"] =  df_test["Fare"].fillna(-10)

100.0 * df_clean.isnull().sum() / len(df_clean)
100.0 * df_test.isnull().sum() / len(df_test)
Y = df_clean["Survived"].values
X = df_clean.drop(columns="Survived")

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

print("ratio: ", len(y_test) / (len(y_test) + len(y_train)))
from sklearn.metrics import accuracy_score

# try Random Forest for baseline

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
# Nice baseline! To compare, ratio of non-balanced is 

df.query("Survived == 1").shape[0] / df.shape[0]
(pd.Series(clf.feature_importances_, index=X.columns)
   .nlargest(10)
   .plot(kind='barh'));        # some method chaining, because it's sexy!
# try XGBoost

from xgboost import XGBRFClassifier, plot_importance

xgb_clf = XGBRFClassifier()
xgb_clf.fit(X_train, y_train)

y_pred = xgb_clf.predict(X_test)
accuracy_score(y_test, y_pred)
plot_importance(xgb_clf);
from lightgbm import LGBMClassifier 

clf = LGBMClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)

# 0.7985074626865671
# Best: Random Forest 

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
df_test.head()
preds = clf.predict(df_test.drop(columns="PassengerId"))
preds[:10]
#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not
submission = pd.DataFrame({'PassengerId':df_test['PassengerId'],'Survived':preds})

#Visualize the first 5 rows
submission.head(20)
#Convert DataFrame to a csv file that can be uploaded
#This is saved in the same directory as your notebook
filename = 'titanic_predictions_1.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)
!kaggle competitions submit -c titanic -f titanic_predictions_1.csv