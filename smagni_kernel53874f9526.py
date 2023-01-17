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
train=pd.read_csv("/kaggle/input/sf-crime/train.csv.zip")
test=pd.read_csv("/kaggle/input/sf-crime/test.csv.zip")
test.head()
train.head()
train.isnull().sum()
test.info()
# error
# train.loc[train["Resolution"]=="NONE","Resolution"]=0
# train.loc[train["Resolution"]=="ARREST, BOOKED","Resolution"]=1

print(train["Resolution"].unique())
len(train["Resolution"].unique()) 
def bargraph(feature):
    booked=train[train["Resolution"]==1][feature].value_counts()
    left=train[train["Resolution"]==0][feature].value_counts()
    df=pd.DataFrame([booked,left])
    df.plot(kind="bar",stacked=True)
bargraph("Category")
bargraph("X")

bargraph("Y")
bargraph("PdDistrict")
bargraph("DayOfWeek")
bargraph("PdDistrict")
train.drop(["Dates" , "Category" , "Descript" , "DayOfWeek","Address"] , inplace=True , axis=1)
test.drop(["Id" , "Dates" , "DayOfWeek" , "Address"] , inplace=True , axis=1)
print(test.nunique())
train.nunique()
train.head()
from sklearn.preprocessing import LabelEncoder
le_PdDistrict = LabelEncoder()
le_Resolution = LabelEncoder()
le_X = LabelEncoder()
le_Y = LabelEncoder()
# train.loc[train["PdDistrict"]=="TENDERLOIN","PdDistrict"]=0
# train.loc[train["PdDistrict"]=="SOUTHERN","PdDistrict"]=1
# train.loc[train["PdDistrict"]=="MISSION","PdDistrict"]=2
# train.loc[train["PdDistrict"]=="NORTHERN","PdDistrict"]=3
# train.loc[train["PdDistrict"]=="BAYVIEW","PdDistrict"]=4
# train.loc[train["PdDistrict"]=="CENTRAL","PdDistrict"]=5
# train.loc[train["PdDistrict"]=="INGLESIDE","PdDistrict"]=6
# train.loc[train["PdDistrict"]=="PARK","PdDistrict"]=7
# train.loc[train["PdDistrict"]=="TARAVAL","PdDistrict"]=8
# train.loc[train["PdDistrict"]=="RICHMOND","PdDistrict"]=9

# test.loc[test["PdDistrict"]=="TENDERLOIN","PdDistrict"]=0
# test.loc[test["PdDistrict"]=="SOUTHERN","PdDistrict"]=1
# test.loc[test["PdDistrict"]=="MISSION","PdDistrict"]=2
# test.loc[test["PdDistrict"]=="NORTHERN","PdDistrict"]=3
# test.loc[test["PdDistrict"]=="BAYVIEW","PdDistrict"]=4
# test.loc[test["PdDistrict"]=="CENTRAL","PdDistrict"]=5
# test.loc[test["PdDistrict"]=="INGLESIDE","PdDistrict"]=6
# test.loc[test["PdDistrict"]=="PARK","PdDistrict"]=7
# test.loc[test["PdDistrict"]=="TARAVAL","PdDistrict"]=8
# test.loc[test["PdDistrict"]=="RICHMOND","PdDistrict"]=9

train["PdDistrict"] = le_PdDistrict.fit_transform(train["PdDistrict"])
train["Resolution"] = le_PdDistrict.fit_transform(train["Resolution"])
train["X"] = le_PdDistrict.fit_transform(train["X"])
train["Y"] = le_PdDistrict.fit_transform(train["Y"])
test["PdDistrict"] = le_PdDistrict.transform(test["PdDistrict"])
# test["Resolution"] = le_PdDistrict.transform(test["Resolution"])
test["X"] = le_PdDistrict.transform(test["X"])
test["Y"] = le_PdDistrict.transform(test["Y"])
test


train
test
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()
clf.fit(train[['PdDistrict','X','Y']],train["Resolution"])
pred = clf.predict(train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']])