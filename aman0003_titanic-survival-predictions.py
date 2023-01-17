# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/titanic-machine-learning-from-disaster/train.csv")

df.head()
df.shape
df.isna().sum()
df = df[df["Age"].isna() == False]

#df.reset_index(drop = True, inplace = True)

df.reset_index(drop=True, inplace = True)

df
pd.Series(df["Cabin"]).value_counts()

df["Cabin"].isna().sum()

df.drop(["Cabin"], axis = 1, inplace = True)
df_dummy = pd.get_dummies(df["Sex"], prefix = "Sex_")

df_final = pd.concat([df_dummy, df], sort = True, axis = 1)

df_final = df_final.drop(["Sex", "Sex__male"], axis = 1)

df_final.head()
df_dummy2 = pd.get_dummies(df["Embarked"], prefix = "embarked")

df_f = pd.concat([df_final, df_dummy2], axis = 1)

df_f = df_f.drop(["Embarked", "embarked_S"], axis = 1)

df_f.head(10)
list1 = pd.Series(df_f["SibSp"])

list2 = pd.Series(df_f["Parch"])

list = []

for i in range(len(list1)):

    if list1[i] != 0 and list2[i] != 0:

        list.append(1)

    else : 

        list.append(0)

            

df_f["family"] = list
names = tuple(df_f["Name"])

titles = tuple(map(lambda x : x.split(",")[1].split(".")[0], names))

df_f["titles"] = titles

#df_dummy3 = pd.get_dummies(df_f["titles"], prefix = "title")

#df_f2 = pd.concat([df_f, df_dummy3], axis = 1)
list_wo_post = []

list_w_post = []

for i in range(len(df_f["titles"])):

    if df_f["titles"][i] == " Mr" or df_f["titles"][i] == " Miss" or df_f["titles"][i] == " Mrs" or df_f["titles"][i] == " Ms":

        list_wo_post.append(1)

        list_w_post.append(0)

    else:

        list_wo_post.append(0)

        list_w_post.append(1)

df_f["with_post"] = list_w_post

df_f["w/o_post"] = list_wo_post
df_f.head()
features = df_f.drop(["Name", "Ticket", "Survived", "titles"] , axis = 1)

target = df["Survived"]

print(features.columns)

from sklearn.model_selection import train_test_split as tts

from sklearn.linear_model import LogisticRegression as lr



x_train, x_test, y_train, y_test = tts(features, target, random_state = 2)

reg = lr()

reg.fit(x_train, y_train)

predictions = reg.predict(x_test)

reg.score(x_test, y_test)
from sklearn.tree import DecisionTreeClassifier as dtc

#x_train, x_test, y_train, y_test = tts(features, target, random_state = 2)

reg2 = dtc()

reg2.fit(x_train, y_train)

predictions = reg2.predict(x_test)

reg2.score(x_test, y_test)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)

clf.fit(x_train, y_train)

predictions = clf.predict(x_test)

clf.score(x_test, y_test)
from sklearn.svm import SVC

svc = SVC(gamma='auto')

svc.fit(x_train, y_train)

predictions = svc.predict(x_test)

svc.score(x_test, y_test)