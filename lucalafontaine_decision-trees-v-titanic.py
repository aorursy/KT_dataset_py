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
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_test = pd.read_csv("/kaggle/input/titanic/test.csv")
from sklearn.tree import DecisionTreeClassifier

from sklearn import preprocessing

from sklearn.model_selection import train_test_split
df_train.head(5)
y = df_train["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
dums = df_train[features].values

dums_test = df_test[features].values



le_sex = preprocessing.LabelEncoder()

le_sex.fit(['male','female'])

dums[:,1] = le_sex.transform(dums[:,1]) 

dums_test[:,1] = le_sex.transform(dums_test[:,1])



dums[0:5]

X_trainset, X_testset, y_trainset, y_testset = train_test_split(dums, y, test_size=0.3, random_state=3)
survival_DTC = DecisionTreeClassifier(criterion="entropy", max_depth = 12)

survival_DTC # it shows the default parameters
survival_DTC.fit(dums,y)
survive_pred = survival_DTC.predict(dums_test)
from sklearn import metrics

import matplotlib.pyplot as plt

print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, survive_pred))
final_df = pd.DataFrame({'PassengerId' : df_test['PassengerId'], 'Survived': survive_pred})
final_df.head(10)
final_df.to_csv('DecisionTree.csv', index=False)

print("Your submission was successfully saved!")

final_df.head()