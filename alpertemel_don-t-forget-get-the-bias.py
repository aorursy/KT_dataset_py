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
train = pd.read_csv("/kaggle/input/health-insurance-cross-sell-prediction/train.csv")

train.head()
train.isnull().sum()
train["Response"].value_counts()
import seaborn as sns



sns.countplot(train["Response"])
one = train.loc[train["Response"] == 1]

zero = train.loc[train["Response"] == 0]



zero = zero.iloc[0:len(one), :]



data = pd.concat([one, zero], axis = 0)



import seaborn as sns



sns.countplot(data["Response"])
data.head()
sns.scatterplot(x = "Region_Code", y = "Annual_Premium", hue = "Response", data = data)
sns.distplot(data["Age"])
sns.countplot(data["Gender"])
sns.distplot(data.Vintage)
df = data.copy()



gender = pd.get_dummies(data["Gender"])

vehicle_age = pd.get_dummies(data["Vehicle_Age"])

vehicle_damage = pd.get_dummies(data["Vehicle_Damage"])

driving_license = pd.get_dummies(data["Driving_License"])



df = pd.concat([df, gender, vehicle_age, vehicle_damage, driving_license], axis = 1)



df = df.drop(["Gender", "Vehicle_Age", "Vehicle_Damage", "Driving_License"], axis = 1)

del df["id"]





x = df.iloc[:, 0:6]

x2 = df.iloc[:, 7:]

x = pd.concat([x, x2], axis = 1)

x["a"] = x["> 2 Years"]

x["b"] = x["< 1 Year"]

x = x.drop(["> 2 Years", "< 1 Year"], axis = 1)

y = df["Response"]



from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 34)



x_train.head()
from sklearn.metrics import accuracy_score, confusion_matrix

from xgboost import XGBClassifier



xgb = XGBClassifier()

xgb.fit(x_train, y_train)

xgb_pred = xgb.predict(x_test)



print(accuracy_score(y_test, xgb_pred))

print(confusion_matrix(y_test, xgb_pred))
from sklearn.metrics import roc_auc_score 

preds = xgb.predict_proba(x_train)

clas = xgb.predict(x_train)

score = roc_auc_score(y_train, preds[:,1])

print(score)
from sklearn.metrics import precision_recall_curve, auc, roc_curve, recall_score

import matplotlib.pyplot as plt



y_score = xgb.predict_proba(x_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, y_score)





print ('Area under curve (AUC): ', auc(fpr,tpr))

print("Thanks!")