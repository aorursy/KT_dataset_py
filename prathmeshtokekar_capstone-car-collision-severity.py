# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #For data visualization
import numpy as np #For data processing on series
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print("Hello Capstone Project Course!")
df = pd.read_csv("../input/capstone-car-accident-serveity/Data_Collisions.csv")
df.info() #Analysing data types of each column
df.nunique() #Analysing number of unique values per column
df.isna().sum() #Finding total number of missing values in the data.
sns.heatmap(df.isnull(), cbar=False) #Visualizing the missing values
df.describe()
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')
df["ST_COLCODE"] = df["ST_COLCODE"].fillna(df["ST_COLCODE"].mode()[0])
df["UNDERINFL"] = df["UNDERINFL"].fillna(df["UNDERINFL"].mode()[0])
df["X"] = df["X"].fillna(df["X"].median())
df["Y"] = df["Y"].fillna(df["Y"].median())
df["LIGHTCOND"] = df["LIGHTCOND"].fillna(df["LIGHTCOND"].mode()[0])
df = df.drop(["SDOTCOLNUM","INTKEY","COLDETKEY","SEVERITYCODE.1","SPEEDING" ,"EXCEPTRSNDESC","PEDROWNOTGRNT","INATTENTIONIND","EXCEPTRSNCODE","LOCATION","INCDATE","INCDTTM","OBJECTID","REPORTNO","SDOT_COLDESC","ST_COLDESC","ST_COLCODE","SEVERITYDESC"], axis=1)
df.info()
df.dtypes

#Making a list of all categorical variables
clmn = {"STATUS","ADDRTYPE","COLLISIONTYPE","JUNCTIONTYPE","UNDERINFL","WEATHER","ROADCOND","LIGHTCOND","HITPARKEDCAR"}

#Converting them into dummy variables
df = pd.get_dummies(data=df,columns=clmn,prefix=clmn)

#Concatnating them to the dataframe
df.dtypes
df.shape
X = df.drop(["SEVERITYCODE"],axis=1)
y = df["SEVERITYCODE"]

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


LR = LogisticRegression(max_iter=100000)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20)

LR.fit(X_train,y_train)
score = LR.score(X_test, y_test)
print(score)

#y_pred = LR.predict(X_test)


plot_confusion_matrix(LR,X_test,y_test)

y_pred = LR.predict(X_test) 
print(classification_report(y_test, y_pred))

from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import  cross_val_score

DT = DecisionTreeClassifier()

DT.fit(X_train,y_train)
score_1 = DT.score(X_test, y_test)
print(score_1)
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(DT,X_test,y_test)

y_pred_1 = DT.predict(X_test) 

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_1))

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

random_forest.fit(X_train,y_train)
score_2 = random_forest.score(X_test, y_test)
print(score_2)

plot_confusion_matrix(random_forest,X_test,y_test)


y_pred_2 = random_forest.predict(X_test) 

print(classification_report(y_test, y_pred_2))



from xgboost import XGBClassifier

xgb = XGBClassifier()

xgb.fit(X_train,y_train)
score_3 = xgb.score(X_test, y_test)
print(score_3)

plot_confusion_matrix(xgb,X_test,y_test)


y_pred_3 = xgb.predict(X_test) 

print(classification_report(y_test, y_pred_3))