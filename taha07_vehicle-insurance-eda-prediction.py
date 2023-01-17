# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio
import plotly.express as px
import missingno as msno
import pandas_profiling
plt.style.use("fivethirtyeight")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/health-insurance-cross-sell-prediction/train.csv")
df.head()
df.info()
n = msno.bar(df,color="gold")
sns.countplot("Gender",data=df)
plt.show()
plt.style.use("fivethirtyeight")
sns.countplot(x = "Gender",data = df,hue = "Response")
plt.show()
facet = sns.FacetGrid(df,hue="Response",aspect = 4)
facet.map(sns.kdeplot,"Age",shade = True)
facet.set(xlim = (0,df["Age"].max()))
facet.add_legend()
plt.show()
sns.countplot("Driving_License",data = df,hue = "Response")
plt.show()
df["Response"].value_counts()
sns.countplot(x = "Previously_Insured",data = df,hue = "Response")
plt.show()
df.groupby("Response")["Previously_Insured"].value_counts()
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
sns.countplot("Vehicle_Age", data = df,hue = "Response")
plt.xlabel("Fig: i",color="coral")
plt.subplot(1,2,2)
plt.rcParams['figure.figsize']=(6,8)
color = ['yellowgreen','gold',"lightskyblue"]
df['Vehicle_Age'].value_counts().plot.pie(y="Vehicle_Age",colors=color,explode=(0.02,0,0.3),startangle=50,shadow=True,autopct="%0.1f%%")
plt.axis('on')
plt.xlabel("Fig: ii",color="coral")
plt.show()
plt.figure(figsize=(8,6))
sns.countplot("Vehicle_Damage", data = df,hue = "Response")
plt.show()
facet = sns.FacetGrid(df,hue="Response",aspect = 4)
facet.map(sns.kdeplot,"Annual_Premium",shade = True)
facet.set(xlim = (0,df["Annual_Premium"].max()))
facet.add_legend()
plt.show()
facet = sns.FacetGrid(df,hue="Response",aspect = 4)
facet.map(sns.kdeplot,"Vintage",shade = True)
facet.set(xlim = (0,df["Vintage"].max()))
facet.add_legend()
plt.show()
sns.catplot(x="Vehicle_Age", hue="Vehicle_Damage", col="Response",
                data=df, kind="count",
                height=6, aspect=.7)
plt.show()
sns.catplot(x="Vehicle_Damage", hue="Previously_Insured", col="Response",
                data=df, kind="count",
                height=6, aspect=.7)
plt.show()
df.drop(["id","Region_Code","Policy_Sales_Channel"],axis = 1,inplace=True)
df.head()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["Vehicle_Damage"] = le.fit_transform(df["Vehicle_Damage"])
df = pd.get_dummies(df,drop_first=True)
df.head()
x = df.drop("Response",axis=1)
y = df["Response"]
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(x,y)
plt.figure(figsize=(8,6))
important_features = pd.Series(model.feature_importances_,index = x.columns)
important_features.nlargest(7).plot(kind = "bar")
plt.show()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

dt_clf = DecisionTreeClassifier(criterion='entropy',max_depth = 100,random_state=0)
dt_clf.fit(x_train,y_train)
from sklearn.metrics import accuracy_score,classification_report
dt_pred = dt_clf.predict(x_test)
dt_accuracy = accuracy_score(y_test,dt_pred)
dt_accuracy
rf_clf = RandomForestClassifier(n_estimators = 200,random_state=0)
rf_clf.fit(x_train,y_train)
rf_pred = rf_clf.predict(x_test)
rf_accuracy = accuracy_score(y_test,rf_pred)
rf_accuracy
lr_clf = LogisticRegression(random_state=0)
lr_clf.fit(x_train,y_train)
lr_pred = lr_clf.predict(x_test)
lr_accuracy = accuracy_score(y_test,lr_pred)
lr_accuracy
lgbm_clf = LGBMClassifier(n_estimators=1000,learning_rate=0.007,random_state=0)#1000
lgbm_clf.fit(x_train,y_train)
lgbm_pred = lgbm_clf.predict(x_test)
lgbm_accuracy = accuracy_score(y_test,lgbm_pred)
lgbm_accuracy
knn_clf = KNeighborsClassifier(n_neighbors=20)
knn_clf.fit(x_train,y_train)
knn_pred = knn_clf.predict(x_test)
knn_accuracy = accuracy_score(y_test,knn_pred)
knn_accuracy
acc_df = pd.DataFrame({"Decision Tree":dt_accuracy,"Random Forest":rf_accuracy,
                       "LightGBM":lgbm_accuracy,"Logistic Regression" : lr_accuracy,"KNN":knn_accuracy},index = ["Accuracy"])
acc_df.style.background_gradient(cmap = "Reds")
