# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")

df.head()
df.shape
df.info()
df.isnull().sum()
cat_col = df.select_dtypes(include="object")
cat_col.nunique()


fig,ax = plt.subplots(3,3, figsize=(10,10))               # 'ax' has references to all the four axes

sns.distplot(df['TotalWorkingYears'], ax = ax[0,0]) 

sns.distplot(df['YearsAtCompany'], ax = ax[0,1]) 

sns.distplot(df['DistanceFromHome'], ax = ax[0,2]) 

sns.distplot(df['YearsInCurrentRole'], ax = ax[1,0]) 

sns.distplot(df['YearsWithCurrManager'], ax = ax[1,1]) 

sns.distplot(df['YearsSinceLastPromotion'], ax = ax[1,2]) 

sns.distplot(df['PercentSalaryHike'], ax = ax[2,0]) 

sns.distplot(df['YearsSinceLastPromotion'], ax = ax[2,1]) 

sns.distplot(df['TrainingTimesLastYear'], ax = ax[2,2]) 

plt.show()
cat_col.columns
fig,axes = plt.subplots(1,2,figsize=(20,5))

pd.crosstab(df['Department'],df['Attrition']).plot(kind='bar',ax=axes[0])

pd.crosstab(df['BusinessTravel'],df['Attrition']).plot(kind='bar',ax=axes[1])

plt.show()
fig,axes = plt.subplots(1,2,figsize=(20,5))

pd.crosstab(df['EducationField'],df['Attrition']).plot(kind='bar',ax=axes[0])

pd.crosstab(df['Gender'],df['Attrition']).plot(kind='bar',ax=axes[1])

plt.show()
fig,axes = plt.subplots(1,2,figsize=(20,5))

pd.crosstab(df['OverTime'],df['Attrition']).plot(kind='bar',ax=axes[0])

pd.crosstab(df['MaritalStatus'],df['Attrition']).plot(kind='bar',ax=axes[1])

plt.show()
pd.crosstab(df['JobLevel'],df['Attrition']).plot(kind='bar',figsize=(18,5))

plt.show()
pd.crosstab(df['JobRole'],df['Attrition']).plot(kind='bar',figsize=(18,5))

plt.show()
plt.figure(figsize=(15,5))

sns.barplot(df['Attrition'],df['DistanceFromHome'],hue=df['JobLevel'])

plt.show()
plt.figure(figsize=(15,5))

sns.boxplot(df['EducationField'],df['MonthlyIncome'],hue=df['Attrition'])

plt.show()
df = df.drop(["Over18","EmployeeCount","StandardHours"],axis=1)
plt.figure(figsize=(30, 30))

sns.heatmap(df.corr(), annot=True, cmap="RdYlGn",annot_kws={"size":15})
df.columns
cat_col.columns
cat_col = cat_col.drop(["Attrition","Over18"],axis=1)
df["Attrition"] = df["Attrition"].map({"Yes":1,"No":0})
from sklearn.preprocessing import LabelEncoder

lr = LabelEncoder()



for i in cat_col:

    df[i]=lr.fit_transform(df[i])
df[cat_col.columns].head()
X = df.drop('Attrition',axis=1)

y = df['Attrition']
from sklearn.model_selection import train_test_split

X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1)
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(random_state=10)
dtc.fit(X_train,y_train)
y_train_pred = dtc.predict(X_train)

y_train_prob = dtc.predict_proba(X_train)[0,1]
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve,confusion_matrix, f1_score

print("Decision Tree Accuracy Score for Train", accuracy_score(y_train, y_train_pred))
from sklearn.model_selection import GridSearchCV

dtc=DecisionTreeClassifier()

max_depth=[2,3,4,5,6,7,8,10,15,20]

min_samples_split=[2,3,4,5]

min_samples_leaf=[6,7,8,9,10,11,12,13,14,15,16,17,18]

criterion = ['gini','entropy']

param_grid={'max_depth':max_depth,'min_samples_split':min_samples_split,'min_samples_leaf':min_samples_leaf,'criterion':criterion}

dtc = DecisionTreeClassifier(random_state=10)

gridSearchCV = GridSearchCV(dtc, param_grid =param_grid,scoring='accuracy',n_jobs=-1,cv=3)
gridSearchCV.fit(X_train,y_train)
gridSearchCV.best_params_
dtc = DecisionTreeClassifier(**gridSearchCV.best_params_)

dtc.fit(X_train, y_train)

y_train_pred = dtc.predict(X_train)

y_train_prob = dtc.predict_proba(X_train)[:,1]
print("Decision Tree Accuracy Score for Train", accuracy_score(y_train, y_train_pred))
y_test_pred = dtc.predict(X_test)

y_test_prob = dtc.predict_proba(X_test)[:,1]

print("Decision Tree Accuracy Score for Train", accuracy_score(y_test, y_test_pred))
confusion_matrix(y_test,y_test_pred)
dfpr, dtpr, dthreshold = roc_curve(y_test, y_test_prob)
roc_auc_score(y_test, y_test_prob)
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc.fit(X_train, y_train)

rfc_y_train_pred = rfc.predict(X_train)

rfc_y_train_prob = rfc.predict_proba(X_train)[:,1]
print("Random Forest Classifier Accuracy Score for Train", accuracy_score(y_train, rfc_y_train_pred))
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint as sp_randint

# rfc = RandomForestClassifier(random_state=10)

params = {

    'n_estimators':sp_randint(5,150),#nooftrees

    'max_features':sp_randint(1,31),#choose any values betwenn 1-31 features excluding Attrition

    'max_depth':sp_randint(2,10),

    'min_samples_leaf':sp_randint(1,50),

    'min_samples_split':sp_randint(2,50),

    'criterion':['gini','entropy']

}

rsearch = RandomizedSearchCV(rfc, param_distributions=params, n_iter=100, scoring='roc_auc',cv=3, n_jobs=-1)
rsearch.fit(X_train, y_train)
rsearch.best_params_
rfc = RandomForestClassifier(**rsearch.best_params_,random_state=10)

rfc.fit(X_train, y_train)

rfc_y_train_pred = rfc.predict(X_train)

rfc_y_train_prob = rfc.predict_proba(X_train)[:,1]
print("Random Forest Classifier Accuracy Score for Train", accuracy_score(y_train, rfc_y_train_pred))
rfc_y_test_pred = rfc.predict(X_test)

rfc_y_test_prob = rfc.predict_proba(X_test)[:,1]

print("Random Forest Classifier Accuracy Score for Test", accuracy_score(y_test, rfc_y_test_pred))
confusion_matrix(y_test,rfc_y_test_pred)
roc_auc_score(y_test,rfc_y_test_prob)
rfpr,rtpr,rthreshold = roc_curve(y_test, rfc_y_test_prob)
plt.plot(dfpr,dtpr,c='g',label='DecisionTree')

plt.plot(rfpr,rtpr,c='b',label='RFC')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")



plt.legend()

plt.show()
rfc_fi = pd.DataFrame(rfc.feature_importances_,index=X.columns,columns=['RFC Score'])

rfc_feat = rfc_fi.sort_values(by='RFC Score',ascending=False)

rfc_feat.plot(kind='bar',figsize=(20,5))

plt.show()