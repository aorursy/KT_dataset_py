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
!pip install pyforest   # it automatically imports few important libraries including numpy, pandas and matplotlib
import pyforest
import matplotlib
import warnings
warnings.filterwarnings("ignore")

from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
from sklearn.model_selection import KFold
# matplotlib.rcParams['figure.figsize'] = (13.0, 7.0)
df=pd.read_csv("../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")
df.head()
plt.figure(figsize=(13,5))
sns.heatmap(df.isna())
plt.figure(figsize=(15,5))
df.Attrition.value_counts().plot(kind='bar')
fig, axes =plt.subplots(3,3, figsize=(15,15))
axes = axes.flatten()
object_bol = df.dtypes == 'object'
for ax, catplot in zip(axes, df.dtypes[object_bol].index):
    sns.countplot(hue=catplot,x=df.Attrition ,data=df, ax=ax,)

plt.tight_layout()  
plt.show()
fig, axes = plt.subplots(5,5, figsize=(15,15))
axes = axes.flatten()
object_bol = df.dtypes == 'int'
for ax, catplot in zip(axes, df.dtypes[object_bol].index):
    sns.boxplot(y=df[catplot],x= df["Attrition"],ax=ax)

plt.tight_layout()  
plt.show()
df1=df[['Attrition','Age','BusinessTravel', 'DistanceFromHome', 'MonthlyIncome','PercentSalaryHike','TotalWorkingYears']]
sns.pairplot(df1,hue='Attrition',)
plt.figure(figsize=(15,5))
sns.distplot(df.Age,bins=20, kde=False)
plt.xlabel("Age")
plt.ylabel("Counts")
plt.title("Age Counts")
plt.show()
s = (df.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)
from sklearn.preprocessing import LabelEncoder

# Make copy to avoid changing original data 
label_data = df.copy()

# Apply label encoder to each column with categorical data
label_encoder = LabelEncoder()
for col in object_cols:
    label_data[col] = label_encoder.fit_transform(df[col])
label_data.head()
rf = RandomForestClassifier(class_weight="balanced", n_estimators=500) 
rf.fit(label_data.drop(['Attrition'],axis=1), hr.Attrition)
importances = rf.feature_importances_
names = label_data.columns
importances, names = zip(*sorted(zip(importances, names)))

# Lets plot this
plt.figure(figsize=(15,8))
plt.barh(range(len(names)), importances, align = 'center')
plt.yticks(range(len(names)), names)
plt.xlabel('Importance of features')
plt.ylabel('Features')
plt.title('Importance of each feature')
plt.show()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(label_data.drop(['Attrition'],axis=1), label_data.Attrition,
                                                    test_size=0.25, random_state=42)
def kfold_and_confusion_matrix(model, t_model):
    kfold = KFold(n_splits=5)
    model_kfold = model
    results_kfold = model_selection.cross_val_score(model_kfold, X_train, y_train,  cv=kfold)
    print("K Fold Accuracy: %.2f%%" % (results_kfold.mean()*100.0)) 
    
    plt.figure(figsize=(15,5))
    y_pred=t_model.predict(X_test)
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)

print("Normal Accuracy:",model.score(X_test, y_test))
kfold_and_confusion_matrix(LogisticRegression(), model)
# SMOTE for imbalanced data
oversampler=SMOTE(random_state=0)
X, y = oversampler.fit_sample(label_data.drop(['Attrition'],axis=1), label_data.Attrition,)   #label_data[list(names)[14:], using less features are reducing the accuracy
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.25)
model=LogisticRegression()
model.fit(X_train, y_train, )
print("Normal Accuracy: %.2f%%" % ( model.score(X_test, y_test)*100))

kfold_and_confusion_matrix(LogisticRegression(), model)

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

logregpipe = Pipeline([('scale', StandardScaler()),
                   ('logreg',LogisticRegression())])

# Gridsearch to determine the value of C
param_grid = {'logreg__C':np.arange(0.01,100,10)}
logreg_cv = GridSearchCV(logregpipe,param_grid,cv=5,return_train_score=True)
logreg_cv.fit(X_train,y_train)
print(logreg_cv.best_params_)


bestlogreg = logreg_cv.best_estimator_
bestlogreg.fit(X_train,y_train)
bestlogreg.coef_ = bestlogreg.named_steps['logreg'].coef_
print("Normal Accuracy: %.2f%%" % (bestlogreg.score(X_train,y_train)*100))

kfold_and_confusion_matrix(logreg_cv.best_estimator_, bestlogreg)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(class_weight="balanced", n_estimators=500)
model.fit(X_train, y_train)
print("Normal Accuracy:",(model.score(X_test, y_test)*100))

kfold_and_confusion_matrix(RandomForestClassifier(), model)
from sklearn.ensemble import  GradientBoostingClassifier
model = GradientBoostingClassifier()
model.fit(X_train, y_train)
print("Normal Accuracy:",(model.score(X_test, y_test)*100))

kfold_and_confusion_matrix(GradientBoostingClassifier(), model)
import xgboost as xgb
from sklearn import metrics
# Parameter Tuning
model = xgb.XGBClassifier()
param_dist = {"max_depth": [10,30,50],
              "min_child_weight" : [1,3,6],
              "n_estimators": [200],
              "learning_rate": [0.05, 0.1,0.16],}
grid_search = GridSearchCV(model, param_grid=param_dist, cv = 3, 
                                   verbose=10, n_jobs=-1)
grid_search.fit(X_train, y_train)
model=grid_search.best_estimator_
# model = xgb.XGBClassifier(max_depth=50, min_child_weight=1,  n_estimators=200,\
#                           n_jobs=-1 , verbose=1,learning_rate=0.16)
model.fit(X_train,y_train)
print("Normal Accuracy:",(model.score(X_test, y_test)*100))

kfold_and_confusion_matrix(grid_search.best_estimator_, model)
import lightgbm as lgb
from sklearn import metrics


lg = lgb.LGBMClassifier(silent=False)
param_dist = {"max_depth": [25,50, 75],
              "learning_rate" : [0.01,0.05,0.1],
              "num_leaves": [300,900,1200],
              "n_estimators": [200]
             }
grid_search = GridSearchCV(lg, n_jobs=-1, param_grid=param_dist, cv = 3, scoring="roc_auc", verbose=5)
grid_search.fit(X_train,y_train)
grid_search.best_estimator_
model= grid_search.best_estimator_
model.fit(X_train,y_train)
print("Normal Accuracy:",(model.score(X_test, y_test)*100))

kfold_and_confusion_matrix(grid_search.best_estimator_, model)