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
import matplotlib.pyplot as plt

import seaborn as sns

import sklearn
df = pd.read_csv("/kaggle/input/heart-disease-prediction-using-logistic-regression/framingham.csv")

display(df.head())
df["glucose"].median()
display(df.shape)

display(df.info())





fig,ax=plt.subplots(figsize=(20,10))



sns.heatmap(df)



display(df.isnull().sum())
columns = list(df.columns)



n_largest = df.corr().nlargest(n=8,columns="TenYearCHD")["TenYearCHD"].index[1:]
fig,ax2=plt.subplots(figsize=(20,10))

sns.heatmap(df.corr(),annot=True,ax=ax2)
sns.distplot(df["education"])

sns.distplot(df["cigsPerDay"])
from sklearn.impute import SimpleImputer,KNNImputer

knn_imputer = KNNImputer(weights='distance')

simp_imp_mode = SimpleImputer(strategy = "most_frequent")

simp_imp_mean = SimpleImputer(strategy = "mean")

simp_imp_median = SimpleImputer(strategy = "median")
df["education"] = knn_imputer.fit_transform(df[["education"]])

df["education"].isnull().sum()
df["cigsPerDay"] = knn_imputer.fit_transform(df[["cigsPerDay"]])

df["cigsPerDay"].isnull().sum()


df["BPMeds"] = knn_imputer.fit_transform(df[["BPMeds"]])

df["BPMeds"].isnull().sum()


df[["totChol","BMI","heartRate","glucose" ]] = simp_imp_median.fit_transform(df[["totChol","BMI","heartRate","glucose" ]])

df[["totChol","BMI","heartRate","glucose" ]].isnull().sum()
df.isnull().sum().sum()
#sns.pairplot(df[n_largest])

#plt.show()


y = df["TenYearCHD"] 

X = df.drop("TenYearCHD",axis=1)



display(X.shape, y.shape)
n_largest
X_modified = X.copy()

#X_modified = X_modified[n_largest]
X_modified["bp"] = X_modified["sysBP"] + X_modified["diaBP"] + (X_modified["sysBP"] + X_modified["diaBP"])*(X_modified["prevalentHyp"])









X_modified.drop(["sysBP","diaBP","prevalentHyp"],inplace=True,axis=1)
X_modified.head()
X.head()
from sklearn import preprocessing,metrics,model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV





log_reg = LogisticRegression(max_iter=10000)                 #max_iter=10000,penalty="l1",solver="liblinear"

params = {"penalty":["l2","none"], "solver" : ["lbfgs"]}

log_reg_grid = GridSearchCV(log_reg,params,cv=5)

log_reg_grid.fit(X,y)

print(log_reg_grid.best_params_)



log_reg2 = LogisticRegression(max_iter=10000,penalty="none",solver="lbfgs")

log_reg2.fit(X_modified,y)



display(model_selection.cross_val_score(log_reg2,X,y,cv=5).mean())

print("ROC_AUC = " + str(model_selection.cross_val_score(log_reg2,X,y,cv=5,scoring="roc_auc").mean()))

knn_clf = KNeighborsClassifier(weights="distance",n_neighbors= 34)

knn_clf.fit(X,y)

display(model_selection.cross_val_score(knn_clf,X,y,cv=5).mean())


