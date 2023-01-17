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
df = pd.read_csv('../input/breast-cancer-prediction-dataset/Breast_cancer_data.csv')
df.head()
df.info()
df.corr()
import seaborn as sns
sns.heatmap(df.corr(),annot= True,linewidths=0.5)
sns.lmplot(x="mean_area",y="mean_radius", data = df)
df.plot(kind="scatter",x="mean_area",y="mean_radius")
X = df.iloc[:,0:-1]
X
y =df.iloc[:,-1]
y
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state= 0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
y_test.head()
y_train.head()
from sklearn.linear_model import LogisticRegression

from  sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from xgboost import  XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb


lgr = LogisticRegression().fit(X_train,y_train)
dtc = DecisionTreeClassifier(random_state=0).fit(X_train,y_train)
rfc = RandomForestClassifier(n_estimators= 250,random_state=0).fit(X_train,y_train)
xgb = XGBClassifier().fit(X_train,y_train)
svm = SVC().fit(X_train,y_train)
gnb = GaussianNB().fit(X_train,y_train)
KNN = KNeighborsClassifier().fit(X_train,y_train)
cat = CatBoostClassifier(iterations=70).fit(X_train,y_train)
lgb = lgb.LGBMClassifier().fit(X_train,y_train)
abc = AdaBoostClassifier(n_estimators=200).fit(X_train,y_train)
lgr_pred = lgr.predict(X_test)
dtc_pred = dtc.predict(X_test)
rfc_pred = rfc.predict(X_test)
xgb_pred = xgb.predict(X_test)
svm_pred = svm.predict(X_test)
gnb_pred = gnb.predict(X_test)
knn_pred = KNN.predict(X_test)
cat_pred = cat.predict(X_test)
lgb_pred =lgb.predict(X_test)
abc_pred = abc.predict(X_test)
from sklearn.metrics import accuracy_score

print("The accuracy is LogisticRegressioin",accuracy_score(y_test,lgr_pred)*100)
print("The accuracy is Tree",accuracy_score(y_test,dtc_pred)*100)
print("The accuracy is RandomForest",accuracy_score(y_test,rfc_pred)*100)
print("The accuracy is XGBoost",accuracy_score(y_test,xgb_pred)*100)
print("The accuracy is SVM",accuracy_score(y_test,svm_pred)*100)
print("The accuracy is Gaussian NB", accuracy_score(y_test,gnb_pred)*100)
print("The accuaracy is KNN", accuracy_score(y_test,knn_pred)*100)
print("The accuracy is Catboost",accuracy_score(y_test,cat_pred)*100)
print("The accuracy is Lightboost",accuracy_score(y_test,lgb_pred)*100)
print("The accuracy is Adaboost",accuracy_score(y_test,abc_pred)*100)