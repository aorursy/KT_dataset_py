# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/HR-Employee-Attrition.csv")
df.info()
import warnings
warnings.filterwarnings('ignore')

df.Attrition.replace({"Yes":1,"No":0}, inplace=True)
df.drop(columns=['EmployeeCount','StandardHours'], inplace=True)
df.columns
cat_col = df.select_dtypes(exclude=np.number).columns
num_col = df.select_dtypes(include=np.number).columns
for i in cat_col:
    print(df[i].value_counts())
    print("------------------------------------")
encoded_cat_col = pd.get_dummies(df[cat_col])
final_model = pd.concat([df[num_col],encoded_cat_col], axis = 1)
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
x = final_model.drop(columns="Attrition")
y = final_model["Attrition"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train, y_train)
train_Pred = model.predict(x_train)
metrics.confusion_matrix(y_train,train_Pred)
Accuracy_percent_train = (metrics.accuracy_score(y_train,train_Pred))*100
Accuracy_percent_train
test_Pred = model.predict(x_test)
metrics.confusion_matrix(y_test,test_Pred)
Accuracy_percent_test = (metrics.accuracy_score(y_test,test_Pred))*100
Accuracy_percent_test
print(classification_report(y_test, test_Pred))
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from lightgbm import LGBMClassifier
knn = KNeighborsClassifier(n_neighbors=15)
clf = knn.fit(x_train, y_train)
y_pred = clf.predict(x_test)
acc_knb_model=roc_auc_score(y_test, y_pred)*100
acc_knb_model
lr = LogisticRegression(C = 0.2)
clf1 = lr.fit(x_train, y_train)
y_pred1 = clf1.predict(x_test)
acc_log_reg=roc_auc_score(y_test, y_pred1)*100
acc_log_reg
clf2 = GaussianNB().fit(x_train, y_train)
y_pred2 = clf2.predict(x_test)
acc_nb=roc_auc_score(y_test, y_pred2)*100
acc_nb
clf3 = tree.DecisionTreeClassifier().fit(x_train, y_train)
y_pred3 = clf3.predict(x_test)
acc_dt=roc_auc_score(y_test, y_pred3)*100
acc_dt
clf4 = RandomForestClassifier(max_depth=5, random_state=0).fit(x_train, y_train)
y_pred4 = clf4.predict(x_test)
acc_rmf_model=roc_auc_score(y_test, y_pred4)*100
acc_rmf_model
clf5 = SVC(gamma='auto').fit(x_train, y_train)
y_pred5 = clf5.predict(x_test)
acc_svm_model=roc_auc_score(y_test, y_pred5)*100
acc_svm_model
sgd_model=SGDClassifier()
sgd_model.fit(x_train,y_train)
sgd_pred=sgd_model.predict(x_test)
acc_sgd=round(sgd_model.score(x_train,y_train)*100,10)
acc_sgd
xgb_model=XGBClassifier()
xgb_model.fit(x_train,y_train)
xgb_pred=xgb_model.predict(x_test)
acc_xgb=round(xgb_model.score(x_train,y_train)*100,10)
acc_xgb
lgbm = LGBMClassifier()
lgbm.fit(x_train,y_train)
lgbm_pred=lgbm.predict(x_test)
acc_lgbm=round(lgbm.score(x_train,y_train)*100,10)
acc_lgbm
regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)
regr_pred=regr.predict(x_test)
acc_regr=round(regr.score(x_train,y_train)*100,10)
acc_regr
results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest','Stochastic Gradient Decent','Linear Regression','Naive Bayes','XGBoost','LightGBM','Decision Tree'],
    'Score': [acc_svm_model, acc_knb_model, acc_log_reg, 
              acc_rmf_model,acc_sgd,acc_regr,acc_nb,acc_xgb,acc_lgbm,acc_dt]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df
