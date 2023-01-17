# Importing Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

ds = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
ds
ds.info()
ds.isnull().sum()
mean_salary = ds['salary'].mean()
ds.fillna({'salary' : mean_salary}, inplace=True)
ds.isnull().sum()
from sklearn.preprocessing import LabelEncoder

gender_n = LabelEncoder()
ssc_b_n = LabelEncoder()
hsc_b_n = LabelEncoder()
hsc_s_n = LabelEncoder()
degree_t_n = LabelEncoder()
workex_n = LabelEncoder()
specialisation_n = LabelEncoder()
status_n = LabelEncoder()

ds['gender_n'] = gender_n.fit_transform(ds['gender'])
ds['ssc_b_n'] = ssc_b_n.fit_transform(ds['ssc_b'])

ds['hsc_b_n'] = hsc_b_n.fit_transform(ds['hsc_b'])
ds['hsc_s_n'] = hsc_s_n.fit_transform(ds['hsc_s'])

ds['degree_t_n'] = degree_t_n.fit_transform(ds['degree_t'])
ds['workex_n'] = workex_n.fit_transform(ds['workex'])

ds['specialisation_n'] = specialisation_n.fit_transform(ds['specialisation'])
ds['status_n'] = status_n.fit_transform(ds['status'])
ds.drop(['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation', 'status','sl_no'], axis=1, inplace=True)
ds
ds['status_n'].value_counts()
ds_0 = ds[ds['status_n'] == 0]
ds_1 = ds[ds['status_n'] == 1]

ds_1 = ds_1.sample(ds_0.shape[0])

ds = ds_0.append(ds_1, ignore_index = True)
ds['status_n'].value_counts()
x = ds.drop(['status_n'], axis=1)
y = ds['status_n']
from sklearn.feature_selection import SelectKBest, chi2
best_feature = SelectKBest(score_func= chi2, k = 'all')

fit = best_feature.fit(x,y)

ofscore = pd.DataFrame(fit.scores_)
ofcolumn = pd.DataFrame(x.columns)
feature_score = pd.concat([ofcolumn, ofscore], axis=1)
feature_score.columns = ['spec', 'score']
feature_score
x.drop(['mba_p','gender_n','ssc_b_n','hsc_b_n','hsc_s_n','degree_t_n'], axis=1, inplace=True)
x
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.18)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
rf = RandomForestClassifier()
rf.fit(x_train,y_train)
rf.score(x_test,y_test)
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
dt.score(x_test,y_test)
kn = KNeighborsClassifier()
kn.fit(x_train,y_train)
kn.score(x_test,y_test)
svm = SVC()
svm.fit(x_train,y_train)
svm.score(x_test,y_test)
xg = XGBClassifier()
xg.fit(x_train,y_train)
xg.score(x_test,y_test)
nb = GaussianNB()
nb.fit(x_train,y_train)
nb.score(x_test,y_test)
from sklearn.metrics import confusion_matrix, classification_report

y_pred = xg.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print('Confusion Matrix\n',cm)
plt.figure(figsize=(7,5))
sns.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('truth')
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
print(classification_report(y_test,y_pred, target_names=['Class 0','Class 1']))