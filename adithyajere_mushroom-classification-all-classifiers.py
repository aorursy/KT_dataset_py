import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
DATA_PATH = '/kaggle/input/mushroom-classification/'
file_path = os.path.join(DATA_PATH,'mushrooms.csv')
pd.set_option('display.max_columns',30)
df = pd.read_csv(file_path)
print(f'shape of csv file: {df.shape}')
df.head()
df.columns = ['target', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
       'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
       'stalk-surface-below-ring', 'stalk-color-above-ring',
       'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
       'ring-type', 'spore-print-color', 'population', 'habitat']
for i in df.columns:
    print(f'{i} -> {df[i].unique()}')
for i in df.columns:
    if df[i].dtype == 'object':
        df[i] = pd.factorize(df[i])[0]
df.groupby(['cap-shape'])['target'].value_counts()
pd.crosstab(df['cap-shape'],df['target'])
fig = px.violin(df,
          x = df['cap-shape'],
          y=df['target'])
fig.show()
fig = px.violin(df,
          x = df['cap-surface'],
          y=df['target'])
fig.show()
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.feature_selection import mutual_info_classif
y = df.target
df.drop('target',axis =1,inplace=True)
x = df
vrt = VarianceThreshold(threshold=0.01)
vrt.fit(x,y)
sum(vrt.get_support())
X = vrt.transform(df)
chi2_selector = SelectKBest(chi2, k=11)
X_kbest = chi2_selector.fit_transform(X, y)
X_kbest.shape
mut_feat = mutual_info_classif(X_kbest,y)
mut_feat
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
X_train,X_test,y_train,y_test = train_test_split(X_kbest,y,test_size=0.15,random_state=1)
lr = LogisticRegression(max_iter=200)
lr.fit(X_train,y_train)
lr.score(X_train,y_train)
cross_val_score(lr,X_train,y_train,cv=5)
lr.score(X_test,y_test)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_features=9,max_depth=5,n_estimators=10)
rf.fit(X_train,y_train)
rf.score(X_train,y_train)
cross_val_score(rf,X_train,y_train,cv=5)
rf.feature_importances_
rf.score(X_test,y_test)
from sklearn.metrics import classification_report,roc_auc_score,roc_curve,auc
y_pred = rf.predict(X_test)
print(classification_report(y_test,y_pred))
roc_auc_score(y_test,y_pred)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr,tpr)
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title(f'tpr vs fpr plot with auc: {roc_auc_score(y_test,y_pred)}')
plt.show()
from sklearn import tree
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
clf = knn.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc_knb_model=roc_auc_score(y_test, y_pred)*100
acc_knb_model
lr = LogisticRegression(C = 0.2)
clf1 = lr.fit(X_train, y_train)
y_pred1 = clf1.predict(X_test)
acc_log_reg=roc_auc_score(y_test, y_pred1)*100
acc_log_reg
clf2 = GaussianNB().fit(X_train, y_train)
y_pred2 = clf2.predict(X_test)
acc_nb=roc_auc_score(y_test, y_pred2)*100
acc_nb
clf3 = tree.DecisionTreeClassifier().fit(X_train, y_train)
y_pred3 = clf3.predict(X_test)
acc_dt=roc_auc_score(y_test, y_pred3)*100
acc_dt
clf4 = RandomForestClassifier(max_depth=5, random_state=0).fit(X_train, y_train)
y_pred4 = clf4.predict(X_test)
acc_rmf_model=roc_auc_score(y_test, y_pred4)*100
acc_rmf_model
clf5 = SVC(gamma='auto').fit(X_train, y_train)
y_pred5 = clf5.predict(X_test)
acc_svm_model=roc_auc_score(y_test, y_pred5)*100
acc_svm_model
sgd_model=SGDClassifier()
sgd_model.fit(X_train,y_train)
sgd_pred=sgd_model.predict(X_test)
acc_sgd=round(sgd_model.score(X_train,y_train)*100,10)
acc_sgd
xgb_model=XGBClassifier()
xgb_model.fit(X_train,y_train)
xgb_pred=xgb_model.predict(X_test)
acc_xgb=round(xgb_model.score(X_train,y_train)*100,10)
acc_xgb
lgbm = LGBMClassifier()
lgbm.fit(X_train,y_train)
lgbm_pred=lgbm.predict(X_test)
acc_lgbm=round(lgbm.score(X_train,y_train)*100,10)
acc_lgbm
regr = linear_model.LinearRegression()
regr.fit(X_train,y_train)
regr_pred=regr.predict(X_test)
acc_regr=round(regr.score(X_train,y_train)*100,10)
acc_regr
results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest','Stochastic Gradient Decent','Linear Regression','Naive Bayes','XGBoost','LightGBM','Decision Tree'],
    'Score': [acc_svm_model, acc_knb_model, acc_log_reg, 
              acc_rmf_model,acc_sgd,acc_regr,acc_nb,acc_xgb,acc_lgbm,acc_dt]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df
