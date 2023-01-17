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
df = pd.read_csv('../input/minor-project-2020/train.csv')

# df.head()
df.drop(['id'],axis=1,inplace=True)
X=df.iloc[:,:-1]

y=df.iloc[:,-1]
# df.head()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=200)
from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()

scaled_X_train = scalar.fit_transform(X_train)

scaled_X_test = scalar.transform(X_test)
# from sklearn.model_selection import cross_val_score,cross_val_predict,GridSearchCV

# from sklearn.model_selection import RepeatedStratifiedKFold

# from xgboost import XGBClassifier

# xgb = XGBClassifier()

# parameters = {'criterion': ("gini", "entropy"), 'max_depth': [50]}

# clf = GridSearchCV(xgb, parameters, verbose=1)

# clf.fit(scaled_X_train, y_train)
# from sklearn.tree import DecisionTreeClassifier

# dt_cv = DecisionTreeClassifier()

# clf = GridSearchCV(dt_cv, parameters, verbose=1)

# clf.fit(scaled_X_train, y_train)
# from imblearn.over_sampling import SMOTE

# from sklearn.metrics import roc_auc_score

# from xgboost import XGBClassifier

# sm = SMOTE(random_state=10)

# X_train_sm, y_train_sm = sm.fit_sample(X_train, y_train)

# xgbsm = XGBClassifier()

# xgbsm.fit(X_train_sm, y_train_sm)

# y_predsm=xgbsm.predict_proba(X_test)[:,1]

# auc_score5 = roc_auc_score(y_test, y_predsm)

# print(auc_score5)
# xgbsm = XGBClassifier()

# xgbsm.fit(scaled_X_train_sm, y_train_sm)
# y_predsm=xgbsm.predict_proba(scaled_X_test)[:,1]
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=10)

scaled_X_train_sm, y_train_sm = sm.fit_sample(scaled_X_train, y_train)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

# clf_log = LogisticRegression(solver='saga',C=0.3,random_state=10,max_iter=200).fit(scaled_X_train_sm, y_train_sm)

clf_log = LogisticRegression().fit(scaled_X_train_sm, y_train_sm)

y_pred_log=clf_log.predict(scaled_X_test)

auc_score2 = roc_auc_score(y_test, y_pred_log)

print(auc_score2)
y_pred_log1=clf_log.predict_proba(scaled_X_test)[:,1]

auc_score2 = roc_auc_score(y_test, y_pred_log1)

print(auc_score2)

# from sklearn.model_selection import GridSearchCV
# parameters = {'criterion': ["gini"], 'max_depth': [5]}

# dt_cv_log = LogisticRegression()

# clf_log = GridSearchCV(dt_cv_log, parameters, verbose=1)

# clf_log.fit(scaled_X_train_sm, y_train_sm)
# grid={"C":np.logspace(-3,3,7), "penalty":["l2"]}

# logreg=LogisticRegression()

# logreg_cv=GridSearchCV(logreg,grid,cv=5)

# logreg_cv.fit(scaled_X_train_sm, y_train_sm)
# y_pred_logcv=logreg_cv.predict_proba(scaled_X_test)[:,1]

# auc_score3 = roc_auc_score(y_test, y_pred_logcv)

# print(auc_score3)
# from sklearn.tree import DecisionTreeClassifier

# from sklearn.model_selection import GridSearchCV

# from sklearn.metrics import roc_auc_score

# #from xgboost import XGBClassifier

# #from sklearn.ensemble import GradientBoostingClassifier

# parameters = {'criterion': ("gini", "entropy"), 'max_depth': [100]}

# dt_cv = DecisionTreeClassifier()

# # xgb = XGBClassifier()

# # gbclf = GradientBoostingClassifier(random_state=0)

# clf_dt = GridSearchCV(dt_cv, parameters, verbose=1)

# clf_dt.fit(scaled_X_train_sm, y_train_sm)

# y_pred_dtcv1=clf_dt.predict_proba(scaled_X_test)[:,1]

# auc_score4 = roc_auc_score(y_test, y_pred_dtcv1)

# print(auc_score4)
# cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)



# resy = cross_val_predict(xgb, X, y, cv=cv, n_jobs=-1)
# from sklearn.metrics import roc_auc_score

# #print('Mean ROC AUC: %.5f' % np.mean(scores))


# xgb = XGBClassifier()

#xgb.fit(scaled_X_train, y_train)


#res = xgb.predict(scaled_X_test)

# auc_score1 = roc_auc_score(y_test, y_predsm)

# print(auc_score1)
dfTest=pd.read_csv('../input/minor-project-2020/test.csv')
# dfTest.head()
xmainTest=dfTest.drop('id',axis=1)

scaled_xmainTest=scalar.transform(xmainTest)

#ymainPred=clf.predict(scaled_xmainTest)

# ymainPred=xgbsm.predict_proba(scaled_xmainTest)[:,1]

#ymainpred_log=clf_log.predict_proba(scaled_xmainTest)[:,1]

ymainpred_log=clf_log.predict(scaled_xmainTest)
df_out = pd.DataFrame(list(zip(dfTest.id, ymainpred_log)), 

               columns =['id', 'target'])
df_out.to_csv('outputpredfile.csv', index=False)