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
import numpy as np 
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from xgboost import plot_importance
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
aa = pd.read_csv('mem_data.csv')
bb = pd.read_csv('mem_transaction.csv')
cc = pd.read_csv('store_info.csv')
print(aa.info())
print(bb.info())
print(cc.info())
ba = pd.merge(aa, bb, on = 'MEM_ID',how= 'outer')
df = ba.copy()
df
# df = df.drop(['ZIP_CD','RGST_DT','MEMP_DT','LAST_VST_DT','M_STORE_ID','BIRTH_DT','BIRTH_SL'],axis =1)
df.SMS = (df.SMS=='Y').astype(int)
df.SMS.value_counts()
df.BIRTH_SL = (df.BIRTH_SL=='S').astype(int)
df.BIRTH_SL.value_counts()
df['BIRTH_DT'] = df['BIRTH_DT'].fillna(0)
obj = ['BIRTH_DT','RGST_DT','ZIP_CD','LAST_VST_DT','MEMP_DT','MEMP_STY','MEMP_TP']
df[obj] = df[obj].apply(lambda x: x.astype('category').cat.codes)
q = df.GENDER!='UNKNOWN'
mdf = df[q] 
mdf.GENDER = (mdf.GENDER=='M').astype(int)
w = df.GENDER =='UNKNOWN'
test= df[w].sort_values('GENDER')
corr_matrix = mdf.corr()
corr_matrix["GENDER"].sort_values(ascending=False)
from pandas.plotting import scatter_matrix
num = ['USED_PNT','ACC_PNT','USABLE_INIT','STORE_ID','SELL_DT','SELL_AMT','GENDER']
num1 = ['USABLE_PNT','M_STORE_ID','VISIT_CNT','SALES_AMT','GENDER']
num2 = ['BIRTH_SL','VISIT_CNT','ACC_PNT','SALES_AMT','USABLE_PNT','GENDER']
scatter_matrix(mdf[num2] , figsize=(12,8))
mdf.plot(kind='scatter', x='USABLE_PNT', y='GENDER', alpha=0.1)
mdf.plot(kind='scatter', x='ACC_PNT', y='GENDER', alpha=0.1)
f = mdf.ACC_PNT.where(mdf.ACC_PNT>=0, other=0) # 음수처리
f = np.log(f+1)
mdf.ACC_PNT = f
f = mdf.USABLE_PNT.where(mdf.USABLE_PNT>=0, other=0) # 음수처리
f = np.log(f+1)
mdf.USABLE_PNT = f
f = mdf.SALES_AMT.where(mdf.SALES_AMT>=0, other=0) # 음수처리
f = np.log(f+1)
mdf.SALES_AMT = f
#f = mdf.VISIT_CNT.where(mdf.VISIT_CNT>=0, other=0) # 음수처리
#f = np.log(f+1)
#mdf.VISIT_CNT = f
mdf.plot(kind='scatter', x='M_STORE_ID', y='GENDER', alpha=0.1)
mdf.plot(kind='scatter', x='VISIT_CNT', y='GENDER', alpha=0.1)
mdf.plot(kind='scatter', x='SALES_AMT', y='GENDER', alpha=0.1)
mdf_gender = mdf['GENDER']
plt.figure(figsize=(6,4))
sns.distplot(mdf_gender, kde=False)
mdf.head()
#mdf.to_csv('mdf.csv')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import glob
kfold = StratifiedKFold(n_splits=5) # 하이퍼 파라미터 지정
n_it = 12
t_final = test.copy()
t_final.head()
mdf = mdf.dropna()
test = test.drop(['GENDER','MEM_ID'], axis=1)
target = mdf.GENDER.values
mdf = mdf.drop(['GENDER','MEM_ID'], axis=1)
np.random.seed(123)
params = {'max_features':list(np.arange(1, mdf.shape[1])), 'bootstrap':[False], 'n_estimators': [50], 'criterion':['gini','entropy']}
model = RandomizedSearchCV(RandomForestClassifier(), param_distributions=params, n_iter=n_it, cv=kfold, scoring='roc_auc',n_jobs=-1, verbose=1)
print('MODELING.............................................................................')
model.fit(mdf, target)
print('========BEST_AUC_SCORE = ', model.best_score_)
model = model.best_estimator_
t_final.GENDER = model.predict_proba(test.values)[:,1]


0.9627 # 기본 
0.9629 # 카운트 로그화
0.9592
0.9590
mdf.info()
t_final.head(40)
t_final1 = t_final.groupby('MEM_ID').agg({'GENDER':'mean'}).reset_index()
t_final1
t_final1.to_csv('output_data.csv', index=False)
print('COMPLETE')
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.metrics import roc_curve, classification_report
X_train, X_test, y_train, y_test = train_test_split(mdf, target, test_size=0.25, random_state = 0)
def bestGBDTNextModel(model, isKfold, nfold, searchCV, Xtrain, ytrain, Xtest, ytest, nIter, scoring, errScore, verbose, nJobs):
    # GridSearchCV을 위해 파라미터 값을 제한함.
    grd_prams = {}
    classifier = XGBClassifier(random_state=0, objective='binary:logistic')
    cv = KFold(n_splits=nfold, shuffle=True, random_state=0)
    
    if model == 'LGBM':
        # 그래디언트 부스팅 결정 트리(GBDT)    
        grd_prams.update({'max_depth': [50, 100],
              'learning_rate' : [0.01, 0.05],
              'num_leaves': [150, 200],
              'n_estimators': [300, 400],
              'num_boost_round':[4000, 5000],
              'subsample': [0.5, 1],
              'reg_alpha': [0.01, 0.1],
              'reg_lambda': [0.01, 0.1],
              'min_data_in_leaf': [20, 30],
              'lambda_l1': [0.01, 0.1],
              'lambda_l2': [0.01, 0.1]
            })
        
        #grd_prams.update({'max_depth': [50, 75, 90, 100],
        #      'learning_rate' : [0.01, 0.05, 0.07, 0.1],
        #      'num_leaves': [300,600,900,1200],
        #      'n_estimators': [100, 300, 500, 900],
        #      'num_boost_round':[1000, 2000, 3000, 4000],
        #      'num_leaves': [30, 60, 120, 150, 200],
        #      'reg_alpha': [0.01, 0.1, 0.5, 0.7, 1.0],
        #      'min_data_in_leaf': [50, 100, 300, 800],
        #      'lambda_l1': [0, 0.1, 0.5, 1.0],
        #      'lambda_l2': [0, 0.01, 1.0]})
        
        classifier = LGBMClassifier(random_state=0, boosting_type='gbdt', objective='binary', metric='auc')
        
    elif model == 'XGB':
        grd_prams.update({'n_estimators': [300, 500],
            'learning_rate': [0.001, 0.01],
            'subsample': [0.5, 1],
            'max_depth': [5, 6],
            'colsample_bytree': [0.97, 1.24],
            'min_child_weight': [1, 2],
            'gamma': [0.001, 0.005],
            'nthread': [3, 4],
            'reg_lambda': [0.5, 1.0],
            'reg_alpha': [0.01, 0.1]
          })
        
        #grd_prams.update({'n_estimators': [300, 500, 700],
        #    'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.09],
        #    'subsample': [0.5, 1],
        #    'max_depth': [4, 5, 6, 7, 8, 9, 10],
        #    'colsample_bytree': [0.52, 0.97, 1,55, 2.32, 3.46],
        #    'min_child_weight': [1, 2, 3, 4],
        #    'gamma': [0.001, 0.01, 0.1, 0, 1],
        #    'nthread': [3, 4, 5],
        #    'reg_lambda': [0.01, 0.1, 0.5, 0.7, 1.0],
        #    'reg_alpha': [0.01, 0.1, 0.5, 0.7, 1.0]
        #  })
    
    if isKfold == False:
        cv = StratifiedShuffleSplit(n_splits=nfold, test_size=0.2, random_state=0)
    
    grid_ = RandomizedSearchCV(classifier, param_distributions=grd_prams, n_iter=nIter, scoring=scoring, error_score=errScore, verbose=verbose, n_jobs=nJobs, cv=cv)

    # 속도 이슈
    if searchCV == 'GRID': 
        grid_ = GridSearchCV(classifier, param_grid=grd_prams, n_jobs=nJobs, scoring=scoring, verbose=verbose, cv=cv)
    
    grid_.fit(Xtrain, ytrain)
    score_ = grid_.score(Xtest, ytest)
    
    #best = {"best_param":grid_.best_params_, 
    #        "best_score":grid_.best_score_, 
    #        "best_estimator":grid_.best_estimator_,
    #        "test_score":score_
    #       }
    
    print("{} grid_.best_score {}".format(model, np.round(grid_.best_score_,3)))
    print("{} grid_.best_score {}".format(model, np.round(score_,3)))
    print("{} best_estimator {}".format(model, grid_.best_estimator_))

    return grid_.best_params_
best_param1 = bestGBDTNextModel('LGBM', False, 5, 'RANDOM', X_train, y_train, X_test, y_test, 15, 'roc_auc', 0, 3, -1)
lgbm1 = LGBMClassifier(**best_param1)
score_lgbm1 = lgbm1.fit(X_train, y_train).score(X_test, y_test)
print("score_lgbm1 ::: {}".format(score_lgbm1))
print("-----------------------------------")
y_lgbm1 = lgbm1.predict(X_test)
print(classification_report(y_test, y_lgbm1))
best_model1 = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
               importance_type='split', lambda_l1=0.1, lambda_l2=0.01,
               learning_rate=0.01, max_depth=50, metric='auc',
               min_child_samples=20, min_child_weight=0.001,
               min_data_in_leaf=20, min_split_gain=0.0, n_estimators=400,
               n_jobs=-1, num_boost_round=5000, num_leaves=200,
               objective='binary', random_state=0, reg_alpha=0.01,
               reg_lambda=0.01, silent=True, subsample=0.5,
               subsample_for_bin=200000, subsample_freq=0)

score_best1 = best_model1.fit(X_train, y_train).score(X_test, y_test)
y_best1 = best_model1.predict(X_test)

print("best_model1 -----------------------------{}".format(score_best1))
print(classification_report(y_test, y_best1))
t_final.GENDER = best_model1.predict_proba(test.values)[:,1]
t_final1 = t_final.groupby('MEM_ID').agg({'GENDER':'mean'}).reset_index()
t_final1
t_final1.to_csv('output_data1.csv', index=False)
print('COMPLETE')
t_final1.head(40)
