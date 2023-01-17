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
import pandas as pd
import numpy as np
df1 = pd.read_csv('mem_data.csv')
df1
df1['GENDER'].value_counts()
df2 = pd.read_csv('mem_transaction.csv')
df2
df3 = pd.read_csv('store_info.csv')
df3
data = df1.merge(df2, how='left')
data
all_data = data.merge(df3, how='left')
all_data
all_data.info()
# 결측값 제거

alldata = all_data.fillna(0)
alldata.info()
alldata.BIRTH_SL.value_counts()
# 양력은 1, 음력은 0
alldata.BIRTH_SL = (alldata.BIRTH_SL=='S').astype(int)
alldata.BIRTH_SL.value_counts()
alldata.SMS.value_counts()
# YES는 1, NO는 0
alldata.SMS = (alldata.SMS=='Y').astype(int)
alldata.SMS.value_counts()
alldata.MEMP_STY.value_counts()
# O는 1, M는 0
alldata.MEMP_STY = (alldata.MEMP_STY=='O').astype(int)
alldata.MEMP_STY.value_counts()
obj = ['BIRTH_DT', 'ZIP_CD', 'RGST_DT', 'LAST_VST_DT', 'MEMP_DT', 'MEMP_TP']
alldata[obj] = alldata[obj].apply(lambda x: x.astype('category').cat.codes)
alldata.info()
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

fm.get_fontconfig_fonts()
font_location = 'C:\\Windows\\Fonts\\malgun.ttf'
font_name = fm.FontProperties(fname=font_location).get_name()
plt.rc('font', family=font_name)
import seaborn as sns
sns.heatmap(alldata.corr(), annot=True)
alldata['GENDER'].value_counts()
tr = alldata.GENDER!='UNKNOWN'
train = alldata[tr]
train.GENDER = (train.GENDER=='M').astype(int)
te = alldata.GENDER=='UNKNOWN'
test = alldata[te].sort_values('MEM_ID')
test.head()
train.corr()
display(train.head())
display(test.head())
t_final = test.copy()
t_final.head()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import glob
kfold = StratifiedKFold(n_splits=5) # 하이퍼 파라미터 지정
n_it = 12
test = test.drop(['GENDER','MEM_ID'], axis=1)
target = train.GENDER.values
train = train.drop(['GENDER','MEM_ID'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.25, random_state=0)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
print(train.shape, test.shape)
params = {'max_features':list(np.arange(1, train.shape[1])), 'bootstrap':[False], 'n_estimators': [50], 'criterion':['gini','entropy']}
model = RandomizedSearchCV(RandomForestClassifier(), param_distributions=params, n_iter=n_it, cv=kfold, scoring='roc_auc',n_jobs=-1, verbose=1)
print('MODELING.............................................................................')
model.fit(train, target)
print('========BEST_AUC_SCORE = ', model.best_score_)
model = model.best_estimator_
t_final.GENDER = model.predict_proba(test.values)[:,1]
import numpy as np 
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from matplotlib import font_manager, rc
import warnings
from sklearn.metrics import roc_curve
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import *
from sklearn.decomposition import PCA
from imblearn.combine import SMOTETomek
from lightgbm import LGBMClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split  
from sklearn.metrics import classification_report
# !pip install xgboost
warnings.filterwarnings('ignore')
%matplotlib inline
mpl.rcParams['axes.unicode_minus'] = False
plt.style.use('fivethirtyeight')

fm.get_fontconfig_fonts()
font_location = 'C:\\Windows\\Fonts\\malgun.ttf'
font_name = fm.FontProperties(fname=font_location).get_name()
plt.rc('font', family=font_name)

import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

from xgboost import plot_importance
%matplotlib inline
# 파라미터 서치
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
# scoring="roc_auc|f1" => 0.755
# categories_indices = [X_train.columns.get_loc(col) for col in ['주구매코너']]
best_param1 = bestGBDTNextModel('LGBM', False, 5, 'RANDOM', X_train, y_train, X_test, y_test, 15, 'roc_auc', 0, 3, -1)
lgbm1 = LGBMClassifier(**best_param1)
score_lgbm1 = lgbm1.fit(X_train, y_train).score(X_test, y_test)
print("score_lgbm1 ::: {}".format(score_lgbm1))
print("-----------------------------------")
y_lgbm1 = lgbm1.predict(X_test)
print(classification_report(y_test, y_lgbm1))

# 최고=0.755 Score LGBM best_estimator
#  LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
#         importance_type='split', lambda_l1=0.01, lambda_l2=0,
#         learning_rate=0.01, max_depth=50, metric='auc',
#         min_child_samples=20, min_child_weight=0.001, min_data_in_leaf=20,
#         min_split_gain=0.0, n_estimators=300, n_jobs=-1,
#         num_boost_round=4000, num_leaves=150, objective='binary',
#         random_state=0, reg_alpha=0.1, reg_lambda=0.0, silent=True,
#         subsample=1.0, subsample_for_bin=200000, subsample_freq=0)
best_model1 = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
               importance_type='split', lambda_l1=0.1, lambda_l2=0.01,
               learning_rate=0.05, max_depth=50, metric='auc',
               min_child_samples=20, min_child_weight=0.001,
               min_data_in_leaf=20, min_split_gain=0.0, n_estimators=400,
               n_jobs=-1, num_boost_round=4000, num_leaves=200,
               objective='binary', random_state=0, reg_alpha=0.01,
               reg_lambda=0.1, silent=True, subsample=0.5,
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
