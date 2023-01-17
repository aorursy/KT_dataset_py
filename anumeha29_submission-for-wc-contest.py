import os
print((os.listdir('../input/')))
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
%matplotlib inline
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


from scipy import stats
from scipy.stats import norm, skew #for some statistics


pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points


from subprocess import check_output

df_train = pd.read_csv('../input/web-club-recruitment-2018/train.csv')
df_test = pd.read_csv('../input/web-club-recruitment-2018/test.csv')
corr_matrix=df_train.corr()
corr_matrix['Y'].sort_values()

df_train=df_train.drop('X12',axis=1)
df_test=df_test.drop('X12',axis=1)
train_ID = df_train['id']
test_ID = df_test['id']
df_train=df_train.drop('id',axis=1)
df_test=df_test.drop('id',axis=1)
df_train['X1']=np.log1p(df_train['X1'])
df_test['X1']=np.log1p(df_test['X1'])
df_train
from sklearn.preprocessing import Imputer
imputer=Imputer(strategy="median")
col=df_train.columns
cols=df_test.columns
df_train=imputer.fit_transform(df_train)
df_train=pd.DataFrame(df_train,columns=col)
df_test=imputer.fit_transform(df_test)
df_test=pd.DataFrame(df_test,columns=cols)

train_X = df_train.loc[:, 'X1':'X23']
train_y = df_train.loc[:, 'Y']

col_mask=train_X.isnull().any(axis=0) 
col_mask

dev_X, val_X, dev_y, val_y = train_test_split(train_X, train_y, test_size = 0.2, random_state = 42)
params = {'objective': 'binary:logistic','eval_metric': 'rmse', 'eta': 0.005, 'max_depth': 10, 'subsample': 0.7, 'colsample_bytree': 0.5, 'alpha':0, 'silent': True, 'random_state':5}
# {'objective': 'reg:linear','eval_metric': 'rmse','eta': 0.005,'max_depth': 15,'subsample': 0.7,'colsample_bytree': 0.5,'alpha':0,'random_state':42,'silent': True}
    
tr_data = xgb.DMatrix(train_X, train_y)
va_data = xgb.DMatrix(val_X, val_y)

watchlist = [(tr_data, 'train'), (va_data, 'valid')]

model_xgb = xgb.train(params, tr_data, 2000, watchlist, maximize=False, early_stopping_rounds = 30, verbose_eval=100)

dft = xgb.DMatrix(df_test)
xgb_pred_y = np.log1p(model_xgb.predict(dft, ntree_limit=model_xgb.best_ntree_limit))







result = pd.DataFrame()
result['id']=test_ID
result['predicted_val']=xgb_pred_y

print(result.head())
result.to_csv('output.csv',index=False)

