import pandas as pd

import numpy as np

import xgboost as xgb

import lightgbm as lgb

pd.options.mode.chained_assignment = None
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00510/Grisoni_et_al_2016_EnvInt88.csv')

df
df.SMILES.value_counts()
import re

df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

df['SMILES'] = df['SMILES'].apply(lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
df.SMILES.value_counts()
df.head()
df.Set.value_counts(100)
x = df.drop(['Class','Set'],axis=1)

y = df[['Class']]
x,y
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,

                                                    y,

                                                    test_size = 0.25,

                                                    random_state=29)
pd.DataFrame(np.arange(12).reshape((4,3)),columns=['a', 'b', 'c'])
xgb.DMatrix(pd.DataFrame(np.arange(12).reshape((4,3)), columns=['a', 'b', 'c']))
xgb.DMatrix(X_train)
xgb.DMatrix(pd.get_dummies(df['Set']))
lgb.LGBMClassifier(boosting_type='gbdt',n_jobs=-1,).fit(X_train.select_dtypes('float'),np.ravel(y_train))
lgb.LGBMClassifier(n_jobs=-1).fit(X_train,np.ravel(y_train))
X_train[['CAS','SMILES']] = X_train[['CAS','SMILES']].apply(lambda x: x.astype('category'),axis=0)

X_test[['CAS','SMILES']] = X_test[['CAS','SMILES']].apply(lambda x: x.astype('category'),axis=0)
X_train.dtypes
lgb.LGBMClassifier(n_jobs=-1).fit(X_train,np.ravel(y_train))
X_train_na = X_train.copy()
X_train_na.isna().sum().sum()
X_train_na = X_train_na.replace(to_replace=0,value=np.nan)
X_train_na.isna().sum().sum()
xgb.DMatrix(X_train_na['N072'].replace(to_replace=0,value=np.nan), missing=np.nan)
xgb.DMatrix(X_train_na['N072'], missing=np.nan)
xgb.XGBRFClassifier().fit(pd.get_dummies(X_train_na,columns=['CAS','SMILES']),np.ravel(y_train))
lgb.LGBMClassifier(n_jobs=-1,use_missing=True).fit(X_train,np.ravel(y_train))
from xgboost import plot_importance



plot_importance(xgb.XGBRFClassifier().fit(pd.get_dummies(X_train_na,

                                                         columns=['CAS','SMILES']),np.ravel(y_train)));
lgb.plot_importance(lgb.LGBMClassifier(n_jobs=-1,

                                       use_missing=True).fit(X_train,np.ravel(y_train)));
%%time

xgb.XGBRegressor(colsample_bytree = 0.3, 

                 learning_rate = 0.1,max_depth = 5, alpha = 10, 

                 n_estimators = 10).fit(pd.get_dummies(X_train_na,columns=['CAS','SMILES']),np.ravel(y_train))
%%time

lgb.LGBMClassifier(n_jobs=-1,use_missing=True,categorical_feature=True,max_depth=1,learning_rate=0.1).fit(X_train_na,np.ravel(y_train),verbose=True)
from sklearn.metrics import f1_score,precision_score,recall_score,mean_squared_error,classification_report
X_train_xgboost, X_test_xgboost, y_train, y_test = train_test_split(pd.get_dummies(x,columns=['SMILES','CAS']),

                                                    y,

                                                    test_size = 0.25,

                                                    random_state=29)
X_train_xgboost.shape
%%time

xgb_c = xgb.XGBRFClassifier(n_jobs=-1,).fit(X_train_xgboost,np.ravel(y_train))

xgb_predict = xgb_c.predict(X_test_xgboost)

print("\n Model Parameters: ",xgb_c)

print("\nF1 Score: ",f1_score(y_test, xgb_predict,average='weighted'))

print("\nPrecision Score : ",precision_score(y_test, xgb_predict,average='weighted'))

print("\nMSE SCORE : ",mean_squared_error(y_test, xgb_predict))

print("\nclassification report: \n",classification_report(y_test,xgb_predict))
%%time

lgb_c = lgb.LGBMClassifier(n_jobs=-1).fit(X_train,np.ravel(y_train),verbose=True)

lgb_predict = lgb_c.predict(X_test)

print("\nModel Parameters: ",lgb_c)

print("\nF1 Score: ",f1_score(y_test, lgb_predict,average='weighted'))

print("\nPrecision Score : ",precision_score(y_test, lgb_predict,average='weighted'))

print("\nMSE SCORE : ",mean_squared_error(y_test, lgb_predict))

print("\nclassification report: \n",classification_report(y_test,lgb_predict))