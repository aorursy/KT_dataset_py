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
import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
# from plotly import tools
# import plotly.tools as tls
application_train = pd.read_csv('../input/application_train.csv')
POS_CASH_balance = pd.read_csv('../input/POS_CASH_balance.csv')
bureau_balance = pd.read_csv('../input/bureau_balance.csv')
previous_application = pd.read_csv('../input/previous_application.csv')
installments_payments = pd.read_csv('../input/installments_payments.csv')
credit_card_balance = pd.read_csv('../input/credit_card_balance.csv')
bureau = pd.read_csv('../input/bureau.csv')
application_test = pd.read_csv('../input/application_test.csv')
sample = pd.read_csv('../input/sample_submission.csv')
sample.head(2)
print('Size of application_train data', application_train.shape)
print('Size of POS_CASH_balance data', POS_CASH_balance.shape) ## like 30,000 unique 
print('Size of bureau_balance data', bureau_balance.shape)
print('Size of previous_application data', previous_application.shape) ## 338857 unique 
print('Size of installments_payments data', installments_payments.shape)
print('Size of credit_card_balance data', credit_card_balance.shape)
print('Size of bureau data', bureau.shape)
#bureau_df = bureau_balance.merge(bureau,on='SK_ID_BUREAU')
#print(bureau_df.shape)
#bureau_df.head(10)
#len(previous_application['SK_ID_CURR'].unique())
#POS_CASH_balance.head(2)
#POS_CASH_balance['SK_DPD_DEF'].value_counts()
#pos_grp_by_1 = POS_CASH_balance.groupby('SK_ID_CURR')['MONTHS_BALANCE','CNT_INSTALMENT','CNT_INSTALMENT_FUTURE','SK_DPD','SK_DPD_DEF'].mean()
#pos_grp_by_1 = pos_grp_by_1.reset_index()
application_train.shape,application_test.shape
'''
application_train['Is_train'] = 'Yes'
application_test['Is_train'] = 'No'
target = application_train['TARGET']
del application_train['TARGET']
frames = [application_train,application_test]
full_application_df = pd.concat(frames)
print (full_application_df.head(2))
print (full_application_df.shape)
'''
#full_application_df['Is_train'].value_counts()
#a = full_application_df.merge(previous_application,on='SK_ID_CURR',how='inner')
#a.shape
#len(a['SK_ID_CURR'].unique())
#merged = pd.merge(full_application_df,previous_application, on=['SK_ID_CURR'])
#merged.shape
application_train.head(2)
categorical_mask = (application_train.dtypes==object)
categorical_mask
categorical_column = application_train.columns[categorical_mask].tolist()
categorical_column
application_train[categorical_column].head(2)
target = application_train['TARGET']
print (target.value_counts())
del application_train['TARGET']
numerical_column = application_train.select_dtypes(exclude=['object']).columns.tolist()
numerical_column
application_train[numerical_column].head(2)
print ('CATEGORICAL-COLS-IN-APPLICATION-TRAIN:',len(categorical_column))
print ('NUMERICAL-COLS-IN-APPLICATION-TRAIN:',len(numerical_column))
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline,FeatureUnion,make_pipeline

app_train = application_train[numerical_column+categorical_column]
app_test = application_test[numerical_column+categorical_column]
print(app_train.shape,app_test.shape)
app_train.head(2)
print (app_train.isnull().sum())
print (app_test.isnull().sum())
from sklearn_pandas import DataFrameMapper
from sklearn_pandas import CategoricalImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
import xgboost as xgb 
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import gc
import lightgbm as gbm
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold,RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
X = app_train.copy()
print(X.shape)
y = target
X_test = app_test.copy()
print(X_test.shape)
# separate dataset into train and test
X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.3,random_state=1)
print (X_train.shape, X_test.shape)
print (y_train.value_counts())
print(y_val.value_counts())
del X_train['SK_ID_CURR']
del X_val['SK_ID_CURR']
test_ids = X_test['SK_ID_CURR']
del X_test['SK_ID_CURR']
del X['SK_ID_CURR']
# Create a boolean mask for categorical columns
categorical_feature_mask = X.dtypes == object
CATEGORICAL_COLS = X.columns[categorical_feature_mask].tolist()
NON_CATEGORICAL_COLS = X.columns[~categorical_feature_mask].tolist()
print ('CATEGORICAL_COLS:',len(CATEGORICAL_COLS))
print ('NON_CATEGORICAL_COLS:',len(NON_CATEGORICAL_COLS))
numeric_imputation_mapper = DataFrameMapper(
    [([numeric_feature],Imputer(strategy='median')) for numeric_feature in NON_CATEGORICAL_COLS],input_df=True,df_out=True)
# Apply categorical imputer( it will effect all categorocal columns)
categorical_imputation_mapper = DataFrameMapper(
                                                [(category_feature, CategoricalImputer()) for category_feature in CATEGORICAL_COLS],
                                                input_df=True,
                                                df_out=True
                                               )
numerical_categorical_union = FeatureUnion([('num_mapper',numeric_imputation_mapper),('cat_mapper',categorical_imputation_mapper)])
    

full_pipeline_rf = Pipeline([
                         ("feature_union",numerical_categorical_union),
                         ("clf", RandomForestClassifier(max_depth=3,class_weight='balanced'))
])
full_pipeline_xg = Pipeline([
                         ("feature_union",numerical_categorical_union),
                         ("cl", xgb.XGBClassifier(max_depth=3))
])
full_pipeline_logis = Pipeline([
                         ("feature_union",numerical_categorical_union),
                         ("scaleing",MaxAbsScaler ),
                         ("logs",LogisticRegression(C=100,class_weight='balanced'))
])
print("Loading data...\n")
lb=LabelEncoder()
def LabelEncoding_Cat(df):
    df=df.copy()
    Cat_Var=df.select_dtypes('object').columns.tolist()
    for col in Cat_Var:
        df[col]=lb.fit_transform(df[col].astype('str'))
    return df
X = LabelEncoding_Cat(X)
X_train = LabelEncoding_Cat(X_train)
X_val = LabelEncoding_Cat(X_val)
X_test = LabelEncoding_Cat(X_test)
print ('X-shape:',X.shape)
print ('X_train-shape:',X_train.shape)
print ('X_test-shape:',X_test.shape)
print ('X_val.shape:',X_val.shape)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold,RepeatedKFold
#print ('--Logistic Performance--')
#cv_logis = cross_val_score(full_pipeline_logis,X,y,scoring="roc_auc",cv=3)
#print (cv_logis)
'''
print ('---RandomForest Performance---')
cv_rf = cross_val_score(full_pipeline_rf,X,y,scoring="roc_auc",cv=3)
print (cv_rf)
print('---Xgboost Performance---')
cv_xg = cross_val_score(full_pipeline_xg,X,y,scoring="roc_auc",cv=3)
print(cv_xg)
'''
#print (cv_rf)
#print (cv_xg)
from sklearn.metrics import roc_auc_score,confusion_matrix,classification_report
from sklearn.tree import DecisionTreeClassifier
'''
roc_values = []
for feature in X_train.columns:
    clf = DecisionTreeClassifier()
    clf.fit(X_train[feature].fillna(0).to_frame(), y_train)
    y_scored = clf.predict_proba(X_val[feature].fillna(0).to_frame())
    roc_values.append(roc_auc_score(y_val, y_scored[:, 1]))
roc_values = pd.Series(roc_values)
roc_values.index = X_train.columns
roc_values.sort_values(ascending=False).plot.bar(figsize=(20,8))
'''
#selected_feat = roc_values[roc_values>0.54]
#len(selected_feat), X_train.shape[1]
'''
X_selec = X[selected_feat.index]
print (X_selec.shape)
X_train_selec = X_train[selected_feat.index]
print (X_train_selec.shape)
X_val_selec = X_val[selected_feat.index]
print (X_val_selec.shape)
X_test_selec = X_test[selected_feat.index]
print (X_test_selec.shape)
'''
#X_test_selec.head(2)
#full_pipeline_rf
rf = RandomForestClassifier(class_weight='balanced')
#xg = xgb.XGBClassifier(max_depth=3)
''''
print ('---RandomForest Performance---')
cv_rf = cross_val_score(rf,X_selec.fillna(0),y,scoring="roc_auc",cv=3)
print (cv_rf)
print('---Xgboost Performance---')
cv_xg = cross_val_score(xg,X_selec.fillna(0),y,scoring="roc_auc",cv=3)
print(cv_xg)
'''
rf.fit(X.fillna(0),y)
#pred_val = rf.predict(y_val.values)
#print ('--Confusion_Matrix---',confusion_matrix(y_val,pred_val))
#print ('--Classification_report--',classification_report(y_val,pred_val))
pred_test = rf.predict_proba(X_test.fillna(0))[:,1]
submission = pd.DataFrame()
submission['SK_ID_CURR'] = test_ids
submission['TARGET'] = pred_test
submission.to_csv('Step-2-RF-selected_feat.csv',index=False)
dtrain = xgb.DMatrix(X.values,label=y.values)
dtest = xgb.DMatrix(X_test.values)
#dval = xgb.DMatrix(X_val.values,label=y_val.values)
train_labels = dtrain.get_label()
params = {'objective':'binary:logistic',
          'n_estimators':5000,
         'max_depth':7,
         'eta':1}
num_rounds = 200 
ratio = float(np.sum(train_labels== 0))/np.sum(train_labels == 1)
params['scale_pos_weight'] = ratio
bst = xgb.train(params,dtrain,num_rounds)

#xg_prediction_val = (bst.predict(dval) > 0.5).astype(int)
#print (confusion_matrix(y_val,xg_prediction_val))
#print (classification_report(y_val,xg_prediction_val))

#roc_auc_score(y_val,xg_prediction_val)*100
xg_prediction = bst.predict(dtest)

xg_prediction
submission1 = pd.DataFrame()
submission1['SK_ID_CURR'] = test_ids
submission1['TARGET'] = xg_prediction
submission1.to_csv('Step-2-XG-selected_feat.csv',index=False)





#cv = cross_val_score(full_pipeline,X,y,cv=3)
gbm_param_grid = {
    'clf__learning_rate': np.arange(.05, 1, .05),
    'clf__max_depth': np.arange(3,10, 1),
    'clf__n_estimators': np.arange(50, 200, 50)}
#gbm_param_grid['scale_pos_weight'] = np.arange(1,100)
# Perform RandomizedSearchCV
randomized_roc_auc = RandomizedSearchCV(estimator=full_pipeline,
                                        param_distributions=gbm_param_grid,
                                        n_iter=2, scoring='roc_auc', cv=2, verbose=1)
randomized_roc_auc.fit(X, y)
predictions = randomized_roc_auc.predict_proba(X_test)[:,1]
predictions
#test_ids = X_test['SK_ID_CURR']
submission = pd.DataFrame()
submission['SK_ID_CURR'] = test_ids
submission['TARGET'] = predictions
submission.to_csv('Step-1-RF-RandmSearch_all_feat.csv')




