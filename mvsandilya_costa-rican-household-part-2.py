import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
costa = pd.read_csv('../input/train.csv')
costa_test = pd.read_csv('../input/test.csv')
median = costa['meaneduc'].median()
costa['meaneduc'].fillna(median, inplace = True)
col_list = []
for feature in costa.columns: # Loop through all columns in the dataframe
    if costa[feature].isnull().sum() > 0: # Only apply for columns with categorical strings
        col_list.append(feature)
col_list
X = costa.drop(col_list, axis =1)
columns_list = ['rooms', 'r4t1','r4t2','r4m1', 'r4m2', 'r4h1','r4h2','v14a','refrig', 'v18q', 'meaneduc','r4h3', 'r4m3', 'r4t3', 'escolari','paredblolad', 'cielorazo', 'epared1', 'epared2', 'epared3', 'etecho1', 'etecho2' , 'etecho3', 'eviv1', 'eviv2', 'eviv3', 'hogar_adul', 'hogar_mayor', 'Target']
new = X[X['parentesco1']==1]
final = new[columns_list]
features = final[[i for i in list(final.columns) if i != 'Target']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features,final['Target'],
                                                    test_size=0.30)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))
median = costa_test['meaneduc'].median()
costa_test['meaneduc'].fillna(median, inplace = True)
col1_list = []
for feature in costa_test.columns: # Loop through all columns in the dataframe
    if costa_test[feature].isnull().sum() > 0: # Only apply for columns with categorical strings
        col1_list.append(feature)
col1_list
X1 = costa_test.drop(col1_list, axis =1)
newtest = X1[X1['parentesco1']==1]
columns_list1 = ['Id', 'idhogar','rooms', 'r4t1','r4t2','r4m1', 'r4m2', 'r4h1','r4h2','v14a','refrig', 'v18q', 'meaneduc','r4h3', 'r4m3', 'r4t3', 'escolari','paredblolad', 'cielorazo', 'epared1', 'epared2', 'epared3', 'etecho1', 'etecho2' , 'etecho3', 'eviv1', 'eviv2', 'eviv3', 'hogar_adul', 'hogar_mayor']
final1 = newtest[columns_list1]
final2 = final1.drop('Id', axis =1)
final3 = final2.drop('idhogar', axis=1)
final1.reset_index(inplace = True)
rfc_pred1 = rfc.predict(final3)
my_submission = pd.DataFrame({'Target': rfc_pred1})
final_submit = final1.join(my_submission)
my_submit1 = final_submit[['Target', 'idhogar']]
submission_base = X1[['Id', 'idhogar']].copy()
submission = submission_base.merge(my_submit1, 
                                       on = 'idhogar',
                                       how = 'left').drop(columns = ['idhogar'])
submission['Target'].fillna(median, inplace = True)
submission['Target'] = submission['Target'].astype(int)
submission.info()
#submission.to_csv('submission.csv', index = False)
from xgboost.sklearn import XGBClassifier 
xclas = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=1000,
       n_jobs=1, nthread=None, objective='multi:softprob', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)
xclas.fit(X_train, y_train)  
xgb_pred1 = xclas.predict(final3)
my_submission_xgb = pd.DataFrame({'Target': xgb_pred1})
final_submit_xgb = final1.join(my_submission_xgb)
my_submit_xgb = final_submit_xgb[['Target', 'idhogar']]
submission_base_xgb = X1[['Id', 'idhogar']].copy()
submission_xgb = submission_base_xgb.merge(my_submit_xgb, 
                                       on = 'idhogar',
                                       how = 'left').drop(columns = ['idhogar'])
median = submission_xgb['Target'].median()
submission_xgb['Target'].fillna(median, inplace= True)
submission_xgb['Target'] = submission_xgb['Target'].astype(int)
submission_xgb.to_csv('submission_xgb.csv', index = False)
