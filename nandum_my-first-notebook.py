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
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
train.shape
test.shape
plt.figure(figsize=(10,10))
sns.distplot(train['target'] , bins = 25 )
plt.show()
plt.figure(figsize=(10,12))
sns.distplot(np.log10( 1 + train['target'].values) , bins=25)
plt.show()
train.isnull().values.any()
cols = list(train.columns)
cols.remove('ID')
cols.remove('target')
len(cols)
correlations = train[cols].corr()
corr_with_target = []
for col in cols:
        corr_with_target.append(np.corrcoef(train[col].values , train['target'].values)[0,1])
correlation_matrix = pd.DataFrame({'cols' : cols , 'correlation_value' : corr_with_target})
print("no:of columns with corr value > 0.1 : " +str(len(correlation_matrix[(correlation_matrix['correlation_value'] > 0.1) | (correlation_matrix['correlation_value'] < -0.1)])))
print("no:of columns with corr value > 0.2 : " +str(len(correlation_matrix[(correlation_matrix['correlation_value'] > 0.2) | (correlation_matrix['correlation_value'] < -0.2)])))
print("no:of columns with corr value > 0.3 : " +str(len(correlation_matrix[(correlation_matrix['correlation_value'] > 0.3) | (correlation_matrix['correlation_value'] < -0.3)])))
features = list(correlation_matrix.cols[(correlation_matrix['correlation_value'] > 0.15) | (correlation_matrix['correlation_value'] < -0.15)].values)
unique_df = train[features].nunique().reset_index()
unique_df.columns = ["col_name" , "count"]
unique_df[unique_df['col_name'] == 1].shape
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import cross_val_score
import lightgbm as lgb
def execute_rf_model(train_x, train_y, val_x, val_y , test_final):
    rf_clf = RandomForestRegressor(n_estimators=200, min_samples_split= 50, n_jobs=-1 , random_state=0)
    rf_clf.fit(train_x , train_y)
    val_predicted = rf_clf.predict(val_x)
    print("MSE :"+ str(mean_squared_error(val_y , val_predicted)))
    y_pred = rf_clf.predict(test_final)
    return y_pred , rf_clf
    
X_train = train[features]
y_train = np.log(1+train['target'].values)
train_x , val_x, train_y , val_y = train_test_split(X_train, y_train)
y_final_rf, rf_model = execute_rf_model(train_x, train_y, val_x, val_y, test[features])
lgb_params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 50,
        "learning_rate" : 0.01,
        "bagging_fraction" : 0.6,
        "bagging_frequency" : 5,
        "bagging_seed" : 2000
    }
def execute_lgbmodel(params , train_x , train_y , val_x , val_y , test_final):
    train_lgb = lgb.Dataset(train_x, label=train_y)
    test_lgb = lgb.Dataset(val_x, label=val_y)
    evals_result = {}
    model = lgb.train(params, train_lgb, 5000, 
                      valid_sets=[test_lgb], 
                      early_stopping_rounds=100, 
                      evals_result=evals_result)
    
    pred_test_y = model.predict(test_final, num_iteration=model.best_iteration)
    return pred_test_y, model, evals_result
y_final_lgb, _ , _ = execute_lgbmodel(lgb_params , train_x , train_y , val_x , val_y , test[features])
y_final_lgb = np.exp(1 + y_final_lgb)
y_final_rf = np.exp(1 + y_final_rf)

y_final = 0.75*y_final_lgb + 0.25*y_final_rf
test['target'] = y_final
test[['ID' , 'target']].to_csv('./sub1.csv' , index = False)
