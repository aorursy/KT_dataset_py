# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder

# File system manangement
import os
import numpy as np
import pandas as pd
# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn import metrics
# Training data
app_train = pd.read_csv('../input/public.train.csv')
print('Training data shape: ', app_train.shape)
#app_train.head()
# Training data
app_test = pd.read_csv('../input/public.test.csv')
print('Training data shape: ', app_test.shape)
#app_train.head()
train_id=app_train[['ID']]
test_id=app_test[['ID']]
app_train_test = [app_train, app_test]
app_train_test = pd.concat(app_train_test)
app_train_test=app_train_test.mask(app_train_test.sub(app_train_test.mean()).div(app_train_test.std()).abs().gt(3))
app_train_test=app_train_test.fillna(method='ffill')
app_train= train_id.merge(app_train_test, on='ID', how='left')
app_train.shape
app_test= test_id.merge(app_train_test, on='ID', how='left')
app_test=app_test.drop(columns='发电量')
app_test.shape
#domine feature
'''a'''
app_train['理论输出']=app_train['光照强度']*app_train['转换效率']
app_test['理论输出']=app_test['光照强度']*app_test['转换效率']
'''b'''
app_train['温差']=app_train['板温']-app_train['现场温度']
app_test['温差']=app_test['板温']-app_test['现场温度']
'''c'''
app_train['实际功率']=app_train['转换效率']*app_train['平均功率']
app_test['实际功率']=app_test['转换效率']*app_test['平均功率']
'''d'''
app_train['风力X风向']=app_train['风向']*app_train['风速']
app_test['风力X风向']=app_test['风向']*app_test['风速']

app_train['实际温度']=app_train['转换效率']*app_train['现场温度']
app_test['实际温度']=app_test['转换效率']*app_test['现场温度']
#app_train['abk']=(app_train['电流B']/app_train['转换效率B']+app_train['电流A']/app_train['转换效率A']+app_train['电流C']/app_train['转换效率C'])/3
#3app_test['abk']=(app_test['电流B']/app_test['转换效率B']+app_test['电流A']/app_test['转换效率A']+app_test['电流C']/app_test['转换效率C'])/3
app_train['cde']=app_train['电压A']/app_train['转换效率A']
app_test['cde']=app_test['电压A']/app_test['转换效率A']
app_train['cde1']=app_train['电压B']/app_train['转换效率B']
app_test['cde1']=app_test['电压B']/app_test['转换效率B']
app_train['cde2']=app_train['电压C']/app_train['转换效率C']
app_test['cde2']=app_test['电压C']/app_test['转换效率C']

#app_train['iuo']=app_train['光照强度']*np.cos((app_train['ID']))
#app_test['iuo']=app_test['光照强度']*np.cos((app_test['ID']))
app_train['cde']=app_train['cde']*app_train['cde']
app_test['cde']=app_test['cde']*app_test['cde']
app_train['cde1']=app_train['cde1']*app_train['cde1']
app_test['cde1']=app_test['cde1']*app_test['cde1']
app_train['cde2']=app_train['cde2']*app_train['cde2']
app_test['cde2']=app_test['cde2']*app_test['cde2']



#app_train['风力X风向1']=app_train['转换效率']*app_train['风力X风向']
#app_test['风力X风向1']=app_test['转换效率']*app_test['风力X风向']

#app_test['abk']=(app_test['电流B']/app_test['转换效率B']+app_test['电流C']/app_test['转换效率C'])/2
app_train=app_train.drop(columns='ID')

app_test=app_test.drop(columns='ID')
final_y=app_train.pop('发电量')
#print(final_y)
final_x=app_train
final_x.columns
import xgboost as xgb
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(final_x, final_y, test_size=0.2, random_state=42)
X_train.head()
from xgboost import XGBRegressor

#my_model = XGBRegressor()
# Add silent=True to avoid printing out updates with each cycle
#my_model.fit(X_train, y_train, verbose=False)
# make predictions
#predictions = my_model.predict(X_test)

from sklearn.metrics import mean_squared_error
#print("rmse : " + str(np.sqrt(((predictions - y_test) ** 2).mean())))
import math
def my_scorer(y_true, y_predicted):
    rmse = math.sqrt(np.mean((y_true - y_predicted)**2))

    return rmse
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import KFold
loss  = make_scorer(my_scorer, greater_is_better=False)
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
cv_params = {'n_estimators': [20000],
            'max_depth':[10,8,6],
             'subsample': [0.6],
              'colsample_bytree':[0.8],

            }
other_params = {'learning_rate': 0.005, 'seed': 0,
                   'gamma': 0}
# To be used within GridSearch (5 in your case)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=98)
model = xgb.XGBRegressor(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring=loss,cv=inner_cv, verbose=1, n_jobs=4)
optimized_GBM.fit(X_train, y_train)
#evalute_result=optimized_GBM.cv_results_
#print('每轮迭代运行结果:{0}'.format(evalute_result))
print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
#def logregobj(preds, dtrain):
 #   labels = dtrain.get_label()
  #  preds = 1.0 / (1.0 + np.exp(-preds))
   # grad = preds - labels
    #hess = preds * (1.0 - preds)
 #   return grad, hess

#model = xgb.XGBRegressor(colsample_bytree=0.6, max_depth=8, n_estimators=2000, reg_alpha=0, reg_lambda=1, subsample= 0.6,learning_rate= 0.005,  min_child_weight= 1, 
 #                   gamma= 0)
#model.fit(X_train, y_train)
#app_test.shape
# make predictions
predictions =optimized_GBM.predict(app_test)

len(predictions)

app_test = pd.read_csv('../input/public.test.csv')
print('Training data shape: ', app_test.shape)

test_id=app_test['ID']
submit=pd.DataFrame({'ID': test_id, '发电量': predictions})
submit.to_csv('xgboost_5.csv',index=False)