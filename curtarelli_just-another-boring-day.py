import numpy as np
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.info()
test.info()
pd.options.display.float_format = '{:.2f}'.format
train.describe()
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(figsize=(30,10))
sns.heatmap(train.replace(0,np.NaN).isnull(), cbar=False)
train['target'].plot(kind='hist',bins=50)
np.log(train['target']).plot(kind='hist', bins=50)
x=train.drop(['ID','target'], axis=1)
y=np.log(train['target'])

from sklearn.model_selection import train_test_split

x_train, x_cv, y_train, y_cv = train_test_split( x, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_cv = scaler.transform(x_cv)
from sklearn.decomposition import PCA
pca = PCA(.90)
pca.fit(x_train)

x_train = pca.transform(x_train)
x_cv = pca.transform(x_cv)

test_matrix = test.drop(['ID'], axis=1).values
test_matrix = pca.transform(test_matrix)
print(x_train.shape)
print(x_cv.shape)
print(test_matrix.shape)
import xgboost as xgb
md,lr,ne = [3,6,9,12],[0.01,0.10,0.20,0.50,1.00],[100,150,200,250,300]
params = [[x,y,z] for x in md for y in lr for z in ne]
print(len(params))
def rmsle(a,b):
    return np.sqrt(np.mean(np.square( np.log( (np.exp(a)) + 1 ) - np.log((np.exp(b))+1) )))
params_dict = {}
"""
%%time
for i in range(len(params)):
    error_rate = []
    dtrain=xgb.DMatrix(x_train,label=y_train)
    dcv=xgb.DMatrix(x_cv,label=y_cv)
    dtest =xgb.DMatrix(x_cv)
    
    watchlist = [(dtrain,'train-rmse'), (dcv, 'eval-rmse')]
    parameters={'max_depth':params[i][0], 'silent':1,'objective':'reg:linear','eval_metric':'rmse','learning_rate':params[i][1]}
    num_round=params[i][2]
    xg=xgb.train(parameters,dtrain,num_boost_round = num_round,evals = watchlist,early_stopping_rounds = 7,verbose_eval =False) 
    y_pred=xg.predict(dtest) 
    rmsle_calculated = rmsle(y_pred,y_cv)
    error_rate.append(rmsle_calculated)
    
    params_dict[str(params[i])] = round(np.mean(error_rate),5)
    if i % 5 ==0:
        print(i)
"""
"""
params_df = pd.Series(params_dict)
print(len(params_dict))
params_df = params_df.sort_values(ascending=True)
params_df[:20]
"""
%%time
error_rate = []
dtrain=xgb.DMatrix(x_train,label=y_train)
dcv=xgb.DMatrix(x_cv,label=y_cv)
dtest =xgb.DMatrix(x_cv)

watchlist = [(dtrain,'train-rmse'), (dcv, 'eval-rmse')]
parameters={'max_depth':3, 'silent':1,'objective':'reg:linear','eval_metric':'rmse','learning_rate':0.1}
num_round=250
xg=xgb.train(parameters,dtrain,num_boost_round = num_round,evals = watchlist,early_stopping_rounds = 5,verbose_eval =False) 
y_pred=xg.predict(dtest) 
rmsle_calculated = rmsle(y_pred,y_cv)
error_rate.append(rmsle_calculated)
plt.figure(figsize=(30,5))
xgb.plot_importance(xg,max_num_features=100)
dtest =xgb.DMatrix(test_matrix)
y_pred=xg.predict(dtest)
y_pred=pd.Series(np.exp(y_pred))
sample_sub = pd.read_csv('../input/sample_submission.csv')
del sample_sub['target']
sample_sub['target'] = y_pred
sample_sub.to_csv('first_try.csv',index=False)