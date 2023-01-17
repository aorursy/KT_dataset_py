import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from catboost import Pool, CatBoostRegressor
data=pd.read_csv('../input/train_V2.csv')
gamer_counts=data.groupby(['matchId']).size().reset_index(name='counts')
train_data=pd.merge(data,gamer_counts, on='matchId')
train_data.head()
mtype_val=['squad-fpp','duo-fpp','squad','solo-fpp','duo','solo','normal-squad-fpp','crashfpp','normal-duo-fpp','flaretpp','normal-solo-fpp','flarefpp','normal-squad','crashtpp','normal-solo','normal-duo']
mtype_s=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
mtype_typ=dict(zip(mtype_val,mtype_s))
train_data=train_data.replace({'matchType':mtype_typ})
train_data['winPlacePerc']=train_data['winPlacePerc'].fillna(train_data['winPlacePerc'].mean())
train_sub=train_data['winPlacePerc']
train_data=train_data.drop(['winPlacePerc','Id','groupId','matchId'],axis=1)
scaler = MinMaxScaler()
scaler.fit(train_data)
train=scaler.transform(train_data)
train_X, test_X, train_y, test_y = train_test_split(train, train_sub, test_size=0.5, random_state=42)
train_pool = Pool(train_X, train_y)
test_pool = Pool(test_X, test_y.values) 
model = CatBoostRegressor(
    iterations=5000,
    depth=10,
    learning_rate=0.1,
    l2_leaf_reg= 2,#def=3
    loss_function='RMSE',
    eval_metric='MAE',
    random_strength=0.1,
    bootstrap_type='Bernoulli',#Poisson (supported for GPU only);Bayesian;Bernoulli;No
    #bagging_temperature=1,#for Bayesian bootstrap_type; 1=exp;0=1
    leaf_estimation_method='Gradient', #Gradient;Newton
    leaf_estimation_iterations=1,
    boosting_type='Plain' #Ordered-small data sets; Plain
    ,task_type = "GPU"
    ,feature_border_type='GreedyLogSum' #Median;Uniform;UniformAndQuantiles;MaxLogSum;MinEntropy;GreedyLogSum
    ,random_seed=1234
)
model.fit(train_pool, eval_set=test_pool, plot=True)
fea_imp = pd.DataFrame({'imp': model.feature_importances_, 'col': train_data.columns})
fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]
fea_imp.plot(kind='barh', x='col', y='imp', figsize=(20, 10))
test_data=pd.read_csv('../input/test_V2.csv')
gamer_counts=test_data.groupby(['matchId']).size().reset_index(name='counts')
test=pd.merge(test_data,gamer_counts, on='matchId')
test=test.drop(['Id','groupId','matchId'],axis=1)
test=test.replace({'matchType':mtype_typ})
test=scaler.transform(test)
pred_test=model.predict(test)
res=test_data.filter(['Id'],axis=1)
res['winPlacePerc']=pred_test
res.to_csv('submission.csv', index=False)