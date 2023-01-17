import numpy as np
import pandas as pd
from catboost import Pool, CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
data=pd.read_csv('../input/diamonds.csv')
data.head()
data.describe()
data['cut'].value_counts()
data['color'].value_counts()
data['clarity'].value_counts()
#cut
cut_val=['Ideal','Premium','Very Good','Good','Fair']
cut_s=[0,1,2,3,4]
cut_typ=dict(zip(cut_val,cut_s))
data=data.replace({'cut':cut_typ})

#color
color_val=['D','E','F','G','H','I','J']
color_s=[0,1,2,3,4,5,6]
color_typ=dict(zip(color_val,color_s))
data=data.replace({'color':color_typ})

#clarity
clar_val=['SI1','VS2','SI2','VS1','VVS2','VVS1','IF','I1']
clar_s=[0,1,2,3,4,5,6,7]
clar_typ=dict(zip(clar_val,clar_s))
data=data.replace({'clarity':clar_typ})
sub=data['price']
train_data=data.filter(['carat','cut','color','clarity','depth','table','x','y','z'],axis=1)
train_X, test_X, train_y, test_y = train_test_split(train_data, sub, test_size=0.5, random_state=42)
train_X['price']=train_y
train_X.head()
len(train_X)
train_y=train_X['price']
train_X=train_X.drop(['price'],axis=1)
scaler = MinMaxScaler()
scaler.fit(train_X)
train_X=scaler.transform(train_X)
test_X=scaler.transform(test_X)
train_pool = Pool(train_X, train_y)
test_pool = Pool(test_X, test_y.values) 
model2 = CatBoostRegressor(
    iterations=10000,
    depth=10,
    learning_rate=0.001,
    l2_leaf_reg= 0.1,#def=3
    loss_function='RMSE',
    eval_metric='MAPE',
    random_strength=0.001,
    bootstrap_type='Bayesian',#Poisson (supported for GPU only);Bayesian;Bernoulli;No
    bagging_temperature=1,#for Bayesian bootstrap_type; 1=exp;0=1
    leaf_estimation_method='Newton', #Gradient;Newton
    leaf_estimation_iterations=2,
    boosting_type='Ordered' #Ordered-small data sets; Plain
    ,task_type = "GPU"
    ,feature_border_type='Median' #Median;Uniform;UniformAndQuantiles;MaxLogSum;MinEntropy;GreedyLogSum
    ,random_seed=1234
)
model2.fit(train_pool, eval_set=test_pool, plot=True)
train_data=scaler.transform(train_data)
y_pred=model2.predict(train_data)
np.mean(np.abs((sub - y_pred) / sub))