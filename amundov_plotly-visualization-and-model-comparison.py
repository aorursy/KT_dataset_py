

import numpy as np

import pandas as pd 

import math





from scipy import stats

from scipy.stats import norm,skew



from statistics import mode

from sklearn.preprocessing import RobustScaler

from sklearn.metrics import explained_variance_score, mean_squared_log_error, mean_squared_error



import matplotlib.pyplot as plt

import seaborn as sns



import plotly

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, plot, iplot

init_notebook_mode(connected = True)





import warnings

warnings.simplefilter('ignore')



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



n_train = len(train)
traces = []

for value in np.sort(train.OverallQual.unique()):

    df = train[train.OverallQual == value]

    traces.append(

        go.Scatter(x=df['1stFlrSF'], y = df['SalePrice'], name = ('Quality: ' + str(value)), mode = 'markers',opacity = 0.75)          

    )



iplot({

    'data': traces,

    'layout': go.Layout({'xaxis':{'title':'1st floor square feet'},

                         'yaxis':{'title': 'SalePrice'}

        

    })

})
all_data = pd.concat([train,test],axis = 0)



#Finding mssing values

total_missing = all_data.isnull().sum().sort_values(ascending = False)

percent_missing = total_missing/len(all_data)

missing_data = pd.concat((total_missing,percent_missing,),axis = 1, keys =

                    ['TotalMissing','PercentMissing'])



#Dropping data

remove = (missing_data[missing_data['PercentMissing']>0.005]).drop('SalePrice')

all_data = all_data.drop((remove).index, axis = 1)

all_data = all_data.drop(['Utilities'], axis = 1)



#Filling

fillers= (

'MSZoning','BsmtHalfBath','Electrical','Functional','BsmtFullBath','BsmtFinSF2',

'BsmtFinSF1','Exterior2nd','BsmtUnfSF','TotalBsmtSF','SaleType',

'Exterior1st','KitchenQual','GarageArea','GarageCars')

for key in fillers:

    all_data[key] = all_data[key].fillna(train[key].mode()[0])

total_missing = all_data.drop(['SalePrice'],axis = 1).isnull().sum().sort_values(ascending = False)

percent_missing = total_missing/len(all_data)





missing_data = pd.concat((total_missing,percent_missing,),axis = 1, keys =

                    ['TotalMissing','PercentMissing'])





print('Misisng values:  ',total_missing.sum())





numeric_feats = all_data.drop(['SalePrice','Id'],axis = 1).dtypes[all_data.dtypes != 'object'].index

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending = False)



skewness = pd.DataFrame({'Skew': skewed_feats})

skewness = skewness[abs(skewness)>0.75]



from scipy.special import boxcox1p



skewness_features = skewness.index

for feat in skewness_features:

    all_data[feat] = boxcox1p(all_data[feat],0.15)



    

y_train = all_data.SalePrice[:n_train]



all_data = pd.get_dummies(all_data.drop(['SalePrice','Id'],axis = 1))



y_train = [math.log(x) for x in y_train]

y_train = pd.Series(y_train)


scaler = RobustScaler().fit(all_data)

all_data = scaler.transform(all_data)







x_train = all_data[:n_train]

x_test = all_data[n_train:]







#Creating dummy traninig set with "known" y_test

x_train_dummy = x_train[250:]

x_test_dummy = x_train[:250]



y_train_dummy = y_train[250:]

y_test_dummy = y_train[:250]











print('Shapes of train and test:', x_train.shape, x_test.shape)
x_train.shape
import lightgbm as lgb

import xgboost as xgb



from sklearn.svm import SVR



from sklearn.linear_model import LassoLars

from sklearn.linear_model import Lasso, Ridge, BayesianRidge

from sklearn.ensemble import BaggingRegressor,RandomForestRegressor, ExtraTreesRegressor, VotingRegressor



def xgb_params():

    return {

    'eval_metric':'rmse',

    'objective': 'reg:squarederror',

    'eta': 0.01,

    'max_depth': 6,

    'colsample_bytree': 0.6,

    'colsample_bylevel':0.6,

    'colsample_bynode':0.6,

    'verbosity':0

        

    }

    

    

def lgb_params():

    return { 'metric': 'rmse',

    'objective': 'regression',

    'eta': 0.01,

    'max_depth': 6,

    'feature_fraction':0.3,

    'bagging_fraction':0.6,

    'bagging_freq': 2,

    'min_data_in_leaf': 2,

    'verbosity':0

    }



def runLGB_cv(params,dataset,rounds):

    hist = lgb.cv(params,dataset, num_boost_round = rounds)

    return hist

def runXBG_cv(params,dataset,rounds):

    hist = xgb.cv(params,dataset, num_boost_round = rounds)

    return hist



def makeLGB(params,dataset,rounds):

    return lgb.train(params,dataset,num_boost_round = rounds)

def makeXGB(params,dataset,rounds):

    return xgb.train(params,dataset,num_boost_round = rounds)



def stackLGB(params,dataset,rounds,depths):

    boosters = []

    for depth in depths:

        params['max_depth']=depth

        booster = lgb.train(params, dataset, num_boost_round = rounds)

        boosters.append(booster)

    return boosters



def stackXGB(params,dataset,rounds,leaves):

    boosters = []

    for depth in depths:

        params['max_depth']=depth

        booster = xgb.train(params, dataset, num_boost_round = rounds)

        boosters.append(booster)

    return boosters

    

base_lgb_params = lgb_params()

base_xgb_params = xgb_params()



lgb_train = lgb.Dataset(x_train, y_train)

lgb_train_dummy = lgb.Dataset(x_train_dummy, y_train_dummy)



xgb_train = xgb.DMatrix(x_train, y_train)

xgb_train_dummy = xgb.DMatrix(x_train_dummy, y_train_dummy)



xgb_test = xgb.DMatrix(x_test)

xgb_test_dummy = xgb.DMatrix(x_test_dummy)

all_scores = pd.DataFrame()






"""

    

boosters = stackXGB(base_xgb_params,xgb_train_dummy,1000,depths)

for i,booster in enumerate(boosters):

    all_scores['XGB - depth: '+str(depths[i])] = booster.predict(xgb_test_dummy)



boosters = stackLGB(base_lgb_params,lgb_train_dummy,1000,depths)

for i,booster in enumerate(boosters):

    all_scores['LGBM - depth: '+str(depths[i])] = booster.predict(x_test_dummy)



"""



    

                                     

                                     

    

    
models = [

    Ridge(), BayesianRidge(),

    BaggingRegressor(n_estimators = 1000, max_samples = 0.6, max_features = 0.6),

    RandomForestRegressor(n_estimators = 1000),

    ExtraTreesRegressor(n_estimators = 1000)

             ]



labels = ['Ridge','BayesianRidge','BaggingRegressor','RandomForest','ExtraRandomForest']

  





    

Cs = [0.05,0.1,0.3,0.7]

for c in Cs:

    labels.append('SVM - C: '+str(c))

    models.append(SVR(kernel = 'linear',gamma = 'scale',epsilon = 0.01, C = c))



depths = [3,4,5,6]

for depth in depths:

    models.append(xgb.XGBRegressor(max_depth = depth,

                                     learning_rate = 0.01,

                                     n_estimators = 1000,

                                     verbosity = 0,

                                     booster = 'gbtree',

                                     colsample_bytree = 0.6,

                                     colsample_bylevel = 0.6,

                                     colsample_bynode = 0.6,

                                     base_score = 12

                                    )

                   )

    labels.append('XGB - depth: ' + str(depth))

    models.append(lgb.LGBMRegressor(num_leaves = 64,

                                    max_depth = depth,

                                    n_estimators = 1000,

                                    min_child_samples = 3,

                                    learning_rate = 0.01,

                                    subsample = 0.5,

                                    subsample_freq = 1,

                                    colsample_bytree = 0.5                          

                            

                                

                                   ))

    labels.append('LGB - depth: ' + str(depth))





for model,label in zip(models,labels):

    model.fit(x_train_dummy,y_train_dummy)

    all_scores[label]=model.predict(x_test_dummy)



"""



voting_models = list(zip(labels,models))

voter = VotingRegressor(voting_models).fit(x_train_dummy,y_train_dummy)

all_scores['voter'] = voter.predict(x_test_dummy)

"""
all_scores['mean'] = all_scores.mean(axis = 1)

all_scores['median'] = all_scores.drop(['mean'],axis = 1).median(axis = 1)

all_scores['true_price'] =y_test_dummy.values
for col in all_scores.columns:

    all_scores[col] = [math.exp(x) for x in all_scores[col]]

        
all_scores[:10]
#First plot - just random values

start = 0

stop = 60



traces = []

plot_scores = all_scores[start:stop]



for label in plot_scores.drop('true_price',axis=1).columns:

    traces.append(go.Scatter(x = all_scores[label][start:stop].index,

                            y = plot_scores[label], mode = 'markers', name = label,opacity = 0.7))

traces.append(go.Scatter(x = all_scores.true_price[start:stop].index,

                        y = plot_scores.true_price, mode = 'lines', name = 'True Price',opacity = 1))



iplot(

    {'data': traces,

    'layout': go.Layout({'title':'Model performances - interactive plot - select models in the list',

                        'xaxis':{'title': '"Randomly" Selected Unseen Test Indices'},

                        'yaxis':{'title': 'Price'}})

    }

)

#Second plot - Sorted by sale price

start = 10

stop = 105



traces = []

plot_scores = all_scores[start:stop].sort_values(by='true_price')



for label in plot_scores.drop('true_price',axis=1).columns:

    traces.append(go.Scatter(x = all_scores[label][start:stop].index,

                            y = plot_scores[label], mode = 'markers', name = label,opacity = 0.7))

traces.append(go.Scatter(x = all_scores.true_price[start:stop].index,

                        y = plot_scores.true_price, mode = 'lines', name = 'True Price',opacity = 1))



iplot(

    {'data': traces,

    'layout': go.Layout({'title':'Model performances - interactive plot - select models in the list',

                        'xaxis':{'title': '"Randomly" Selected Unseen Test Indices - sorted by sale price'},

                        'yaxis':{'title': 'Price'}})

    }

)




y_true = all_scores.true_price





exp_var = []

msle = []

for col in all_scores.drop('true_price',axis = 1).columns:

    exp_var.append(explained_variance_score(y_true,all_scores[col]))

    msle.append(mean_squared_log_error(y_true,all_scores[col]))



df_accuracies = pd.DataFrame({'Explained variance': exp_var,'Mean Squared Log Error': msle}, index = all_scores.drop('true_price',1).columns)



df_accuracies.sort_values('Mean Squared Log Error', ascending = True)
final_scores = pd.DataFrame()







for model,label in zip(models,labels):

    model.fit(x_train,y_train)

    final_scores[label]=model.predict(x_test)











final_scores['mean'] = final_scores.mean(axis = 1)

final_scores['median'] = final_scores.median(axis = 1)

for col in final_scores.columns:

    final_scores[col] = [math.exp(x) for x in final_scores[col]]
my_submission1 = pd.DataFrame({'Id': test.Id, 'SalePrice': final_scores['SVM - C: 0.3']})

my_submission1.to_csv('submission1.csv', index=False)



my_submission2 = pd.DataFrame({'Id': test.Id, 'SalePrice': final_scores['mean']})

my_submission2.to_csv('submission2.csv', index=False)



my_submission3 = pd.DataFrame({'Id': test.Id, 'SalePrice': final_scores['Ridge']})

my_submission3.to_csv('submission3.csv', index=False)



my_submission4 = pd.DataFrame({'Id': test.Id, 'SalePrice': final_scores['ExtraRandomForest']})

my_submission4.to_csv('submission4.csv', index=False)



my_submission5 = pd.DataFrame({'Id': test.Id, 'SalePrice': final_scores[['LGB - depth: 3','LGB - depth: 4','LGB - depth: 5','LGB - depth: 6']].mean(axis=1)})

my_submission5.to_csv('submission5.csv', index=False)