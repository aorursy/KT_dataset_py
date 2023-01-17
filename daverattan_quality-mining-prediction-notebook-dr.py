#Dave Rattan: Notebook
import numpy as np
import pandas as pd
# pd.options.plotting.backend = 'plotly'
import plotly.offline
import cufflinks as cf
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# cf.go_offline()
# cf.set_config_file(offline = False, world_readable = True)
# plotly.offline.init_notebook_mode (connected = True)

from datetime import datetime
import os
dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

df = pd.read_csv("/kaggle/input/quality-prediction-in-a-mining-process/MiningProcess_Flotation_Plant_Database.csv",decimal = ',',\
                 parse_dates = ['date'],date_parser = dateparse)
df.set_index('date', inplace = True)
df = df.rename_axis(None)
df.info()
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(df.corr(), annot = True, ax = ax)
from statsmodels.tsa.stattools import adfuller
silica_concentrate = df['% Silica Concentrate']
silica_concentrate = silica_concentrate["2017-04-01 00:00:00":"2017-09-09 23:00:00"]
#output = adfuller(silica_concentrate)
#print('t-Stat: ', output[0])
#print('p-value: ',output[1])
# print('Critical Values: ')
# for key, value in output[4].items():
#     print('\t%s: %.3f' %(key, value))
from statsmodels.stats.outliers_influence import variance_inflation_factor
def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)
calc_vif(df.drop(['% Iron Concentrate'], axis = 1))
df_range = df["2017-04-01 00:00:00":"2017-09-09 23:00:00"]
flotations_col_air_one_three = pd.concat([df_range['Flotation Column 01 Air Flow'], df_range['Flotation Column 02 Air Flow'], df_range['Flotation Column 03 Air Flow']], axis = 1)
flotations_col_air_six_seven = pd.concat([df_range['Flotation Column 06 Air Flow'], df_range['Flotation Column 07 Air Flow']], axis  = 1)

flotations_col_lvl_one_three = pd.concat([df_range['Flotation Column 01 Level'], df_range['Flotation Column 02 Level'], df_range['Flotation Column 03 Level']], axis = 1)
flotations_col_lvl_four_seven = pd.concat([df_range['Flotation Column 04 Level'], df_range['Flotation Column 05 Level'], df_range['Flotation Column 06 Level'], df_range['Flotation Column 07 Level']], axis = 1)
fig, ax = plt.subplots(figsize=(20,10))
plt.plot(flotations_col_air_one_three, label = 'Flotations_air_flow_1-3')
plt.title('Flotations Air Flow Columns 1-3')
plt.legend(loc = "best")
fig, ax = plt.subplots(figsize=(20,10))
plt.plot(flotations_col_air_six_seven, label = 'Flotations_air_flow_6-7')
plt.title('Flotations Air Flow Columns 6-7')
plt.legend(loc = "best")
fig, ax = plt.subplots(figsize=(20,10))
plt.plot(flotations_col_lvl_one_three, label = 'Flotations_level_1-3')
plt.title('Flotations Level Columns 1-3')
plt.legend(loc = "best")
fig, ax = plt.subplots(figsize=(20,10))
plt.plot(flotations_col_lvl_four_seven, label = 'Flotations_level_4-7')
plt.title('Flotations Level Columns 4-7')
plt.legend(loc = "best")
flotations_col_air_one_three_ewm = flotations_col_air_one_three.ewm(span = 60480, adjust = False).mean()
flotations_col_air_six_seven_ewm = flotations_col_air_six_seven.ewm(span = 60480, adjust = False).mean()

flotations_col_lvl_one_three_ewm = flotations_col_lvl_one_three.ewm(span = 60480, adjust = False).mean()
flotations_col_lvl_four_seven_ewm = flotations_col_lvl_four_seven.ewm(span = 60480, adjust = False).mean()
fig, ax = plt.subplots(figsize=(20,10))
plt.plot(flotations_col_air_one_three_ewm, label = 'Flotations_air_flow_1-3_ema')
plt.title('Flotations Air Flow Columns 1-3 with EMA')
plt.legend(loc = "best")
fig, ax = plt.subplots(figsize=(20,10))
plt.plot(flotations_col_air_six_seven_ewm, label = 'Flotations_air_flow_6-7_ema')
plt.title('Flotations Air Flow Columns 6-7 with EMA')
plt.legend(loc = "best")
fig, ax = plt.subplots(figsize=(20,10))
plt.plot(flotations_col_lvl_one_three_ewm, label = 'Flotations_level_1-3_ema')
plt.title('Flotations Level Columns 1-3 EMA')
plt.legend(loc = "best")
fig, ax = plt.subplots(figsize=(20,10))
plt.plot(flotations_col_lvl_four_seven_ewm, label = 'Flotations_level_4-7_ema')
plt.title('Flotations Level Columns 4-7 EMA')
plt.legend(loc = "best")
df_range = df_range.ewm(span = 60480).mean()
flotations_col_air_one_three_ewm['Mean'] = flotations_col_air_one_three_ewm.mean(axis = 1)
flotations_col_air_six_seven_ewm['Mean'] = flotations_col_air_six_seven_ewm.mean(axis = 1)

flotations_col_lvl_one_three_ewm['Mean'] = flotations_col_lvl_one_three_ewm.mean(axis = 1)
flotations_col_lvl_four_seven_ewm['Mean'] = flotations_col_lvl_four_seven_ewm.mean(axis = 1)
df_range['Mean_flotations_col_air_one_three_ewm'] = flotations_col_air_one_three_ewm['Mean']
df_range['Mean_flotations_col_air_six_seven_ewm'] = flotations_col_air_six_seven_ewm['Mean']

df_range['Mean_flotations_col_lvl_one_three_ewm'] = flotations_col_lvl_one_three_ewm['Mean']
df_range['Mean_flotations_col_lvl_four_seven_ewm'] = flotations_col_lvl_four_seven_ewm['Mean']

df_range
df_range_test_1 = df_range.drop(['% Iron Concentrate', 'Flotation Column 01 Air Flow', 'Flotation Column 02 Air Flow', 'Flotation Column 03 Air Flow', 'Flotation Column 06 Air Flow', 'Flotation Column 07 Air Flow', 'Flotation Column 01 Level', 'Flotation Column 02 Level', 'Flotation Column 03 Level', 'Flotation Column 04 Level', 'Flotation Column 05 Level', 'Flotation Column 06 Level', 'Flotation Column 07 Level'], axis = 1)
calc_vif(df_range_test_1)
df_range_test_2 = df_range_test_1.drop(['Flotation Column 04 Air Flow', 'Flotation Column 05 Air Flow'], axis = 1)
calc_vif(df_range_test_2)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(df_range_test_1.corr(), annot = True, ax = ax)
df_range_test_3 = df_range_test_2.drop(['Mean_flotations_col_air_one_three_ewm', 'Mean_flotations_col_lvl_one_three_ewm'], axis = 1)
calc_vif(df_range_test_3)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(df_range_test_3.corr(), annot = True, ax = ax)
df_range_test_4 = df_range_test_3.drop(['% Iron Feed', '% Silica Feed','Ore Pulp pH', 'Starch Flow'], axis = 1)
calc_vif(df_range_test_4)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(df_range_test_4.corr(), annot = True, ax = ax)
df_range_test_5 = df_range_test_1.drop(['% Iron Feed', '% Silica Feed','Ore Pulp pH', 'Starch Flow', 'Flotation Column 04 Air Flow', 'Flotation Column 05 Air Flow'], axis = 1)
# fig, ax = plt.subplots(figsize=(10,10))
# sns.heatmap(df_range_test_5.corr(), annot = True, ax = ax)
#standardised train-test split of 50% 
train_size  = int(len(df_range_test_1)*0.55)
valid_size = int(len(df_range_test_1)*0.70)
train, valid, test = df_range_test_1[0:train_size], df_range_test_1[train_size:valid_size],df_range_test_1[valid_size: len(df_range_test_1)]
fin_train = df_range_test_1[0:valid_size]
fig, ax = plt.subplots(figsize=(20,10))
plt.title('% Silica Concentrate Train(55%)-Valid(15%)-Test(30%) split')
plt.plot(train['% Silica Concentrate'], label = 'Train')
plt.plot(valid["% Silica Concentrate"], label = 'Valid')
plt.plot(test['% Silica Concentrate'], label = 'Test')
plt.legend(loc = "best")
plt.show()
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_squared_error

Xtrain = train.drop(['% Silica Concentrate'], axis = 1)
ytrain = train['% Silica Concentrate']

Xvalid = valid.drop(["% Silica Concentrate"], axis = 1)
yvalid = valid['% Silica Concentrate']

XFinTrain = fin_train.drop(["% Silica Concentrate"], axis = 1)
yFinTrain = fin_train["% Silica Concentrate"]

Xtest = test.drop(["% Silica Concentrate"], axis = 1)
ytest = test["% Silica Concentrate"]

model_list = [LinearRegression(), Ridge(), ElasticNet()]
def feature_importance(model, XTrain, yTrain, XTest, yTest):
    from sklearn.feature_selection import RFE
    from warnings import simplefilter
    
    simplefilter(action='ignore', category=FutureWarning)
    num_of_features = np.arange(1, len(XTrain.columns))
    #print(num_of_features)
    high_score = 1
    score = 0
    optimum_num_features = 0
    score_list = []
    
    for i in range(len(num_of_features)):
        rfe = RFE(model, num_of_features[i])
        XTrain_rfe = rfe.fit_transform(XTrain, yTrain)
        XTest_rfe = rfe.transform(XTest)
        model.fit(XTrain_rfe, yTrain)
        pred = model.predict(XTest_rfe)
        score = mean_squared_error(yTest, pred)
        print("MAE during training: ", score)
        score_list.append(score)
        
        if (score < high_score):
            high_score = score
            optimum_num_features = num_of_features[i]
            data_zip = list(zip(rfe.support_, rfe.ranking_))
            temp = pd.Series(data_zip, index = Xtrain.columns)
            #print(temp)
    print("Optimum number of features %d" % optimum_num_features)
    print("Score with %d features: %f" %(optimum_num_features, high_score))
    print("Best features are: ", temp)
    
model_trained = []
for i in model_list:
    model = i
    print("Training model: ", i)
    model_trained.append(model.fit(Xtrain, ytrain))      
model_trained[0]
pred_linear_model = model_trained[0].predict(Xvalid)
mean_squared_error(yvalid, pred_linear_model)
fig, ax = plt.subplots(figsize=(20,10))
yvalid.reset_index(drop = True, inplace = True)
plt.title('LinearRegression: Forecasted % Silica vs Validation % Silica')
plt.plot(pred_linear_model, label = 'Forecasted')
plt.plot(yvalid, label = 'Ground Truth')
plt.legend(loc = "best")
plt.show()
model_trained[1]
pred_ridge_model = model_trained[1].predict(Xvalid)
mean_squared_error(yvalid, pred_ridge_model)
fig, ax = plt.subplots(figsize=(20,10))
yvalid.reset_index(drop = True, inplace = True)
plt.title('Ridge Regression: Forecasted % Silica vs Validation % Silica')
plt.plot(pred_ridge_model, label = 'Forecasted')
plt.plot(yvalid, label = 'Ground Truth')
plt.legend(loc = "best")
plt.show()
model_trained[2]
pred_elastic_model = model_trained[2].predict(Xvalid)
mean_squared_error(yvalid, pred_elastic_model)
fig, ax = plt.subplots(figsize=(20,10))
yvalid.reset_index(drop = True, inplace = True)
plt.title('ElasticNet: Forecasted % Silica vs Validation % Silica')
plt.plot(pred_elastic_model, label = 'Forecasted')
plt.plot(yvalid, label = 'Ground Truth')
plt.legend(loc = "best")
plt.show()
feature_importance(model_trained[0], Xtrain, ytrain, Xvalid, yvalid)
feature_importance(model_trained[1], Xtrain, ytrain, Xvalid, yvalid)
feature_importance(model_trained[2], Xtrain, ytrain, Xvalid, yvalid)
model_trained_final = []
for i in model_list:
    model = i
    print("Training model: ", i)
    model_trained_final.append(model.fit(XFinTrain, yFinTrain))  
pred_linear_model_final = model_trained_final[0].predict(Xtest)
mean_squared_error(ytest, pred_linear_model_final)
fig, ax = plt.subplots(figsize=(20,10))
ytest.reset_index(drop = True, inplace = True)
plt.title('LinearRegression: Forecasted % Silica vs Validation % Silica')
plt.plot(pred_linear_model_final, label = 'Forecasted')
plt.plot(ytest, label = 'Ground Truth')
plt.legend(loc = "best")
plt.show()
pred_ridge_model_final = model_trained_final[1].predict(Xtest)
mean_squared_error(ytest, pred_ridge_model_final)
fig, ax = plt.subplots(figsize=(20,10))
ytest.reset_index(drop = True, inplace = True)
plt.title('Ridge: Forecasted % Silica vs Validation % Silica')
plt.plot(pred_ridge_model_final, label = 'Forecasted')
plt.plot(ytest, label = 'Ground Truth')
plt.legend(loc = "best")
plt.show()
pred_elastic_model_final = model_trained_final[2].predict(Xtest)
mean_squared_error(ytest, pred_elastic_model_final)
fig, ax = plt.subplots(figsize=(20,10))
ytest.reset_index(drop = True, inplace = True)
plt.title('ElasticNet: Forecasted % Silica vs Validation % Silica')
plt.plot(pred_elastic_model_final, label = 'Forecasted')
plt.plot(ytest, label = 'Ground Truth')
plt.legend(loc = "best")
plt.show()
from sklearn import preprocessing
from statsmodels.tsa.stattools import pacf

import math
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping