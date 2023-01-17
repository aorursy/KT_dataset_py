# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import scipy
from sklearn.ensemble import AdaBoostRegressor
import tensorflow as tf
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import statsmodels.api as sm
import xgboost

print(os.listdir("../input"))
import warnings 
warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.
test_correlation = pd.read_csv('../input/train_correlation.csv',encoding='gbk')
test_fund_benchmark_return = pd.read_csv('../input/train_fund_benchmark_return.csv',encoding='gbk')
test_fund_return = pd.read_csv('../input/train_fund_return.csv',encoding='gbk')
index_return = pd.read_csv('../input/train_index_return.csv',encoding='gbk')
test_correlation.index = test_correlation['Unnamed: 0']
test_fund_benchmark_return.index = test_fund_benchmark_return['Unnamed: 0']
index_return.index = index_return['Unnamed: 0']
test_fund_return.index = test_fund_return['Unnamed: 0']
cor_data = test_correlation.T
fund_ben_data = test_fund_benchmark_return.T
index_data = index_return.T
test_fund_return = test_fund_return.T
cor_data = cor_data.drop(['Unnamed: 0'])
fund_ben_data = fund_ben_data.drop(['Unnamed: 0'])
index_data = index_data.drop(['Unnamed: 0'])
fund_ret_data = test_fund_return.drop(['Unnamed: 0'])

del test_fund_benchmark_return
del test_correlation
col_list = cor_data.columns
fund_list = list(fund_ben_data.columns)
fund_list[0] = 'fund_ID'
test_correlation_1 = pd.read_csv('../input/test_correlation.csv',encoding='gbk')
test_fund_benchmark_return_1 = pd.read_csv('../input/test_fund_benchmark_return.csv',encoding='gbk')
test_fund_return_1 = pd.read_csv('../input/test_fund_return.csv',encoding='gbk')
index_return_1 = pd.read_csv('../input/test_index_return.csv',encoding='gbk')
test_correlation_1.index = test_correlation_1['Unnamed: 0']
test_fund_benchmark_return_1.index = test_fund_benchmark_return_1['Unnamed: 0']
index_return_1.index = index_return_1['Unnamed: 0']
test_fund_return_1.index = test_fund_return_1['Unnamed: 0']
cor_data_1 = test_correlation_1.T
fund_ben_data_1 = test_fund_benchmark_return_1.T
index_data_1 = index_return_1.T
test_fund_return_1 = test_fund_return_1.T
cor_data_1 = cor_data_1.drop(['Unnamed: 0'])
fund_ben_data_1 = fund_ben_data_1.drop(['Unnamed: 0'])
index_data_1 = index_data_1.drop(['Unnamed: 0'])
fund_ret_data_1 = test_fund_return_1.drop(['Unnamed: 0'])

del test_fund_benchmark_return_1
del test_correlation_1
col_list_1 = cor_data_1.columns
fund_list_1 = list(fund_ben_data_1.columns)
fund_list_1[0] = 'fund_ID'
ada = AdaBoostRegressor(learning_rate=0.01,n_estimators=2000)
gbdt = GradientBoostingRegressor(learning_rate=0.01,n_estimators=2000)
rf = RandomForestRegressor(max_depth=2, random_state=0)
svr = SVR(C=1.0, epsilon=0.2)
# print(gbdt.feature_importances_)
# print(rf.feature_importances_)
# plt.subplot(3,1,1)
# plt.hist(Ry)
# plt.subplot(3,1,2)
# plt.hist(rprel)
# plt.subplot(3,1,3)
# plt.hist(gprel)
# mae = np.sum(np.abs(np.array(rprel) - np.array(Ry)))/len(rprel)
# tmape = np.sum(np.abs((np.array(rprel) - np.array(Ry))/(1.5-np.array(Ry))))/len(rprel)
# score = (2/(2+mae+tmape))*(2/(2+mae+tmape))
# print('rf -- mae:', mae, '     tmape:', tmape, '      score:', score)
# mae = np.sum(np.abs(np.array(gprel) - np.array(Ry)))/len(gprel)
# tmape = np.sum(np.abs((np.array(gprel) - np.array(Ry))/(1.5-np.array(Ry))))/len(gprel)
# score = (2/(2+mae+tmape))*(2/(2+mae+tmape))
# print('gbdt -- mae:', mae, '     tmape:', tmape, '      score:', score)
cor_data_2 = pd.concat([cor_data, cor_data_1])
fund_ben_data_2 = pd.concat([fund_ben_data, fund_ben_data_1])
fund_ret_data_2 = pd.concat([fund_ret_data, fund_ret_data_1])
index_data_2 = pd.concat([index_data, index_data_1])
col_list_2 = cor_data_2.columns
# cor_data_2.head()
index_data_2.tail()
# 时间序列分析
# X = []
# Ry = []
# rprel = []
# gprel = []
# for i in col_list[:1]:
#     X = []
#     y = []
#     gbdt = GradientBoostingRegressor()
#     rf = RandomForestRegressor(max_depth=2, random_state=0)
#     data = cor_data_2[[i]]
#     funds = i.split('-')
#     fund1 = funds[0].strip()
#     fund2 = funds[1].strip()
#     fund_data = fund_ret_data_2[[fund1]]
#     data = data.merge(fund_data, left_index=True, right_index=True, how='right')
#     fund_data = fund_ret_data_2[[fund2]]
#     data = data.merge(fund_data, left_index=True, right_index=True, how='left')
#     data = data.merge(index_data_2, left_index=True, right_index=True, how='left')
#     data = data.iloc[:,:].values
#     data[pd.isnull(np.array(data[:,0], dtype=float)),0] = 0
# #     print(data.shape)
#     temp_X = []
#     for k in range(0, data.shape[0]):
#         if k%5!=0: 
#             continue
#         temp_X=temp_X+list(data[k,1:])
#         if(k > 61):
#             for j in range(1, data.shape[1]):
#                 temp_X.pop(0)
# #             print(len(temp_X))
#             X.append(temp_X)
#             y.append(data[k,0])
#     rf.fit(X[:-1], y[:-1])
#     gbdt.fit(X[:-1], y[:-1])
#     rprel.append(rf.predict([X[-1]])[0])
#     gprel.append(gbdt.predict([X[-1]])[0])
#     Ry.append(y[-1])
#     if(len(Ry)%10 == 0):
#         print(len(Ry))
# print(len(X))
# for i in range(0,100):
#     print('%s,%s'%(col_list[i], str(gprel[i])))
# 计算相关系数，数据填充的做法

add_cor_data = cor_data.copy()
add_cor_data.head()
        
# 数据填充
X = []
Ry = []
rprel = []
gprel = []
for i in col_list[:10]:
    X = []
    y = []
    gbdt = GradientBoostingRegressor()
    rf = RandomForestRegressor(max_depth=2, random_state=0)
    data = add_cor_data[[i]]
    funds = i.split('-')
    fund1 = funds[0].strip()
    fund2 = funds[1].strip()
    fund_data_1 = fund_ret_data[[fund1]]
    fund_data_2 = fund_ret_data[[fund2]]
    a = list(fund_data_1.iloc[-60:,0])
    b = list(fund_data_2.iloc[-60:,0])
    for k in range(data.shape[0]-60, data.shape[0]-2):
        temp = scipy.stats.pearsonr(a,b)[0]
        data.iloc[k, 0] = temp
        a.pop(0)
        b.pop(0)
        add_cor_data[[i]] = data
#         print(temp, add_cor_data[[i]].iloc[k].values,cor_data[[i]].iloc[k].values)
#         print(cor_data[[i]].iloc[k])
    add_cor_data[[i]] = data
#     print(add_cor_data[[i]],cor_data[[i]])
    print(i)
# 数据填充2 
# for i in range(61, 2, -1):
#     temp_fund_ret_data = fund_ret_data.iloc[-i:,:]
#     temp_fund_ret_data = temp_fund_ret_data.astype('float64')
#     res_cor = temp_fund_ret_data.corr()
# #     print(add_cor_data.index[-i])
#     res_cor = res_cor*(0.95+0.05*i/61)
#     for row in res_cor.index[:1]:
#         for column in res_cor.columns[:20]:
#             if row == column:
#                 continue
#             add_cor_data.loc[add_cor_data.index[-i], '%s-%s'%(row,column)] = res_cor.loc[row, column]
# #             print(res_cor.loc[row, column], add_cor_data.loc[add_cor_data.index[-i], '%s-%s'%(row,column)])
#     print(i)
# 基金和指数的关系
# result = []
# col = list(fund_ret_data.columns)
# for i in range(0, 200):
#     temp_fund = fund_ret_data.iloc[:, i:i+1]
#     temp_index = index_data.iloc[:, :]
#     temp_fund = temp_fund.merge(temp_index, left_index=True, right_index=True, how='left')
# #     print(temp_fund)
#     temp_fund = temp_fund.astype('float64')
#     temp_corr = temp_fund.corr()
# #     temp_corr = temp_corr*(0.95+0.05*(min(i+61,300)-i)/61)
# #     print(temp_corr)
# #     print(temp_corr.loc['Fund 1'])
#     result.append(list(temp_corr[col[i]])[1:])
# fund_index_corr_total = np.array(result)
# fund_index_corr_total = pd.DataFrame(fund_index_corr_total, columns = list(index_data.columns), index = col)
# plt.subplot(3,1,1)
# plt.hist(list(cor_data.iloc[-60,0:20]))
# plt.subplot(3,1,2)
# plt.hist(list(add_cor_data.iloc[-60,0:20]))
# plt.subplot(3,1,3)
# plt.hist(cor_data.iloc[-50,:])
# for i in col:
#     print(np.argmax(fund_index_corr_total.loc[i]))
# 时间序列分析
# X = []
# Ry = []
# rprel = []
# gprel = []
# xprel = []
# cor_series_pred = []
# for i in col_list[:3]:
#     cor_se = []
#     X = []
#     y = []
#     gbdt = GradientBoostingRegressor(loss='huber', max_depth=2, min_samples_split=2)
#     rf = RandomForestRegressor(max_depth=2, random_state=0)
#     xgb = xgboost.XGBRegressor()
#     data = cor_data[[i]]
#     funds = i.split('-')
#     fund1 = funds[0].strip()
#     fund2 = funds[1].strip()
#     fund_data = fund_ret_data[[fund1]]
#     data = data.merge(fund_data, left_index=True, right_index=True, how='left')
#     fund_data = fund_ret_data[[fund2]]
#     data = data.merge(fund_data, left_index=True, right_index=True, how='left')
#     data = data.merge(index_data, left_index=True, right_index=True, how='left')
    
#     #fund_index_data_1 = list(fund_index_corr_total.loc[fund1])
#     #fund_index_data_2 = list(fund_index_corr_total.loc[fund2])
    
#     data = data.iloc[:,:].values
#     data[pd.isnull(np.array(data[:,0], dtype=float)),0] = 0
# #     print(data.shape)
# #     print(data)
#     temp_X = []
#     for k in range(0, data.shape[0]-60):
# #         if k%5!=0: 
# #             continue
#         if(k > 5):
#             temp_X=temp_X+list(data[k,1:])
# #             print(len(temp_X))
#             temp = temp_X.copy()
#             #temp += fund_index_data_1
#             #temp += fund_index_data_2
#             X.append(temp)
#             y.append(data[k,0])
#             for j in range(1, data.shape[1]):
#                 temp_X.pop()
#             temp_X.pop(0)
# #             print(len(temp_X))
#         temp_X.append(data[k,0])
#     rf.fit(X[:], y[:])
#     gbdt.fit(X[:], y[:])
#     xgb.fit(X[:],y[:])
#     cor_se += list(gbdt.predict(X[:]))
# #     随机森林
#     temp_pre = 0.0
#     temp_rf_X = temp_X.copy()
#     for k in range(data.shape[0]-60, data.shape[0]-1):
#         temp_rf_X=temp_rf_X+list(data[k,1:])
#         temp = temp_rf_X.copy()
#         #temp += fund_index_data_1
#         #temp += fund_index_data_2
# #         print(temp)
#         temp_pre = rf.predict([temp])[0]
#         for j in range(1, data.shape[1]):
#             temp_rf_X.pop()
#         temp_rf_X.pop(0)
#         temp_rf_X.append(temp_pre)
# #         cor_se.append(temp_pre)
#     rprel.append(temp_pre)
# #      gbdt
#     temp_pre = 0.0
#     for k in range(data.shape[0]-60, data.shape[0]-1):
#         temp_X=temp_X+list(data[k,1:])
#         temp = temp_X.copy()
#         #temp += fund_index_data_1
#         #temp += fund_index_data_2
#         temp_pre = gbdt.predict([temp])[0]
#         for j in range(1, data.shape[1]):
#             temp_X.pop()
#         temp_X.pop(0)
#         temp_X.append(temp_pre)
#     gprel.append(temp_pre)
#     # xgb
#     temp_pre = 0.0
#     temp_xgb_X = temp_X.copy()
#     for k in range(data.shape[0]-60, data.shape[0]-1):
#         temp_xgb_X=temp_xgb_X+list(data[k,1:])
#         temp = temp_xgb_X.copy()
#         #temp += fund_index_data_1
#         #temp += fund_index_data_2
# #         print(temp)
#         temp_pre = xgb.predict([temp])[0]
#         for j in range(1, data.shape[1]):
#             temp_xgb_X.pop()
#         temp_xgb_X.pop(0)
#         temp_xgb_X.append(temp_pre)
#         cor_se.append(temp_pre)
#     xprel.append(temp_pre)
# #     不添加相关系数的时候的做法
# #     rprel.append(rf.predict([X[-1]])[0])
# #     gprel.append(gbdt.predict([X[-1]])[0])
#     Ry.append(data[-1,0])
#     cor_series_pred.append(cor_se)
#     if(len(Ry)%10 == 0):
#         print(len(Ry))
# print(len(X))
# print(cor_data.index)
# 时间序列分析 new
X = []
Ry = []
rprel = []
gprel = []
xprel = []
tprel = []
cor_series_pred = []
for i in col_list[:10]:
    cor_se = []
    X = []
    y = []
    gbdt = GradientBoostingRegressor(loss='huber', max_depth=2, min_samples_split=2)
    rf = RandomForestRegressor(max_depth=2, random_state=0)
    xgb = xgboost.XGBRegressor(max_depth=3, learning_rate=0.01, n_estimators=1000)
    data = add_cor_data[[i]]
    data = data.iloc[:,:].values
    data[pd.isnull(np.array(data[:,0], dtype=float)),0] = 0
#     temp_X_series = []
#     for k in range(0, data.shape[0]-40):
#         if(k > 300 and k < data.shape[0]-60):
#             temp = []
#             for x in range(k-300, k-1, 2):
#                 temp.append(temp_X_series[x])
#             X.append(temp)
#             y.append(data[k,0])
#         temp_X_series.append(data[k,0])
#         cor_se.append(data[k, 0])
# #     print(X[0])
#     rf.fit(X[:], y[:])
#     gbdt.fit(X[:], y[:])
#     xgb.fit(X[:], y[:])
# #     cor_se += list(gbdt.predict(X[:]))
# #     随机森林
#     temp_rf_X = temp_X_series.copy()
#     temp_pre = 0.0
#     for k in range(data.shape[0]-60, data.shape[0]-1):
#         temp_pre = 0.0
#         temp = []
#         for x in range(k-300, k-1, 2):
#             temp.append(temp_rf_X[x])
#         temp_pre = rf.predict([temp])[0]
#         temp_rf_X.append(temp_pre)
# #         cor_se.append(temp_pre)
#     rprel.append(temp_pre)
#     # xgb
#     temp_pre = 0.0
#     temp_xgb_X = temp_X_series.copy()
#     for k in range(data.shape[0]-60, data.shape[0]-1):
#         temp_pre = 0.0
#         temp = []
#         for x in range(k-300, k-1, 2):
#             temp.append(temp_xgb_X[x])
#         temp_pre = rf.predict([temp])[0]
#         temp_xgb_X.append(temp_pre)
#         cor_se.append(temp_pre)
#     xprel.append(temp_pre)
# #      gbdt
#     for k in range(data.shape[0]-60, data.shape[0]-1):
#         temp_pre = 0.0
#         temp = []
#         for x in range(k-300, k-1, 2):
#             temp.append(temp_X_series[x])
#         temp_pre = gbdt.predict([temp])[0]
#         temp_X_series.append(temp_pre)
# #         cor_se.append(temp_pre)
#     gprel.append(temp_pre)
    Ry.append(data[-1,0])
    tprel.append(data[-30, 0])
    cor_series_pred.append(cor_se)
    if(len(Ry)%100 == 0):
        print(len(Ry))
print(len(X))
mae = np.sum(np.abs(np.array(tprel) - np.array(Ry)))/len(tprel)
tmape = np.sum(np.abs((np.array(tprel) - np.array(Ry))/(1.5-np.array(Ry))))/len(tprel)
score = (2/(2+mae+tmape))*(2/(2+mae+tmape))
print('gess -- mae:', mae, '     tmape:', tmape, '      score:', score)
# print(gbdt.feature_importances_)
# print(rf.feature_importances_)
# print(xgb.feature_importances_)
# print(cor_series_pred)
# plt.plot([cor_data.iloc[i, 0] for i in range(100, 201, 1)])
# plt.plot(list(cor_data.iloc[300:, 0]))
# plt.plot(list(add_cor_data.iloc[300:400, 4]))
# plt.plot(cor_series_pred[0][300:])
# plt.subplot(3,1,1)
# plt.hist(Ry)
# plt.subplot(3,1,2)
# plt.hist(rprel)
# plt.subplot(3,1,3)
# plt.hist(gprel)
# mae = np.sum(np.abs(np.array(rprel) - np.array(Ry)))/len(rprel)
# tmape = np.sum(np.abs((np.array(rprel) - np.array(Ry))/(1.5-np.array(Ry))))/len(rprel)
# score = (2/(2+mae+tmape))*(2/(2+mae+tmape))
# print('rf -- mae:', mae, '     tmape:', tmape, '      score:', score)
# mae = np.sum(np.abs(np.array(gprel) - np.array(Ry)))/len(gprel)
# tmape = np.sum(np.abs((np.array(gprel) - np.array(Ry))/(1.5-np.array(Ry))))/len(gprel)
# score = (2/(2+mae+tmape))*(2/(2+mae+tmape))
# print('gbdt -- mae:', mae, '     tmape:', tmape, '      score:', score)
# mae = np.sum(np.abs(np.array(xprel) - np.array(Ry)))/len(xprel)
# tmape = np.sum(np.abs((np.array(xprel) - np.array(Ry))/(1.5-np.array(Ry))))/len(xprel)
# score = (2/(2+mae+tmape))*(2/(2+mae+tmape))
# print('xgb -- mae:', mae, '     tmape:', tmape, '      score:', score)
mae = np.sum(np.abs(np.array(tprel) - np.array(Ry)))/len(tprel)
tmape = np.sum(np.abs((np.array(tprel) - np.array(Ry))/(1.5-np.array(Ry))))/len(tprel)
score = (2/(2+mae+tmape))*(2/(2+mae+tmape))
print('gess -- mae:', mae, '     tmape:', tmape, '      score:', score)
# 计算相关系数，数据填充的做法，测试部分

add_cor_data_2 = cor_data_2.copy()
add_cor_data_2.head()
add_cor_data_2.index
# 数据填充2 
for i in range(40, 41, 1):
    add_cor_data_2[fund_ret_data_2.index[-i]] = 0
    temp_fund_ret_data = fund_ret_data_2.iloc[-i:,:]
    temp_fund_ret_data = temp_fund_ret_data.astype('float64')
    res_cor = temp_fund_ret_data.corr()
#     res_cor = res_cor*(0.95+0.05*i/61)
#     print(res_cor.columns)
    for row in res_cor.index[:]:
        for column in res_cor.columns[:]:
            if row == column:
                continue
            add_cor_data_2.loc[fund_ret_data_2.index[-i], '%s-%s'%(row,column)] = res_cor.loc[row, column]
#             print(fund_ret_data_2.index[-i], '%s-%s'%(row,column),res_cor.loc[row, column], add_cor_data_2.loc[fund_ret_data_2.index[-i], '%s-%s'%(row,column)])
#     print(i)
# 基金和指数的关系

# add_cor_data_2.tail(60)
# cor_data_2.tail(60)
# 时间序列分析 old
# X = []
# Ry = []
# rprel = []
# gprel = []
# for i in col_list[:]:
#     X = []
#     y = []
#     gbdt = GradientBoostingRegressor(loss='huber', max_depth=2, min_samples_split=2)
#     rf = RandomForestRegressor(max_depth=2, random_state=0)
#     data = add_cor_data_2[[i]]
#     funds = i.split('-')
#     fund1 = funds[0].strip()
#     fund2 = funds[1].strip()
#     fund_data = fund_ret_data_2[[fund1]]
#     data = data.merge(fund_data, left_index=True, right_index=True, how='right')
#     fund_data = fund_ret_data_2[[fund2]]
#     data = data.merge(fund_data, left_index=True, right_index=True, how='left')
#     data = data.merge(index_data_2, left_index=True, right_index=True, how='left')
#     data = data.iloc[:,:].values
#     data[pd.isnull(np.array(data[:,0], dtype=float)),0] = 0
# #     print(data.shape)
# #     print(data)
#     temp_X = []
#     for k in range(0, data.shape[0]-30):
# #         if k%5!=0: 
# #             continue
#         if(k > 61):
#             temp_X=temp_X+list(data[k,1:])
# #             print(len(temp_X))
#             X.append(temp_X.copy())
#             y.append(data[k,0])
#             for j in range(1, data.shape[1]):
#                 temp_X.pop()
#             temp_X.pop(0)
# #             print(len(temp_X))
#         temp_X.append(data[k,0])
#     rf.fit(X[:], y[:])
#     gbdt.fit(X[:], y[:])
# #     随机森林
#     temp_pre = 0.0
#     temp_rf_X = temp_X.copy()
#     for k in range(data.shape[0]-30, data.shape[0]-1):
# #         print(temp_pre)
#         temp_rf_X=temp_rf_X+list(data[k,1:])
# #         print(temp_rf_X)
# #         temp_pre = (rf.predict([temp_rf_X])[0]+data[k, 0])/2
#         temp_pre = rf.predict([temp_rf_X])[0]
# #         temp_pre = rf.predict([temp_rf_X])[0]*(1-0.05*(k-data.shape[0]+60)/60) + data[k, 0]*0.05*(k-data.shape[0]+60)/60
#         for j in range(1, data.shape[1]):
#             temp_rf_X.pop()
#         temp_rf_X.pop(0)
#         temp_rf_X.append(temp_pre)
#     rprel.append(temp_pre)
# #      gbdt
#     temp_pre = 0.0
#     for k in range(data.shape[0]-30, data.shape[0]-1):
#         temp_X=temp_X+list(data[k,1:])
# #         temp_pre = (gbdt.predict([temp_X])[0]+data[k, 0])/2
#         temp_pre = gbdt.predict([temp_X])[0]
# #         temp_pre = rf.predict([temp_X])[0]*(1-0.05*(k-data.shape[0]+60)/60) + data[k, 0]*0.05*(k-data.shape[0]+60)/60
#         for j in range(1, data.shape[1]):
#             temp_X.pop()
#         temp_X.pop(0)
#         temp_X.append(temp_pre)
#     gprel.append(temp_pre)
# #     不添加相关系数的时候的做法
# #     rprel.append(rf.predict([X[-1]])[0])
# #     gprel.append(gbdt.predict([X[-1]])[0])
#     Ry.append(data[-1,0])
#     if(len(Ry)%10 == 0):
#         print(len(Ry))
# print(len(X))
# X = []
# Ry = []
# rprel = []
# gprel = []
# cor_series_pred = []
# for i in col_list[:]:
#     cor_se = []
#     X = []
#     y = []
#     gbdt = GradientBoostingRegressor(loss='huber', max_depth=2, min_samples_split=2)
#     rf = RandomForestRegressor(max_depth=2, random_state=0)
#     data = cor_data_2[[i]]
#     data = data.iloc[:,:].values
#     data[pd.isnull(np.array(data[:,0], dtype=float)),0] = 0
#     temp_X_series = []
#     for d in data:
#         temp_X_series.append(d[0])
#     print(data.shape[0])
#     for k in range(data.shape[0]):
#         if(k > 100 and k < data.shape[0]):
#             temp = []
#             for x in range(k-100, k-1, 2):
#                 temp.append(temp_X_series[x])
#             X.append(temp)
#             y.append(data[k,0])
#         cor_se.append(data[k, 0])
#     rf.fit(X[:], y[:])
#     gbdt.fit(X[:], y[:])
# #     随机森林
#     temp_rf_X = temp_X_series.copy()
#     temp_pre = 0.0
#     for k in range(data.shape[0], data.shape[0]+61):
#         temp_pre = 0.0
#         temp = []
#         for x in range(k-100, k-1, 2):
#             temp.append(temp_rf_X[x])
#         temp_pre = rf.predict([temp])[0]
#         temp_rf_X.append(temp_pre)
#     rprel.append(temp_pre)
# #      gbdt
#     for k in range(data.shape[0], data.shape[0]+61):
#         temp_pre = 0.0
#         temp = []
#         for x in range(k-100, k-1, 2):
#             temp.append(temp_X_series[x])
#         temp_pre = gbdt.predict([temp])[0]
#         temp_X_series.append(temp_pre)
#         cor_se.append(temp_pre)
#     gprel.append(temp_pre)
#     cor_series_pred.append(cor_se)
# print(len(X))
X = []
Ry = []
tprel = []
for i in col_list[:]:
    cor_se = []
    X = []
    y = []
    gbdt = GradientBoostingRegressor(loss='huber', max_depth=2, min_samples_split=2)
    rf = RandomForestRegressor(max_depth=2, random_state=0)
    data = add_cor_data_2[[i]]
    data = data.iloc[:,:].values
    data[pd.isnull(np.array(data[:,0], dtype=float)),0] = 0
    tprel.append(data[-1, 0])
print(len(X))
# plt.plot(list(cor_data_2.iloc[300:, 8]))
# # plt.plot(list(add_cor_data.iloc[300:400, 4]))
# plt.plot(cor_series_pred[8][300:])
plt.subplot(2,2,1)
plt.hist(tprel)
# plt.subplot(2,2,2)
# plt.hist(rprel)
result = pd.DataFrame(columns=['ID', '2018-03-19'])
result['ID'] = col_list[:]
result['2018-03-19'] = tprel
result.to_csv('result_gess.csv', header=True,index=False)
# result = pd.DataFrame(columns=['ID', '2018-03-19'])
# result['ID'] = col_list[:]
# result['2018-03-19'] = rprel
# result.to_csv('result_rf.csv', header=True,index=False)
# result = pd.DataFrame(columns=['ID', '2018-03-19'])
# result['ID'] = col_list[:]
# result['2018-03-19'] = gprel
# result.to_csv('result_gbdt.csv', header=True,index=False)
# 基金基准的相关性

# print(len(index_data))
# index_corr_list = []
# for i in range(0, 300):
#     temp_index = index_data.iloc[i:i+61,:]
#     temp_index = temp_index.astype('float64')
#     temp_corr = temp_index.corr()
#     temp_list = []
#     for k in temp_corr.index:
#         for j in temp_corr.columns:
#             temp_list.append(temp_corr.loc[k, j])
#     index_corr_list.append(temp_list)
# index_corr_list = np.array(index_corr_list)
# # print(index_corr_list)
# for i in index_corr_list:
#     print(i[0])
# print(index_corr_list[:,0])
# plt.plot(index_corr_list[:, 1])
# 基金和指数的相关性

# result = []
# for i in range(0, 300):
#     temp_fund = fund_ret_data.iloc[i:i+61, 0:1]
#     temp_index = index_data.iloc[i:i+61, :]
#     temp_fund = temp_fund.merge(temp_index, left_index=True, right_index=True, how='left')
# #     print(temp_fund)
#     temp_fund = temp_fund.astype('float64')
#     temp_corr = temp_fund.corr()
# #     print(temp_corr)
# #     print(temp_corr.loc['Fund 1'])
#     result.append(list(temp_corr['Fund 1']))
# fund_index_corr = np.array(result)
# plt.plot(fund_index_corr[:, 1])
# 基金收益和基准收益率的相关性

# result = []
# for i in range(0, 300):
#     temp_ret = fund_ret_data.iloc[i:i+61, 0:1]
#     temp_ben = fund_ben_data.iloc[i:i+61, 0:1]
#     temp_ret = temp_ret.merge(temp_ben, left_index=True, right_index=True)
#     temp_ret = temp_ret.astype('float64')
#     temp_corr = temp_ret.corr()
#     temp_list = []
#     for k in temp_corr.index:
#         for j in temp_corr.columns:
#             temp_list.append(temp_corr.loc[k, j])
#     result.append(temp_list)
# fund_ret_ben_corr = np.array(result)
# plt.subplot(2,1,1)
# plt.plot(fund_ret_ben_corr[:, 2])
# plt.subplot(2,1,2)
# plt.plot(fund_ret_ben_corr[:, 2])
# for i in range(1,2):
#     plt.plot(fund_index_corr[:, i])
# plt.plot(list(cor_data.iloc[:300,0]))
# plt.plot(fund_index_corr[:, 5])
# plt.plot(index_corr_list[:, 40])
# # 基准收益率和基金基准的相关性

# ben_index_corr = []
# for i in range(0,329):
#     temp_ben = fund_ben_data.iloc[i:i+61, 0:1]
#     temp_index = index_data.iloc[i:i+61, :]
#     temp_ben = temp_ben.merge(temp_index, left_index=True, right_index=True)
#     temp_ben = temp_ben.astype('float64')
#     temp_corr = temp_ben.corr()
# #     print(temp_ben)
#     ben_index_corr.append(list(temp_corr['Fund 1']))
# ben_index_corr = np.array(ben_index_corr)
# plt.plot(ben_index_corr[:, 1])
# # 基金基准预测
# X = []
# y = []
# Ry = []
# rprel = []
# gprel = []
# prelist = []
# for i in range(100):
#     data = fund_index_corr[:, i:i+1]
#     if(data.shape[1] <= 0 ):
#         break
#     X = []
#     y = []
#     gbdt = GradientBoostingRegressor()
#     rf = RandomForestRegressor()
# #     data = data.iloc[:,:].values
# #     data[pd.isnull(np.array(data[:,0], dtype=float)),0] = 0
# #     print(data)
#     temp_X = []
#     temp_list = []
#     for k in range(0, data.shape[0]-20):
#         if(k > 100):
#             temp_X.pop(0)
# #             print(temp_X, data[k,0])
#             X.append(temp_X.copy())
#             y.append(data[k,0])
#         temp_X.append(data[k,0])
#         temp_list.append(data[k,0])
# #     print(len(X), len(y))
# #     print(X[-1], y[-1])
#     rf.fit(X[:], y[:])
#     gbdt.fit(X[:], y[:])
# #     print(gbdt.feature_importances_)
# #     print(rf.feature_importances_)
# #     随机森林
#     temp_pre = 0.0
#     temp_rf_X = temp_X.copy()
#     for k in range(data.shape[0]-20, data.shape[0]):
# #         print(temp_pre)
# #         print(k)
#         temp_rf_X.pop(0)
#         temp_pre = rf.predict([temp_rf_X])[0]
#         temp_rf_X.append(temp_pre)
# #         temp_list.append(temp_pre)
#     rprel.append(temp_pre)
# #      gbdt
#     temp_pre = 0.0
#     for k in range(data.shape[0]-20, data.shape[0]):
#         temp_X.pop(0)
#         temp_pre = gbdt.predict([temp_X])[0]
#         temp_X.append(temp_pre)
#         temp_list.append(temp_pre)
#     gprel.append(temp_pre)
#     prelist.append(temp_list)
# #     不添加相关系数的时候的做法
# #     rprel.append(rf.predict([X[-1]])[0])
# #     gprel.append(gbdt.predict([X[-1]])[0])
#     Ry.append(data[-1,0])
#     if(len(Ry)%1000 == 0):
#         print(len(Ry))
# print(len(X))
# print(gbdt.feature_importances_)
# print(rf.feature_importances_)
# plt.plot(fund_index_corr[:, 3])
# plt.plot(prelist[:][3])
# mae = np.sum(np.abs(np.array(rprel) - np.array(Ry)))/len(rprel)
# tmape = np.sum(np.abs((np.array(rprel) - np.array(Ry))/(1.5-np.array(Ry))))/len(rprel)
# score = (2/(2+mae+tmape))*(2/(2+mae+tmape))
# print('rf -- mae:', mae, '     tmape:', tmape, '      score:', score)
# mae = np.sum(np.abs(np.array(gprel) - np.array(Ry)))/len(gprel)
# tmape = np.sum(np.abs((np.array(gprel) - np.array(Ry))/(1.5-np.array(Ry))))/len(gprel)
# score = (2/(2+mae+tmape))*(2/(2+mae+tmape))
# print('gbdt -- mae:', mae, '     tmape:', tmape, '      score:', score)
# # 基金相关性填充效果

# result = []
# for i in range(0, 290):
#     temp_fund = fund_ret_data.iloc[i:min(i+61,300), 0:1]
#     temp_index = index_data.iloc[i:i+61, :]
#     temp_fund = temp_fund.merge(temp_index, left_index=True, right_index=True, how='left')
# #     print(temp_fund)
#     temp_fund = temp_fund.astype('float64')
#     temp_corr = temp_fund.corr()
#     temp_corr = temp_corr*(0.95+0.05*(min(i+61,300)-i)/61)
# #     print(temp_corr)
# #     print(temp_corr.loc['Fund 1'])
#     result.append(list(temp_corr['Fund 1']))
# fund_index_corr_limit = np.array(result)
# k = 20
# mae = np.sum(np.abs(np.array(fund_index_corr_limit[-k,:]) - np.array(fund_index_corr[-k,:])))/len(fund_index_corr_limit[-k,:])
# tmape = np.sum(np.abs((np.array(fund_index_corr_limit[-k,:]) - np.array(fund_index_corr[-k,:]))/(1.5-np.array(fund_index_corr[-k,:]))))/len(fund_index_corr_limit[-k,:])
# score = (2/(2+mae+tmape))*(2/(2+mae+tmape))
# print('mae:', mae, '     tmape:', tmape, '      score:', score)
# plt.plot(fund_index_corr[:280, 7])
# plt.plot(fund_index_corr_limit[:280, 7])
# # 基金基准预测
# X = []
# y = []
# Ry = []
# rprel = []
# gprel = []
# prelist = []
# for i in range(100):
#     data = fund_index_corr_limit[:, i:i+1]
#     if(data.shape[1] <= 0 ):
#         break
#     X = []
#     y = []
#     gbdt = GradientBoostingRegressor()
#     rf = RandomForestRegressor()
# #     data = data.iloc[:,:].values
# #     data[pd.isnull(np.array(data[:,0], dtype=float)),0] = 0
# #     print(data)
#     temp_X = []
#     temp_list = []
#     for k in range(0, data.shape[0]-20):
#         if(k > 100):
#             temp_X.pop(0)
# #             print(temp_X, data[k,0])
#             X.append(temp_X.copy())
#             y.append(data[k,0])
#         temp_X.append(data[k,0])
#         temp_list.append(data[k,0])
# #     print(len(X), len(y))
# #     print(X[-1], y[-1])
#     rf.fit(X[:], y[:])
#     gbdt.fit(X[:], y[:])
# #     print(gbdt.feature_importances_)
# #     print(rf.feature_importances_)
# #     随机森林
#     temp_pre = 0.0
#     temp_rf_X = temp_X.copy()
#     for k in range(data.shape[0]-20, data.shape[0]+10):
# #         print(temp_pre)
# #         print(k)
#         temp_rf_X.pop(0)
#         temp_pre = rf.predict([temp_rf_X])[0]
#         temp_rf_X.append(temp_pre)
# #         temp_list.append(temp_pre)
#     rprel.append(temp_pre)
# #      gbdt
#     temp_pre = 0.0
#     for k in range(data.shape[0]-20, data.shape[0]+10):
#         temp_X.pop(0)
#         temp_pre = gbdt.predict([temp_X])[0]
#         temp_X.append(temp_pre)
#         temp_list.append(temp_pre)
#     gprel.append(temp_pre)
#     prelist.append(temp_list)
# #     不添加相关系数的时候的做法
# #     rprel.append(rf.predict([X[-1]])[0])
# #     gprel.append(gbdt.predict([X[-1]])[0])
#     data = fund_index_corr[:, i:i+1]
#     Ry.append(data[-1,0])
#     if(len(Ry)%1000 == 0):
#         print(len(Ry))
# print(len(X))
# mae = np.sum(np.abs(np.array(rprel) - np.array(Ry)))/len(rprel)
# tmape = np.sum(np.abs((np.array(rprel) - np.array(Ry))/(1.5-np.array(Ry))))/len(rprel)
# score = (2/(2+mae+tmape))*(2/(2+mae+tmape))
# print('rf -- mae:', mae, '     tmape:', tmape, '      score:', score)
# mae = np.sum(np.abs(np.array(gprel) - np.array(Ry)))/len(gprel)
# tmape = np.sum(np.abs((np.array(gprel) - np.array(Ry))/(1.5-np.array(Ry))))/len(gprel)
# score = (2/(2+mae+tmape))*(2/(2+mae+tmape))
# print('gbdt -- mae:', mae, '     tmape:', tmape, '      score:', score)
# plt.plot(fund_index_corr[:, 2])
# plt.plot(prelist[:][2])
# # 基金和指数的关系
# result = []
# col = list(fund_ret_data.columns)
# for i in range(0, 200):
#     temp_fund = fund_ret_data.iloc[:, i:i+1]
#     temp_index = index_data.iloc[:, :]
#     temp_fund = temp_fund.merge(temp_index, left_index=True, right_index=True, how='left')
# #     print(temp_fund)
#     temp_fund = temp_fund.astype('float64')
#     temp_corr = temp_fund.corr()
# #     temp_corr = temp_corr*(0.95+0.05*(min(i+61,300)-i)/61)
# #     print(temp_corr)
# #     print(temp_corr.loc['Fund 1'])
#     result.append(list(temp_corr[col[i]])[1:])
# fund_index_corr_total = np.array(result)
# fund_index_corr_total = pd.DataFrame(fund_index_corr_total, columns = list(index_data.columns), index = col)
# index_data.columns
# fund_index_corr_total
# fund_index_corr_total = pd.DataFrame(fund_index_corr_total, columns = list(index_data.columns), index = col)
# fund_index_corr_total
# plt.plot(fund_index_corr_limit[1, :])
