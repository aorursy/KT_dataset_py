import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from xgboost import plot_importance
import xgboost as xgb
from sklearn import metrics
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split,cross_val_predict,cross_val_score, ShuffleSplit
from sklearn.model_selection import GridSearchCV,KFold
import matplotlib.pylab as plt
import matplotlib.pyplot as plot
from matplotlib.pyplot import savefig
from matplotlib.pylab import rcParams
import time
import seaborn as sns
from scipy import stats,integrate
import datetime
def rad(d):
    return d * np.pi / 180.0
def GetDistance(lon1,lat1,lon2,lat2):
    EARTH_RADIUS = 6378137 #赤道半径
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lon1) - rad(lon2)
    s = 2 * np.arcsin(np.sqrt((np.sin(a/2)**2)+np.cos(radLat1)*np.cos(radLat2)*(np.sin(b/2)**2)))
    s = s * EARTH_RADIUS
    return s

def GetDate(date_time):
    return date_time[:10]

def GetDate_time(date_time):
    if(int(date_time[-9:-7]) > 30):
        return (int(date_time[-12:-10])+1)% 24
    else:
        return int(date_time[-12:-10])
def GetDate_year(x):
    return int(x[:4])
def GetDate_month(x):
    return int(x[5:7])
def GetDate_day(x):
    return int(x[8:])
def func(x,y):
    if(x > 0.00 and y > 0.00):
        return y/x
    else:
        return -1
def week_get(date):    
    return int(datetime.datetime(int(date[:4]),int(date[5:7]),int(date[8:])).strftime("%w")) # strftime("%a")

def data_deal(chunk,tag):
    chunk['distance'] = GetDistance(chunk['pickup_longitude'],chunk['pickup_latitude'],chunk['dropoff_longitude'],chunk['dropoff_latitude'])/1000.0
    chunk['date'] = chunk['pickup_datetime'].apply(lambda x:GetDate(x))
    chunk['year'] = chunk['date'].apply(lambda x:GetDate_year(x))
    chunk['month'] = chunk['date'].apply(lambda x:GetDate_month(x))
    chunk['day'] = chunk['date'].apply(lambda x:GetDate_day(x))
    chunk['time'] = chunk['pickup_datetime'].apply(lambda x:GetDate_time(x))
    chunk['week'] = chunk.apply(lambda x: week_get(x.date), axis = 1)
    if(tag == True):        
        chunk['price'] = chunk.apply(lambda x: func(x.distance, x.fare_amount), axis = 1)
    return chunk

def getData_train(train_chunks,allow):
    chunks = []
    if(allow == False):        
        count = 0        
        while(count < 7):               
            chunk = train_chunks.get_chunk(10000)
            chunks.append(data_deal(chunk,True))
            count += 1
    else:
        loop = True    
        while loop:
            try:
                chunk = train_chunks.get_chunk(10000)
                chunks.append(data_deal(chunk,True))
            except StopIteration:
                loop = False
                print("Iteration is stopped.")
    return chunks
start = time.time()
# train_chunks = pd.read_csv('./data/train.csv',iterator = True,engine='python')         
train_chunks = pd.read_csv('../input/train.csv',iterator = True,engine='python')
chunks = getData_train(train_chunks,False)
data_train = pd.concat(chunks, ignore_index=True)
data_ = data_train[data_train['price'] >= 0]
data_.index = range(len(data_))
print(data_.head())
end = time.time()
print(end - start)
plt.title("I'm a scatter diagram.") 
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(xmax=24,xmin=-1)
plt.ylim(ymax=50,ymin=0)
plt.plot(data_.time,data_.price,'ro')
plt.show()
# 某个时间的最高价能看出随时间的大致分布
data = data_.loc[:,'price'].values
# 统计输出信息
percentile_result = np.percentile(data, [25, 50, 75])
num = 0
for i in list(data > percentile_result[2] * 1.5):
    if(i == True):
        num+=1
print('离群点个数：',num,'\n四分位数Q3：',percentile_result[2])
print(num/len(list(data)))
# 显示图例
plt.boxplot(x=data,showmeans=True,meanline=True,whis=1.5)
plt.legend()
# 显示图形
plt.show()
plt.close()

print(data_.shape[0])
result = data_[data_['price'] > percentile_result[2] * 1.5].index.tolist()
data_.drop(data_.index[result],inplace=True)
data_.index = range(len(data_))
result = data_[data_['price'] < percentile_result[0] / 1.5].index.tolist()
data_.drop(data_.index[result],inplace=True)
data_.index = range(len(data_))
print(data_.shape[0])
from pandas import Categorical
ordered_days = data_.week.value_counts().index
print(ordered_days)
# ordered_days = Categorical(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
ordered_days = Categorical([i for i in range(7)])
# FacetGrid传数据需要是pandas格式
g = sns.FacetGrid(data_,row='week',row_order=ordered_days,size=2,aspect=3)
g.map(sns.boxplot,"price")
days = [ 'Mon','Tue','Wed','Thu','Fri','Sat','Sun']
days_ = [i for i in range(0,7)]
# print(np.unique(list(data_['week'].values)))
percentile_day1 = []
percentile_day2 = []
percentile_day0 = []
percentile_day = dict().fromkeys([i for i in days_])
pds = []
for i in days_:
    data = data_[data_['week']== i].price
#     print(data)
    # 统计输出信息
    percentile_result = np.percentile(data, [25, 50, 75])
    percentile_day0.append(percentile_result[0])
    percentile_day1.append(percentile_result[1])
    percentile_day2.append(percentile_result[2])
    percentile_day[i] = percentile_result
    num = 0
    for j in list(data > percentile_result[2] * 1.5):
        if(j == True):
            num += 1
    pds.append([i,num,percentile_result[0],percentile_result[1],percentile_result[2],num/len(list(data))])
pas = pd.DataFrame(pds,columns=['week','离群点个数','Q1','Q2','Q3','离群点占比'])
pas
#     print('离群点个数：',num,'\n四分位数Q3：',percentile_result[2])    
# 一起绘制对比
week_days = [i for i in range(1,8)]
plt.title('Week Analysis')
mak = ['*','^','v','o','s','<','>']
plt.plot(week_days, percentile_day0, color='green', label='25%',marker=mak[0])
plt.plot(week_days, percentile_day1, color='#873018', label='50%',marker=mak[1])
plt.plot(week_days, percentile_day2,  color='skyblue', label='75%',marker=mak[2])
plt.xticks(week_days,days)#将数字转换为字符显示
plt.xlabel('days')
plt.ylabel('price')
plt.show()
# 分开来看
def draw_point_plt(data):
    plt.plot(week_days, data, color='green', label='25%',marker=mak[0])
    plt.xticks(week_days,days)#将数字转换为字符显示
    plt.xlabel('days')
    plt.ylabel('price')
    plt.show()
    plt.close()
draw_point_plt(percentile_day0)
draw_point_plt(percentile_day1)
draw_point_plt(percentile_day2)
d = []
d.append(days_)
d.append(percentile_day)
plt.title('Week Analysis')
color_ = ['#587123','#581223','#587199','#327123','#007123','#127453','#918652']
mak = ['*','^','v','o','s','<','>']
# print(percentile_day)

plt.xticks(days_,days)#将数字--->字符显示
for i in range(7):
    day = [i,i,i]
    plt.plot(day, percentile_day[i], color=color_[i], marker=mak[i])
plt.plot(days_, percentile_day0, color='green', label='25%')
plt.plot(days_, percentile_day1, color='#873018', label='50%')
plt.plot(days_, percentile_day2,  color='skyblue', label='75%')
plt.legend() # 显示图例
plt.xlabel('days')
plt.ylabel('price')
plt.show()
data_.columns.tolist()
year = []
for i in range(2009,2016):
    year.append(i)
percentile_year = dict().fromkeys([i for i in year])
percentile_year0 = []
percentile_year1 = []
percentile_year2 = []
pds = []
for i in year:    
    data = data_[data_['year']==i].price
    # 统计输出信息
    percentile_result = np.percentile(data, [25, 50, 75])
    percentile_year0.append(percentile_result[0])
    percentile_year1.append(percentile_result[1])
    percentile_year2.append(percentile_result[2])
    percentile_year[i] = percentile_result
    num = 0
    for j in list(data > percentile_result[2] * 1.5):
        if(j == True):
            num+=1
    pds.append([i,num,percentile_result[0],percentile_result[1],percentile_result[2],num/len(list(data))])
pas = pd.DataFrame(pds,columns=['year','离群点个数','Q1','Q2','Q3','离群点占比'])
pas
#     print('离群点个数：',num,'\n四分位数Q3：',percentile_result[2])    
# 一起绘制对比
year_ = [i for i in range(7)]
plt.title('Year Analysis')
mak = ['*','^','v','o','s','<','>']
plt.plot(year_, percentile_year0, color='green', label='25%',marker=mak[0])
plt.plot(year_, percentile_year1, color='#873018', label='50%',marker=mak[1])
plt.plot(year_, percentile_year2, color='skyblue', label='75%',marker=mak[2])
plt.xticks(year_,year)#将数字--->字符显示
plt.xlabel('year')
plt.ylabel('price')
plt.show()
# 分开来看
def draw_point_plt(data):
    plt.plot(year_, data, color='skyblue', label='75%',marker=mak[2])
    plt.xticks(year_,year)#将数字--->字符显示
    plt.xlabel('days')
    plt.ylabel('price')
    plt.show()
    plt.close()
draw_point_plt(percentile_year0)
draw_point_plt(percentile_year1)
draw_point_plt(percentile_year2)
# 先训练一波
def modelfit(alg, data, labels_, cols, target, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    # 可以返回n_estimates的最佳数目，为什么呢, 哪里返回？
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(data, label=labels_)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='rmse', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    #Fit the algorithm on the data
    seed = 10
    # seed=10从0.11升为了0.2566
    # Model Report
    # r2_score : 0.2566
    # MAE:  0.4723899992310908 %
    test_size = 0.3
    x_train,x_test,y_train,y_test = train_test_split(data, labels_, test_size=test_size,random_state=seed)    
    print(x_train.shape[1],y_train.shape[1])    
    eval_set = [(x_test,y_test)]
    alg.fit(x_train, y_train, early_stopping_rounds=early_stopping_rounds, eval_metric='rmse',eval_set=eval_set,verbose=True)        
    #Predict training set:
    dtrain_predictions = alg.predict(x_test)

    # print(type(dtrain_predictions),type(labels_))
    y_true = list(y_test)
    y_pred = list(dtrain_predictions)
    
    #Print model report:
    print("\nModel Report")
    print("r2_score : %.4g" % metrics.r2_score(y_true, y_pred))
    mae_y = 0.00
#     for i in range(len(y_true)):
#         mae_y += np.abs(np.float(y_true[i])-y_pred[i])
#     print("MAE: ", (mae_y*4799+6)/len(y_true))
    # Model Report
    # r2_score : 0.9673
    # MAE:  0.636517748270864 %
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)   

    # feat_imp.plot(kind='bar', title='Feature Importances')
    # plt.ylabel('Feature Importance Score')
    fig, ax = plt.subplots(1, 1, figsize=(8, 13))    
    plot_importance(alg, max_num_features=25, height=0.5, ax=ax)
    plt.show()
    # 重要性筛选
    feat_sel = list(feat_imp.index)
    feat_val = list(feat_imp.values)
    featur = []
    for i in range(len(feat_sel)):
        featur.append([cols[int(feat_sel[i][1:])],feat_val[i]])
    print('所有特征的score:\n',featur)

    feat_sel2 = list(feat_imp[feat_imp.values > target].index)    
    featur2 = []
    for i in range(len(feat_sel2)):
        featur2.append(cols[int(feat_sel2[i][1:])])    
    return featur2
def MAE_(xgb1,train_x,train_y):
    y_pre = list(xgb1.predict(train_x))
    train_y = train_y.as_matrix()    
    num = 0
    for i in range(len(y_pre)):        
        num += np.abs(y_pre[i] - train_y[i])
    print((num*4799+6)/len(y_pre))
#     1.9692270331443802 7.559401862892015
def RMSE(xgb1,train_x,train_y):
    y_pre = list(xgb1.predict(train_x))
#     train_y = train_y.as_matrix()
    num = 0
    print(len(y_pre),len(train_y))
    for i in range(len(y_pre)):
        num += ((y_pre[i] - train_y[i])*5.5902+1.9692)**2
    print(np.sqrt(num*5.5902+1.9692)/len(y_pre))
#     print(np.sqrt(num)/ len(y_pre))
    

def xgboost_select_feature(data_, labels_,cols,target):# # 特征选择
    xgb1 = XGBRegressor(learning_rate =0.1,max_depth=5,min_child_weight=1,n_estimators=1000,
                    gamma=0,subsample=0.7,colsample_bytree=0.75,objective= 'reg:logistic',
                        nthread=4,scale_pos_weight=1,seed=27)       
    feature_ = list(modelfit(xgb1, data_.values,labels_.values,cols,target)) # 特征选择    
    return feature_
# def mini_xgboost_train(train_x, train_y):    
#     # # 半手动调参-------------------是个过程------调参成功需要注释掉----------------------------------------------------
#     param_test1 = {
# #         'n_estimators':[64,65,66,67]
# #         'max_depth':[i for i in range(3,11)]
# #         'subsample':[i/100 for i in range(60,100,10)],
# #         'colsample_bytree':[j/100 for j in range(60,100,10)]
# #           'subsample':[i/100 for i in range(80,100,5)],
# #           'colsample_bytree':[j/100 for j in range(80,100,5)]
#         'learning_rate':[0.1,0.05,0.08,0.07,0.2]
#     }
#     gsearch1 = GridSearchCV(estimator = XGBRegressor(learning_rate=0.1,n_estimators=138,max_depth=4,min_child_weight=1,
#                        colsample_bytree=0.9,subsample=0.9,gamma=0,objective= 'reg:logistic', nthread=4, seed=27), 
#                     param_grid = param_test1,scoring='neg_mean_squared_error',n_jobs=4, iid=False, cv=5)
#     gsearch1.fit(train_x,train_y)
#     print(gsearch1.best_params_,gsearch1.best_score_) 
def xgboost_train(train_x, train_y):
#     train_x = train_x.as_matrix()
    xgb1 = XGBRegressor(learning_rate=0.1,n_estimators=138,max_depth=4,min_child_weight=1,
                       colsample_bytree=0.9,subsample=0.9,gamma=0,objective= 'reg:logistic', nthread=4, seed=27)
    xgb1.fit(train_x,train_y)
    RMSE(xgb1,train_x,train_y)
       
    return xgb1
colum = ['distance','year','month','day', 'time', 'week','passenger_count','price']
data_ = data_[colum].sort_values(by=['year','month','day','time'])
data_.index = range(len(data_))
colum.remove('price')
data = data_[colum]
labels_ = data_[['price']]
# print(data.head())
# print(labels_.head())

# reg:logistic要求对label归一化
minn = labels_['price'].min(axis=0)
maxx = labels_['price'].max(axis=0)
print(minn,maxx)
labels_['price_'] = labels_['price'].apply(lambda x: (x - minn)/(maxx - minn))
labels_ = labels_.drop('price',axis=1)
# print(labels_['price'])
test_size = 0.3
seed = 10
data = pd.DataFrame(data.values,columns = colum)
x_train,x_test,y_train,y_test = train_test_split(data, labels_, test_size=test_size,random_state=seed)
# xgboost_select_feature(data, labels_,colum,100)
# mini_xgboost_train(x_train.values,y_train.values)
xgb_ = XGBRegressor()
xgb_ = xgboost_train(x_train.values,y_train.values)

# -------------------------prediction-------------------------
# test = pd.read_csv('./data/test.csv')
test = pd.read_csv('../input/test.csv')
test_data = data_deal(test,False)
test_data = test_data[colum_]
test_data = pd.DataFrame(test_data.values,columns = colum)
test_data = test_data.as_matrix()
# test_data['week'] = test_data['week'].apply(lambda x: int(x))
# print(test_data['week'].head())
y_pre = list(xgb_.predict(test_data))
test_pre_ = pd.DataFrame(y_pre).apply(lambda x: (maxx - minn)*x + minn)
print(test_pre_.head())