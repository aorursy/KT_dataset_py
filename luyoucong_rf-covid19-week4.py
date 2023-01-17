#Load required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook", style="darkgrid", palette="deep", font="sans-serif", font_scale=1, color_codes=True)
## 随机森林：Ensembel（集成学习中），bagging的代表算法。
## 算法描述：以单决策树为基础，多棵决策树构成随机森林。通过多个不那么好的模型进行组合(投票)组成一个新的模型。避免单棵决策树过深过拟合的情况
## 随机体现在：1.数据抽取方面——有放回的抽取。每棵决策树训练的时候数据不一样，保证树木的独立性。
##           2.构建分支的时候，从所有特征中抽取部分特征进行决定当前层的决定条件.--Random Forest
##           3.或者从最好的几个决策条件中抽取一个作为决策条件——Extra Trees
df_train=pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")
df_test=pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")
df_info=pd.read_csv("../input/countryinfo/covid19countryinfo.csv")
df_train.head()
df_info
df_info.isnull().sum()
df_test.head()
df_train['Date']=pd.to_datetime(df_train['Date'])
df_test['Date']=pd.to_datetime(df_test['Date'])
## 查看两个数据集合时间

# print(df_train['Date'].max())
# print(df_test['Date'].min())
## 发现有12天重复时间，因此应该将训练集中3-12 to 3-24部分数据去掉
# date_filter=df_train['Date']<df_test['Date'].min()
# df_train=df_train.loc[date_filter]
# df_train
## 将所有地区唯一识别成place_id
def genplace_id(x):
    try:
        place_id=x['Country_Region']+'/'+x['Province_State']
    except:
        place_id=x['Country_Region']
    return place_id

df_train['place_id']=df_train.apply(lambda x:genplace_id(x),axis=1)
df_test['place_id']=df_test.apply(lambda x:genplace_id(x),axis=1)
print("地区个数==>"+str(len(df_train['place_id'].unique())))
def genplace_id2(x):
    try:
        place_id=x['country']+'/'+x['region']
    except:
        place_id=x['country']
    return place_id
df_info['place_id']=df_info.apply(lambda x:genplace_id2(x),axis=1)
print("地区个数==>"+str(len(df_info['place_id'].unique())))
df_train=pd.merge(df_train,df_info,how='left')
df_test=pd.merge(df_test,df_info,how='left')
df_train.isnull().sum()
# df_test.isnull().sum()
df_train.columns
df_test.isnull().sum()
## 将df_train中空的且是数值类型的数据用中位数填补
temp1=[]


for column in df_train.columns:
    try:
        median=df_train[column].median()
        temp1.append(column)
    except:
        print(column+"不是数值类型")

for col in temp1:
    df_train[col].fillna(df_train[col].median(),inplace=True)
    
df_train
## 将df_test中空的且是数值类型的数据用中位数填补
temp2=[]
temp_no=[]


for column in df_test.columns:
    try:
        median=df_test[column].median()
        temp2.append(column)
    except:
        temp_no.append(column)
        print(column+"不是数值类型")

for col in temp2:
    df_test[col].fillna(df_test[col].median(),inplace=True)
    
df_test.isnull().sum()
temp_list=[ 'newcases30',
 'newcases31',
 'newcases1',
 'newcases2',
 'newcases3']
def change_type(df):
    for col in temp_list:
        df[col]=df[col].fillna("0").apply(lambda x:x.replace(",","")).astype("int")
change_type(df_train)
change_type(df_test)

def fillna(df):
    for col in temp_list:
        df[col].fillna(df[col].mean(),inplace=True)
fillna(df_train)
fillna(df_test)
# 将pop,newcases的na补充完毕
df_train['pop']=df_train['pop'].fillna("0").apply(lambda x:x.replace(",","")).astype("int")
df_test['pop']=df_test['pop'].fillna("0").apply(lambda x:x.replace(",","")).astype("int")
# df_train['density'].fillna(df_train['density'].median(),inplace=True)
df_train['pop'].fillna(df_train['pop'].median(),inplace=True)

# df_test['density'].fillna(df_train['density'].median(),inplace=True)
df_test['pop'].fillna(df_train['pop'].median(),inplace=True)
## 数据清洗后的样子
print(df_train)
print(df_test)
train=df_train.copy()
test=df_test.copy()

train['Date'].dt.day
def create_features(df):
    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['quarter'] = df['Date'].dt.quarter
    df['weekofyear'] = df['Date'].dt.weekofyear
    return df
## 创造变量特征 
train=create_features(train)
test=create_features(test)
## 多加day数据，日期——一年的第几天 Day 变量相当于dayofyear
train['Day']=train['Date'].apply(lambda x:x.dayofyear).astype('int')
test['Day']=test['Date'].apply(lambda x:x.dayofyear).astype('int')
train
print(train['Day'].max())  # 5/15
print(train['Date'].max()) 
print(test['Day'].min())  # 4/02
print(test['Date'].min()) 
## week 4不存在地区经纬度，使用地区标签代替
i=0
train['place_label']=0
test['place_label']=0
places=train['place_id'].unique()
for place in places:
    train['place_label'][train['place_id']==place]=i
    test['place_label'][test['place_id']==place]=i
    i=i+1
print(train['place_label'].unique())
print(test['place_label'].unique())
## 引入变量 Confirm/day, Fatal/day #没用上
train['Confirm/day']=0
train['Fatal/day']=0
places=train['place_id'].unique()
for place in places:
    temp=train['ConfirmedCases'][train['place_id']==place].values
    temp[1:]-=temp[:-1]
    train['Confirm/day'][train['place_id']==place]=temp
    
    temp=train['Fatalities'][train['place_id']==place].values
    temp[1:]-=temp[:-1]
    train['Fatal/day'][train['place_id']==place]=temp
train[:50]
    
## 将因变量换成 Confirm/day, Fatal/day #没用上
test['Confirm/day']=0
test['Fatal/day']=0
test
# 模型参数：class sklearn.ensemble.RandomForestRegressor(n_estimators=10, criterion='mse', 
# max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
# max_features='auto',max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, 
# random_state=None, verbose=0, warm_start=False)

# 一些重要参数：
# n_estimators:森林中树木的数量——越多越好
# criterion: 分裂时候的决策算法：mse:均方误差。只支持mse。如果是RandomForestClassifier还支持gini等
# max_features: 选取特征时候抽取的数量。可以是（int,float,string）可以是一个数目，或者sqrt表示所有特征取方根，auto表示max_features=n_features
# min_samples_leaf: 树木叶子节点最少包含的样本数目
# n_jobs: 表示并行的进程数目

# 更详尽的连接：
#http://lijiancheng0614.github.io/scikit-learn/modules/generated/sklearn.ensemble.RandomForestRegressor.html

## 划分测试集和验证集——用于调参
from sklearn.model_selection import train_test_split
col_var=[ 
#     'Confirm/day',
    'day',  
#     'month',  
    'dayofweek',
    'dayofyear',
#     'quarter',
#     'weekofyear',
    'place_label',
    'pop', 
    # OK
#          'tests',  # 每天病例测试数
#        'testpop',  # poppulation/tests
#          'density', #人口密度
#          'medianage', #地区中位数年龄
#          'urbanpop',  #城市人口
    
#          'quarantine', 
#          'schools',
#        'publicplace', 
#          'gatheringlimit', 
#          'gathering', 
#          'nonessential',
   # OK
#        'hospibed', 
#          'smokers', #吸烟者
#          'sex0', 
#          'sex14', 
#          'sex25', 
#          'sex54', 
#          'sex64',
#        'sex65plus', 
#          'sexratio', 
#          'lung',    #因肺类疾病的死亡率 
#          'femalelung', 
#          'malelung',
    
#          'gdp2019',
#        'healthexp', 

    #    OK
#          'healthperpop', 
         'fertility', #平均children/women
#          'avgtemp', # 1-3月平均温度
    
#          'avghumidity',
#        'firstcase', 
#          'totalcases', 
    
#          'active30', 
#          'active31', 
#          'active1', 
#          'active2',
#        'active3', 
    #OK
         'newcases30',  #3/30新增病例数
         'newcases31', 
         'newcases1', 
         'newcases2',
#        'newcases3', 
    
#          'deaths', 
#          'newdeaths30', # 3/30新增死亡病例
#          'newdeaths31', 
#          'newdeaths1',
#        'newdeaths2', 
#          'newdeaths3', 
#          'recovered', 
#          'critical30',  #3/30新增重症病例
#          'critical31',
#        'critical1',
#          'critical2',
#          'critical3', 
#          'casediv1m', 
#          'deathdiv1m',
      ]

train.columns
X=train[col_var]
Y1=train['ConfirmedCases']
Y2=train['Fatalities']
X_pred=test[col_var]
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
## 旧的划分训练集和验证集的方式：
# x_train,x_test,y1_train,y1_test=train_test_split(x,y1,test_size=0.3,random_state=44)
##因为是一个预测趋势的问题，所以将Day<93的部分当作训练集，Day>=93且<107的当作验证集来模拟预测的模型训练
x_train=X[X['dayofyear']<93]
x_valid=X[(X['dayofyear']>92)&(X['dayofyear']<107)]  #4/02后两周作为验证集
y1_train=train['ConfirmedCases'][train['dayofyear']<93] 
y1_valid=train['ConfirmedCases'][(train['dayofyear']>92)&(train['Day']<107)]
y1_train.values
##  确诊病例
rf=RandomForestRegressor(n_estimators=600)
rf.fit(x_train,y1_train.values.ravel())

## 发现前3个变量太强别的几乎变量没有作用
# display feature importance
tmp = pd.DataFrame()
tmp["feature"] = col_var
tmp["importance"] = rf.feature_importances_
tmp = tmp.sort_values('importance', ascending=False)
tmp

## 评分良好暂时不需要调参
print(rf.score(x_train,y1_train))
print(rf.score(x_valid,y1_valid))

## 预测单个地区的可视化：
## 当选择确诊病例作为因变量，画出的预测曲线明显与实际拟合的不好
## Day=93 是4/02
place='China/Hubei'
data=train[train['place_id']==place].copy()
data_test=test[test['place_id']==place].copy()

sns.lineplot(x=data['dayofyear'][data['dayofyear']<108],y=data['ConfirmedCases'],label='true')
pred=rf.predict(train[col_var][(train['place_id']==place)&(train['dayofyear']>92)&(train['dayofyear']<108)])
sns.lineplot(x=data['dayofyear'][(data['dayofyear']>92)&(train['dayofyear']<108)],y=pred,label='pred')

# pred2=rf.predict(test[col_var][(test['place_id']==place)])
# sns.lineplot(x=data_test['Day'],y=pred2,label='pred2')
plt.title(place)

plt.show()
#画出所有地区的预测图形

places=train['place_id'].unique()
for place in places:
   
    data=train[train['place_id']==place].copy()
    data_test=test[test['place_id']==place].copy()

    sns.lineplot(x=data['dayofyear'],y=data['ConfirmedCases'],label='true_case')
    pred=rf.predict(train[col_var][(train['place_id']==place)&(train['dayofyear']>92)])
    sns.lineplot(x=data['dayofyear'
                       ][data['dayofyear']>92],y=pred,label='pred_case')

    # pred2=rf.predict(test[col_var][(test['place_id']==place)])
    # sns.lineplot(x=data_test['Day'],y=pred2,label='pred2')
    plt.title(place)

    plt.show()
rf_predcase=rf.predict(X_pred)
plt.plot(rf_predcase)
test['Pred_case']=rf_predcase
test
rf.predict(X_pred)

col_var2=[ 
     'day',
    'month',
    'dayofweek',
    'dayofyear',
    'quarter',
    'weekofyear',
    'place_label',
    'pop', 
#     'Fatal/day',
         'tests',
#     OK
       'testpop', 
         'density', 
         'medianage',
         'urbanpop', 
    
#          'quarantine', 
#          'schools',
#        'publicplace', 
#          'gatheringlimit', 
#          'gathering', 
#          'nonessential',
    # OK
       'hospibed', 
         'smokers', 
         'sex0', 
         'sex14', 
         'sex25', 
         'sex54', 
         'sex64',
       'sex65plus', 
         'sexratio', 
         'lung', 
         'femalelung', 
         'malelung',
    
#          'gdp2019',
#        'healthexp', 

    #     OK
         'healthperpop', 
         'fertility', 
         'avgtemp', 
    
#          'avghumidity',
#        'firstcase', 
#          'totalcases', 
    
#          'active30', 
#          'active31', 
#          'active1', 
#          'active2',
#        'active3', 
#          'newcases30', 
#          'newcases31', 
#          'newcases1', 
#          'newcases2',
#        'newcases3', 
#          'deaths', 
    # OK
         'newdeaths30', 
         'newdeaths31', 
         'newdeaths1',
       'newdeaths2', 
         'newdeaths3', 
#          'recovered', 
#          'critical30', 
#          'critical31',
#        'critical1',
#          'critical2',
#          'critical3', 
#          'casediv1m', 
#          'deathdiv1m',
      ]

X2=train[col_var2]
X_pred2=test[col_var2]
##因为是一个预测趋势的问题，所以将Day<93的部分当作训练集，Day>=93的当作验证集来模拟预测的模型训练
x_train=X2[X2['dayofyear']<93]
x_valid=X2[(X2['dayofyear']>92)&(X2['dayofyear']<107)]
y2_train=train['Fatalities'][train['dayofyear']<93]   #4/02后两周作为验证集
y2_valid=train['Fatalities'][(train['dayofyear']>92)&(train['dayofyear']<107)]


rf2=RandomForestRegressor(n_estimators=600)
rf2.fit(x_train,y2_train.values.ravel())
# 前4个变量有明显作用
tmp = pd.DataFrame()
tmp["feature"] = col_var2
tmp["importance"] = rf2.feature_importances_
tmp = tmp.sort_values('importance', ascending=False)
tmp

print(rf2.score(x_train,y2_train))
print(rf2.score(x_valid,y2_valid))



## 预测单个地区的可视化：
## 当选择确诊病例作为因变量，画出的预测曲线明显与实际拟合的不好
## Day=93 是4/02
place='China/Hubei'
data=train[train['place_id']==place].copy()
data_test=test[test['place_id']==place].copy()

sns.lineplot(x=data['dayofyear'],y=data['Fatalities'],label='true_death')
pred=rf2.predict(train[col_var2][(train['place_id']==place)&(train['dayofyear']>92)])
sns.lineplot(x=data['dayofyear'][data['dayofyear']>92],y=pred,label='pred_death')

# pred2=rf.predict(test[col_var][(test['place_id']==place)])
# sns.lineplot(x=data_test['Day'],y=pred2,label='pred2')
plt.title(place)

plt.show()
rf_predfatal=rf2.predict(X_pred2)
plt.plot(rf_predfatal)
test['Pred_death']=rf_predfatal
train[['place_id','Date','ConfirmedCases','Fatalities']]
result_train=train[['place_id','Date','ConfirmedCases','Fatalities']]
result_test=test[['Date','place_id','Country_Region','Pred_case','Pred_death']]
result=pd.merge(result_train,result_test,how='left')
result
result.to_csv('RF_result.csv')


submission = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")
submission['ConfirmedCases'] = rf_predcase
submission['ConfirmedCases'] = submission['ConfirmedCases'].astype("int")
submission['Fatalities'] = rf_predfatal
submission['Fatalities'] = submission['Fatalities'].astype("int")

submission.to_csv('submission.csv', index = False)

