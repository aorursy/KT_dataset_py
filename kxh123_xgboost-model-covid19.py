# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import sys

np.set_printoptions(threshold=sys.maxsize)
#############这和LightGBM是一个类似的算法

############LightGBM似乎是他的一个升级版

###########XG将K个树的结果凑在一起作为预测值

############还是之前的那一组数据

###########来看看XGBOOST:https://www.jianshu.com/p/a62f4dce3ce8

##########和之前的LGBM有一个显著的不同就是最优分割点的选择不同：

##########XGB有两种选取分割点的方法：一种是贪心分割

#########贪心分割就是枚举每个潜在的分割点选最好的，计算消耗大，会首先对特征进行排序

########第二种就是根据特征分布的N个百分位数提出N个最佳分割点，然后把数据放在桶里，然会计算最佳分割点

########有两种分割方式：1.局部近似 2.全局近似

########局部近似是我们在每个节点分裂的时候再考虑其分位点并划分样本

########全局近似是我们在一棵树还没有产生的时候我们就考虑分位点并划分样本
train_path=r"../input/covid19-global-forecasting-week-1/train.csv"

test_path=r"../input/covid19-global-forecasting-week-1/test.csv"
df_train=pd.read_csv(train_path)

df_test=pd.read_csv(test_path)
df_train.head()
df_test.head()
#####################EDA分析

######################Kaggle提供的地区跨度
places=df_train["Country/Region"].unique()
places.shape
########################Kaggle提供的时间跨度

########################是差不多一个月的跨度

########################需要转化时间序列的属性
df_train["Date"]=pd.to_datetime(df_train["Date"])   
print(len(df_train["Date"].unique()))  #查看有多少天
#######################每天世界的已确诊人数总和
train_cases_conf = df_train.groupby(['Date'])['ConfirmedCases'].sum()

train_cases_conf
#####################上升期-稳定期-爆发期
train_cases_conf.plot(figsize = (5,4), title = 'Worldwide Confirmed Cases')

####################死亡人数在确诊人数进入爆发期后同样进入了爆发期
train_fatal = df_train.groupby(['Date'])['Fatalities'].sum()

train_fatal.plot(figsize = (5,4), title = 'Worldwide Fatalaties')
#########################观察不同地区的实际情况

#########################对比我们国家和美国

#########################完全不同的局势啊，我们前期高爆后期稳定,美国前期平稳后期爆发
def country_stats(country, df):

    country_filt = (df['Country/Region'] == country)

    df_cases = df.loc[country_filt].groupby(['Date'])['ConfirmedCases'].sum()

    df_fatal = df.loc[country_filt].groupby(['Date'])['Fatalities'].sum()

    fig, axes = plt.subplots(nrows = 2, ncols= 1, figsize=(5,5))

    df_cases.plot(ax = axes[0])

    df_fatal.plot(ax = axes[1])

country_stats('China', df_train)

country_stats('US', df_train)
########这里比较的是不同国家每个地区疫情最严重城市的确诊人数

##########为什么意大利比我们确诊人数多？

######因为意大利本身就一个地区,单独疫情最严重的湖北地区确诊人数比不上整个意大利的确诊人数
train_case_country = df_train.groupby(['Country/Region'], as_index=False)['ConfirmedCases'].max()

###########################

train_case_country.sort_values('ConfirmedCases', ascending=False, inplace = True)

train_case_country

plt.figure(figsize=(8,6))

plt.bar(train_case_country['Country/Region'][:5], train_case_country['ConfirmedCases'][:5], color = ['red', 'yellow','black','blue','green'])
##################这里比较的才是全国确诊人数
def case_day_country (Date, df):

    df = df.groupby(['Country/Region', 'Date'], as_index = False)['ConfirmedCases'].sum()

    date_filter = (df['Date'] == Date)

    df = df.loc[date_filter]

    df.sort_values('ConfirmedCases', ascending = False, inplace = True)

    sns.catplot(x = 'Country/Region', y = 'ConfirmedCases' , data = df.head(10), height=3,aspect=4, kind = 'bar')

case_day_country('2020-03-23', df_train)
################数据处理部分
##################测试集和训练集有重合,这会导致什么情况？

##################1.这样会夸大模型对于数据的拟合效果

##################2.其实我们拟合的数据更加多了，应该能做出更好的模型了

##################3.但是为了保证衡量模型的公正性：

##################我们应该这么做：

##################1.手动剪裁训练集

##################2.手动剪裁测试集,但是测试集合是不可能剪辑的，这会导致无法提交

##############所以只能手动剪辑训练集合
df_train.Date=pd.to_datetime(df_train["Date"])

df_test.Date=pd.to_datetime(df_test["Date"])
print(df_train['Date'].max())

print(df_test['Date'].min())        
date_filter = df_train['Date'] < df_test['Date'].min()

df_train = df_train.loc[date_filter]
train_country_date = df_train.groupby(['Country/Region', 'Date', 'Lat', 'Long'], as_index=False)['ConfirmedCases', 'Fatalities'].sum()
train_country_date.head() 

#############"Province"的缺失值太多了,我们丢弃了，丢弃了缺失值
################特征工程开始
df_test.drop('Province/State', axis = 1, inplace = True)

df_test.Date = pd.to_datetime(df_test['Date'])
df_test.info()
##################直接处理时间数据很麻烦，都转化为日期排序数据
train_country_date['Month'] = train_country_date['Date'].dt.month

train_country_date['Day'] = train_country_date['Date'].dt.day

train_country_date['Day_Week'] = train_country_date['Date'].dt.dayofweek

train_country_date['quarter'] = train_country_date['Date'].dt.quarter

train_country_date['dayofyear'] = train_country_date['Date'].dt.dayofyear

train_country_date['weekofyear'] = train_country_date['Date'].dt.weekofyear

df_test['Month'] = df_test['Date'].dt.month

df_test['Day'] = df_test['Date'].dt.day

df_test['Day_Week'] = df_test['Date'].dt.dayofweek

df_test['quarter'] = df_test['Date'].dt.quarter

df_test['dayofyear'] = df_test['Date'].dt.dayofyear

df_test['weekofyear'] = df_test['Date'].dt.weekofyear
########特征工程是基于原有的特征构造新的特征

########训练集有的特征测试集也要有

########所以不如放在一起做特征,就不需要做两次了
labels = ['Country/Region', 'Lat', 'Long', 'Date', 'Month', 'Day', 'Day_Week','quarter', 'dayofyear', 'weekofyear']

df_train_clean = train_country_date[labels]

df_test_clean = df_test[labels]

data_clean = pd.concat([df_train_clean, df_test_clean], axis = 0)
#######################地区是字符串属性,同样需要转化属性,再编码一次？
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()

data_clean['Country'] = enc.fit_transform(data_clean['Country/Region'])

data_clean.head()               ##############为地区编号
#################原来的国家数据可以丢掉了
data_clean.drop(['Country/Region', 'Date'], axis = 1, inplace=True)
#########################划分训练集和测试集，之前合并了
index_split = df_train.shape[0]

data_train_clean = data_clean[:index_split]

data_test_clean = data_clean[index_split:]
######################查看训练集以及测试集数据
data_train_clean.head()
data_test_clean.head()
########################分离属性变量以及目标变量
x = data_train_clean[['Lat', 'Long', 'Month', 'Day', 'Day_Week','quarter', 'dayofyear', 'weekofyear', 'Country']]

y_case = df_train['ConfirmedCases']

y_fatal = df_train['Fatalities']
###########################73开分离训练数据以及测试数据

##########################注意啊，我们现在在调树的数目阶段

##########################我们是用训练集的一部分调参数
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y_case, test_size = 0.3, random_state = 42)
x_train_fatal, x_test_fatal, y_train_fatal, y_test_fatal = train_test_split(x, y_fatal, test_size = 0.3, random_state = 42)
####################建模
import xgboost as xgb

from sklearn.metrics import mean_squared_error
reg = xgb.XGBRegressor(n_estimators=1000,min_child_weight=1,max_depth=6)
reg.fit(x_train, y_train)
reg.score(x_train, y_train)
reg_y_pred = reg.predict(x_train)
mean_squared_error(y_train, reg_y_pred)
reg.score(x_test, y_test)
reg_y_test_pred = reg.predict(x_test)

mean_squared_error(y_test, reg_y_test_pred)
reg.fit(x, y_case)
y_train_pred = reg.predict(x)
mean_squared_error(y_case, y_train_pred)
xgb_pred_case = reg.predict(data_test_clean)
xgb_pred_case
reg.fit(x, y_fatal)
xgb_pred_fatal = reg.predict(data_test_clean)
xgb_pred_fatal
plt.plot(y_case)
plt.plot(y_train_pred)          ################这不拟合的挺好的
plt.plot(xgb_pred_case)        ###############这个负数是啥啊
plt.plot(reg.predict(x))
plt.plot(y_fatal)
#################################submit_kaggle这是提交文件

######################################提交模块
XBGoostpath=r"../input/covid19-global-forecasting-week-1/submission.csv"

XBGoost=pd.read_csv(XBGoostpath)

XBGoost['ConfirmedCases'] = xgb_pred_case

XBGoost['Fatalities'] = xgb_pred_fatal

XBGoost.to_csv('submission.csv', index = False)