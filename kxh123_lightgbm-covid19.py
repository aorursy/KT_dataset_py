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
import matplotlib.pyplot as plt
pd.set_option("display.max_columns",100)

train_path=r"../input/covid19-global-forecasting-week-1/train.csv"

test_path=r"../input/covid19-global-forecasting-week-1/test.csv"
df_train=pd.read_csv(train_path)

df_test=pd.read_csv(test_path)
print(df_train.shape)

print(df_test.shape) 
df_train.head()
df_test.head() 
df_traintest=pd.concat([df_train,df_test])
df_traintest.shape
df_traintest.head()
def func(x):

    try:

        x_new=x["Country/Region"]+"/"+x["Province/State"]

    except:

        x_new=x["Country/Region"] 

    return str(x_new)    
df_traintest["place_id"]=df_traintest.apply(lambda x:func(x),axis=1)
df_traintest[888:890]     ################成功生成了国家和城市的id
tmp=np.sort(df_traintest["place_id"].unique())
print("一共有{}个地区的调查数据".format(len(tmp)))  

print(tmp[:10])
df_traintest["Date"]=pd.to_datetime(df_traintest["Date"])          ###############将天的数据转换
df_traintest.head()                  ################################训练测试集
df_traintest["day"]=df_traintest["Date"].apply(lambda x: x.dayofyear).astype(np.int16)

df_traintest.head()                     ##########################将日期排序
df_traintest[:-10] 
df_traintest[df_traintest["place_id"]=="Afghanistan"]  ######################各地区一直到4.23的数据
import copy  #######新的模块,deepcopy是完全独立的复制

places=df_traintest["place_id"].unique()########对每一个地区进行新生追踪
df_traintest2=copy.deepcopy(df_traintest)

df_traintest2["cases/day"]=0

df_traintest2["fatal/day"]=0

for place in places:

    tmp=df_traintest2["ConfirmedCases"][df_traintest2["place_id"]==place].values

    tmp[1:]=tmp[1:]-tmp[:-1]   #####################每天的新增确诊数目是隔日之差

    df_traintest2["cases/day"][df_traintest2["place_id"]==place]=tmp

    tmp=df_traintest2["Fatalities"][df_traintest2["place_id"]==place].values

    tmp[1:]=tmp[1:]-tmp[:-1]

    df_traintest2["fatal/day"][df_traintest2["place_id"]==place]=tmp
df_traintest2[df_traintest2["place_id"]=="China/Hubei"].head()   ##########得到了湖北的感染以及致命人数变化
def df_aggregation(df,col,mean_range):

    df_new=copy.deepcopy(df)  

    col_new='{}-({}-{})'.format(col,mean_range[0],mean_range[1])###############

    df_new[col_new]=0

    tmp=df_new[col].rolling(mean_range[1]-mean_range[0]+1).mean() #################都是每7天滚动求一次均值

    df_new[col_new][mean_range[0]:]=tmp[:-(mean_range[0])]    ##################手动延后时间序列

    df_new[col_new][pd.isna(df_new[col_new])]=0       

    return df_new[[col_new]].reset_index(drop=True)  #####################完全把原来的索引丢弃掉，设立新的数字索引
def do_aggregations(df):

    df=(pd.concat([df,df_aggregation(df,'cases/day',[1,1])],axis=1)).reset_index(drop=True)

    df=(pd.concat([df,df_aggregation(df,'cases/day',[1,7])],axis=1)).reset_index(drop=True)

    df=(pd.concat([df,df_aggregation(df,'cases/day',[8,14])],axis=1)).reset_index(drop=True)

    df=(pd.concat([df,df_aggregation(df,'cases/day',[15,21])],axis=1)).reset_index(drop=True)

    df=(pd.concat([df,df_aggregation(df,'fatal/day',[1,1])],axis=1)).reset_index(drop=True)

    df=(pd.concat([df,df_aggregation(df,'fatal/day',[1,7])],axis=1)).reset_index(drop=True)

    df=(pd.concat([df,df_aggregation(df,'fatal/day',[8,14])],axis=1)).reset_index(drop=True)

    df=(pd.concat([df,df_aggregation(df,'fatal/day',[15,21])],axis=1)).reset_index(drop=True)

    for threshold in[1,10,100]: ################设立不同的阈值求和

        days_under_threshold=(df['ConfirmedCases']<threshold).sum()

        tmp=df["day"]-22-days_under_threshold

        tmp[tmp<0]=0                            ###########照顾到22号之前已经爆发的地区，比如中国湖北

        df["days_since_{}cases".format(threshold)]=tmp

    for threshold in [1, 10, 100]:

        days_under_threshold = (df['Fatalities']<threshold).sum()

        tmp = df['day'].values - 22 - days_under_threshold

        tmp[tmp<=0] = 0

        df['days_since_{}fatal'.format(threshold)] = tmp

    if df['place_id'][0]=='China/Hubei':             #################湖北爆发时间比其他地区早，需要特别调整

        df['days_since_1cases'] += 35 # 2019/12/8

        df['days_since_10cases'] += 35-13 # 2019/12/8-2020/1/2 assume 2019/12/8+13

        df['days_since_100cases'] += 4 # 2020/1/18

        df['days_since_1fatal'] += 13 # 2020/1/9

    return df
df_traintest3=[]

for place in places:

    df_tmp=df_traintest2[df_traintest2["place_id"]==place].reset_index(drop=True)

    df_tmp=do_aggregations(df_tmp)

    df_traintest3.append(df_tmp)

df_traintest3=pd.concat(df_traintest3).reset_index(drop=True)

df_traintest3[df_traintest3["place_id"]=="China/Hubei"].head()
smoke_path=r"../input/smokingstats/share-of-adults-who-smoke.csv"
df_smoking=pd.read_csv(smoke_path)

df_smoking.head()
df_smoking_recent=df_smoking.sort_values("Year",ascending=False).reset_index(drop=True) 

####################同个地区多个年份的数据重复

df_smoking_recent=df_smoking_recent[df_smoking_recent["Entity"].duplicated()==False]

###########################改了两列的名字,没什么大变动,方便之后的连接

df_smoking_recent['Country/Region'] = df_smoking_recent['Entity']

df_smoking_recent['SmokingRate'] = df_smoking_recent['Smoking prevalence, total (ages 15+) (% of adults)']

df_smoking_recent.head()
df_traintest4 = pd.merge(df_traintest3,df_smoking_recent[["Country/Region","SmokingRate"]],on="Country/Region",how="left")

print(df_traintest4.shape)
df_traintest4[df_traintest4["place_id"]=="China/Hubei"].head()
SmokingRate=df_smoking_recent["SmokingRate"][df_smoking_recent["Entity"]=="World"].values[0]

df_traintest4["SmokingRate"][pd.isna(df_traintest4["SmokingRate"])]=SmokingRate

df_traintest4.head()
smoke_path=r"../input/smokingstats/WEO.csv"
df_weo=pd.read_csv(smoke_path)
print(df_weo['Subject Descriptor'].unique())
subs=df_weo["Subject Descriptor"].unique()[:-1]  ###########去掉最后一个空缺值

df_weo_agg=df_weo[["Country"]][df_weo["Country"].duplicated()==False].reset_index(drop=True)

for sub in subs[:]:

    df_tmp=df_weo[["Country","2019"]][df_weo["Subject Descriptor"]==sub].reset_index(drop=True)

    df_tmp=df_tmp[df_tmp["Country"].duplicated()==False].reset_index(drop=True)

    df_tmp.columns=["Country",sub]          ##############把表头的2019改了

    df_weo_agg=df_weo_agg.merge(df_tmp,on="Country",how="left")

df_weo_agg.columns=["".join(c if c.isalnum() else "_" for c in str(x))for x in df_weo_agg.columns]

df_weo_agg.columns

df_weo_agg['Country/Region'] = df_weo_agg['Country']

df_weo_agg.head()        #####################各个经济指标的数据
df_traintest5 = pd.merge(df_traintest4, df_weo_agg, on='Country/Region', how='left')

print(df_traintest5.shape)

df_traintest5.head()      #####################有空缺值很正常，很粗的数据
life_path=r"../input/smokingstats/Life expectancy at birth.csv"

df_life=pd.read_csv(life_path)
df_life.head()
tmp=df_life.iloc[:,1].values.tolist()

df_life=df_life[["Country","2018"]]

def func(x):

    try:

        x_new=float(x.replace(",",""))

    except:

        print(x)

        x_new=np.nan 

    return x_new

df_life["2018"]=df_life["2018"].apply(lambda x:func(x))

df_life.head()
df_life = df_life[['Country', '2018']]

df_life.columns = ['Country/Region', 'LifeExpectancy']  #############表头转换
df_traintest6 = pd.merge(df_traintest5, df_life, on='Country/Region', how='left')

print(len(df_traintest6))

df_traintest6.head()
country_path=r"../input/countryinfo/covid19countryinfo.csv"

df_country=pd.read_csv(country_path)[["country","pop","tests","testpop","density","medianage","urbanpop","quarantine","schools","hospibed","smokers","sex0","sex14","sex25","sex54","sex64","sex65plus","sexratio","lung","femalelung","malelung"]]

df_country.head()
df_country["femalelung"][df_country["country"]=="China"] ####################中国，中国香港，中国澳门，湖北特别疫情地区
df_country["Country/Region"]=df_country["country"]

df_country=df_country[df_country["country"].duplicated()==False]    ############把不同的称呼的同一地区合并
print(df_country[df_country['country'].duplicated()].shape)    ############确认无重复
df_country["femalelung"][df_country["Country/Region"]=="China"]       #####中国地区数据变成了香港的NaN

####手动删除香港  
df_country["femalelung"][df_country["Country/Region"]=="China"]    ###########正常数据
df_traintest7=pd.merge(df_traintest6,df_country.drop(["country","testpop","tests"],axis=1),on=["Country/Region"],how="left")

print(df_traintest7.shape)

df_traintest7.head()

df_traintest7[df_traintest7["Country/Region"]=="China"]
df_traintest7[df_traintest7["place_id"]=="China/Hubei"].head()
def encode_label(df, col, freq_limit=0):

    df[col][pd.isna(df[col])] = 'nan'

    tmp = df[col].value_counts()

    cols = tmp.index.values               ######################cols是索引值

    freq = tmp.values                   ######################freq是对应的值

    num_cols = (freq>=freq_limit).sum()

    print("col: {}, num_cat: {}, num_reduced: {}".format(col, len(cols), num_cols))

    col_new = '{}_le'.format(col)

    df_new = pd.DataFrame(np.ones(len(df), np.int16)*(num_cols-1), columns=[col_new])

    for i, item in enumerate(cols[:num_cols]):

        df_new[col_new][df[col]==item] = i

    return df_new



def get_df_le(df, col_index, col_cat):

    df_new = df[[col_index]]

    for col in col_cat:

        df_tmp = encode_label(df, col)

        df_new = pd.concat([df_new, df_tmp], axis=1)

    return df_new



df_traintest7['id'] = np.arange(len(df_traintest7))

df_le = get_df_le(df_traintest7, 'id', ['Country/Region', 'Province/State'])

df_traintest8 = pd.merge(df_traintest7, df_le, on='id', how='left')
df_traintest8['cases/day'] = df_traintest8['cases/day'].astype(np.float)

df_traintest8['fatal/day'] = df_traintest8['fatal/day'].astype(np.float)
# covert object type to float

def func(x):

    x_new = 0

    try:

        x_new = float(x.replace(",", ""))

    except:

        x_new = np.nan

    return x_new

cols = [

    'Gross_domestic_product__constant_prices', 

    'Gross_domestic_product__current_prices', 

    'Gross_domestic_product__deflator', 

    'Gross_domestic_product_per_capita__constant_prices', 

    'Gross_domestic_product_per_capita__current_prices', 

    'Output_gap_in_percent_of_potential_GDP', 

    'Gross_domestic_product_based_on_purchasing_power_parity__PPP__valuation_of_country_GDP', 

    'Gross_domestic_product_based_on_purchasing_power_parity__PPP__per_capita_GDP', 

    'Gross_domestic_product_based_on_purchasing_power_parity__PPP__share_of_world_total', 

    'Implied_PPP_conversion_rate', 'Total_investment', 

    'Gross_national_savings', 'Inflation__average_consumer_prices', 

    'Inflation__end_of_period_consumer_prices', 

    'Six_month_London_interbank_offered_rate__LIBOR_', 

    'Volume_of_imports_of_goods_and_services', 

    'Volume_of_Imports_of_goods', 

    'Volume_of_exports_of_goods_and_services', 

    'Volume_of_exports_of_goods', 'Unemployment_rate', 'Employment', 'Population', 

    'General_government_revenue', 'General_government_total_expenditure', 

    'General_government_net_lending_borrowing', 'General_government_structural_balance', 

    'General_government_primary_net_lending_borrowing', 'General_government_net_debt', 

    'General_government_gross_debt', 'Gross_domestic_product_corresponding_to_fiscal_year__current_prices', 

    'Current_account_balance', 'pop'

]

for col in cols:

    df_traintest8[col] = df_traintest8[col].apply(lambda x: func(x))  

print(df_traintest8['pop'].dtype)
df_traintest8[df_traintest8['place_id']=='China/Hubei'].head()
df_traintest8.to_csv("final_data.csv")
#######################到这一步数据处理初步完成

################首先得到一个模型的指标

################metrics.mean_squared_error文档：https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error

###############仔细讲解下面的函数：metric是sk中的一个指标类,mean_squared_error平均回归偏误

###############numpy.clip函数控制预测的数据范围，可能模型抛出一个无穷大的值或负值，要截取预测值的上下限，

from sklearn import metrics

def calc_score(y_true, y_pred):

    y_true[y_true<0] = 0

    score = metrics.mean_squared_error(np.log(y_true.clip(0, 1e10)+1), np.log(y_pred[:]+1))**0.5

    return score
import lightgbm as lgb
SEED = 42

params = {'num_leaves': 8, ###########叶节点数目

          'min_data_in_leaf': 5,  # 42, ################每个叶子最少的样本数：减少过拟合

          'objective': 'regression',############目标：回归拟合

          'max_depth': 8,  ###############深度限制防止过拟合

          'learning_rate': 0.02,    ############学习速率

          'boosting': 'gbdt',         #############梯度下降算法

          'bagging_freq': 5,  # 5           ##########每迭代5次做一次重新取样

          'bagging_fraction': 0.8,  # 0.5,  ############每次迭代重新取样的比例

          'feature_fraction': 0.8201,      #########每次迭代选择键树的参数数目

          'bagging_seed': SEED,       ###########随机种子数

          'reg_alpha': 1,  # 1.728910519108444,       ############正则化项参数

          'reg_lambda': 4.9847051755586085,     #############正则化部分参数

          'random_state': SEED,     ##############我的理解是随机种子数

          'metric': 'mse',      #############模型自我评价的指标:mse: mean squared error

          'verbosity': 100,          ########实在不懂,显示训练资讯详细程度(0~3)，默认：1,就是最后训练结果的报告信息

          'min_gain_to_split': 0.02,  # 0.01077313523861969,      ##############分裂的最小信息增益？？？应该也是防止过拟合

          'min_child_weight': 5,  # 19.428902804238373,   ################叶节点样本权重小于此值，停止分列，防止过拟合

          'num_threads': 6,         ############线程数目，与训练速度有关

          }
# train model to predict fatalities/day

col_target = 'fatal/day'

col_var = [

    'Lat', 'Long',                  #####保留经纬数据？

    'days_since_1cases', 

    'days_since_10cases', 

    'days_since_100cases',

    'days_since_1fatal', 

    'days_since_10fatal', 'days_since_100fatal',

 ################################下面是一组十分重要的变量，我认为他在拟合不同的传染阶段，人为调整爆发期

    'cases/day-(1-1)',  ####################第一天的确诊人数

    'cases/day-(1-7)',  ################从第七天开始统计的滚动确诊每日平均数目

    'cases/day-(8-14)',   #################从第八天开始统计的滚动确诊每日平均数目

    'cases/day-(15-21)',  ##################同理

    'fatal/day-(1-1)', 

    'fatal/day-(1-7)',         #########同理

    'fatal/day-(8-14)', 

    'fatal/day-(15-21)', 

    'SmokingRate',

    'Gross_domestic_product__constant_prices',

    'Gross_domestic_product__current_prices',

    'Gross_domestic_product__deflator',

    'Gross_domestic_product_per_capita__constant_prices',

    'Gross_domestic_product_per_capita__current_prices',

    'Output_gap_in_percent_of_potential_GDP',

    'Gross_domestic_product_based_on_purchasing_power_parity__PPP__valuation_of_country_GDP',

    'Gross_domestic_product_based_on_purchasing_power_parity__PPP__per_capita_GDP',

    'Gross_domestic_product_based_on_purchasing_power_parity__PPP__share_of_world_total',

    'Implied_PPP_conversion_rate', 'Total_investment',

    'Gross_national_savings', 'Inflation__average_consumer_prices',

    'Inflation__end_of_period_consumer_prices',

    'Six_month_London_interbank_offered_rate__LIBOR_',

    'Volume_of_imports_of_goods_and_services', 'Volume_of_Imports_of_goods',

    'Volume_of_exports_of_goods_and_services', 'Volume_of_exports_of_goods',

    'Unemployment_rate',                            ###############失业率

    'Employment', 'Population',

    'General_government_revenue', 'General_government_total_expenditure',

    'General_government_net_lending_borrowing',

    'General_government_structural_balance',

    'General_government_primary_net_lending_borrowing',

    'General_government_net_debt', 'General_government_gross_debt',

    'Gross_domestic_product_corresponding_to_fiscal_year__current_prices',

    'Current_account_balance', 

    'LifeExpectancy',

    'pop',

    'density', 

    'medianage', 

    'urbanpop', 

    'hospibed', 'smokers',

]

col_cat = []

#####################当然是划分训练集和验证集

#####################日期序号为61天前的数据作为训练集合

#####################日期学号为61-72天的数据作为验证集合

df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<61)]

df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']>=61) & (df_traintest8['day']<72)]

df_test = df_traintest8[pd.isna(df_traintest8['ForecastId'])==False]

X_train = df_train[col_var]

X_valid = df_valid[col_var]

#######################为什么要取对数：取对数可以缩小特别大特别小的数据之间的差别，比如1和e10,相当于排除了特殊点？

#######################最后得到的最佳参数肯定是不变的

y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)

#######################lgb封装自身的训练集

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)

valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)

num_round = 15000

model = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data],

                  verbose_eval=100, ###########每100次输出一次评测结果

                  early_stopping_rounds=150,)      #############如果连续150轮都无法优化，那么就提前停下



best_itr = model.best_iteration
y_true = df_valid['fatal/day'].values

y_pred = np.exp(model.predict(X_valid))-1

score = calc_score(y_true, y_pred)

print("{:.6f}".format(score))       ###################越大越差
tmp = pd.DataFrame()

tmp["feature"] = col_var

tmp["importance"] = model.feature_importance()

tmp = tmp.sort_values('importance', ascending=False)

tmp[:10]
col_var = [

#     'Lat', 'Long',                  #####保留经纬数据？

    'days_since_1cases', 

    'days_since_10cases', 

    'days_since_100cases',

    'days_since_1fatal', 

    'days_since_10fatal', 

#     'days_since_100fatal',

#  ################################下面是一组十分重要的变量，拟合不同的传染阶段，人为调整爆发期

    'cases/day-(1-1)',  ####################第一天的确诊人数

    'cases/day-(1-7)',  ################从第七天开始统计的滚动确诊每日平均数目

    'cases/day-(8-14)',   #################从第八天开始统计的滚动确诊每日平均数目

#     'cases/day-(15-21)',  ##################同理

    'fatal/day-(1-1)', 

    'fatal/day-(1-7)',         #########同理

]
col_cat = []

#####################当然是划分训练集和验证集

#####################日期序号为61天前的数据作为训练集合

#####################日期学号为61-72天的数据作为验证集合

df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<61)]

df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']>=61) & (df_traintest8['day']<72)]

df_test = df_traintest8[pd.isna(df_traintest8['ForecastId'])==False]

X_train = df_train[col_var]

X_valid = df_valid[col_var]

#######################为什么要取对数：取对数可以缩小特别大特别小的数据之间的差别，比如1和e10,相当于排除了特殊点？

#######################最后得到的最佳参数肯定是不变的

y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)

#######################lgb封装自身的训练集

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)

valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)

num_round = 15000

model = lgb.train(params, train_data, num_round, 

                  valid_sets=[train_data, valid_data],

                  verbose_eval=100, ###########每100次输出一次评测结果

                  early_stopping_rounds=150,)      #############如果连续150轮都无法优化，那么就提前停下



best_itr = model.best_iteration
tmp = pd.DataFrame()

tmp["feature"] = col_var

tmp["importance"] = model.feature_importance()

tmp = tmp.sort_values('importance', ascending=False)

tmp[:10]
y_true = df_valid['fatal/day'].values

y_pred = np.exp(model.predict(X_valid))-1

score = calc_score(y_true, y_pred)

print("{:.6f}".format(score)) ####################只是好了一点点，这代表前面几个变量是解释的因变量中主力军
col_var = [

#     'Lat', 'Long',                  #####保留经纬数据？

#     'days_since_1cases', 

#     'days_since_10cases', 

#     'days_since_100cases',

#     'days_since_1fatal', 

#     'days_since_10fatal', 

#     'days_since_100fatal',

#  ################################下面是一组十分重要的变量，我认为他在拟合不同的传染阶段，人为调整爆发期

    'cases/day-(1-1)',  ####################第一天的确诊人数

    'cases/day-(1-7)',  ################从第七天开始统计的滚动确诊每日平均数目

    'cases/day-(8-14)',   #################从第八天开始统计的滚动确诊每日平均数目

    'cases/day-(15-21)',  ##################同理

    'fatal/day-(1-1)', 

    'fatal/day-(1-7)',         #########同理保留了所有的趋势数据

    'fatal/day-(8-14)', 

    'fatal/day-(15-21)', 

]
col_cat = []

#####################当然是划分训练集和验证集

#####################日期序号为61天前的数据作为训练集合

#####################日期学号为61-72天的数据作为验证集合

df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<61)]

df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']>=61) & (df_traintest8['day']<72)]

df_test = df_traintest8[pd.isna(df_traintest8['ForecastId'])==False]

X_train = df_train[col_var]

X_valid = df_valid[col_var]

#######################为什么要取对数：取对数可以缩小特别大特别小的数据之间的差别，比如1和e10,相当于排除了特殊点？

#######################最后得到的最佳参数肯定是不变的

y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)

#######################lgb封装自身的训练集

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)

valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)

num_round = 15000

model = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data],

                  verbose_eval=100, ###########每100次输出一次评测结果

                  early_stopping_rounds=150,)      #############如果连续150轮都无法优化，那么就提前停下



best_itr = model.best_iteration
y_true = df_valid['fatal/day'].values

y_pred = np.exp(model.predict(X_valid))-1

score = calc_score(y_true, y_pred)

print("{:.6f}".format(score)) ####################变得好很多,继续引入更加多的参数
col_var = [

#     'Lat', 'Long',                  #####保留经纬数据？

    'days_since_1cases', 

    'days_since_10cases', 

    'days_since_100cases',

    'days_since_1fatal', 

    'days_since_10fatal', 

    'days_since_100fatal',

#  ################################下面是一组十分重要的变量，我认为他在拟合不同的传染阶段，人为调整爆发期

#     'cases/day-(1-1)',  ####################第一天的确诊人数

#     'cases/day-(1-7)',  ################从第七天开始统计的滚动确诊每日平均数目

#     'cases/day-(8-14)',   #################从第八天开始统计的滚动确诊每日平均数目

#     'cases/day-(15-21)',  ##################同理

#     'fatal/day-(1-1)', 

#     'fatal/day-(1-7)',         #########同理保留了所有的趋势数据

#     'fatal/day-(8-14)', 

#     'fatal/day-(15-21)', 

]
col_cat = []

#####################当然是划分训练集和验证集

#####################日期序号为61天前的数据作为训练集合

#####################日期学号为61-72天的数据作为验证集合

df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<61)]

df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']>=61) & (df_traintest8['day']<72)]

df_test = df_traintest8[pd.isna(df_traintest8['ForecastId'])==False]

X_train = df_train[col_var]

X_valid = df_valid[col_var]

#######################为什么要取对数：取对数可以缩小特别大特别小的数据之间的差别，比如1和e10,相当于排除了特殊点？

#######################最后得到的最佳参数肯定是不变的

y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)

#######################lgb封装自身的训练集

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)

valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)

num_round = 15000

model = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data],

                  verbose_eval=100, ###########每100次输出一次评测结果

                  early_stopping_rounds=150,)      #############如果连续150轮都无法优化，那么就提前停下



best_itr = model.best_iteration
y_true = df_valid['fatal/day'].values

y_pred = np.exp(model.predict(X_valid))-1

score = calc_score(y_true, y_pred)

print("{:.6f}".format(score)) ####################变得好很多,继续引入更加多的参数
col_var = [

#     'Lat', 'Long',                  #####需要保留经纬数据吗，经度纬度可能影响气候？

#  ################################下面是一组十分重要的变量，我认为他在拟合不同的传染阶段，人为调整爆发期

    'cases/day-(1-1)',  ####################第一天的确诊人数

    'cases/day-(1-7)',  ################从第七天开始统计的滚动确诊每日平均数目

    'cases/day-(8-14)',   #################从第八天开始统计的滚动确诊每日平均数目

    'cases/day-(15-21)',  ##################同理

    'fatal/day-(1-1)', 

    'fatal/day-(1-7)',         #########同理

     'SmokingRate',

    'Gross_domestic_product__constant_prices',

    'LifeExpectancy',

    'pop',

    'density', 

    'medianage', 

    'urbanpop', 

    'hospibed', 'smokers'

]
col_cat = []

#####################当然是划分训练集和验证集

#####################日期序号为61天前的数据作为训练集合

#####################日期学号为61-72天的数据作为验证集合

df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<61)]

df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']>=61) & (df_traintest8['day']<72)]

df_test = df_traintest8[pd.isna(df_traintest8['ForecastId'])==False]

X_train = df_train[col_var]

X_valid = df_valid[col_var]

#######################为什么要取对数：取对数可以缩小特别大特别小的数据之间的差别，比如1和e10,相当于排除了特殊点？

#######################最后得到的最佳参数肯定是不变的

y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)

#######################lgb封装自身的训练集

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)

valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)

num_round = 15000

model = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data],

                  verbose_eval=100, ###########每100次输出一次评测结果

                  early_stopping_rounds=150,)      #############如果连续150轮都无法优化，那么就提前停下



best_itr = model.best_iteration
y_true = df_valid['fatal/day'].values

##################记住之前加了一个1

y_pred = np.exp(model.predict(X_valid))-1

score = calc_score(y_true, y_pred)

print("{:.6f}".format(score)) ####################难受，更差了
tmp = pd.DataFrame()

tmp["feature"] = col_var

tmp["importance"] = model.feature_importance()

tmp = tmp.sort_values('importance', ascending=False)

tmp[:10]
col_var = [

    'Lat', 'Long',                  #####需要保留经纬数据吗，经度纬度可能影响气候？试一下

#  ################################下面是一组十分重要的变量，我认为他在拟合不同的传染阶段，人为调整爆发期

    'cases/day-(1-1)',  ####################第一天的确诊人数

    'cases/day-(1-7)',  ################从第七天开始统计的滚动确诊每日平均数目

#     'cases/day-(8-14)',   #################从第八天开始统计的滚动确诊每日平均数目

#     'cases/day-(15-21)',##################去掉这个最差劲的趋势数据，试一下结果会不会好一点，太长的周期可能不会起作用，换为致命的15-21滚动

   'fatal/day-(1-7)',

    'fatal/day-(8-14)', 

    'fatal/day-(15-21)',         #########同理

     'SmokingRate',

     'density', 

]
col_cat = []

df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<61)]

df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']>=61) & (df_traintest8['day']<72)]

df_test = df_traintest8[pd.isna(df_traintest8['ForecastId'])==False]

X_train = df_train[col_var]

X_valid = df_valid[col_var]

y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)

valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)

num_round = 15000

model = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data],

                  verbose_eval=100,

                  early_stopping_rounds=150,)



best_itr = model.best_iteration
y_true = df_valid['fatal/day'].values

y_pred = np.exp(model.predict(X_valid))-1

score = calc_score(y_true, y_pred)

print("{:.6f}".format(score))
tmp = pd.DataFrame()

tmp["feature"] = col_var

tmp["importance"] = model.feature_importance()

tmp = tmp.sort_values('importance', ascending=False)

tmp[:10]          
col_cat = []

#####################当然是划分训练集和验证集

#####################日期序号为61天前的数据作为训练集合

#####################日期学号为61-72天的数据作为验证集合

df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<61)]

df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']>=61) & (df_traintest8['day']<72)]

df_test = df_traintest8[pd.isna(df_traintest8['ForecastId'])==False]

X_train = df_train[col_var] 

X_valid = df_valid[col_var]

#######################为什么要取对数：取对数可以缩小特别大特别小的数据之间的差别，比如1和e10,相当于排除了特殊点？

#######################最后得到的最佳参数肯定是不变的

y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)

#######################lgb封装自身的训练集

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)

valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)

num_round = 15000

model = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data],

                  verbose_eval=100, ###########每100次输出一次评测结果

                  early_stopping_rounds=150,)      #############如果连续150轮都无法优化，那么就提前停下



best_itr = model.best_iteration
y_true = df_valid['fatal/day'].values

y_pred = np.exp(model.predict(X_valid))-1

score = calc_score(y_true, y_pred)

print("{:.6f}".format(score))
df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<72)]

df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<72)]

df_test = df_traintest8[pd.isna(df_traintest8['ForecastId'])==False]

X_train = df_train[col_var]

X_valid = df_valid[col_var]

y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)

valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)

model = lgb.train(params, train_data, best_itr, valid_sets=[train_data, valid_data],

                  verbose_eval=100,

                  early_stopping_rounds=150,)
col_target2 = 'cases/day'

col_var2 = [

    'Lat', 'Long',

    'cases/day-(1-1)', 

    'cases/day-(1-7)', 

    'cases/day-(8-14)',  

    'cases/day-(15-21)', 

    'days_since_10cases'

]
col_cat = []

df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<61)]

df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']>=61) & (df_traintest8['day']<72)]

# df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']>=61)]

df_test = df_traintest8[pd.isna(df_traintest8['ForecastId'])==False]

X_train = df_train[col_var2]

X_valid = df_valid[col_var2]

y_train = np.log(df_train[col_target2].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target2].values.clip(0, 1e10)+1)

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)

valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)

model2 = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data],

                  verbose_eval=100,

                  early_stopping_rounds=150,)

best_itr = model2.best_iteration
y_true = df_valid['cases/day'].values

y_pred = np.exp(model2.predict(X_valid))-1

score = calc_score(y_true, y_pred)

print("{:.6f}".format(score))                ##################还可以
df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<72)]

df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<72)]

# df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']>=61)]

df_test = df_traintest8[pd.isna(df_traintest8['ForecastId'])==False]

X_train = df_train[col_var2]

X_valid = df_valid[col_var2]

y_train = np.log(df_train[col_target2].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target2].values.clip(0, 1e10)+1)

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)

valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)

model2 = lgb.train(params, train_data, best_itr, valid_sets=[train_data, valid_data],

                  verbose_eval=100,

                  early_stopping_rounds=150,)
y_true = df_valid['cases/day'].values

y_pred = np.exp(model2.predict(X_valid))-1

score = calc_score(y_true, y_pred)

print("{:.6f}".format(score))
df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<72)]

df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']>=72)]

df_test = df_traintest8[pd.isna(df_traintest8['ForecastId'])==False]

X_train = df_train[col_var]

X_valid = df_valid[col_var]

y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)

valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)

model_pri = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data],

                  verbose_eval=100,

                  early_stopping_rounds=150,)

best_itr = model_pri.best_iteration
df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId']))]

df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId']))]

# df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']>=61)]

df_test = df_traintest8[pd.isna(df_traintest8['ForecastId'])==False]

X_train = df_train[col_var]

X_valid = df_valid[col_var]

y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)

valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)

model_pri = lgb.train(params, train_data, best_itr, valid_sets=[train_data, valid_data],

                  verbose_eval=100,

                  early_stopping_rounds=150,)
# train model to predict fatalities/day

df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<72)]

df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']>=72)]

# df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']>=61)]

df_test = df_traintest8[pd.isna(df_traintest8['ForecastId'])==False]

X_train = df_train[col_var2]

X_valid = df_valid[col_var2]

y_train = np.log(df_train[col_target2].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target2].values.clip(0, 1e10)+1)

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)

valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)

model2_pri = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data],

                  verbose_eval=100,

                  early_stopping_rounds=150,)

best_itr = model2_pri.best_iteration
df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId']))]

df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId']))]

# df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']>=61)]

df_test = df_traintest8[pd.isna(df_traintest8['ForecastId'])==False]

X_train = df_train[col_var2]

X_valid = df_valid[col_var2]

y_train = np.log(df_train[col_target2].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target2].values.clip(0, 1e10)+1)

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)

valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)

model2_pri = lgb.train(params, train_data, best_itr, valid_sets=[train_data, valid_data],

                  verbose_eval=100,

                  early_stopping_rounds=150,)

best_itr = model2_pri.best_iteration
df_tmp = df_traintest8[(df_traintest8['day']<72) | (pd.isna(df_traintest8['ForecastId'])==False)].reset_index(drop=True)

################################删除之前聚合的数据项

df_tmp = df_tmp.drop([

    'cases/day-(1-1)', 'cases/day-(1-7)', 'cases/day-(8-14)', 'cases/day-(15-21)', 

    'fatal/day-(1-1)', 'fatal/day-(1-7)', 'fatal/day-(8-14)', 'fatal/day-(15-21)',

    'days_since_1cases', 'days_since_10cases', 'days_since_100cases',

    'days_since_1fatal', 'days_since_10fatal', 'days_since_100fatal',

                               ],  axis=1)

df_traintest9 = []

######重新聚合

for i, place in enumerate(places[:]):

    df_tmp2 = df_tmp[df_tmp['place_id']==place].reset_index(drop=True)

    df_tmp2 = do_aggregations(df_tmp2)

    df_traintest9.append(df_tmp2)

df_traintest9 = pd.concat(df_traintest9).reset_index(drop=True)

df_traintest9[df_traintest9['day']>68].head()
########################怎么说呢，训练集很准，到真正测试集合上就不行了

import seaborn as sns#################这是我们以前python学过的包
###############注意啊，这是什么模型的曲线?

#############这当然是确诊数目的曲线，但是确诊我训练了两个模型

########这是哪个模型？

#########这是我们拟合72天内的模型啊

##########可以看到72天内效果及其优秀，72天后呵呵

place = 'Iran'

df_interest_base = df_traintest9[df_traintest9['place_id']==place].reset_index(drop=True)

df_interest = copy.deepcopy(df_interest_base)

df_interest['ConfirmedCases'] = df_interest['ConfirmedCases'].astype(np.float)

df_interest['cases/day'] = df_interest['cases/day'].astype(np.float)

df_interest['fatal/day'] = df_interest['fatal/day'].astype(np.float)

df_interest['Fatalities'] = df_interest['Fatalities'].astype(np.float)

df_interest['cases/day'][df_interest['day']>=72] = -1

df_interest['fatal/day'][df_interest['day']>=72] = -1

len_known = (df_interest['cases/day']!=-1).sum()

len_unknown = (df_interest['cases/day']==-1).sum()

print("len train: {}, len prediction: {}".format(len_known, len_unknown))

X_valid = df_interest[col_var][df_interest['day']>=72]

X_valid2 = df_interest[col_var2][df_interest['day']>=72]

pred_f =  np.exp(model.predict(X_valid))-1

pred_c = np.exp(model2.predict(X_valid2))-1

df_interest['fatal/day'][df_interest['day']>=72] = pred_f.clip(0, 1e10)

df_interest['cases/day'][df_interest['day']>=72] = pred_c.clip(0, 1e10)

df_interest['Fatalities'] = np.cumsum(df_interest['fatal/day'].values)

df_interest['ConfirmedCases'] = np.cumsum(df_interest['cases/day'].values)

for j in range(len_unknown): # use predicted cases and fatal for next days' prediction

    X_valid = df_interest[col_var].iloc[j+len_known]

    X_valid2 = df_interest[col_var2].iloc[j+len_known]

    pred_f = model.predict(X_valid)

    pred_c = model2.predict(X_valid2)

    pred_c = (np.exp(pred_c)-1).clip(0, 1e10)

    pred_f = (np.exp(pred_f)-1).clip(0, 1e10)

    df_interest['fatal/day'][j+len_known] = pred_f

    df_interest['cases/day'][j+len_known] = pred_c

    df_interest['Fatalities'][j+len_known] = df_interest['Fatalities'][j+len_known-1] + pred_f

    df_interest['ConfirmedCases'][j+len_known] = df_interest['ConfirmedCases'][j+len_known-1] + pred_c

    df_interest = df_interest.drop([

        'cases/day-(1-1)', 'cases/day-(1-7)', 'cases/day-(8-14)', 'cases/day-(15-21)', 

        'fatal/day-(1-1)', 'fatal/day-(1-7)', 'fatal/day-(8-14)', 'fatal/day-(15-21)', 

        'days_since_1cases', 'days_since_10cases', 'days_since_100cases',

        'days_since_1fatal', 'days_since_10fatal', 'days_since_100fatal',], axis=1)

    df_interest = do_aggregations(df_interest.reset_index(drop=True))

# visualize

tmp = df_interest['cases/day'].values

tmp = np.cumsum(tmp)

sns.lineplot(x=df_interest_base['day'], y=tmp, label='pred')

tmp = df_traintest8['ConfirmedCases'][(df_traintest8['place_id']==place)& (pd.isna(df_traintest8['ForecastId']))].values

print(len(tmp), tmp)

sns.lineplot(x=df_traintest8['day'][(df_traintest8['place_id']==place)& (pd.isna(df_traintest8['ForecastId']))].values,

             y=tmp, label='true')

print(place)

plt.show()
last_day_train = df_traintest8['day'][pd.isna(df_traintest8['ForecastId'])].max()

print(last_day_train)

df_tmp = df_traintest8[

    (pd.isna(df_traintest8['ForecastId'])) |

    ((df_traintest8['day']>last_day_train) & (pd.isna(df_traintest8['ForecastId'])==False))].reset_index(drop=True)

df_tmp = df_tmp.drop([

    'cases/day-(1-1)', 'cases/day-(1-7)', 'cases/day-(8-14)', 'cases/day-(15-21)', 

    'fatal/day-(1-1)', 'fatal/day-(1-7)', 'fatal/day-(8-14)', 'fatal/day-(15-21)',

    'days_since_1cases', 'days_since_10cases', 'days_since_100cases',

    'days_since_1fatal', 'days_since_10fatal', 'days_since_100fatal',

                               ],  axis=1)

df_traintest10 = []

############################再次聚合，记不记得，我们又动了数据了

for i, place in enumerate(places[:]):

    df_tmp2 = df_tmp[df_tmp['place_id']==place].reset_index(drop=True)

    df_tmp2 = do_aggregations(df_tmp2)

    df_traintest10.append(df_tmp2)

df_traintest10 = pd.concat(df_traintest10).reset_index(drop=True)

df_traintest10[df_traintest10['day']>last_day_train-5].head(10) 
place = places[np.random.randint(len(places))]

place = "Iran"

df_interest_base = df_traintest9[df_traintest9['place_id']==place].reset_index(drop=True)

df_interest = copy.deepcopy(df_interest_base)

df_interest['ConfirmedCases'] = df_interest['ConfirmedCases'].astype(np.float)

df_interest['cases/day'] = df_interest['cases/day'].astype(np.float)

df_interest['fatal/day'] = df_interest['fatal/day'].astype(np.float)

df_interest['Fatalities'] = df_interest['Fatalities'].astype(np.float)

df_interest['cases/day'][df_interest['day']>=72] = -1

df_interest['fatal/day'][df_interest['day']>=72] = -1

len_known = (df_interest['cases/day']!=-1).sum()

len_unknown = (df_interest['cases/day']==-1).sum()

print("len train: {}, len prediction: {}".format(len_known, len_unknown))

X_valid = df_interest[col_var][df_interest['day']>=72]

X_valid2 = df_interest[col_var2][df_interest['day']>=72]

pred_f =  np.exp(model.predict(X_valid))-1

pred_c = np.exp(model2.predict(X_valid2))-1

df_interest['fatal/day'][df_interest['day']>=72] = pred_f.clip(0, 1e10)

df_interest['cases/day'][df_interest['day']>=72] = pred_c.clip(0, 1e10)

df_interest['Fatalities'] = np.cumsum(df_interest['fatal/day'].values)

df_interest['ConfirmedCases'] = np.cumsum(df_interest['cases/day'].values)

for j in range(len_unknown): # use predicted cases and fatal for next days' prediction

    X_valid = df_interest[col_var].iloc[j+len_known]

    X_valid2 = df_interest[col_var2].iloc[j+len_known]

    pred_f = model.predict(X_valid)

    pred_c = model2.predict(X_valid2)

    pred_c = (np.exp(pred_c)-1).clip(0, 1e10)

    pred_f = (np.exp(pred_f)-1).clip(0, 1e10)

    df_interest['fatal/day'][j+len_known] = pred_f

    df_interest['cases/day'][j+len_known] = pred_c

    df_interest['Fatalities'][j+len_known] = df_interest['Fatalities'][j+len_known-1] + pred_f

    df_interest['ConfirmedCases'][j+len_known] = df_interest['ConfirmedCases'][j+len_known-1] + pred_c

    df_interest = df_interest.drop([

    'cases/day-(1-1)', 'cases/day-(1-7)', 'cases/day-(8-14)', 'cases/day-(15-21)', 

    'fatal/day-(1-1)', 'fatal/day-(1-7)', 'fatal/day-(8-14)', 'fatal/day-(15-21)',

    'days_since_1cases', 'days_since_10cases', 'days_since_100cases',

    'days_since_1fatal', 'days_since_10fatal', 'days_since_100fatal',

                                   

                                   ],  axis=1)

    df_interest = do_aggregations(df_interest.reset_index(drop=True))



# visualize

tmp = df_interest['fatal/day'].values

tmp = np.cumsum(tmp)

sns.lineplot(x=df_interest_base['day'], y=tmp, label='pred')

tmp = df_traintest8['Fatalities'][(df_traintest8['place_id']==place)& (pd.isna(df_traintest8['ForecastId']))].values

print(len(tmp), tmp)

sns.lineplot(x=df_traintest8['day'][(df_traintest8['place_id']==place)& (pd.isna(df_traintest8['ForecastId']))].values,

             y=tmp, label='true')

print(place)

plt.show()
place = 'Iran'

df_interest_base = df_traintest10[df_traintest10['place_id']==place].reset_index(drop=True)

df_interest = copy.deepcopy(df_interest_base)

df_interest['ConfirmedCases'] = df_interest['ConfirmedCases'].astype(np.float)

df_interest['cases/day'] = df_interest['cases/day'].astype(np.float)

df_interest['fatal/day'] = df_interest['fatal/day'].astype(np.float)

df_interest['Fatalities'] = df_interest['Fatalities'].astype(np.float)

df_interest['cases/day'][df_interest['day']>last_day_train] = -1

df_interest['fatal/day'][df_interest['day']>last_day_train] = -1

len_known = (df_interest['cases/day']!=-1).sum()

len_unknown = (df_interest['cases/day']==-1).sum()

print("len train: {}, len prediction: {}".format(len_known, len_unknown))

X_valid = df_interest[col_var][df_interest['day']>84]

X_valid2 = df_interest[col_var2][df_interest['day']>84]

pred_f =  np.exp(model.predict(X_valid))-1

pred_c = np.exp(model2.predict(X_valid2))-1

df_interest['fatal/day'][df_interest['day']>last_day_train] = pred_f.clip(0, 1e10)

df_interest['cases/day'][df_interest['day']>last_day_train] = pred_c.clip(0, 1e10)

df_interest['Fatalities'] = np.cumsum(df_interest['fatal/day'].values)

df_interest['ConfirmedCases'] = np.cumsum(df_interest['cases/day'].values)

for j in range(len_unknown): # use predicted cases and fatal for next days' prediction

    X_valid = df_interest[col_var].iloc[j+len_known]

    X_valid2 = df_interest[col_var2].iloc[j+len_known]

    pred_f = model_pri.predict(X_valid)

    pred_c = model2_pri.predict(X_valid2)

    pred_c = (np.exp(pred_c)-1).clip(0, 1e10)

    pred_f = (np.exp(pred_f)-1).clip(0, 1e10)

    df_interest['fatal/day'][j+len_known] = pred_f

    df_interest['cases/day'][j+len_known] = pred_c

    df_interest['Fatalities'][j+len_known] = df_interest['Fatalities'][j+len_known-1] + pred_f

    df_interest['ConfirmedCases'][j+len_known] = df_interest['ConfirmedCases'][j+len_known-1] + pred_c

    df_interest = df_interest.drop([

         'cases/day-(1-1)', 'cases/day-(1-7)', 'cases/day-(8-14)', 'cases/day-(15-21)', 

    'fatal/day-(1-1)', 'fatal/day-(1-7)', 'fatal/day-(8-14)', 'fatal/day-(15-21)',

    'days_since_1cases', 'days_since_10cases', 'days_since_100cases',

    'days_since_1fatal', 'days_since_10fatal', 'days_since_100fatal',],  axis=1)

    df_interest = do_aggregations(df_interest.reset_index(drop=True))



# visualize

tmp = df_interest['cases/day'].values

tmp = np.cumsum(tmp)

sns.lineplot(x=df_interest_base['day'], y=tmp, label='pred')

tmp = df_traintest10['ConfirmedCases'][(df_traintest10['place_id']==place)& (pd.isna(df_traintest10['ForecastId']))].values

print(len(tmp), tmp)

sns.lineplot(x=df_traintest10['day'][(df_traintest10['place_id']==place)& (pd.isna(df_traintest10['ForecastId']))].values,

             y=tmp, label='true')

print(place)

plt.show()
place = 'Bhutan'

place = places[np.random.randint(len(places))]

# place = 'Iran'

df_interest_base = df_traintest10[df_traintest10['place_id']==place].reset_index(drop=True)

df_interest = copy.deepcopy(df_interest_base)

df_interest['ConfirmedCases'] = df_interest['ConfirmedCases'].astype(np.float)

df_interest['cases/day'] = df_interest['cases/day'].astype(np.float)

df_interest['fatal/day'] = df_interest['fatal/day'].astype(np.float)

df_interest['Fatalities'] = df_interest['Fatalities'].astype(np.float)

df_interest['cases/day'][df_interest['day']>last_day_train] = -1

df_interest['fatal/day'][df_interest['day']>last_day_train] = -1

###################标记训练与预测数据

len_known = (df_interest['cases/day']!=-1).sum()

len_unknown = (df_interest['cases/day']==-1).sum()

print("len train: {}, len prediction: {}".format(len_known, len_unknown))

X_valid = df_interest[col_var][df_interest['day']>84]

X_valid2 = df_interest[col_var2][df_interest['day']>84]

pred_f =  np.exp(model.predict(X_valid))-1

pred_c = np.exp(model2.predict(X_valid2))-1

df_interest['fatal/day'][df_interest['day']>last_day_train] = pred_f.clip(0, 1e10)

df_interest['cases/day'][df_interest['day']>last_day_train] = pred_c.clip(0, 1e10)

########累计确诊是每日确诊相加

df_interest['Fatalities'] = np.cumsum(df_interest['fatal/day'].values)

df_interest['ConfirmedCases'] = np.cumsum(df_interest['cases/day'].values)

for j in range(len_unknown): # use predicted cases and fatal for next days' prediction

    X_valid = df_interest[col_var].iloc[j+len_known]

    X_valid2 = df_interest[col_var2].iloc[j+len_known]

    pred_f = model_pri.predict(X_valid)

    pred_c = model2_pri.predict(X_valid2)

    pred_c = (np.exp(pred_c)-1).clip(0, 1e10)

    pred_f = (np.exp(pred_f)-1).clip(0, 1e10)

    df_interest['fatal/day'][j+len_known] = pred_f

    df_interest['cases/day'][j+len_known] = pred_c

    df_interest['Fatalities'][j+len_known] = df_interest['Fatalities'][j+len_known-1] + pred_f

    df_interest['ConfirmedCases'][j+len_known] = df_interest['ConfirmedCases'][j+len_known-1] + pred_c

    df_interest = df_interest.drop([

        'cases/day-(1-1)', 'cases/day-(1-7)', 'cases/day-(8-14)', 'cases/day-(15-21)', 

        'fatal/day-(1-1)', 'fatal/day-(1-7)', 'fatal/day-(8-14)', 'fatal/day-(15-21)', 

        'days_since_1cases', 'days_since_10cases', 'days_since_100cases',

        'days_since_1fatal', 'days_since_10fatal', 'days_since_100fatal',

                                   

                                   ],  axis=1)

    df_interest = do_aggregations(df_interest.reset_index(drop=True))



# visualize

tmp = df_interest['fatal/day'].values

tmp = np.cumsum(tmp)

sns.lineplot(x=df_interest_base['day'], y=tmp, label='pred')

tmp = df_traintest10['Fatalities'][(df_traintest10['place_id']==place)& (pd.isna(df_traintest10['ForecastId']))].values

print(len(tmp), tmp)

sns.lineplot(x=df_traintest10['day'][(df_traintest10['place_id']==place)& (pd.isna(df_traintest10['ForecastId']))].values,

             y=tmp, label='true')

print(place)

plt.show()
df_interest.head()
# predict test data in public

day_before_public = 71

df_preds = []

for i, place in enumerate(places[:]):

#     if place!='Japan' and place!='Afghanistan' :continue

    df_interest = copy.deepcopy(df_traintest9[df_traintest9['place_id']==place].reset_index(drop=True))

    df_interest['cases/day'][(pd.isna(df_interest['ForecastId']))==False] = -1

    df_interest['fatal/day'][(pd.isna(df_interest['ForecastId']))==False] = -1

    len_known = (df_interest['day']<=day_before_public).sum()

    len_unknown = (day_before_public<df_interest['day']).sum()

    for j in range(len_unknown): # use predicted cases and fatal for next days' prediction

        X_valid = df_interest[col_var].iloc[j+len_known]

        X_valid2 = df_interest[col_var2].iloc[j+len_known]

        pred_f = model.predict(X_valid)

        pred_c = model2.predict(X_valid2)

        pred_c = (np.exp(pred_c)-1).clip(0, 1e10)

        pred_f = (np.exp(pred_f)-1).clip(0, 1e10)

        df_interest['fatal/day'][j+len_known] = pred_f

        df_interest['cases/day'][j+len_known] = pred_c

        df_interest['Fatalities'][j+len_known] = df_interest['Fatalities'][j+len_known-1] + pred_f

        df_interest['ConfirmedCases'][j+len_known] = df_interest['ConfirmedCases'][j+len_known-1] + pred_c

        df_interest = df_interest.drop([

            'cases/day-(1-1)', 'cases/day-(1-7)', 'cases/day-(8-14)', 'cases/day-(15-21)', 

            'fatal/day-(1-1)', 'fatal/day-(1-7)', 'fatal/day-(8-14)', 'fatal/day-(15-21)',

            'days_since_1cases', 'days_since_10cases', 'days_since_100cases',

            'days_since_1fatal', 'days_since_10fatal', 'days_since_100fatal',



                                       ],  axis=1)

        df_interest = do_aggregations(df_interest)

    if (i+1)%10==0:

        print("{:3d}/{}  {}, len known: {}, len unknown: {}".format(i+1, len(places), place, len_known, len_unknown), df_interest.shape)

    df_interest['fatal_pred'] = np.cumsum(df_interest['fatal/day'].values)

    df_interest['cases_pred'] = np.cumsum(df_interest['cases/day'].values)

    df_preds.append(df_interest)
# concat prediction

df_preds= pd.concat(df_preds)

df_preds = df_preds.sort_values('day')

col_tmp = ['place_id', 'ForecastId', 'day', 'cases/day', 'cases_pred', 'fatal/day', 'fatal_pred',]

df_preds[col_tmp][(df_preds['place_id']=='Afghanistan') & (df_preds['day']>75)].head(10)
# predict test data in public

day_before_private = 84

df_preds_pri = []

for i, place in enumerate(places[:]):

#     if place!='Japan' and place!='Afghanistan' :continue

    df_interest = copy.deepcopy(df_traintest10[df_traintest10['place_id']==place].reset_index(drop=True))

    df_interest['cases/day'][(pd.isna(df_interest['ForecastId']))==False] = -1

    df_interest['fatal/day'][(pd.isna(df_interest['ForecastId']))==False] = -1

    len_known = (df_interest['day']<=day_before_private).sum()

    len_unknown = (day_before_private<df_interest['day']).sum()

    for j in range(len_unknown): # use predicted cases and fatal for next days' prediction

        X_valid = df_interest[col_var].iloc[j+len_known]

        X_valid2 = df_interest[col_var2].iloc[j+len_known]

        pred_f = model_pri.predict(X_valid)

        pred_c = model2_pri.predict(X_valid2)

        pred_c = (np.exp(pred_c)-1).clip(0, 1e10)

        pred_f = (np.exp(pred_f)-1).clip(0, 1e10)

        df_interest['fatal/day'][j+len_known] = pred_f

        df_interest['cases/day'][j+len_known] = pred_c

        df_interest['Fatalities'][j+len_known] = df_interest['Fatalities'][j+len_known-1] + pred_f

        df_interest['ConfirmedCases'][j+len_known] = df_interest['ConfirmedCases'][j+len_known-1] + pred_c

        df_interest = df_interest.drop([

            'cases/day-(1-1)', 'cases/day-(1-7)', 'cases/day-(8-14)', 'cases/day-(15-21)', 

            'fatal/day-(1-1)', 'fatal/day-(1-7)', 'fatal/day-(8-14)', 'fatal/day-(15-21)',

            'days_since_1cases', 'days_since_10cases', 'days_since_100cases',

            'days_since_1fatal', 'days_since_10fatal', 'days_since_100fatal',



                                       ],  axis=1)

        df_interest = do_aggregations(df_interest)

    if (i+1)%10==0:

        print("{:3d}/{}  {}, len known: {}, len unknown: {}".format(i+1, len(places), place, len_known, len_unknown), df_interest.shape)

    df_interest['fatal_pred'] = np.cumsum(df_interest['fatal/day'].values)

    df_interest['cases_pred'] = np.cumsum(df_interest['cases/day'].values)

    df_preds_pri.append(df_interest)
# concat prediction

df_preds_pri= pd.concat(df_preds_pri)

df_preds_pri = df_preds_pri.sort_values('day')

col_tmp = ['place_id', 'ForecastId', 'Date', 'day', 'cases/day', 'cases_pred', 'fatal/day', 'fatal_pred',]

df_preds_pri[col_tmp][(df_preds_pri['place_id']=='Japan') & (df_preds_pri['day']>79)].head(10)
# merge 2 preds

df_preds[df_preds['day']>last_day_train] = df_preds_pri[df_preds['day']>last_day_train]
df_preds.to_csv("df_preds.csv", index=None)
# load sample submission

df_sub = pd.read_csv("../input/covid19-global-forecasting-week-1/submission.csv")

print(len(df_sub))

df_sub.head()
# merge prediction with sub

df_sub = pd.merge(df_sub, df_traintest3[['ForecastId', 'place_id', 'day']])

df_sub = pd.merge(df_sub, df_preds[['place_id', 'day', 'cases_pred', 'fatal_pred']], on=['place_id', 'day',], how='left')

df_sub.head(10)
df_sub['ConfirmedCases'] = df_sub['cases_pred']

df_sub['Fatalities'] = df_sub['fatal_pred']

df_sub = df_sub[['ForecastId', 'ConfirmedCases', 'Fatalities']]

df_sub.to_csv("submission.csv", index=None)

df_sub.head(10)