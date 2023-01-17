# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns#################画图包

import matplotlib.pyplot as plt 



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train=pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")

df_test=pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")
df_traintest=pd.concat([df_train,df_test])
## 将所有地区唯一识别成place_id

def genplace_id(x):

    try:

        place_id=x['Country_Region']+'/'+x['Province_State']

    except:

        place_id=x['Country_Region']

    return place_id



df_traintest['place_id']=df_traintest.apply(lambda x:genplace_id(x),axis=1)

print("地区个数==>"+str(len(df_traintest['place_id'].unique())))
## 将时间类型转换：

df_traintest['Date']=pd.to_datetime(df_traintest['Date'])
## 生成名为dayofyear的"day"变量

df_traintest["day"]=df_traintest["Date"].apply(lambda x: x.dayofyear).astype(np.int16)
## 生成疫情变化速率变量

places=df_traintest['place_id'].unique()

df_traintest2=df_traintest.copy()

df_traintest2["cases/day"]=0

df_traintest2["fatal/day"]=0

for place in places:

    tmp=df_traintest2["ConfirmedCases"][df_traintest2["place_id"]==place].values

    tmp[1:]=tmp[1:]-tmp[:-1]   #####################每天的新增确诊数目是隔日之差

    df_traintest2["cases/day"][df_traintest2["place_id"]==place]=tmp

    tmp=df_traintest2["Fatalities"][df_traintest2["place_id"]==place].values

    tmp[1:]=tmp[1:]-tmp[:-1]

    df_traintest2["fatal/day"][df_traintest2["place_id"]==place]=tmp
df_traintest2[df_traintest2["place_id"]=="China/Hubei"].head() 
import copy
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
## 生成趋势数据

df_traintest3=[]

for place in places:

    df_tmp=df_traintest2[df_traintest2["place_id"]==place].reset_index(drop=True)

    df_tmp=do_aggregations(df_tmp)

    df_traintest3.append(df_tmp)

df_traintest3=pd.concat(df_traintest3).reset_index(drop=True)

df_traintest3[df_traintest3["place_id"]=="China/Hubei"].head()
df_countryinfo=pd.read_csv("../input/countryinfo/covid19countryinfo.csv")[["country","pop","tests","testpop","density","medianage","urbanpop","quarantine","schools","hospibed","smokers","sex0","sex14","sex25","sex54","sex64","sex65plus","sexratio","lung","femalelung","malelung"]]

df_countryinfo.head()
df_countryinfo["femalelung"][df_countryinfo["country"]=="China"] ####################中国，中国香港，中国澳门，湖北特别疫情地区
df_countryinfo["femalelung"][df_countryinfo["country"]=="China"]=56.35

df_countryinfo["femalelung"][df_countryinfo["country"]=="China"]
df_countryinfo["Country_Region"]=df_countryinfo["country"]

df_countryinfo=df_countryinfo[df_countryinfo["country"].duplicated()==False]    ############把不同的称呼的同一地区合并
print(df_countryinfo[df_countryinfo['country'].duplicated()].shape)    ############确认无重复

df_traintest4=pd.merge(df_traintest3,df_countryinfo.drop(["country","testpop","tests"],axis=1),on=["Country_Region"],how="left")

print(df_traintest4.shape)

df_traintest4.head()

df_traintest4[df_traintest4["Country_Region"]=="China"]
## 引入吸烟数据
smoke_path=r"../input/smokingstats/share-of-adults-who-smoke.csv"

df_smoking=pd.read_csv(smoke_path)

df_smoking.head()
df_smoking_recent=df_smoking.sort_values("Year",ascending=False).reset_index(drop=True) 

####################同个地区多个年份的数据重复

df_smoking_recent=df_smoking_recent[df_smoking_recent["Entity"].duplicated()==False]

###########################改了两列的名字,没什么大变动,方便之后的连接

df_smoking_recent['Country_Region'] = df_smoking_recent['Entity']

df_smoking_recent['SmokingRate'] = df_smoking_recent['Smoking prevalence, total (ages 15+) (% of adults)']

df_smoking_recent.head()
df_traintest5 = pd.merge(df_traintest4,df_smoking_recent[["Country_Region","SmokingRate"]],on="Country_Region",how="left")

print(df_traintest5.shape)
df_traintest5[df_traintest4["place_id"]=="China/Hubei"].head()
##使用世界平均值填补空值的抽烟率

SmokingRate=df_smoking_recent["SmokingRate"][df_smoking_recent["Entity"]=="World"].values[0]

df_traintest5["SmokingRate"][pd.isna(df_traintest5["SmokingRate"])]=SmokingRate

df_traintest5.head()
## 引入国家经济水平数据
smoke_path=r"../input/smokingstats/WEO.csv"

df_weo=pd.read_csv(smoke_path)

df_weo.head()
print(df_weo['Subject Descriptor'].unique()) ## 查看包含的经济描述项目
## 取国家2019的经济数据然后将数据横竖变换

subs=df_weo["Subject Descriptor"].unique()[:-1]  ###########去掉最后一个空缺的经济指标

df_weo_agg=df_weo[["Country"]][df_weo["Country"].duplicated()==False].reset_index(drop=True)

for sub in subs[:]:

    df_tmp=df_weo[["Country","2019"]][df_weo["Subject Descriptor"]==sub].reset_index(drop=True) 

    df_tmp=df_tmp[df_tmp["Country"].duplicated()==False].reset_index(drop=True)

    df_tmp.columns=["Country",sub]          ##############把表头的2019改了

    df_weo_agg=df_weo_agg.merge(df_tmp,on="Country",how="left")

df_weo_agg.columns=["".join(c if c.isalnum() else "_" for c in str(x))for x in df_weo_agg.columns]

df_weo_agg.columns

df_weo_agg['Country_Region'] = df_weo_agg['Country']

df_weo_agg.head()        #####################各个经济指标的数据
df_traintest6 = pd.merge(df_traintest5, df_weo_agg, on='Country_Region', how='left')

print(df_traintest6.shape)

df_traintest6.head()      ##################### 将经济数据与上一版本数据合并
## 引入平均寿命数据
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

df_life.columns = ['Country_Region', 'LifeExpectancy']  #############表头转换
df_traintest7 = pd.merge(df_traintest6, df_life, on='Country_Region', how='left') ##再次合并

print(len(df_traintest7))

df_traintest7.head()
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

df_le = get_df_le(df_traintest7, 'id', ['Country_Region', 'Province_State'])

df_traintest8 = pd.merge(df_traintest7, df_le, on='id', how='left')
df_traintest8['cases/day'] = df_traintest8['cases/day'].astype(np.float)

df_traintest8['fatal/day'] = df_traintest8['fatal/day'].astype(np.float)
# 转换数据类型从对象至float

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
## 补充给地区标签改成数值类型

i=0

df_traintest8['place_label']=0



places=df_traintest8['place_id'].unique()

for place in places:

    df_traintest8['place_label'][df_traintest8['place_id']==place]=i

    i=i+1

print(df_traintest8['place_label'].unique())

df_traintest8[df_traintest8['place_id']=='China/Hubei'].head()
temp_list=[    'Gross_domestic_product__constant_prices',

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

    'hospibed', 'smokers']

## 将df_train中空的且是数值类型的数据用中位数填补

temp1=[]





for column in temp_list:

    try:

        mean=df_traintest8[column].mean()

        temp1.append(column)

    except:

        print(column+"不是数值类型")



for col in temp1:

    df_traintest8[col].fillna(df_traintest8[col].mean(),inplace=True)

    

df_traintest8[temp_list].isnull()
df_traintest8[temp_list].isnull().sum() ##仍有两各属性存在空值，之后不使用这两各属性
## 输出预测用数据

df_traintest8.to_csv("data_prepared.csv")
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
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
df_traintest.head()
# train model to predict fatalities/day

col_target = 'fatal/day'

col_var = ['day',

     'place_label',                 #####保留地区标签？

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

#     'Six_month_London_interbank_offered_rate__LIBOR_', ##数据质量过差，不适用抛弃

    'Volume_of_imports_of_goods_and_services', 'Volume_of_Imports_of_goods',

    'Volume_of_exports_of_goods_and_services', 'Volume_of_exports_of_goods',

    'Unemployment_rate',                            ###############失业率

#     'Employment',  ##数据质量过差，不适用抛弃

    'Population',

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

    'hospibed', 'smokers'

]

col_cat = []

#####################当然是划分训练集和验证集

#####################日期序号为93天前的数据作为训练集合

#####################日期学号为92-107天的数据作为验证集合

df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<93)] #4/2之前

df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']>92) & (df_traintest8['day']<107)]

df_test = df_traintest8[pd.isna(df_traintest8['ForecastId'])==False]

X_train = df_train[col_var]

X_valid = df_valid[col_var]

#######################为什么要取对数：取对数可以缩小特别大特别小的数据之间的差别，比如1和e10,相当于排除了特殊点？

#######################最后得到的最佳参数肯定是不变的

y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)







rf=RandomForestRegressor(n_estimators=600)

rf.fit(X_train,y_train)
tmp = pd.DataFrame() ## 发现重要的是趋势数据

tmp["feature"] = col_var

tmp["importance"] = rf.feature_importances_

tmp = tmp.sort_values('importance', ascending=False)

tmp


tmp[0:13].sum()
## 评分良好暂时不需要调参

print(rf.score(X_train,y_train))

print(rf.score(X_valid,y_valid))
##根据特征显著性数量级保留前10样特征重新训练
col_var2=tmp['feature'][0:13].tolist()
col_var2
# train model to predict caes/day



col_cat = []

#####################当然是划分训练集和验证集

#####################日期序号为93天前的数据作为训练集合

#####################日期学号为92-107天的数据作为验证集合

df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<93)] #4/2之前

df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']>92) & (df_traintest8['day']<107)]

df_test = df_traintest8[pd.isna(df_traintest8['ForecastId'])==False]

X_train = df_train[col_var2]

X_valid = df_valid[col_var2]

#######################为什么要取对数：取对数可以缩小特别大特别小的数据之间的差别，比如1和e10,相当于排除了特殊点？

#######################最后得到的最佳参数肯定是不变的

y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)







rf2=RandomForestRegressor(n_estimators=600)

rf2.fit(X_train,y_train)
## 评分良好暂时不需要调参

print(rf2.score(X_train,y_train))

print(rf2.score(X_valid,y_valid))
tmp2 = pd.DataFrame() ## 发现重要的是趋势数据

tmp2["feature"] = col_var2

tmp2["importance"] = rf2.feature_importances_

tmp2 = tmp2.sort_values('importance', ascending=False)

tmp2
## 预测case/day
# train model to predict fatalities/day

col_target = 'cases/day'

col_var = ['day',

     'place_label',                 #####保留地区标签？

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

#     'Six_month_London_interbank_offered_rate__LIBOR_', ##数据质量过差，不适用抛弃

    'Volume_of_imports_of_goods_and_services', 'Volume_of_Imports_of_goods',

    'Volume_of_exports_of_goods_and_services', 'Volume_of_exports_of_goods',

    'Unemployment_rate',                            ###############失业率

#     'Employment',  ##数据质量过差，不适用抛弃

    'Population',

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

    'hospibed', 'smokers'

]

col_cat = []

#####################当然是划分训练集和验证集

#####################日期序号为93天前的数据作为训练集合

#####################日期学号为92-107天的数据作为验证集合

df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<93)] #4/2之前

df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']>92) & (df_traintest8['day']<107)]

df_test = df_traintest8[pd.isna(df_traintest8['ForecastId'])==False]

X_train = df_train[col_var]

X_valid = df_valid[col_var]

#######################为什么要取对数：取对数可以缩小特别大特别小的数据之间的差别，比如1和e10,相当于排除了特殊点？

#######################最后得到的最佳参数肯定是不变的

y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)

#######################lgb封装自身的训练集

# train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)

# valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)

# num_round = 15000

# model = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data],

#                   verbose_eval=100, ###########每100次输出一次评测结果

#                   early_stopping_rounds=150,)      #############如果连续150轮都无法优化，那么就提前停下



# best_itr = model.best_iteration





rf3=RandomForestRegressor(n_estimators=600)

rf3.fit(X_train,y_train)
## 评分良好暂时不需要调参

print(rf3.score(X_train,y_train))

print(rf3.score(X_valid,y_valid))
tmp3 = pd.DataFrame() ## 发现重要的是趋势数据

tmp3["feature"] = col_var

tmp3["importance"] = rf3.feature_importances_

tmp3 = tmp3.sort_values('importance', ascending=False)

tmp3
tmp3[0:13].sum()
## 保留前30的特征

col_var2=tmp3['feature'][0:13].tolist()

col_var2
#重新训练
# train model to predict caess/day again

col_target = 'cases/day'

col_cat = []

#####################当然是划分训练集和验证集

#####################日期序号为93天前的数据作为训练集合

#####################日期学号为92-107天的数据作为验证集合

df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<93)] #4/2之前

df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']>92) & (df_traintest8['day']<107)]

df_test = df_traintest8[pd.isna(df_traintest8['ForecastId'])==False]

X_train = df_train[col_var2]

X_valid = df_valid[col_var2]

#######################为什么要取对数：取对数可以缩小特别大特别小的数据之间的差别，比如1和e10,相当于排除了特殊点？

#######################最后得到的最佳参数肯定是不变的

y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)







rf4=RandomForestRegressor(n_estimators=600)

rf4.fit(X_train,y_train)
## 评分良好暂时不需要调参

print(rf4.score(X_train,y_train))

print(rf4.score(X_valid,y_valid))
df_traintest8[df_traintest8['day']==93]
df_tmp = df_traintest8[(df_traintest8['day']<93) | (pd.isna(df_traintest8['ForecastId'])==False)].reset_index(drop=True)

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
df_tmp
col_target1='fatal/day'

col_target='cases/day'



col_var1=tmp['feature'][0:13].tolist()

col_var2=tmp3['feature'][0:13].tolist()



print(col_var1)

print(col_var2)
###############注意啊，这是什么模型的曲线?

#############这当然是确诊数目的曲线，但是确诊我训练了两个模型

########这是哪个模型？

#########这是我们拟合72天内的模型啊

##########可以看到72天内效果及其优秀，72天后呵呵

place = 'Japan'

df_interest_base = df_traintest9[df_traintest9['place_id']==place].reset_index(drop=True)

df_interest = copy.deepcopy(df_interest_base)

df_interest['ConfirmedCases'] = df_interest['ConfirmedCases'].astype(np.float)

df_interest['cases/day'] = df_interest['cases/day'].astype(np.float)

df_interest['fatal/day'] = df_interest['fatal/day'].astype(np.float)

df_interest['Fatalities'] = df_interest['Fatalities'].astype(np.float)

df_interest['cases/day'][df_interest['day']>92] = -1

df_interest['fatal/day'][df_interest['day']>92] = -1

len_known = (df_interest['cases/day']!=-1).sum()

len_unknown = (df_interest['cases/day']==-1).sum()

print("len train: {}, len prediction: {}".format(len_known, len_unknown))

X_valid = df_interest[col_var1][df_interest['day']>92]



X_valid2 = df_interest[col_var2][df_interest['day']>92]

pred_f =  np.exp(rf2.predict(X_valid))-1

pred_c = np.exp(rf4.predict(X_valid2))-1

df_interest['fatal/day'][df_interest['day']>92] = pred_f.clip(0, 1e10)

df_interest['cases/day'][df_interest['day']>92] = pred_c.clip(0, 1e10)

df_interest['Fatalities'] = np.cumsum(df_interest['fatal/day'].values)

df_interest['ConfirmedCases'] = np.cumsum(df_interest['cases/day'].values)

for j in range(len_unknown): # use predicted cases and fatal for next days' prediction

    X_valid =np.array(df_interest[col_var1].iloc[j+len_known]).reshape(1,-1)

    X_valid2 = np.array(df_interest[col_var2].iloc[j+len_known]).reshape(1,-1) #就算只有一个样本也要转换成2d矩阵

    pred_f = rf2.predict(X_valid)

    pred_c = rf4.predict(X_valid2)

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

plt.title(place)

plt.show()

## 预测文件生成
# predict test data in public

day_before_public = 92

df_preds = []

for i, place in enumerate(places[:]):

#     if place!='Japan' and place!='Afghanistan' :continue

    df_interest = copy.deepcopy(df_traintest9[df_traintest9['place_id']==place].reset_index(drop=True))

    df_interest['cases/day'][(pd.isna(df_interest['ForecastId']))==False] = -1

    df_interest['fatal/day'][(pd.isna(df_interest['ForecastId']))==False] = -1

    len_known = (df_interest['day']<=day_before_public).sum()

    len_unknown = (day_before_public<df_interest['day']).sum()

    for j in range(len_unknown): # use predicted cases and fatal for next days' prediction

        X_valid = np.array(df_interest[col_var1].iloc[j+len_known]).reshape(1,-1)

        X_valid2 = np.array(df_interest[col_var2].iloc[j+len_known]).reshape(1,-1)

        pred_f = rf2.predict(X_valid)

        pred_c = rf4.predict(X_valid2)

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
df_preds.to_csv("df_preds.csv", index=None)
# load sample submission

df_sub = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")

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