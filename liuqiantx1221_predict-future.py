import pandas as pd

from pandas import DataFrame,Series

import numpy as np

from numpy import nan

import os

from datetime import datetime as dt

import matplotlib.pyplot as plt

%matplotlib inline

from scipy import stats



import seaborn as sns

sns.set_style("darkgrid")

import warnings

warnings.filterwarnings("ignore")



import re

import statsmodels.api as sm

import holidays as hl

pd.set_option('display.float_format', lambda x: '%.2f' % x)
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
print(check_output(["ls", "../input/item-cate-new"]).decode("utf8"))
items2=pd.read_csv('../input/items-translated'+"/items-translated.csv")

item_categories = pd.read_csv("../input/item-cate-new"+"/item_cate_new.csv")

shop = pd.read_csv("../input/shopstranslated"+"/shops-translated.csv")

dataset = "../input/competitive-data-science-predict-future-sales"

transaction = pd.read_csv(dataset+'/sales_train.csv')

items = pd.read_csv(dataset+"/items.csv")

test = pd.read_csv(dataset+"/test.csv")

usd = pd.read_csv("../input/predict-future-sales-supplementary"+"/usd-rub.csv")
def downcast_dtypes(df):

    float_col = [c for c in df if df[c].dtype == "float64"]

    int_col = [c for c in df if df[c].dtype =="int64"]

    df[float_col] = df[float_col].astype(np.float16)

    df[int_col] = df[int_col].astype(np.int16)

    return df



transaction = downcast_dtypes(transaction)

test = downcast_dtypes(test)

items=downcast_dtypes(items)

items2 = downcast_dtypes(items2)

usd = downcast_dtypes(usd)

print(transaction.info())
#移除明显异常的数据

#价格和日销量

transaction.drop([885138,2326930,2909818],inplace=True)
shop.head(2)
#查找超市名称

def get_shop_name(name):

    pattern_name = r'.+(\".+\")'

    regex_name = re.compile(pattern_name,flags=re.IGNORECASE)

    m = regex_name.match(name)

    try:

        return m.groups()[0]

    except:

        return nan





#查找超市地址

def get_shop_city(name):

    pattern_city = r'\!*\s*([A-Za-z\.]+\S)'

    regex_city = re.compile(pattern_city,flags=re.IGNORECASE)

    m = regex_city.match(name)

    try:

        return m.groups()[0]

    except:

        return nan



#查找超市类型

def get_shop_type(name):

    pattern_type =r'\s([A-Z]{2,4})\s'

    regex_type = re.compile(pattern_type,flags=re.IGNORECASE)

    m = regex_type.findall(name)

    try:

        return m[0]

    except:

        return nan

    

city_list = []

name_list = []

type_list = []

for name in shop.shop_name_translated.values:

    city = get_shop_city(name)

    names = get_shop_name(name)

    types = get_shop_type(name)

    city_list.append(city)

    name_list.append(names)

    type_list.append(types)

shop["city"] = city_list

shop["type"] = type_list

shop["name"] = name_list
#检查异常城市名

shop.city.unique()
#检查异常"type"

shop.type.unique()
#更正"type"有误的位置

shop.loc[(shop.type.isin(['and', 'shop', 'Shop','mall'])),"type"] = nan
#查看重复项

shop[shop.duplicated(subset=["city","type","name"],keep=False)]
#修改重复超市数据

transaction.loc[(transaction.shop_id==11),"shop_id"] = 10

shop.loc[shop.shop_id==11,"shop_id"] = 10

shop.drop_duplicates(subset= "shop_id",inplace = True)

test.loc[test.shop_id==11,"shop_id"] = 10
item_categories.head(2)
# 将产品名改为英文，方便识别

items["item_name"] = items2.item_name_translated.values



#移除测试数据中不存在的产品（含所有产品的各属性的销售趋势与仅含测试产品的销售趋势不同）

test_items = test.item_id.unique()

items_combine = items[items.item_id.isin(test_items)].join(item_categories.set_index("item_category_id"),on="item_category_id")

items.head(2)
items_combine.head(2)
paycard_pattern = re.compile(r"\s(([0-9]{3,4}\srub)|([0-9]{1,2}(\s|\-)month))",re.I)

def get_cardtype(x):

    if len(paycard_pattern.findall(x)) !=0:

        if len(paycard_pattern.findall(x))==1:

            cardtype = paycard_pattern.findall(x)[0][0]

            cardtype = re.sub("(\-)"," ",cardtype)

        else:

            cardtype = paycard_pattern.findall(x)[0][0]+"+"+paycard_pattern.findall(x)[1][0]

    else:

        cardtype=nan

    return cardtype

items_combine.loc[(items_combine.attributes

                   =="Payment Cards"),"additional_feature"] = items_combine[(items_combine.attributes

                                                                             =="Payment Cards")].item_name.apply(lambda x:get_cardtype(x))
wireless_pattern = re.compile(r"^(?!.*wireless)",re.I)

headset_pattern = re.compile(r"^.*(stereo.*headset|headset.*stereo).*$",re.I)

def get_headset(x):

    if len(wireless_pattern.findall(x))!=0:

        if len(headset_pattern.findall(x)) != 0:

            headset = "stereo headset"

        else:

            headset = nan

    else:

        headset=nan

    return headset

items_combine.loc[(items_combine.attributes.isin\

                   (["Headsets / Headphones","Accesories"])),"additional_feature"] = items_combine[(items_combine.attributes.isin\

                                                       (["Headsets / Headphones","Accesories"]))].item_name.apply(lambda x:get_headset(x))

items_combine.loc[(items_combine.attributes=="Headsets / Headphones"),"attributes"]=["Accesories"]
control_pattern = re.compile(r"^.*(controller|game(\s)?pad).*$",re.I)

wireless_headset_pattern = re.compile(r"^.*(wireless.*headset|headset.*wireless).*$",re.I)

charge_pattern = re.compile(r"^.*(charge|charging|cable).*$",re.I)

headphone_pattern = re.compile(r"^.*(headphone).*$",re.I)

protect_pattern = re.compile(r"^.*(protect|silicone).*$",re.I)    

def get_access_type(x):

    if len(wireless_pattern.findall(x)) ==0:

        if len(control_pattern.findall(x)) !=0:

            access_type = "controller"

        elif len(wireless_headset_pattern.findall(x)) !=0:

            access_type = "wireless headset"

        else:

            access_type = nan

    elif len(charge_pattern.findall(x)) !=0:

        access_type = "charge"

    elif len(protect_pattern.findall(x)) !=0:

        access_type = "protect"

    elif len(headphone_pattern.findall(x)) !=0:

        access_type = "headphone"   

    else:

        access_type = nan

    return access_type

items_combine.loc[(items_combine.attributes =="Accesories")&(items_combine.additional_feature.isna()),

                  "additional_feature"] = items_combine[(items_combine.attributes =="Accesories")

                                                        &(items_combine.additional_feature.isna())].item_name.apply(lambda x:get_access_type(x))
tb_pattern = re.compile(r"^.+(\s|\()([0-9]{1,4}(\s)?(tb|gb|ГБ)).+$",re.I)

def get_tb(x):

    try:

        tb = tb_pattern.findall(x)[0][1]

        tb = re.sub("(tb|Tb|tB)","TB",tb)

        tb = re.sub("(Gb|gb|gB|ГБ)","GB",tb)

        tb = re.sub(" ","",tb)

        tb = re.sub("(1TB)","1024GB",tb)

    except:

        tb = nan

    return tb

items_combine.loc[(items_combine.attributes=="Game consoles"),

                  "additional_feature"] = items_combine[(items_combine.attributes=="Game consoles")].item_name.apply(lambda x:get_tb(x))
year_pattern = re.compile(r"^.*(\s|\()([0-9]{1,2}(\s)?(month|year)).*",re.I)

year_pattern2 = re.compile(r"^.*(year).*",re.I)

device_pattern = re.compile(r".+(\s|\(|\.)((([0-9](\s|\-)(МУ|dev(ice)?|pc(s)?|pda|desktop|pk|license))|([0-9](МУ|dev(ice)?|pc(s)?|pda|desktop|pk|license)))).*",re.I)

def get_year(x):

    if len(year_pattern.findall(x))==1:

        year = year_pattern.findall(x)[0][1]

        if len(year_pattern2.findall(year))==1:

            num = int(re.findall("([0-9])",year)[0])*12

            year = str(num)+" "+"month"

        else:

            year = year

    else:

        year = nan

    return year

def get_device(x):

    if len(device_pattern.findall(x))==1:

        device = device_pattern.findall(x)[0][1]

        try:

            num = re.findall(r"[0-9]{1,2}",device)[0]

        except:

            num = 1

        device = str(num)+"dev"

    else:

        device = nan

    return device

items_combine.loc[(items_combine.attributes=="Programmes"),"extra_1"] = items_combine[(items_combine.attributes=="Programmes")].item_name.apply(lambda x:get_year(x))

items_combine.loc[(items_combine.attributes=="Programmes"),"extra_2"] = items_combine[(items_combine.attributes=="Programmes")].item_name.apply(lambda x:get_device(x))

items_combine.loc[(items_combine.attributes=="Programmes")&(items_combine.extra_1.notna())&(items_combine.extra_2.isna()),"extra_2"]="1dev"
items_combine.loc[(items_combine.attributes=="Programmes")

                  &(items_combine.extra_1.notna()),"additional_feature"] = items_combine[(items_combine.attributes=="Programmes")

                                                                                         &(items_combine.extra_1.notna())].apply(lambda x:x["extra_1"]+"-"+x["extra_2"],axis=1)
music_type_pattern = re.compile(r"^.*(\s|\-|\()([0-9]{0,1}(lp|cd|dvd|bd))",re.I)

def get_cd_type(x):

    if len(music_type_pattern.findall(x))==1:

        cd_type =music_type_pattern.findall(x)[0][1]

        cd_type = cd_type.lower()

    else:

        cd_type=nan

    return cd_type

items_combine.loc[(items_combine.attributes.isin(['Music','The Games','Movies'])),

                  "extra_1"] = items_combine[(items_combine.attributes.isin(['Music','The Games','Movies']))\

                                            ].item_name.apply(lambda x:get_cd_type(x))
cd_pattern = re.compile(r"((cd|bd|dvd|lp))")

def get_cd_type(x):

    cd_type = cd_pattern.findall(x)[0][0]

    return cd_type

items_combine.loc[(items_combine.attributes.isin(['Music','The Games','Movies']))

                  &(items_combine.extra_1.notna()),"additional_feature"] = items_combine[(items_combine.attributes.isin(['Music','The Games','Movies']))

                                                                                         &(items_combine.extra_1.notna())].extra_1.apply(lambda x:get_cd_type(x))
items_combine.drop(["extra_1","extra_2"],axis=1,inplace=True)
size_pattern = re.compile(r"\s(([0-9\.]{1,3}(см|cm)))($|\s)",re.I)

def get_size(x):

    if len(size_pattern.findall(x))==1:

        size = size_pattern.findall(x)[0][0]

        size = re.sub("(см|cm)","",size)

        try :

            size = int(size)

        except:

            size = size.lower()

    else:

        size = nan

    return size

items_combine.loc[(items_combine.attributes=="Gifts"),

                  "additional_feature"] = items_combine[(items_combine.attributes=="Gifts")].item_name.apply(lambda x:get_size(x))

items_combine.loc[(items_combine.attributes=="Gifts")&(items_combine.additional_feature.notna()),

                  "additional_feature"] = items_combine[(items_combine.attributes=="Gifts")

                                                        &(items_combine.additional_feature.notna())].additional_feature.apply(lambda x:int(x))
items_combine.loc[(items_combine.attributes=="Gifts")

                  &(items_combine.additional_feature.isin([6,  8,  9, 10, 11, 12, 13, 14, 15])),"additional_feature"]="s"

items_combine.loc[(items_combine.attributes=="Gifts")

                  &(items_combine.additional_feature.isin([16, 17, 18, 19, 20, 22, 23, 24])),"additional_feature"]="m"

items_combine.loc[(items_combine.attributes=="Gifts")

                  &(items_combine.additional_feature.isin([25, 26, 27, 28, 29, 30, 32, 33, 35])),"additional_feature"]="l"

items_combine.loc[(items_combine.attributes=="Gifts")

                  &(items_combine.additional_feature.isin([37, 38, 40, 43, 45, 50, 60, 65,75])),"additional_feature"]="xl"
items_combine.drop(["item_name"],axis=1,inplace=True)
sub_model_list = list()

for subclass,model in zip(item_categories[item_categories.subclass.notna()].subclass.values,\

                          item_categories[item_categories.subclass.notna()].model.values):

    if model not in [nan]:

        sub_model = subclass+"-"+model

    else:

        sub_model = subclass

    sub_model_list.append(sub_model)

item_categories.loc[(item_categories.subclass.notna()),"sub_model"] = sub_model_list

item_categories.loc[(item_categories.subclass.isna()),"sub_model"] = item_categories[(item_categories.subclass.isna())].attributes.values

item_categories.drop(["item_category_name"],axis = 1,inplace = True)

item_categories.head(1)



#转换价格（消除汇率的影响）

transaction.date = transaction.date.apply(lambda x:dt.strptime(x, '%d.%m.%Y'))

usd.date = usd.date.apply(lambda x:dt.strptime(x, '%Y-%m-%d'))

transaction = transaction.join(usd.set_index("date"),on="date")

transaction["price"] = transaction.apply(lambda x:x["item_price"]/x["cur_rate"],axis =1)

transaction.drop(["item_price","cur_rate"],axis =1,inplace=True)

transaction.rename(columns={"price":"item_price"},inplace=True)



#移除不存在于测试数据中的产品

test_shop = test.shop_id.unique()

test_items = test.item_id.unique()

transaction = transaction[(transaction.item_id.isin(test_items))&(transaction.shop_id.isin(test_shop))]
#形成与测试数据形式一致的数据集

transaction = transaction.sort_values("date").groupby(["date_block_num","shop_id","item_id"],as_index=False\

                                                     ).agg({"item_price":["mean","std"],"item_cnt_day":["sum","mean","count"]})

transaction.columns=["date_block_num","shop_id","item_id","item_mean_price","price_stability","item_cnt_month","item_cnt_mean","transactions"]
#合并测试集与训练集，形成完整数据集

test["date_block_num"] = 34

test_ID_df = test[["ID"]]

test.drop(["ID"],axis=1,inplace=True)

data_combine = pd.concat([transaction,test],axis=0,ignore_index=False)

data_combine = data_combine.join(items_combine.set_index("item_id"),on="item_id")

data_combine.tail(2)
#移除月均销售值大于20和小于0的部分数据

data_combine.loc[(data_combine.item_cnt_month.isna()),"item_cnt_month"] = 0

data_combine = data_combine.query("item_cnt_month >=0 and item_cnt_month <=20")

data_combine = data_combine.query("item_id not in [20949] and attributes not in ['Delivery of goods']")

#将测试月数据转换回来

data_combine.loc[(data_combine.date_block_num==34),"item_cnt_month"] = nan

data_combine.drop(['item_category_id', 'attributes', 'subclass', 'model', 'additional_feature'],axis=1,inplace=True)

data_combine = downcast_dtypes(data_combine)
shop_id = data_combine.shop_id.unique()

item_id = data_combine.item_id.unique()

full_df = []

for i in range(0,35):

    for sp in shop_id:

        for item in item_id:

            full_df.append([i,sp,item])

full_df = pd.DataFrame(full_df,columns=["date_block_num","shop_id","item_id"])

data_combine = pd.merge(full_df,data_combine,on = ["date_block_num","shop_id","item_id"],how = "left")

data_combine.fillna(0,inplace=True)

data_combine=downcast_dtypes(data_combine)

data_combine.tail(1)
ru_holidays = hl.RU(years = [2013,2014,2015])

holiday_list = list()

for date in list(ru_holidays.keys()):

    holi = ru_holidays[date]

    holiday_list.append(holi)

ru_holiday_df = pd.DataFrame({"date":list(ru_holidays.keys()),"holiday":holiday_list})

ru_holiday_df["date"] = ru_holiday_df["date"].apply(lambda x:dt.strptime(str(x),"%Y-%m-%d"))

ru_holiday_df["year"]=ru_holiday_df.date.apply(lambda x:x.year)

ru_holiday_df["month"]=ru_holiday_df.date.apply(lambda x:x.month)

ru_holiday_df["date_block_num"] = ru_holiday_df.apply(lambda x:(x["year"]-2013)*12+x["month"]-1,axis=1)



ru_holiday_count = ru_holiday_df.groupby(["date_block_num"],as_index=False).agg({"holiday":"count"})

ru_holiday_count.columns=["date_block_num","holiday_cnt"]



year_month = list()

for year in [2013,2014,2015]:

    for month in range(1,13):

        year_month.append([year,month])

year_month = pd.DataFrame(year_month,columns=["year","month"])

year_month.reset_index(inplace=True)

year_month.columns=["date_block_num","year","month"]

ru_holiday_date = pd.merge(year_month,ru_holiday_count,on="date_block_num",how="left")

ru_holiday_date.fillna(0,inplace=True)

ru_holiday_date["holiday_cnt_lag1"] = ru_holiday_date["holiday_cnt"].shift(periods=-1)

ru_holiday_date.fillna(0,inplace=True)

ru_holiday_date.tail(2)
ru_holiday_date["holiday_cnt"] = ru_holiday_date["holiday_cnt"].astype("int64")

ru_holiday_date["holiday_cnt_lag1"] = ru_holiday_date["holiday_cnt_lag1"].astype("int64")

ru_holiday_date = downcast_dtypes(ru_holiday_date)

data_combine = data_combine.join(ru_holiday_date.set_index("date_block_num"),on="date_block_num")

data_combine = data_combine.join(items_combine.set_index("item_id"),on="item_id",rsuffix='_')
#计算最开始销售的月份

train_month2 = data_combine[data_combine.item_mean_price!=0]

items_min_month = train_month2.groupby("item_id",as_index=False).agg({"date_block_num":"min"})

items_min_month.columns = ["item_id","min_month"]

items_min_month =downcast_dtypes(items_min_month)

data_combine = data_combine.join(items_min_month.set_index("item_id"),on = "item_id")

data_combine["min_month"].fillna(34,inplace=True)



#计算至今为止已经销售的月份数

data_combine['month_count'] = data_combine.apply(lambda x:x["date_block_num"]-x["min_month"],axis = 1)

data_combine.loc[(data_combine.month_count<0),'month_count'] = -1

data_combine= downcast_dtypes(data_combine)



#计算各产品最后出现的销售月份

items_max_month = train_month2.groupby("item_id",as_index=False).agg({"date_block_num":"max"})

items_max_month.columns = ["item_id","max_month"]

items_max_month =downcast_dtypes(items_max_month)

data_combine = data_combine.join(items_max_month.set_index("item_id"),on = "item_id")
lag_list = [1,2,3,6,12]

for lag in lag_list:

    lag_name = ("item_cnt_lag%r"%lag)

    data_combine[lag_name] = data_combine.sort_values("date_block_num").groupby(["shop_id","item_id"])["item_cnt_month"].shift(i)

#添加变化趋势

data_combine['month_trend'] = data_combine['item_cnt_month']

for lag in [1,2,3]:

    ft_name = ("item_cnt_lag%r"%lag)

    data_combine['month_trend'] -= data_combine[ft_name]

data_combine['month_trend'] /= len(lag_list) + 1

data_combine['year_trend'] = data_combine['item_cnt_month']

for lag in [1,6,12]:

    ft_name = ("item_cnt_lag%r"%lag)

    data_combine['year_trend'] -= data_combine[ft_name]

data_combine['year_trend'] /= len(lag_list) + 1

data_combine.drop(["item_cnt_lag2"],axis=1,inplace=True)
data_combine.rename(columns={'item_cnt_mean':"mean_item_cnt"},inplace=True)
f_min = lambda x: x.rolling(window=3, min_periods=1).min()

f_max = lambda x: x.rolling(window=3, min_periods=1).max()

f_mean = lambda x: x.rolling(window=3, min_periods=1).mean()

f_std = lambda x: x.rolling(window=3, min_periods=1).std()

function_list = [f_min, f_max, f_mean, f_std]

function_name = ['min', 'max', 'mean', 'std']

for i in range(len(function_list)):

    data_combine[('item_cnt_%s' % function_name[i])] = data_combine.sort_values('date_block_num').groupby(['shop_id',

                                                                                                           'item_id'])['item_cnt_month'].apply(function_list[i])
data_combine["item_cnt_std"].fillna(0,inplace=True)
data_combine[(data_combine.item_cnt_lag1.isna())].sample(2)
data_combine[data_combine.attributes.isin(['The Games','Accesories','Game consoles'])].subclass.unique()
data_combine.head(1)
data_combine[(data_combine.attributes=='The Games')&(data_combine.subclass=='Game accessories')].boxplot(column ="item_mean_price")
train_month2 = data_combine[data_combine.item_mean_price!=0]
## date

date_mean_cnt = data_combine.groupby(["date_block_num"],as_index=False).agg({"item_cnt_month":"mean"})

month_mean_cnt = data_combine.groupby(["month"],as_index=False).agg({"item_cnt_month":"mean"})

f = plt.figure(figsize=(16,9))

ax1 = f.add_subplot(2,1,1)

ax2 = f.add_subplot(2,1,2)

sns.lineplot(x = "date_block_num",y = "item_cnt_month",data=date_mean_cnt ,ax = ax1).set_title("date_mean_cnt")

sns.lineplot(x = "month",y = "item_cnt_month",data=month_mean_cnt ,ax = ax2).set_title("month_mean_cnt")

plt.show()
attribute_date_mean_cnt = data_combine.groupby(["attributes","date_block_num"],as_index=False).agg({"item_cnt_month":"mean"})

fig,axes = plt.subplots(nrows=3,ncols=4,figsize = (18,14))

fig.subplots_adjust(wspace = 0.1,hspace = 0.3)

name_list = list(attribute_date_mean_cnt.attributes.unique())

for attribute,ax in zip(name_list,axes.flat):

    sns.lineplot(x = "date_block_num",y ="item_cnt_month",data = attribute_date_mean_cnt[(attribute_date_mean_cnt.attributes.isin([attribute]))],ax = ax).set_title(attribute)
data = data_combine[(data_combine.attributes=='Programmes')]

data = data.groupby("shop_id",as_index=False).agg({'item_cnt_month':["mean","sum"]})

data.columns = ["shop_id","item_mean_cnt","item_sum_cnt"]

data2 = data_combine[(data_combine.attributes== 'Payment Cards')]

data2 = data2.groupby("shop_id",as_index=False).agg({'item_cnt_month':["mean","sum"]})

data2.columns = ["shop_id","item_mean_cnt","item_sum_cnt"]

data3 = data_combine[(data_combine.attributes=='Books')]

data3 = data3.groupby("shop_id",as_index=False).agg({'item_cnt_month':["mean","sum"]})

data3.columns = ["shop_id","item_mean_cnt","item_sum_cnt"]

data4 = data_combine[(data_combine.attributes=='Service')]

data4 = data4.groupby("shop_id",as_index=False).agg({'item_cnt_month':["mean","sum"]})

data4.columns = ["shop_id","item_mean_cnt","item_sum_cnt"]

data5 = data_combine[(data_combine.attributes=='Batteries')]

data5 = data5.groupby("shop_id",as_index=False).agg({'item_cnt_month':["mean","sum"]})

data5.columns = ["shop_id","item_mean_cnt","item_sum_cnt"]

fig = plt.figure(figsize=(16,20))

ax1 = fig.add_subplot(5,1,1)

ax2 = fig.add_subplot(5,1,2)

ax3 = fig.add_subplot(5,1,3)

ax4 = fig.add_subplot(5,1,4)

ax5 = fig.add_subplot(5,1,5)

sns.barplot(x ="shop_id",y = "item_mean_cnt",data=data,ax=ax1,palette="rocket_r").set_title('Programmes')

sns.barplot(x ="shop_id",y = "item_mean_cnt",data=data3,ax=ax2,palette="rocket_r").set_title('Books')

sns.barplot(x ="shop_id",y = "item_mean_cnt",data=data2,ax=ax3,palette="rocket_r").set_title('Payment Cards')

sns.barplot(x ="shop_id",y = "item_mean_cnt",data=data4,ax=ax4,palette="rocket_r").set_title('Service')

sns.barplot(x ="shop_id",y = "item_mean_cnt",data=data5,ax=ax5,palette="rocket_r").set_title('Batteries')
#产品季节性与趋势天真分解图

x ='Service'

ts = attribute_date_mean_cnt[(attribute_date_mean_cnt.attributes.isin([x]))][["date_block_num","item_cnt_month"]].set_index("date_block_num",drop=True)

res = sm.tsa.seasonal_decompose(ts,freq=12,model="addictive").plot()
#查看已上市月份数与月均销量间的关系（散点图）

have_month_cnt = data_combine[data_combine.month_count>=0]

item_month_cnt = have_month_cnt.groupby(["attributes","item_id","date_block_num"],as_index = False).agg({"item_cnt_month":"mean","month_count":"mean"})

item_month_cnt.drop(["date_block_num"],axis = 1,inplace = True)

fig,axes = plt.subplots(nrows=3,ncols=4,figsize = (18,14))

fig.subplots_adjust(wspace = 0.1,hspace = 0.3)

name_list = list(item_month_cnt.attributes.unique())

for attribute,ax in zip(name_list,axes.flat):

    sns.scatterplot(x = "month_count",y ="item_cnt_month",data = item_month_cnt[(item_month_cnt.attributes.isin([attribute]))],ax = ax).set_title(attribute)
#线图查看平均关系变化

data = item_month_cnt.groupby(["attributes","month_count"],as_index = False).agg({"item_cnt_month":"mean"})

fig,axes = plt.subplots(nrows=3,ncols=4,figsize = (18,14))

fig.subplots_adjust(wspace = 0.1,hspace = 0.3)

name_list = list(data.attributes.unique())

for attribute,ax in zip(name_list,axes.flat):

    sns.lineplot(x = "month_count",y ="item_cnt_month",data = data[(data.attributes.isin([attribute]))],ax = ax).set_title(attribute)
#近一年各属性产品月均销量和总销量的比较

attribute_mean_cnt = data_combine[data_combine.date_block_num.isin(range(22,34))].groupby(["attributes"],as_index=False).agg({"item_cnt_month":"mean"})

attribute_sum_cnt = data_combine[data_combine.date_block_num.isin(range(22,34))].groupby(["attributes"],as_index=False).agg({"item_cnt_month":"sum"})

fig = plt.figure(figsize = (16,10))

ax1 = fig.add_subplot(2,1,1)

ax2 = fig.add_subplot(2,1,2)

sns.barplot(x = "attributes",y = "item_cnt_month",data=attribute_mean_cnt,ax = ax1,palette="rocket_r").set_title("shop_mean_month_cnt_in_last_six_months")

sns.barplot(x = "attributes",y = "item_cnt_month",data=attribute_sum_cnt,ax = ax2,palette="rocket_r").set_title("shop_sum_month_cnt_in_last_six_months")

plt.show()
#出现在train_month中，但是近一年都没有销售的产品，直接可以预测为下一月也没有销售

last_year_items = data_combine[data_combine.max_month.isin(range(22,34))].item_id.unique()
#各型号下近一年产品种类计算

def get_unique(x):

    return len(set(x))

data = data_combine[data_combine.item_id.isin(last_year_items)]

submodel_items_count_year = data.groupby(["attributes","subclass"],as_index = False).agg({"item_id":get_unique,"item_cnt_month":"mean"})

submodel_items_count_year.columns = ["attributes","subclass","items_kinds","item_cnt_month"]

submodel_items_count_year.head(1)
#["The Games","Books","Gifts",'Programmes']

attribute ="The Games"

fig = plt.figure(figsize = (16,15))

ax1 = fig.add_subplot(3,1,1)

ax2 = fig.add_subplot(3,1,2)

ax3 = fig.add_subplot(3,1,3)

sns.barplot(x = "subclass",y ="item_cnt_month",data = submodel_items_count_year[(submodel_items_count_year.attributes.isin([attribute]))],ax = ax1,palette="rocket_r").set_title("month_average_cnt")

train_month2[(train_month2.attributes==attribute)].boxplot(column = "item_mean_price",by = "subclass",ax = ax2).set_title("average_price")

sns.barplot(x = "subclass",y = "items_kinds",data = submodel_items_count_year[submodel_items_count_year.attributes.isin([attribute])],ax = ax3,palette="rocket_r").set_title("item_kinds_in_last_year")

plt.show()
shop_submodel_cnt = data_combine.groupby(["shop_id","attributes","subclass","date_block_num"],as_index=False).agg({"item_cnt_month":"mean"})

shop_submodel_cnt.head(3)
shop_list = [21, 22, 24, 25, 26, 28, 31]

sub_model_list = ['PS', 'Xbox']

attribute ="Accesories"

fig,axes = plt.subplots(nrows=7,ncols=1,figsize = (16,20))

fig.subplots_adjust(hspace = 0.3)

name_list = shop_list

for shop_id,ax in zip(name_list,axes.flat):

    sns.lineplot(x = "date_block_num",y = "item_cnt_month",hue = "subclass",

                 data = shop_submodel_cnt[(shop_submodel_cnt.attributes==attribute)

                                      &(shop_submodel_cnt.shop_id==shop_id)&(shop_submodel_cnt.subclass.isin(sub_model_list))],ax = ax,markers =True).set_title(shop_id)
test_category_list = items_combine.item_category_id.unique()

item_categories[item_categories.item_category_id.isin(test_category_list)][["attributes","subclass","model"]]

data_combine.loc[((data_combine.attributes=='The Games')&(data_combine.subclass=="Game accessories")),"model"] = "Other"

data_combine.loc[((data_combine.attributes=='Payment Cards')&(data_combine.model.isna())),"model"] = "Other"

data_combine.loc[((data_combine.attributes=='Movies')&(data_combine.model.isna())),"model"] = "Other"

data_combine.loc[((data_combine.attributes=='Books')&(data_combine.model.isna())),"model"] = "Other" #仅根据model

#Music仅根据“subclass”

#Gifts仅根据"attribute"

#Programmes仅根据"subclass"

#attr_subclass_model
#attr_subclass_model

price_mean_cnt1 = data_combine[data_combine.attributes.isin(["Accesories",'The Games','Game consoles',

                                                          'Payment Cards','Movies'])].groupby(["attributes","subclass",

                                                                                               "model","item_id"],as_index=False\

                                                                                             ).agg({"item_mean_price":"mean","item_cnt_month":"mean"})

price_mean_cnt2 = data_combine[data_combine.attributes.isin(['Music', 'Programmes'])].groupby(["attributes","subclass","item_id"],as_index=False\

                                                                                             ).agg({"item_mean_price":"mean","item_cnt_month":"mean"})

price_mean_cnt3 = data_combine[data_combine.attributes.isin(['Gifts'])].groupby(["item_id"],as_index=False).agg({"item_mean_price":"mean",

                                                                                                        "item_cnt_month":"mean"})

price_mean_cnt4 = data_combine[data_combine.attributes.isin(['Books'])].groupby(["model","item_id"],as_index=False).agg({"item_mean_price":"mean",

                                                                                                        "item_cnt_month":"mean"})

price_mean_cnt5 = data_combine[data_combine.attributes.isin(['Service','Batteries'])][["attributes","item_mean_price","item_cnt_month"]]
fig,axes = plt.subplots(nrows=2,ncols=3,figsize = (18,10))

fig.subplots_adjust(wspace = 0.1)

name_list = ["Accesories",'The Games','Game consoles','Payment Cards','Movies']

for attribute,ax in zip(name_list,axes.flat):

    sns.scatterplot(x = "item_mean_price",y = "item_cnt_month",data = price_mean_cnt1[(price_mean_cnt1.attributes.isin([attribute]))],ax = ax,hue = "subclass").set_title(attribute)
#'The Games'

data = price_mean_cnt1[(price_mean_cnt1.attributes.isin(['The Games']))]

fig = plt.figure(figsize = (15,4))

ax1 = fig.add_subplot(1,1,1)

sns.scatterplot(x = "item_mean_price",y = "item_cnt_month",data = data,ax = ax1,hue = "subclass").set_title('The Games')

fig2,axes = plt.subplots(nrows=1,ncols=5,figsize = (18,5))

fig2.subplots_adjust(wspace = 0.1)

name_list = ['Android', 'Game accessories', 'PC', 'PS', 'Xbox']

for subclass,ax in zip(name_list,axes.flat):

    sns.scatterplot(x = "item_mean_price",y = "item_cnt_month",data = data[(data.subclass.isin([subclass]))],ax = ax,hue = "model").set_title(subclass)

plt.show()
data = price_mean_cnt1[(price_mean_cnt1.attributes.isin(['Movies']))]

data.model.unique()
#Movies

data = price_mean_cnt1[(price_mean_cnt1.attributes.isin(['Movies']))]

fig,axes = plt.subplots(nrows=1,ncols=3,figsize = (18,4))                                                     

fig.subplots_adjust(wspace = 0.1)

name_list = ['3D', 'Other', 'Collectible']

for sub_model,ax in zip(name_list,axes.flat):

    sns.regplot(x = "item_mean_price",y = "item_cnt_month",data = data[(data.model.isin([sub_model]))],ax = ax).set_title(sub_model)
#"Accesories"，'Game consoles'

fig = plt.figure(figsize=(16,5))

ax1= fig.add_subplot(1,2,1)

ax2 = fig.add_subplot(1,2,2)

sns.scatterplot(x = "item_mean_price",y = "item_cnt_month",data = price_mean_cnt1[(price_mean_cnt1.attributes.isin(["Accesories"]))],ax = ax1,hue = "model").set_title("Accesories")

sns.scatterplot(x = "item_mean_price",y = "item_cnt_month",data = price_mean_cnt1[(price_mean_cnt1.attributes.isin(['Game consoles']))],ax = ax2,hue = "model").set_title('Game consoles')

plt.show()
fig,axes = plt.subplots(nrows=1,ncols=2,figsize = (16,5))

name_list = ['Music', 'Programmes']

for attribute,ax in zip(name_list,axes.flat):

    sns.scatterplot(x = "item_mean_price",y = "item_cnt_month",data = price_mean_cnt2[(price_mean_cnt2.attributes.isin([attribute]))],ax = ax,hue = "subclass").set_title(attribute)
fig = plt.figure(figsize = (18,4))

ax1 = fig.add_subplot(1,4,1)

ax2 = fig.add_subplot(1,4,2)

ax3 = fig.add_subplot(1,4,3)

ax4 = fig.add_subplot(1,4,4)

sns.regplot(x = "item_mean_price",y = "item_cnt_month",data = price_mean_cnt3,ax = ax1).set_title("Gifts")

sns.regplot(x = "item_mean_price",y = "item_cnt_month",data = price_mean_cnt5[price_mean_cnt5.attributes=="Service"],ax = ax2).set_title("Service")

sns.regplot(x = "item_mean_price",y = "item_cnt_month",data = price_mean_cnt5[price_mean_cnt5.attributes=='Batteries'],ax = ax3).set_title('Batteries')

sns.scatterplot(x = "item_mean_price",y = "item_cnt_month",data = price_mean_cnt4,ax = ax4,hue = "model").set_title("Books")

plt.show()
#近1年各超市月均销量和总销量的比较

shop_mean_cnt = data_combine[data_combine.date_block_num.isin(range(22,34))].groupby(["shop_id"],as_index=False).agg({"item_cnt_month":"mean"})

shop_sum_cnt = data_combine[data_combine.date_block_num.isin(range(22,34))].groupby(["shop_id"],as_index=False).agg({"item_cnt_month":"sum"})

fig = plt.figure(figsize = (16,10))

ax1 = fig.add_subplot(2,1,1)

ax2 = fig.add_subplot(2,1,2)

sns.barplot(x = "shop_id",y = "item_cnt_month",data=shop_mean_cnt,ax = ax1,palette="rocket_r").set_title("shop_mean_month_cnt_in_last_year")

sns.barplot(x = "shop_id",y = "item_cnt_month",data=shop_sum_cnt,ax = ax2,palette="rocket_r").set_title("shop_sum_month_cnt_in_last_year")

plt.show()
shop_mean_cnt["rank"] = shop_mean_cnt["item_cnt_month"].rank(ascending = False)

shop_rank = shop.join(shop_mean_cnt.set_index("shop_id"),on = "shop_id")

shop_rank[shop_rank["rank"].notna()].sort_values(by = "rank")[["city","type","rank"]][:3]
shop_date_cnt = data_combine.groupby(["shop_id","date_block_num"],as_index = False).agg({"item_cnt_month":["mean","sum"]})

shop_date_cnt.columns = ["shop_id","date_block_num","item_mean_cnt","item_sum_cnt"]

shop_date_cnt.head(1)
f = plt.figure(figsize=(16,10))

ax1 = f.add_subplot(2,1,1)

ax2 = f.add_subplot(2,1,2)

sns.lineplot(x = "date_block_num",y = "item_cnt_month",data=date_mean_cnt ,ax = ax1).set_title("date_mean_cnt")

data = shop_date_cnt[shop_date_cnt.shop_id.isin([0, 1, 6, 21, 25, 28, 31, 42, 47])]

sns.lineplot(x = "date_block_num",y = "item_mean_cnt",data=data ,ax = ax2,hue = "shop_id").set_title("shop_date_mean_cnt")

plt.show()
f = plt.figure(figsize=(16,10))

ax1 = f.add_subplot(2,1,1)

ax2 = f.add_subplot(2,1,2)

sns.lineplot(x = "date_block_num",y = "item_cnt_month",data=date_mean_cnt ,ax = ax1).set_title("date_mean_cnt")

data = shop_date_cnt[shop_date_cnt.shop_id.isin([2, 3, 10, 34, 36, 39, 44, 45, 49])]

sns.lineplot(x = "date_block_num",y = "item_mean_cnt",data=data ,ax = ax2,hue = "shop_id").set_title("shop_date_mean_cnt")

plt.show()
shop_category_item = pd.DataFrame(columns=["category_kinds","item_kinds"])

shop_list = data_combine.shop_id.unique()

for i in shop_list:

    shop_category_item.loc[i,"category_kinds"] = len(data_combine[(data_combine.shop_id==i)&(data_combine.item_mean_price!=0)].item_category_id.unique())

    shop_category_item.loc[i,"item_kinds"] = len(data_combine[(data_combine.shop_id==i)&(data_combine.item_mean_price!=0)].item_id.unique())

shop_category_item.reset_index(inplace=True)

shop_category_item.columns = ["shop_id","category_kinds","item_kinds"]
shop_category_item=shop_category_item.join(shop_mean_cnt.set_index("shop_id"),on = "shop_id",how="right")

shop_category_item.tail(2)
sns.scatterplot(x="item_kinds",y = "item_cnt_month",data =shop_category_item)
date_shop_attr = data_combine[data_combine.month_count>=0].groupby(["date_block_num",'shop_id','attributes']\

                                                                 ,as_index = False).agg({"item_id":get_unique,"item_cnt_month":"mean"})

date_shop_attr.head(2)
shop_id = 28

fig,axes = plt.subplots(nrows=4,ncols=3,figsize=(16,16))

fig.subplots_adjust(wspace = 0.1,hspace = 0.3)

name_list = date_shop_attr.attributes.unique()

for attribute,ax in zip(name_list,axes.flat):

    sns.scatterplot(x="item_id",y = "item_cnt_month",data = date_shop_attr[(date_shop_attr.shop_id==shop_id)

                                                                           &(date_shop_attr.attributes==attribute)],ax = ax).set_title(attribute)
shop_price_stability = data_combine.groupby(["shop_id","attributes","date_block_num"],as_index = False).agg({'price_stability':"mean","item_cnt_month":"mean"})

shop_price_stability.head(2)
fig,axes = plt.subplots(nrows=4,ncols=3,figsize = (16,16))

fig.subplots_adjust(wspace = 0.1,hspace = 0.3)

name_list = shop_price_stability.attributes.unique()

for attribute,ax in zip(name_list,axes.flat):

    sns.scatterplot(x = "price_stability",y="item_cnt_month",

                    data = shop_price_stability[(shop_price_stability.attributes==attribute)],ax = ax).set_title(attribute)
holiday_month_mean_cnt = data_combine.groupby(["attributes","date_block_num"],\

                                             as_index = False).agg({"item_cnt_month":["mean","sum"]})

holiday_month_mean_cnt.columns = ["attributes","date_block_num","item_mean_cnt","item_sum_cnt"]

holiday_month_mean_cnt.head(1)