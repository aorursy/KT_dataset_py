# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install scipy

!pip install seaborn

import pandas as pd

import numpy as np

import scipy

import matplotlib

import seaborn as sns

import matplotlib.pyplot as plt

import missingno as msno




test_data = pd.read_csv("/kaggle/input/usedcarpred/used_car_testA_20200313.csv",sep = " ")

train_data = pd.read_csv("/kaggle/input/usedcarpred/used_car_train_20200313.csv",sep = " ")

print("---------test_data.head()---------------------")

print(test_data.head().append(test_data.tail()))



print("---------test_data.shape----------------------")



print(test_data.shape)



print("---------train_data.head()---------------------")

print(train_data.head())



print("---------train_data.shape----------------------")



print(train_data.shape)

print("---------test_data.describe()---------------------")

print(test_data.describe())



print("---------test_data.info----------------------")



print(test_data.info)



print("---------train_data.describe()---------------------")

print(train_data.describe())



print("---------train_data.info----------------------")



print(train_data.info)
print("------------test缺值统计--------------------")



# print(test_data.isna().sum())

print(test_data.isnull().sum())



print("-----------train缺值统计--------------------")



# print(train_data.isna().sum())

print(train_data.isnull().sum())
print("---------------train_data缺失数据---------------------")



train_missing = train_data.isna().sum()

train_missing = train_missing[train_missing > 0]

print(train_missing)

train_missing.sort_values(inplace=True)

train_missing.plot.bar(train_missing)





print("---------------test_data缺失数据---------------------")



test_missing = test_data.isna().sum()

test_missing = test_missing[test_missing > 0]

print(test_missing)

test_missing.sort_values(inplace=True)

test_missing.plot.bar(test_missing)
# train 数据的统计



msno.bar(train_data.sample(1000))
# test 数据的统计



msno.bar(test_data.sample(1000))
# train 缺失数据分布情况可视化



msno.matrix(train_data.sample(250))
# train 缺失数据分布情况可视化



msno.matrix(test_data.sample(250))
print("-------------------train_data 相关信息---------------------")

train_data.info()
print("-------------------train_data 相关信息---------------------")

test_data.info()
train_data['notRepairedDamage'].value_counts()
test_data['notRepairedDamage'].value_counts()
train_data['notRepairedDamage'].replace("-",np.nan, inplace = True)

train_data['notRepairedDamage'].value_counts()
test_data['notRepairedDamage'].replace("-",np.nan, inplace = True)

test_data['notRepairedDamage'].value_counts()
print("----------------------train 数据缺失统计-------------------------")

train_data.isna().sum()

print("----------------------test 数据缺失统计-------------------------")

test_data.isna().sum()
train_data["seller"].value_counts()
train_data["offerType"].value_counts()
del train_data["offerType"]

del train_data["seller"]

del test_data["offerType"]

del test_data["seller"]
train_data.isna().sum()
test_data.isna().sum()
train_data["price"]
train_data["price"].value_counts()
import scipy.stats as st

y = train_data["price"]

plt.figure(1)

plt.title("Johnson SU")

sns.distplot(y, kde=False, fit=st.johnsonsu)

plt.figure(2)

plt.title("normal")

sns.distplot(y, kde=False, fit=st.norm)

plt.figure(3)

plt.title("logNorm")

sns.distplot(y, kde=False, fit=st.lognorm)
sns.distplot(train_data["price"])

print("skew :" , train_data["price"].skew())

print("kurt :" , train_data["price"].kurt())
train_data.skew(), train_data.kurt()
sns.distplot(train_data.skew(),color = "blue", axlabel = "skewness")
sns.distplot(train_data.kurt(),color = "red", axlabel = "kurtness")
plt.hist(train_data["price"], orientation = 'vertical')
plt.hist(np.log(train_data["price"]), orientation = 'vertical')
train_data.info()
price_label = train_data["price"]



numeric_features = ['power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13','v_14']



categorical_features = ['name', 'model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'regionCode']
print("---------numeric_features-----------")

for cat in numeric_features:

    print(cat , "特征：")

    print("{}特征有{}个不同的值".format(cat, str(train_data[cat].nunique())))

    print(train_data[cat].value_counts())





    
print("---------numeric_features-----------")

for cat in categorical_features:

    print(cat , "特征：")

    print("{}特征有{}个不同的值".format(cat, str(train_data[cat].nunique())))

    print(train_data[cat].value_counts())
numeric_features.append("price")

numeric_features
train_data.head
price_numeric = train_data[numeric_features]

correlation = price_numeric.corr()

correlation["price"].sort_values(ascending=False)
plt.subplots(figsize = (7,7))

plt.title("correlation of Numeric Features with Price", y = 1, size = 16)

sns.heatmap(correlation,square = True,  vmax=0.8)

del price_numeric['price']
price_numeric
print("------------------特征值及峰值---------------------")

for col in numeric_features:

    print("{:15}".format(col), "Skewness: {:05.2f}".format(train_data[col].skew()),"             ", "Kurtosis:{:05.2f}".format(train_data[col].kurt()))
f = pd.melt(train_data, value_vars = numeric_features)

g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)

g.map(sns.distplot, "value")
sns.set()

columns = ['price', 'v_12', 'v_8' , 'v_0', 'power', 'v_5',  'v_2', 'v_6', 'v_1', 'v_14']

sns.pairplot(train_data[columns], height = 4 ,kind ='scatter',diag_kind='kde')
train_data.columns
price_label
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = plt.subplots(nrows=5, ncols=2, figsize=(24, 20))

# ['v_12', 'v_8' , 'v_0', 'power', 'v_5',  'v_2', 'v_6', 'v_1', 'v_14']

v_12_scatter_plot = pd.concat([price_label,train_data['v_12']],axis = 1)

sns.regplot(x='v_12',y = 'price', data = v_12_scatter_plot,scatter= True, fit_reg=True, ax=ax1)

v_8_scatter_plot = pd.concat([price_label,train_data['v_8']],axis = 1)

sns.regplot(x='v_8',y = 'price', data = v_8_scatter_plot,scatter= True, fit_reg=True, ax=ax2)

v_0_scatter_plot = pd.concat([price_label,train_data['v_0']],axis = 1)

sns.regplot(x='v_0',y = 'price', data = v_0_scatter_plot,scatter= True, fit_reg=True, ax=ax3)

power_scatter_plot = pd.concat([price_label,train_data['power']],axis = 1)

sns.regplot(x='power',y = 'price', data = power_scatter_plot,scatter= True, fit_reg=True, ax=ax4)

v_5_scatter_plot = pd.concat([price_label,train_data['v_5']],axis = 1)

sns.regplot(x='v_5',y = 'price', data = v_5_scatter_plot,scatter= True, fit_reg=True, ax=ax5)

v_2_scatter_plot = pd.concat([price_label,train_data['v_2']],axis = 1)

sns.regplot(x='v_2',y = 'price', data = v_2_scatter_plot,scatter= True, fit_reg=True, ax=ax6)

v_6_scatter_plot = pd.concat([price_label,train_data['v_6']],axis = 1)

sns.regplot(x='v_6',y = 'price', data = v_6_scatter_plot,scatter= True, fit_reg=True, ax=ax7)

v_1_scatter_plot = pd.concat([price_label,train_data['v_1']],axis = 1)

sns.regplot(x='v_1',y = 'price', data = v_1_scatter_plot,scatter= True, fit_reg=True, ax=ax8)

v_14_scatter_plot = pd.concat([price_label,train_data['v_14']],axis = 1)

sns.regplot(x='v_14',y = 'price', data = v_14_scatter_plot,scatter= True, fit_reg=True, ax=ax9)

v_13_scatter_plot = pd.concat([price_label,train_data['v_13']],axis = 1)

sns.regplot(x='v_13',y = 'price', data = v_13_scatter_plot,scatter= True, fit_reg=True, ax=ax10)
for fea in categorical_features:

    print("{}特征有个{}不同的值".format(fea, train_data[fea].nunique()))
categorical_features = [ 'model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage']



for cat in categorical_features:

    train_data[cat] = train_data[cat].astype('category')

    if train_data[cat].isnull().any():

        train_data[cat] = train_data[cat].cat.add_categories(['MISSING'])

        train_data[cat] = train_data[cat].fillna('MISSING')



def boxplot(x, y, **kwargs):

    sns.boxplot(x=x, y=y)

    x=plt.xticks(rotation=90)



f = pd.melt(train_data, id_vars=['price'], value_vars=categorical_features)

g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)

g = g.map(boxplot, "value", "price")
categorical_features
import pandas_profiling
pfr = pandas_profiling.ProfileReport(train_data)

pfr.to_file("./example.html")
import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

from operator import itemgetter
path = './datalab/231784/'

test_data = pd.read_csv("/kaggle/input/usedcarpred/used_car_testA_20200313.csv",sep = " ")

train_data = pd.read_csv("/kaggle/input/usedcarpred/used_car_train_20200313.csv",sep = " ")

print(train_data.shape)

print(test_data.shape)

def outliers_proc(data, col_name, scale=3):

    """

    用于清洗异常值，默认用 box_plot（scale=3）进行清洗

    :param data: 接收 pandas 数据格式

    :param col_name: pandas 列名

    :param scale: 尺度

    :return:

    """



    def box_plot_outliers(data_ser, box_scale):

        """

        利用箱线图去除异常值

        :param data_ser: 接收 pandas.Series 数据格式

        :param box_scale: 箱线图尺度，

        :return:

        """

        iqr = box_scale * (data_ser.quantile(0.75) - data_ser.quantile(0.25)) # 四分位数的间距

        val_low = data_ser.quantile(0.25) - iqr # 最小值

        val_up = data_ser.quantile(0.75) + iqr # 最大值

        print(data_ser.quantile(0.75))

        print(iqr)

        rule_low = (data_ser < val_low)

        rule_up = (data_ser > val_up)

        return (rule_low, rule_up), (val_low, val_up)



    data_n = data.copy()

    data_series = data_n[col_name]

    rule, value = box_plot_outliers(data_series, box_scale=scale)

    print(rule[0])

    index = np.arange(data_series.shape[0])[rule[0] | rule[1]]

    print("Delete number is: {}".format(len(index)))

    data_n = data_n.drop(index)

    data_n.reset_index(drop=True, inplace=True)

    print("Now column number is: {}".format(data_n.shape[0]))

    index_low = np.arange(data_series.shape[0])[rule[0]]

    outliers = data_series.iloc[index_low]

    print("Description of data less than the lower bound is:")

    print(pd.Series(outliers).describe())

    index_up = np.arange(data_series.shape[0])[rule[1]]

    outliers = data_series.iloc[index_up]

    print("Description of data larger than the upper bound is:")

    print(pd.Series(outliers).describe())

    

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))

    sns.boxplot(y=data[col_name], data=data, palette="Set1", ax=ax[0])

    sns.boxplot(y=data_n[col_name], data=data_n, palette="Set1", ax=ax[1])

    return data_n
train_data = outliers_proc(train_data, 'power', scale=3)
train_data["train"] = 0

test_data["train"] = 1

data = pd.concat([train_data, test_data],sort=False)

print(data.head)

print(data.tail)
data.columns
data["used_time"] = (pd.to_datetime(data["creatDate"], format = "%Y%m%d", errors = "coerce") - pd.to_datetime(data["regDate"], format = "%Y%m%d", errors = "coerce")).dt.days
data["used_time"].isnull().sum()
data["city"] = data["regionCode"].apply(lambda x : str(x)[:-3])
data["city"]
data = data

data
train_gb = train_data.groupby("brand")

all_info = {}

for kind, kind_info in train_gb:

    info = {}

    kind_info = kind_info[kind_info["price"] > 0]

    info["brand_amount"] = len(kind_info)

    info["brand_price_max"] = kind_info.price.max()

    info["brand_price_min"] = kind_info.price.min()

    info["brand_price_median"] = kind_info.price.median()

    info["brand_price_sum"] = kind_info.price.sum()

    info["brand_price_std"] = kind_info.price.std()

    info["brand_price_average"] = round(kind_info.price.sum() / (len(kind_info) + 1), 2)

    all_info[kind] = info

brand_fe = pd.DataFrame(all_info).T.reset_index().rename(columns={"index": "brand"})

data = data.merge(brand_fe, how='left', on='brand')

    

     
data
bins = [i*10 for i in range(31)]

data['power_bin'] = pd.cut(data['power'], bins, labels=False)

data[["power_bin","power"]].head()

data = data.drop(["regDate","regionCode","creatDate"],axis =1)
print(data.shape)

print(data.columns)
data.to_csv('data_for_tree.csv', index=0)
data['power'].plot.hist()
train_data["power"].plot.hist()
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()



data['power'] = np.log(data['power'] + 1) 

data['power'] = ((data['power'] - np.min(data['power'])) / (np.max(data['power']) - np.min(data['power'])))

data['power'].plot.hist()
data["kilometer"].plot.hist()
data['kilometer'] = np.log(data['kilometer'] + 1) 

data['kilometer'] = ((data['kilometer'] - np.min(data['kilometer'])) / (np.max(data['kilometer']) - np.min(data['kilometer'])))

data['kilometer'].plot.hist()
data.columns
def max_min(x):

    return (x - np.min(x)) / (np.max(x) - np.min(x))



data['brand_amount'] = ((data['brand_amount'] - np.min(data['brand_amount'])) / 

                        (np.max(data['brand_amount']) - np.min(data['brand_amount'])))

data['brand_price_average'] = ((data['brand_price_average'] - np.min(data['brand_price_average'])) / 

                               (np.max(data['brand_price_average']) - np.min(data['brand_price_average'])))

data['brand_price_max'] = ((data['brand_price_max'] - np.min(data['brand_price_max'])) / 

                           (np.max(data['brand_price_max']) - np.min(data['brand_price_max'])))

data['brand_price_median'] = ((data['brand_price_median'] - np.min(data['brand_price_median'])) /

                              (np.max(data['brand_price_median']) - np.min(data['brand_price_median'])))

data['brand_price_min'] = ((data['brand_price_min'] - np.min(data['brand_price_min'])) / 

                           (np.max(data['brand_price_min']) - np.min(data['brand_price_min'])))

data['brand_price_std'] = ((data['brand_price_std'] - np.min(data['brand_price_std'])) / 

                           (np.max(data['brand_price_std']) - np.min(data['brand_price_std'])))

data['brand_price_sum'] = ((data['brand_price_sum'] - np.min(data['brand_price_sum'])) / 

                           (np.max(data['brand_price_sum']) - np.min(data['brand_price_sum'])))
# 对类别特征进行 OneEncoder

data = pd.get_dummies(data, columns=['model', 'brand', 'bodyType', 'fuelType',

                                     'gearbox', 'notRepairedDamage', 'power_bin'])
data.columns
# 这份数据可以给 LR 用

data.to_csv('data_for_lr.csv', index=0)
# 相关性分析

print(data['power'].corr(data['price'], method='spearman'))

print(data['kilometer'].corr(data['price'], method='spearman'))

print(data['brand_amount'].corr(data['price'], method='spearman'))

print(data['brand_price_average'].corr(data['price'], method='spearman'))

print(data['brand_price_max'].corr(data['price'], method='spearman'))

print(data['brand_price_median'].corr(data['price'], method='spearman'))
# 当然也可以直接看图

data_numeric = data[['power', 'kilometer', 'brand_amount', 'brand_price_average', 

                     'brand_price_max', 'brand_price_median']]

correlation = data_numeric.corr()



f , ax = plt.subplots(figsize = (7, 7))

plt.title('Correlation of Numeric Features with Price',y=1,size=16)

sns.heatmap(correlation,square = True,  vmax=0.8)