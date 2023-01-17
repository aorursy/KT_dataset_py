# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
dt_test = pd.read_csv('../input/test.csv')
dt_train = pd.read_csv('../input/train.csv')
dt_train.info()
dt_train.head()
dt_train.tail()
dt_train.columns
dt_train.corr()
f,ax = plt.subplots(figsize=(20,20))
sns.heatmap(dt_train.corr(), annot = True,linewidths=.4, fmt='.1f', ax=ax)
plt.show()
dt_train.describe()
#descriptive statistics summary
dt_train['SalePrice'].describe()
dt_train.shape
#histogram
sns.distplot(dt_train['SalePrice']);
# Scatter Plot 
# x = SalePrice, y = YearBuilt
dt_train.plot(kind='scatter',x = 'SalePrice', y = 'YearBuilt', alpha=0.5, color='b')
plt.xlabel('SalePrice')
plt.ylabel('YearBuilt')
plt.title('SalePrince or YearBuilt')
dt_train.plot(kind='scatter',x = 'SalePrice', y = 'YearRemodAdd', alpha=0.5, color='r')
plt.xlabel('SalePrice')
plt.ylabel('YearRemodAdd')
plt.title('SalePrince or YearRemodAdd')

# Line Plot
dt_train.SalePrice.plot(kind = 'line', color = 'g',label = 'SalePrice',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
dt_train.LotArea.plot(color = 'r',label = 'LotArea',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              
plt.ylabel('y axis')
plt.title('SalePrice or LotArea')           
plt.show()

# let's look at the average sales price
sale_mean=np.mean(dt_train['SalePrice'])
print("sale_mean:",sale_mean)
# let's look at the average Total rooms above grade
room_mean=np.mean(dt_train['TotRmsAbvGrd'])
print("room_mean:",room_mean)
# Now, let's find the Houses under the average price and the number of rooms above the average.
# Filtering pandas with logical_and
filt1=dt_train[np.logical_and(dt_train['SalePrice']<sale_mean,dt_train['TotRmsAbvGrd']>room_mean)]
filt1.head()
# example of what we learn above
def tuble_ex():
    t = (1,2,3)
    return t
a,b,c = tuble_ex()
print(a,b,c)
# Older than 2006
series_year = dt_train['YrSold']
x=2006
def f():
    x=series_year>2006
    return x
print(x)
print(f())
    
# let's look at unit area prices and mean
series_lvarea=dt_train['GrLivArea']
series_sale=dt_train['SalePrice']
def f():
    y=series_sale/series_lvarea
    return y
print(f())
area_saleprice=np.mean(f())
print('mean_area_saleprice:',area_saleprice)
# How can we learn what is built in scope
import builtins
dir(builtins)
#nested function
def square():
    def add():
        x=5
        y=20
        z=y-x
        return z
    return add()**2
print(square())
        
# quarter prices of houses
def f(series_sale,b=2,c=2):
    y=series_sale/(b+c)
    return y
print(f(series_sale))

# flexi arguments args and kwargs
def f(*args):
    for i in args:
            print(i)
f(series_year)
print("")
def f(**kwargs):
    for key,value in kwargs.items():
        print(key,"",value)
f(year='series_year',sale='series_sale',yes='no')
        
square= lambda x: x**2
print(square(5))
tot= lambda x,y,z: x+y-z
print(tot(15,62,25))
year1 = map(lambda x:x<2007,series_year)
# print(list(year1))
name="levent"
it = iter(name)
print(next(it))
print(next(it))
print(*it)
# year and sale price
list_year = list(series_year)
list_sale1 = list(series_sale)
print(type(list_year))
print(type(list_sale1))
z = zip(list_year,list_sale1)
# print(z)
z_list = list(z)
# print(z_list)
un_zip = zip(*z_list)
un_list1,un_list2=list(un_zip)
# print(un_list1)
# print(un_list2)
print(type(un_list2))
# house year mean
# list comprehension
house_year = [2018-i for i in list_year]
# print(house_year)
np.mean(house_year)
# Conditionals on iterable
num1 = [23,7,15,6,-20]
num2 = [i**2 if i==7 else i+7 if i>0 else i+50 for i in num1]
print(num2)
dt_train.head()
# assessment by number of rooms
rooms = sum(dt_train.TotRmsAbvGrd)/len(dt_train.TotRmsAbvGrd)
dt_train["rooms_level"] = ["roomy" if i > rooms else "scant" for i in dt_train.TotRmsAbvGrd]
dt_train.loc[:10, ["TotRmsAbvGrd","rooms_level"]]
dt_test.tail()
# dt_test or dt_train columns
dt_test.columns
print("test:",dt_test.shape)
print("train:",dt_train.shape)
# verbose info evaluation
dt_train.info(verbose=True)
# for example let's look at the wall coverings of the houses
print(dt_train['MasVnrType'].value_counts(dropna=False))
# and let's look Overall condition rating houses
print(dt_train['OverallCond'].value_counts(dropna=False)) # 10   Very Excellent no house, houses in general 5,6,7 category
dt_train.describe()
# let's find out if the house prices are contradictory
dt_train.boxplot(column='SalePrice')
plt.show()
# for example new data
data_new = dt_train.head()
data_new
# Rooms level melted
melted = pd.melt(frame=data_new, id_vars='YearBuilt', value_vars=['SalePrice','TotalBsmtSF','rooms_level','TotRmsAbvGrd'])
melted
# we have created new data with significant values compared to the year of home construction.
melted.pivot(index='YearBuilt',columns='variable',values='value')
data1 = dt_train.head()
data2 =dt_train.tail()
conc_data = pd.concat([data1,data2], axis=0, ignore_index=True)
conc_data
data3=dt_train['RoofStyle'].head(7)
data4=dt_train['Foundation'].head(7)
conc_data_col = pd.concat([data3,data4], axis=1)
conc_data_col
melted.dtypes
melted['value']=melted['value'].astype('bool')
melted['YearBuilt']=melted['YearBuilt'].astype('float')
melted.dtypes
# dt_train.info()
dt_train['Alley'].value_counts(dropna=False)
# assert statement
dt1=dt_train
dt1['Alley'].dropna(inplace=True)
assert 1==1 # return nothing because it is true
dt1['Alley'].value_counts(dropna=False) # no NaN
assert dt_train['Alley'].notnull().all()
dt_train['Alley'].head()
dt_train.head()
# mssubclass and saleprice new dataframe
class1 = list(dt_train['MSSubClass'])
price1 = list(dt_train['SalePrice'])
list_label = ["class1","price1"]
list_col = [class1,price1]
zipped = list(zip(list_label,list_col))
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df.head()
df["yearsold"] = 0
df.head()
dt1 = dt_train.loc[:,["GarageArea","GrLivArea","LotArea"]]
dt1.plot()
dt1.plot(subplots=True)
plt.show()
dt1.plot(kind="scatter", x="GarageArea", y="GrLivArea")
plt.show()
dt1.plot(kind="hist", y="GarageArea", bins=100, range=(0,1000))
plt.show()
fig, axes = plt.subplots(nrows=2,ncols=1)
dt1.plot(kind="hist",y="GarageArea", bins=100, range=(0,1000), normed=True, ax=axes[0])
dt1.plot(kind="hist",y="GarageArea", bins=100, range=(0,1000), normed=True, ax=axes[1], cumulative=True)
plt.savefig('graph.png')
plt.show()
# savefig problem ???
dt_train.describe()

time_list=["1999-12-24","1999-08-11"]
print(type(time_list))
print(type(time_list[1]))
datetime_object=pd.to_datetime(time_list)
print(type(datetime_object))
datetime_object
import warnings
warnings.filterwarnings("ignore")
# time columns
data2=dt_train.head()
data_list=["2016-10-12","2016-10-05","2016-09-15","2015-02-25","2015-03-07"]
datatime_object=pd.to_datetime(data_list)
data2["date"]=datatime_object
data2=data2.set_index("date")
data2
print(data2.loc["2015-02-25"])
print(data2.loc["2015-02-25":"2016-09-15"])
# year select mean
data2.resample("A").mean()
# monts select mean
data2.resample("M").mean()
# NaN input linear
data2.resample("M").first().interpolate("linear")
data2.resample("M").mean().interpolate("linear")
# read data
data=pd.read_csv('../input/train.csv')
data=data.set_index("Id")
data.head()
data["LotArea"][2]
data.LotArea[2]
data.loc[1,["LotArea"]]
data.loc[1:10,["LotArea","YearBuilt","SalePrice"]]
data.loc[10:1:-1,"MSSubClass":"Street"]
data.loc[1:10,"MoSold":]
boolean = data.LotArea>50000
data[boolean]
first_filter=data.LotArea>50000
second_filter=data.MSSubClass>50
data[first_filter&second_filter]
data.MSSubClass[data.BsmtFinSF1>2000]
def div(n):
    return n/2
data.BsmtFinSF1.apply(div)
data.BsmtFinSF1.apply(lambda n:n+n/2)
data["total_area"]=data.LotArea+data.GrLivArea+data.GarageArea
data.head()
print(data.index.name)
# lets change it
data.index.name= "index_name"
data.head()
data.head()
data3=data.copy()
data3.index.name = range(100,300,10)
data3.head()

data = pd.read_csv('../input/test.csv')
data.head()
data1=data.set_index(["MSZoning","LotConfig"])
data1.head(50)
dic = {"tream":["a","b","a","b"],"gender":["f","m","f","m"],"response":["10","21","61","11"],"age":["12","17","25","32"]}
df=pd.DataFrame(dic)
df
df.pivot(index="age",columns="gender",values="response")
df7=df.set_index(["tream","gender"])
df7
# df7.unstack(level=0)
# df1.unstack(level=1)
df2=df7.swaplevel(0,1)
df2
# MELTING DATA FRAMES
df

pd.melt(df,id_vars="tream", value_vars=["age","response"])

# CATEGORICALS AND GROUPBY
df
df.groupby("tream").mean()
df.groupby("tream").age.max()
df.groupby("tream")[["age","response"]].min()
df.info()