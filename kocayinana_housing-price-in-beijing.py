import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from pandas import DataFrame

from numpy import nan as NA

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

import time,datetime

import re



import os



#Load datasets

filepath="../input/new.csv";

house_data=pd.read_csv("../input/new.csv",encoding="gbk")

name=os.listdir("../input")



print("%s has been loaded as house_data" %str(name)[2:-2])

# Any results you write to the current directory are saved as output.
# 处理tradeTime,将日期变成时间戳，然后再用2017-01-01的时间戳减之

timeStamp=[]

timeSub=time.strptime("2017-01-01", "%Y-%m-%d")

stampSub= int(time.mktime(timeSub))



tradeTime_copy=house_data["tradeTime"]



for i in range(len(house_data["tradeTime"])):

    timeArray = time.strptime(str(tradeTime_copy[i]), "%Y-%m-%d")

    stamp= (int(stampSub-time.mktime(timeArray)))/86400

    timeStamp.append(stamp)

    

house_data["tradeTime"]=timeStamp

    

#处理floor，直接提取里面的数字，也可将里面的“顶”、“高”、“中“、”低“、”底“等关键字提取出来，另外增加一列作为一个新的特征，

#floor里面有异常数据，比如”混凝钢构“

floor_list=[]

floor_copy=house_data["floor"]



for i in range(len(floor_copy)):

    if re.findall('(\d+)',str(floor_copy[i])):

        f1=re.findall('(\d+)',str(floor_copy[i]))

        f1_int=int(f1[0])

        floor_list.append(f1_int)

    else:

        floor_list.append(-1)#32个



house_data["floor"]=floor_list
def process_raw_data(raw_data,columns_selected):

    house_select_columns=raw_data[columns_selected]

       

    with_NAME_row=house_select_columns.index[house_select_columns["livingRoom"]=="#NAME?"].tolist()

    house_select_columns=house_select_columns.drop(with_NAME_row,axis=0)

    with_weizhi_row=house_select_columns.index[house_select_columns["constructionTime"]=="未知"].tolist()

    house_select_columns=house_select_columns.drop(with_weizhi_row,axis=0)

#   with_nan_row=house_select_columns.index[house_select_columns["buildingType"].isnull()].tolist()

#   house_select_columns=house_select_columns.drop(with_nan_row,axis=0)

    

    house_select_columns["drawingRoom"]=house_select_columns["drawingRoom"].astype("float")

    house_select_columns["bathRoom"]=house_select_columns["bathRoom"].astype("float")

    house_select_columns["livingRoom"]=house_select_columns["livingRoom"].astype("float")

    

#   ——————————————————————————————————————————————————————————————————————————————————————————————

    #第一种方法：删除所有含有NaN的行，MAE:5307.469156248709

    house_selected_columns=house_select_columns.dropna(axis=0)

#   ——————————————————————————————————————————————————————————————————————————————————————————————



#   ——————————————————————————————————————————————————————————————————————————————————————————————

    #第二种方法，利用Imputation函数，用平均值代替缺失的值,MAE:4280.556089258759

    my_imputer = SimpleImputer()

    house_selected_columns = pd.DataFrame(my_imputer.fit_transform(house_select_columns))

    house_selected_columns.columns = house_select_columns.columns

#   ——————————————————————————————————————————————————————————————————————————————————————————————



#   ——————————————————————————————————————————————————————————————————————————————————————————————

#     #第三种方法，增加一个列,MAE:4377.901482918298

#     cols_with_missing = [col for col in house_select_columns.columns

#                      if house_select_columns[col].isnull().any()]

        

#     for col in cols_with_missing:

#         house_select_columns[col + '_was_missing'] = house_select_columns[col].isnull()

    

#     my_imputer = SimpleImputer()

#     house_selected_columns = pd.DataFrame(my_imputer.fit_transform(house_select_columns))

#     house_selected_columns.columns = house_select_columns.columns

#   ——————————————————————————————————————————————————————————————————————————————————————————————    



    y_selected=house_selected_columns["price"]

    X_selected=house_selected_columns.drop(["price"],axis=1)

    

    return X_selected,y_selected
columns_selected=["Lng","Lat","square","tradeTime","DOM","followers","livingRoom","drawingRoom","kitchen",

                  "bathRoom","floor","buildingType","constructionTime","renovationCondition","buildingStructure",

                  "ladderRatio","district","price"]

X,y=process_raw_data(house_data,columns_selected)

#X=house_select_columns.select_dtypes(exclude="object")#选择不是object类型的特征

# X.head()
#fiveYearsProperty对价格影响不大

# house_data["fiveYearsProperty"].describe()

# s1=house_data.index[house_data["fiveYearsProperty"]==1].tolist()

# s2=house_data.index[house_data["fiveYearsProperty"]==0].tolist()

# house_data["price"][s1].describe()

# house_data["price"][s2].describe()


# house_data.describe()

# house_data["DOM"].describe()

#with_nan_row=X.index[X["livingRoom"].isnull()].tolist()

#print(len(with_nan_row))

# house_data=house_data.dropna()



# house_data.describe()

#plt.figure(figsize=(16,10))

#sns.scatterplot(x=DOM_data,y=house_data["price"])

#print(len(with_nan_row))

#This dataset has totally 318851 samples,some features like DOM, buildingType,elevator,fiveYearsProperty,

#subway,communityAverage have less than 318851 statistical datas.

#X.describe()

#with_weizhi_row=X.index[X["DOM"]==1].tolist()

#len(with_weizhi_row)



#X.head()

#X.describe()

#y.describe()

#X['drawingRoom'].describe()

#X['livingRoom'].describe()

#X['bathRoom'].describe()

#len(X.index[X["buildingType"].isnull()].tolist())

#X['buildingType'].unique()#有nan

#np.isnan(X['drawingRoom']).unique()

#X.isnull().any()
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)
# Function for comparing different approaches

def score_dataset(X_train, X_valid, y_train, y_valid,n_estimators):

    model = RandomForestRegressor(n_estimators=n_estimators,max_depth=25, min_samples_split=120,min_samples_leaf=20,max_features=7 ,oob_score=True,random_state=1)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    

    #plt.figure(figsize=(10,10))

    #sns.scatterplot(x=preds,y=y_valid)

    #sns.jointplot(x=preds,y=y_valid,kind="hex")

    return mean_absolute_error(y_valid, preds)

# max_depth=25, min_samples_split=120,min_samples_leaf=20,max_features=7 ,oob_score=True
# for i in [20]:

#     print(score_dataset(X_train,X_valid, y_train, y_valid,i))

#house_data["followers"].isnull().unique()

#url	id	Lng	Lat	Cid	tradeTime	DOM	followers

#totalPrice	price	square	livingRoom	drawingRoom	kitchen	bathRoom

#floor	buildingType	constructionTime	renovationCondition	buildingStructure

#ladderRatio	elevator	fiveYearsProperty	subway	district	communityAverage

# house_data.head()

#sns.scatterplot(x=house_data["kitchen"],y=house_data["price"])

house_data["kitchen"].describe()




house_data["floor"].head()
house_data["floor"][:-200]
#查询某一行和和某一列的数据可以通过.iloc实现

#print(house_data.iloc(0)[1])

#print(house_data.iloc(1)[1])
#集中显示每一列数据的描述

house_columns=[house_data.columns]

# for i in range(26):

#     print(house_data[house_columns[0][i]].describe())

#     print("Unique counts in each column",i,":", len(house_data[house_columns[0][i]].unique()))

#     print("Any NULL values?",house_data[house_columns[0][i]].isnull().unique())

#     print("____________________________________________________________________________")

#Lng特征

# plt.figure(figsize=(16,8))

#sns.lineplot(x=house_data["Lng"],y=house_data["price"])

# sns.kdeplot(data=pd.DataFrame(house_data,columns = ["Lng","price"]),shade=True)
#Lat特征

# sns.kdeplot(data=pd.DataFrame(house_data,columns = ["Lat","price"]),shade=True)
#Lat and Lng analysis with plot

#plt.figure(figsize=(10,10))

#cmap = sns.cubehelix_palette(dark=1, light=.8, as_cmap=True)

#sns.scatterplot(x=house_data["Lng"],y=house_data["Lat"],hue=house_data["district"],legend="full",palette='Set3')

# 1东城区 2丰台区 3通州区 4大兴区 5房山区 6昌平区 7朝阳区 8海淀区 9石景山区 10西城区 11平谷区 12门头沟区 13顺义区

# col_n = ["Lat", "Lng", "price"]

# a = pd.DataFrame(house_data,columns = col_n)

#sns.heatmap(a)

# b.head()
#tradeTime特征

# plt.figure(figsize=(16,8))

# sns.lineplot(x=house_data["tradeTime"],y=house_data["price"])
#DOM特征

# plt.figure(figsize=(16,8))

# sns.lineplot(x=house_data["DOM"][:-10000:10],y=house_data["price"][:-10000:10])

#DOM存在NULL值需要进一步处理

import seaborn as sns

import matplotlib.pyplot as plt



#print(len(bjhp_data))

price_copy=bjhp_data["price"][:10000].copy()



for i in range(len(price_copy)):

    #print(price_copy[i])

    

    if price_copy[i]>75000:

        price_copy[i]=11

    elif price_copy[i]>70000 and price_copy[i]<=75000:

        price_copy[i]=10

    elif price_copy[i]>65000 and price_copy[i]<=70000:

        price_copy[i]=9

    elif price_copy[i]>60000 and price_copy[i]<=65000:

        price_copy[i]=8

    elif price_copy[i]>55000 and price_copy[i]<=60000:

        price_copy[i]=7

    elif price_copy[i]>50000 and price_copy[i]<=55000:

        price_copy[i]=6

    elif price_copy[i]>45000 and price_copy[i]<=50000:

        price_copy[i]=5

    elif price_copy[i]<=45000 and price_copy[i]>40000:

        price_copy[i]=4

    elif price_copy[i]<=40000 and price_copy[i]>35000:

        price_copy[i]=3

    elif price_copy[i]<=35000 and price_copy[i]>30000:

        price_copy[i]=2

    else:

        price_copy[i]=1



plt.figure(figsize=(20,16))

sns.jointplot(x=bjhp_data["Lat"][:5000],y=bjhp_data["Lng"][:5000],kind="kde")
plt.figure(figsize=(20,10))

sns.distplot(a=bjhp_data['price'][:10000],kde=False)
plt.figure(figsize=(20,16))

sns.regplot(x=bjhp_data["square"][:1000],y=bjhp_data["followers"][:1000])
plt.figure(figsize=(20,16))

#bjhp_data.describe()

constructionTime=bjhp_data["constructionTime"]

#constructionTime.unique()

#constructionTime.mean()

cons=constructionTime.replace('未知',"2004")

cons=cons.replace('0',"2004")

cons=cons.replace('1',"2004")

cons.unique()

followers=bjhp_data["followers"]

# followers.describe()

sns.jointplot(x=cons[:300],y=bjhp_data["followers"][:300],kind="kde")