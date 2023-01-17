#basic

import numpy as np 

import pandas as pd

#viz

import seaborn as sns

import matplotlib.pyplot as plt

color = sns.color_palette()



#others

import subprocess

from subprocess import check_output

import gc



%matplotlib inline



files=check_output(["ls", "../input"]).decode("utf8")

#Check the number of row of each file

for file in files.split("\n"):

    path='../input/'+file

    popenobj=subprocess.Popen(['wc', '-l', path], stdout=subprocess.PIPE, 

                                              stderr=subprocess.PIPE)

    result,error= popenobj.communicate()

    print("The file :",file,"has :",result.strip().split()[0],"rows")
#train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

stores = pd.read_csv("../input/stores.csv")

items = pd.read_csv("../input/items.csv")

trans = pd.read_csv("../input/transactions.csv")

oil = pd.read_csv("../input/oil.csv")

holiday = pd.read_csv("../input/holidays_events.csv")

print("done")

#check memory use for the two biggest files - train and test

#mem_train = train.memory_usage(index=True).sum()

mem_test=test.memory_usage(index=True).sum()

#print("train dataset uses ",mem_train/ 1024**2," MB")

print("test dataset uses ",mem_test/ 1024**2," MB")

# checking contents in train

test.head()
# optimize test.csv

# First check the contents of train.csv

print(test.max())

print(test.min())

#check datatypes

print(test.dtypes)
#There are only 54 stores

test['store_nbr'] = test['store_nbr'].astype(np.uint8)

# The ID column is a continuous number from 1 to 128867502 in train and 128867503 to 125497040 in test

test['id'] = test['id'].astype(np.uint32)

# item number is unsigned 

test['item_nbr'] = test['item_nbr'].astype(np.uint32)

#Converting the date column to date format

test['date']=pd.to_datetime(test['date'],format="%Y-%m-%d")

#check memory

print(test.memory_usage(index=True))

new_mem_test=test.memory_usage(index=True).sum()

print("test dataset uses ",new_mem_test/ 1024**2," MB after changes")

print("memory saved =",(mem_test-new_mem_test)/ 1024**2," MB")
print(test.memory_usage())



#check range of float 16

min_value = np.finfo(np.float16).min

max_value = np.finfo(np.float16).max

print("range of float16 is",min_value,max_value)
# taking a peak

#train = pd.read_csv("../input/train.csv",nrows=10000)

#print(train.dtypes)



dtype_dict={"id":np.uint32,

            "store_nbr":np.uint8,

            "item_nbr":np.uint32,

            "unit_sales":np.float16

           }



#test for 10000 rows

train_part1 = pd.read_csv("../input/train.csv",nrows=100,dtype=dtype_dict,usecols=[0,2,3,4])

print(train_part1.describe())

print(train_part1.dtypes)









# read in the date col,on promo col

#specify the column number which has the date in the parse_dates as a list

train_part2=pd.read_csv("../input/train.csv",nrows=10000,dtype=dtype_dict,usecols=[1,5],parse_dates=[0])

train_part2['Year'] = pd.DatetimeIndex(train_part2['date']).year

train_part2['Month'] = pd.DatetimeIndex(train_part2['date']).month

train_part2['Day'] =pd.DatetimeIndex(train_part2['date']).day.astype(np.uint8)

del(train_part2['date'])

train_part2['Day']=train_part2['Day'].astype(np.uint8)

train_part2['Month']=train_part2['Month'].astype(np.uint8)

train_part2['Year']=train_part2['Year'].astype(np.uint16)



#impute the missing values to be -1

train_part2["onpromotion"].fillna(0, inplace=True)

train_part2["onpromotion"]=train_part2["onpromotion"].astype(np.int8)

print(train_part2.head())

print(train_part2.dtypes)
# now scaling it to the entire dataset of train



train_part2=pd.read_csv("../input/train.csv",dtype=dtype_dict,usecols=[1,5],parse_dates=[0])

train_part2['Year'] = pd.DatetimeIndex(train_part2['date']).year

train_part2['Month'] = pd.DatetimeIndex(train_part2['date']).month

train_part2['Day'] =pd.DatetimeIndex(train_part2['date']).day.astype(np.uint8)

del(train_part2['date'])

train_part2['Day']=train_part2['Day'].astype(np.uint8)

train_part2['Month']=train_part2['Month'].astype(np.uint8)

train_part2['Year']=train_part2['Year'].astype(np.uint16)



#impute the missing values to be -1

train_part2["onpromotion"].fillna(0, inplace=True)

train_part2["onpromotion"]=train_part2["onpromotion"].astype(np.int8)

print(train_part2.head())

print(train_part2.dtypes)

# scaling part 1 to the entire dataset

dtype_dict={"id":np.uint32,

            "store_nbr":np.uint8,

            "item_nbr":np.uint32,

            "unit_sales":np.float32

           }



train_part1 = pd.read_csv("../input/train.csv",dtype=dtype_dict,usecols=[0,2,3,4])

print(train_part1.dtypes)
# joining part one and two

# For people familiar with R , the equivalent of cbind in pandas is the following command

train = pd.concat([train_part1.reset_index(drop=True), train_part2], axis=1)

#drop temp files

del(train_part1)

del(train_part2)

#Further Id is just an indicator column, hence not required for analysis

id=train['id']

del(train['id'])

# check memory

print(train.memory_usage())

#The extracted train.csv file is approx 5 GB

mem_train=5*1024**3

new_mem_train=train.memory_usage().sum()

print("Train dataset uses ",new_mem_train/ 1024**2," MB after changes")

print("memory saved is approx",(mem_train-new_mem_train)/ 1024**2," MB")
# summary stats

train['unit_sales'].describe()

#check

train['unit_sales'].isnull().sum()
# Using pandas group by and aggregate

#sale_day_store_level=train.groupby(['Year','Month','Day','store_nbr'])['unit_sales'].sum()

#sale_day_item_level=train.groupby(['Year','Month','Day','item_nbr'])['unit_sales'].sum()



#kernal got stuck when trying this piece of code, hence splitting into chunks(chunks of one year) and appending

train_2013=train.loc[train['Year']==2013]

train_2014=train.loc[train['Year'] ==2014]

train_2015=train.loc[train['Year'] ==2015]

train_2016=train.loc[train['Year'] ==2016]

train_2017=train.loc[train['Year'] ==2017]
def aggregate_level1(df):

    '''writing a function to get item and store level summary metrics for a specific year'''

#day-store level

    sale_day_store_level=df.groupby(['Year','Month','Day','store_nbr'],as_index=False)['unit_sales'].agg(['sum','count'])

    #drop index and rename

    sale_day_store_level=sale_day_store_level.reset_index().rename(columns={'sum':'store_sales','count':'item_variety'})

#day-item level  

    sale_day_item_level=df.groupby(['Year','Month','Day','item_nbr'],as_index=False)['unit_sales'].agg(['sum','count'])

    #drop index and rename

    sale_day_item_level=sale_day_item_level.reset_index().rename(columns={'sum':'item_sales','count':'store_spread'})

#store item level   

    sale_store_item_level=df.groupby(['Year','store_nbr','item_nbr'],as_index=False)['unit_sales'].agg(['sum','count'])

    #drop index and rename

    sale_store_item_level=sale_store_item_level.reset_index().rename(columns={'sum':'item_sales','count':'entries'})



    return sale_day_store_level,sale_day_item_level,sale_store_item_level
#run for 2013

sale_day_store_level_2013,sale_day_item_level_2013,sale_store_item_level_2013=aggregate_level1(train_2013)

print(sale_day_store_level_2013.head())

sale_day_item_level_2013.head()

import time

start_time = time.time()

#run for 2014

sale_day_store_level_2014,sale_day_item_level_2014,sale_store_item_level_2014=aggregate_level1(train_2014)

#run for 2015

sale_day_store_level_2015,sale_day_item_level_2015,sale_store_item_level_2015=aggregate_level1(train_2015)

#run for 2016

sale_day_store_level_2016,sale_day_item_level_2016,sale_store_item_level_2016=aggregate_level1(train_2016)

#run for 2017

sale_day_store_level_2017,sale_day_item_level_2017,sale_store_item_level_2017=aggregate_level1(train_2017)



end_time=time.time()

time_taken=end_time-start_time

print("This block took ",time_taken,"seconds")
# appending together

#note: concat expects a list of dfs and not a list of strings

sale_day_store_level=pd.concat([sale_day_store_level_2013,sale_day_store_level_2014,

                                sale_day_store_level_2015,sale_day_store_level_2016,

                                sale_day_store_level_2017])



sale_day_item_level=pd.concat([sale_day_item_level_2013,sale_day_item_level_2014,

                                sale_day_item_level_2015,sale_day_item_level_2016,

                                sale_day_item_level_2017])

sale_store_item_level=pd.concat([sale_store_item_level_2013,sale_store_item_level_2014,

                                sale_store_item_level_2015,sale_store_item_level_2016,

                                sale_store_item_level_2017])
# freeup memory

del(sale_day_store_level_2013)

del(sale_day_store_level_2014)

del(sale_day_store_level_2015)

del(sale_day_store_level_2016)

del(sale_day_store_level_2017)

del(sale_day_item_level_2013)

del(sale_day_item_level_2014)

del(sale_day_item_level_2015)

del(sale_day_item_level_2016)

del(sale_day_item_level_2017)

del(sale_store_item_level_2013)

del(sale_store_item_level_2014)

del(sale_store_item_level_2015)

del(sale_store_item_level_2016)

del(sale_store_item_level_2017)

gc.collect()
sale_day_store_level.to_csv("sale_day_store_level.csv")

sale_day_item_level.to_csv("sale_day_item_level.csv")

sale_store_item_level.to_csv("sale_store_item_level.csv")
#Creating store level metrics

sale_store_level=sale_day_store_level.groupby(['store_nbr'],as_index=False)['store_sales','item_variety'].agg(['sum'])



# Here the group by gives a multiindex , removing that

sale_store_level.columns = sale_store_level.columns.droplevel(1)

sale_store_level=sale_store_level.reset_index()

sale_store_level.head()

#Creating item level metrics

sale_item_level=sale_day_item_level.groupby(['item_nbr'],as_index=False)['item_sales'].agg(['sum'])



sale_item_level=sale_item_level.reset_index()

sale_item_level.head()
# Sorting by sales

temp=sale_store_level.sort_values('store_sales',ascending=False).reset_index(drop=True)

temp=temp.set_index('store_nbr').head(10)



plt.figure(figsize=(12,8))

sns.barplot(temp.index,temp.store_sales, alpha=0.8, color=color[2],)

plt.ylabel('Overall Sales', fontsize=12)

plt.xlabel('Store Number', fontsize=12)

plt.title('Top Stores by Overall sale', fontsize=15)

# plt.xticks(rotation='vertical')

plt.show()

# Sorting by sales

temp1=sale_item_level.sort_values('sum',ascending=False).reset_index(drop=True)

temp1=temp1.set_index('item_nbr').head(10)

plt.figure(figsize=(12,8))

x=temp1.index.values

y=temp1['sum'].values

sns.barplot(x,y, alpha=0.8, color=color[8])

plt.ylabel('Overall Sales', fontsize=12)

plt.xlabel('Store Number', fontsize=12)

plt.title('Top Items by Overall sale', fontsize=15)

plt.show()
print("PLot I wanted to show :(")

print("top 10 items")

temp.iloc[:,0].plot.bar()

plt.show()
#Overall sales

#YOY sales

temp=sale_day_store_level.groupby('Year')['store_sales'].sum()

plt.figure(figsize=(13,4))

sns.pointplot(temp.index,temp.values, alpha=0.8, color=color[1],)

plt.ylabel('Overall Sales', fontsize=12)

plt.xlabel('Year', fontsize=12)

plt.title('Sale YOY', fontsize=15)

plt.xticks(rotation='vertical')



# month over month sales

temp=sale_day_store_level.groupby(['Year','Month'])['store_sales'].sum()

plt.figure(figsize=(13,4))

sns.pointplot(temp.index,temp.values, alpha=0.8, color=color[2],)

plt.ylabel('Overall Sales', fontsize=12)

plt.xlabel('Month', fontsize=12)

plt.title('Monthly sales variation', fontsize=15)

plt.xticks(rotation='vertical')







# also checking the oil price change

oil['date']=pd.to_datetime(oil['date'])

oil['Year']=oil['date'].dt.year

oil['Month']=oil['date'].dt.month 



# Oil price variation over month

temp=oil.groupby(['Year','Month']).agg(['sum','count'])

temp.columns = temp.columns.droplevel(0)

temp['avg']=temp['sum']/temp['count']

#plot

plt.figure(figsize=(13,4))

sns.pointplot(temp.index,temp.avg, alpha=0.8, color=color[4],)

plt.ylabel('Oil price', fontsize=12)

plt.xlabel('Month', fontsize=12)

plt.title('Monthly variation in oil price', fontsize=15)

plt.xticks(rotation='vertical')



plt.show()

plt.show()
# month over month sales

temp=sale_day_store_level.groupby(['Year','Month']).aggregate({'store_sales':np.sum,'Year':np.min,'Month':np.min})

temp=temp.reset_index(drop=True)

sns.set(style="whitegrid", color_codes=True)

# temp

plt.figure(figsize=(15,8))

plt.plot(range(1,13),temp.iloc[0:12,0],label="2013")

plt.plot(range(1,13),temp.iloc[12:24,0],label="2014")

plt.plot(range(1,13),temp.iloc[24:36,0],label="2015")

plt.plot(range(1,13),temp.iloc[36:48,0],label="2015")

plt.ylabel('Overall Sales', fontsize=12)

plt.xlabel('Month', fontsize=12)

plt.title('Monthly sales variation', fontsize=15)

plt.xticks(rotation='vertical')

plt.legend(['2013', '2014', '2015', '2016'], loc='upper left')

plt.show()

#Count of stores in different types and clusters

plt.figure(figsize=(15,12))

#row col plotnumber - 121

plt.subplot(221)

# Count of stores for each type 

temp = stores['cluster'].value_counts()

#plot

sns.barplot(temp.index,temp.values,color=color[5])

plt.ylabel('Count of stores', fontsize=12)

plt.xlabel('Cluster', fontsize=12)

plt.title('Store distribution across cluster', fontsize=15)



plt.subplot(222)

# Count of stores for each type 

temp = stores['type'].value_counts()

#plot

sns.barplot(temp.index,temp.values,color=color[7])

plt.ylabel('Count of stores', fontsize=12)

plt.xlabel('Type of store', fontsize=12)

plt.title('Store distribution across store types', fontsize=15)



plt.subplot(223)

# Count of stores for each type 

temp = stores['state'].value_counts()

#plot

sns.barplot(temp.index,temp.values,color=color[8])

plt.ylabel('Count of stores', fontsize=12)

plt.xlabel('state', fontsize=12)

plt.title('Store distribution across states', fontsize=15)

plt.xticks(rotation='vertical')



plt.subplot(224)

# Count of stores for each type 

temp = stores['city'].value_counts()

#plot

sns.barplot(temp.index,temp.values,color=color[9])

plt.ylabel('Count of stores', fontsize=12)

plt.xlabel('City', fontsize=12)

plt.title('Store distribution across cities', fontsize=15)

plt.xticks(rotation='vertical')

plt.show()

sale_store_level=sale_store_level.iloc[:,0:2]

#print(sale_store_level)

merge=pd.merge(sale_store_level,stores,how='left',on='store_nbr')

#temp



#Sale of stores in different types and clusters

plt.figure(figsize=(15,12))

#row col plotnumber - 121

plt.subplot(221)

# Sale of stores for each type 

temp = merge.groupby(['cluster'])['store_sales'].sum()

#plot

sns.barplot(temp.index,temp.values,color=color[5])

plt.ylabel('Sales', fontsize=12)

plt.xlabel('Cluster', fontsize=12)

plt.title('Cumulative sales across store clusters', fontsize=15)



plt.subplot(222)

# sale of stores for each type 

temp = merge.groupby(['type'])['store_sales'].sum()

#plot

sns.barplot(temp.index,temp.values,color=color[7])

plt.ylabel('sales', fontsize=12)

plt.xlabel('Type of store', fontsize=12)

plt.title('Cumulative sales across store types', fontsize=15)



plt.subplot(223)

# sale of stores for each type 

temp = merge.groupby(['state'])['store_sales'].sum()

#plot

sns.barplot(temp.index,temp.values,color=color[8])

plt.ylabel('sales', fontsize=12)

plt.xlabel('state', fontsize=12)

plt.title('Cumulative sales across states', fontsize=15)

plt.xticks(rotation='vertical')



plt.subplot(224)

# sale of stores for city

temp = merge.groupby(['city'])['store_sales'].sum()

#plot

sns.barplot(temp.index,temp.values,color=color[9])

plt.ylabel('sales', fontsize=12)

plt.xlabel('City', fontsize=12)

plt.title('Cumulative sales across cities', fontsize=15)

plt.xticks(rotation='vertical')

plt.show()

sale_store_level=sale_store_level.iloc[:,0:2]

merge=pd.merge(sale_store_level,stores,how='left',on='store_nbr')



plt.figure(figsize=(15,12))

#row col plotnumber - 121

plt.subplot(221)

#plot

sns.boxplot(x='cluster', y="store_sales", data=merge)

plt.ylabel('Sales', fontsize=12)

plt.xlabel('Cluster', fontsize=12)

plt.title('Variation across store clusters', fontsize=15)



plt.subplot(222)

# sale of stores for each type 

sns.boxplot(x='type', y="store_sales", data=merge)

plt.ylabel('sales', fontsize=12)

plt.xlabel('Type of store', fontsize=12)

plt.title('Variation across store types', fontsize=15)



plt.subplot(223)

# sale of stores for each type 

sns.boxplot(x='state', y="store_sales", data=merge)

plt.ylabel('sales', fontsize=12)

plt.xlabel('state', fontsize=12)

plt.title('Variation across states', fontsize=15)

plt.xticks(rotation='vertical')



plt.subplot(224)

# sale of stores for city

sns.boxplot(x='city', y="store_sales", data=merge)

plt.ylabel('sales', fontsize=12)

plt.xlabel('City', fontsize=12)

plt.title('Variation across cities', fontsize=15)

plt.xticks(rotation='vertical')

plt.show()

#transactions

# month over month sales

trans['date']=pd.to_datetime(trans['date'])

#print(trans.dtypes)

temp=trans.groupby(['date']).aggregate({'store_nbr':'count','transactions':np.sum})

temp=temp.reset_index()

temp_2013=temp[temp['date'].dt.year==2013].reset_index(drop=True)

temp_2014=temp[temp['date'].dt.year==2014].reset_index(drop=True)

temp_2015=temp[temp['date'].dt.year==2015].reset_index(drop=True)

temp_2016=temp[temp['date'].dt.year==2016].reset_index(drop=True)

temp_2017=temp[temp['date'].dt.year==2017].reset_index(drop=True)



#print(temp)

sns.set(style="whitegrid", color_codes=True)

# temp

plt.figure(figsize=(15,14))

plt.subplot(211)

plt.plot(temp_2013['date'],temp_2013.iloc[:,1],label="2013")

plt.plot(temp_2014['date'],temp_2014.iloc[:,1],label="2014")

plt.plot(temp_2015['date'],temp_2015.iloc[:,1],label="2015")

plt.plot(temp_2016['date'],temp_2016.iloc[:,1],label="2016")

plt.plot(temp_2017['date'],temp_2017.iloc[:,1],label="2017")

plt.ylabel('Number of stores open', fontsize=12)

plt.xlabel('Time', fontsize=12)

plt.title('Number of stores open', fontsize=15)

plt.xticks(rotation='vertical')

plt.legend(['2013', '2014', '2015', '2016'], loc='lower right')



plt.subplot(212)

plt.plot(temp_2013.index,temp_2013.iloc[:,1],label="2013")

plt.plot(temp_2014.index,temp_2014.iloc[:,1],label="2014")

plt.plot(temp_2015.index,temp_2015.iloc[:,1],label="2015")

plt.plot(temp_2016.index,temp_2016.iloc[:,1],label="2016")

plt.plot(temp_2017.index,temp_2017.iloc[:,1],label="2017")





plt.ylabel('Number of stores open', fontsize=12)

plt.xlabel('Day of year', fontsize=12)

plt.title('Number of stores open', fontsize=15)

plt.xticks(rotation='vertical')

plt.legend(['2013', '2014', '2015', '2016'], loc='lower right')

plt.show()



temp=trans.groupby(['store_nbr']).agg({'date':[np.min,np.max]}).reset_index()

temp['store_age']=temp['date']['amax']-temp['date']['amin']

temp['open_year']=temp['date']['amin'].dt.year

data=temp['open_year'].value_counts()

#print(data)

plt.figure(figsize=(12,4))

sns.barplot(data.index,data.values, alpha=0.8, color=color[0])

plt.ylabel('Stores', fontsize=12)

plt.xlabel('Store opening Year', fontsize=12)

plt.title('When were the stores started?', fontsize=15)

plt.xticks(rotation='vertical')

plt.show()

store_items=pd.merge(sale_store_item_level,items,on='item_nbr')

store_items=pd.merge(store_items,stores,on='store_nbr')

store_items['item_sales']=store_items['item_sales']



#item

# top selling items by store type

top_items_by_type=store_items.groupby(['type','item_nbr'])['item_sales'].sum()

top_items_by_type=top_items_by_type.reset_index().sort_values(['type','item_sales'],ascending=[True,False])



#get top 5

top_items_by_type=top_items_by_type.groupby(['type']).head(5)





#class

# top selling item class by store type

top_class_by_type=store_items.groupby(['type','class'])['item_sales'].sum()

top_class_by_type=top_class_by_type.reset_index().sort_values(['type','item_sales'],ascending=[True,False])



#get top 5

top_class_by_type=top_class_by_type.groupby(['type']).head(5)





#family

# top selling item family by store type

top_family_by_type=store_items.groupby(['type','family'])['item_sales'].sum()

top_family_by_type=top_family_by_type.reset_index().sort_values(['type','item_sales'],ascending=[True,False])



#get top 5

top_family_by_type=top_family_by_type.groupby(['type']).head(5)

plt.figure(figsize=(12,5))



x=top_family_by_type.pivot(index='type',columns='family')

x.plot.bar(stacked=True,figsize=(12,5))

y=x.columns.droplevel(0).values

#print(y)

plt.ylabel('Sales', fontsize=12)

plt.xlabel('Top 5 item families', fontsize=12)

plt.title('Top 5 item families across different store types', fontsize=15)

plt.xticks(rotation='vertical')

plt.legend(y)

plt.show()
plt.figure(figsize=(12,5))

x=top_class_by_type.pivot(index='type',columns='class')

x.plot.bar(stacked=True,figsize=(12,5))

y=x.columns.droplevel(0).values

#print(y)

plt.ylabel('Sales', fontsize=12)

plt.xlabel('Top 5 item classes', fontsize=12)

plt.title('Top 5 item classes across different store types', fontsize=15)

plt.xticks(rotation='vertical')

plt.legend(y)

plt.show()
plt.figure(figsize=(12,5))

x=top_items_by_type.pivot(index='type',columns='item_nbr')

x.plot.bar(stacked=True,figsize=(12,5))

y=x.columns.droplevel(0).values

#print(y)

plt.ylabel('Sales', fontsize=12)

plt.xlabel('Top 5 items ', fontsize=12)

plt.title('Top 5 items across different store types', fontsize=15)

plt.xticks(rotation='vertical')

plt.legend(y)

plt.show()
top_family_by_type=store_items.groupby(['type','family'])['item_sales'].sum()

top_family_by_type=top_family_by_type.reset_index().sort_values(['type','item_sales'],ascending=[True,False])

x=top_family_by_type.pivot(index='family',columns='type')

cm = sns.light_palette("orange", as_cmap=True)

x = x.style.background_gradient(cmap=cm)

x
top_class_by_type=store_items.groupby(['type','class'])['item_sales'].sum()

top_class_by_type=top_class_by_type.reset_index().sort_values(['type','item_sales'],ascending=[True,False])

top_class_by_type=top_class_by_type.groupby(['class']).head(20)

x=top_class_by_type.pivot(index='class',columns='type')

x['total']=x.sum(axis=1)

x=x.sort_values('total',ascending=False)

del(x['total'])

x=x.head(20)

cm = sns.light_palette("gray", as_cmap=True)

x = x.style.background_gradient(cmap=cm,axis=1)

x
top_items_by_type=store_items.groupby(['type','item_nbr'])['item_sales'].sum()

top_items_by_type=top_items_by_type.reset_index().sort_values(['type','item_sales'],ascending=[True,False])

top_items_by_type=top_items_by_type.groupby(['item_nbr']).head(20)

#print(top_items_by_type)

x=top_items_by_type.pivot(index='item_nbr',columns='type')

x['total']=x.sum(axis=1)

x=x.sort_values('total',ascending=False)

del(x['total'])

x=x.head(30)

cm = sns.light_palette("green", as_cmap=True)

x = x.style.background_gradient(cmap=cm,axis=1)

x