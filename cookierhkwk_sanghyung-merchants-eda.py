import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

merchants = pd.read_csv('../input/merchants.csv')

new_merchant_t = pd.read_csv('../input/new_merchant_transactions.csv')

his_trans = pd.read_csv('../input/historical_transactions.csv')
merchants = pd.read_csv('../input/merchants.csv')
print(train.shape)

print(test.shape)

print(merchants.shape)

print(new_merchant_t.shape)

print(his_trans.shape) 
merchants['merchant_group_id'] = merchants['merchant_group_id'].astype(object)

merchants['merchant_category_id'] = merchants['merchant_category_id'].astype(object)

merchants['subsector_id'] = merchants['subsector_id'].astype(object)

merchants['city_id'] = merchants['city_id'].astype(object)

merchants['state_id'] = merchants['state_id'].astype(object)

merchants['category_2'] = merchants['category_2'].astype(object)
merchants.dtypes
merchants.head(10)
merchants['numerical_1'].describe()
f, ax = plt.subplots(figsize=(10, 8))

sns.distplot(merchants['numerical_1'])
merchants['numerical_2'].describe()
f, ax = plt.subplots(figsize=(10, 8))

sns.distplot(merchants['numerical_2'])
fig, ax = plt.subplots(1, 3, figsize = (16, 6));

merchants['category_1'].value_counts().sort_index().plot(kind='bar', ax=ax[0], color='teal', title='Anonymized_Category_1')

merchants['category_4'].value_counts().sort_index().plot(kind='bar', ax=ax[1], color='blue', title='Anonymized_Category_4')

merchants['category_2'].value_counts().sort_index().plot(kind='bar', ax=ax[2], color='red', title='Anonymized_Category_2')
merchants[merchants['category_1']=='Y'].head(30)
merchants[merchants['category_1']=='Y'].state_id.drop_duplicates()
merchants[merchants['category_1']=='Y'].category_2.drop_duplicates()
merchants[merchants['category_1']=='Y'].city_id.drop_duplicates()
merchants[merchants['state_id']== -1].city_id.drop_duplicates()
merchants[merchants['city_id']== -1].state_id.drop_duplicates()
merchants[merchants['category_2'].isnull()].state_id.drop_duplicates()
data1 = merchants[merchants['category_2']==1]

data2 = merchants[merchants['category_2']==2]

data3 = merchants[merchants['category_2']==3]

data4 = merchants[merchants['category_2']==4]

data5 = merchants[merchants['category_2']==5]
fig, ax = plt.subplots(5, 3, figsize = (16, 21));

data1['avg_sales_lag3'].value_counts().sort_index().plot(kind='hist', ax=ax[0][0], color='teal', title='3_Month');

data1['avg_sales_lag6'].value_counts().sort_index().plot(kind='hist', ax=ax[0][1], color='brown', title='6_Month');

data1['avg_sales_lag12'].value_counts().sort_index().plot(kind='hist', ax=ax[0][2], color='blue', title='12_Month');

data2['avg_sales_lag3'].value_counts().sort_index().plot(kind='hist', ax=ax[1][0], color='teal', title='3_Month');

data2['avg_sales_lag6'].value_counts().sort_index().plot(kind='hist', ax=ax[1][1], color='brown', title='6_Month');

data2['avg_sales_lag12'].value_counts().sort_index().plot(kind='hist', ax=ax[1][2], color='blue', title='12_Month');

data3['avg_sales_lag3'].value_counts().sort_index().plot(kind='hist', ax=ax[2][0], color='teal', title='3_Month');

data3['avg_sales_lag6'].value_counts().sort_index().plot(kind='hist', ax=ax[2][1], color='brown', title='6_Month');

data3['avg_sales_lag12'].value_counts().sort_index().plot(kind='hist', ax=ax[2][2], color='blue', title='12_Month');

data4['avg_sales_lag3'].value_counts().sort_index().plot(kind='hist', ax=ax[3][0], color='teal', title='3_Month');

data4['avg_sales_lag6'].value_counts().sort_index().plot(kind='hist', ax=ax[3][1], color='brown', title='6_Month');

data4['avg_sales_lag12'].value_counts().sort_index().plot(kind='hist', ax=ax[3][2], color='blue', title='12_Month');

data5['avg_sales_lag3'].value_counts().sort_index().plot(kind='hist', ax=ax[4][0], color='teal', title='3_Month');

data5['avg_sales_lag6'].value_counts().sort_index().plot(kind='hist', ax=ax[4][1], color='brown', title='6_Month');

data5['avg_sales_lag12'].value_counts().sort_index().plot(kind='hist', ax=ax[4][2], color='blue', title='12_Month');
fig, ax = plt.subplots(5, 2, figsize = (16, 21));

data1['most_recent_sales_range'].value_counts().sort_index().plot(kind='bar', ax=ax[0][0], color='teal', title='most_recent_sales_range');

data1['most_recent_purchases_range'].value_counts().sort_index().plot(kind='bar', ax=ax[0][1], color='teal', title='most_recent_purchases_range');

data2['most_recent_sales_range'].value_counts().sort_index().plot(kind='bar', ax=ax[1][0], color='teal', title='most_recent_sales_range');

data2['most_recent_purchases_range'].value_counts().sort_index().plot(kind='bar', ax=ax[1][1], color='teal', title='most_recent_purchases_range');

data3['most_recent_sales_range'].value_counts().sort_index().plot(kind='bar', ax=ax[2][0], color='teal', title='most_recent_sales_range');

data3['most_recent_purchases_range'].value_counts().sort_index().plot(kind='bar', ax=ax[2][1], color='teal', title='most_recent_purchases_range');

data4['most_recent_sales_range'].value_counts().sort_index().plot(kind='bar', ax=ax[3][0], color='teal', title='most_recent_sales_range');

data4['most_recent_purchases_range'].value_counts().sort_index().plot(kind='bar', ax=ax[3][1], color='teal', title='most_recent_purchases_range');

data5['most_recent_sales_range'].value_counts().sort_index().plot(kind='bar', ax=ax[4][0], color='teal', title='most_recent_sales_range');

data5['most_recent_purchases_range'].value_counts().sort_index().plot(kind='bar', ax=ax[4][1], color='teal', title='most_recent_purchases_range');
fig, ax = plt.subplots(1, 3, figsize = (16, 6));

data1['avg_sales_lag3'].value_counts().sort_index().plot(kind='hist', ax=ax[0], color='teal', title='3_Month');

data1['avg_sales_lag6'].value_counts().sort_index().plot(kind='hist', ax=ax[1], color='brown', title='6_Month');

data1['avg_sales_lag12'].value_counts().sort_index().plot(kind='hist', ax=ax[2], color='blue', title='12_Month');
merchants['location'] = merchants['category_1'].astype(str) + '_' + merchants['category_2'].astype(str) + '_' + merchants['state_id'].astype(str) + '_' + merchants['city_id'].astype(str)
merchants.head()
fig, ax = plt.subplots(1, 3, figsize = (16, 6));

merchants['active_months_lag3'].value_counts().sort_index().plot(kind='bar', ax=ax[0], color='teal', title='3_Month');

merchants['active_months_lag6'].value_counts().sort_index().plot(kind='bar', ax=ax[1], color='brown', title='6_Month');

merchants['active_months_lag12'].value_counts().sort_index().plot(kind='bar', ax=ax[2], color='blue', title='12_Month');
y1 = merchants.loc[merchants['active_months_lag3']==3]

y2 = merchants.loc[merchants['active_months_lag6']==6]

y3 = merchants.loc[merchants['active_months_lag12']==12]

x1 = merchants.loc[merchants['active_months_lag3']!=3]

x2 = merchants.loc[merchants['active_months_lag6']!=6]

x3 = merchants.loc[merchants['active_months_lag12']!=12]
fig, ax = plt.subplots(3, 4, figsize = (16, 21));

y1['most_recent_sales_range'].value_counts().sort_index().plot(kind='bar', ax=ax[0][0], color='teal', title='most_recent_sales_range');

y1['most_recent_purchases_range'].value_counts().sort_index().plot(kind='bar', ax=ax[0][1], color='teal', title='most_recent_purchases_range');

x1['most_recent_sales_range'].value_counts().sort_index().plot(kind='bar', ax=ax[0][2], color='teal', title='most_recent_sales_range');

x1['most_recent_purchases_range'].value_counts().sort_index().plot(kind='bar', ax=ax[0][3], color='teal', title='most_recent_purchases_range');



y2['most_recent_sales_range'].value_counts().sort_index().plot(kind='bar', ax=ax[1][0], color='teal', title='most_recent_sales_range');

y2['most_recent_purchases_range'].value_counts().sort_index().plot(kind='bar', ax=ax[1][1], color='teal', title='most_recent_purchases_range');

x2['most_recent_sales_range'].value_counts().sort_index().plot(kind='bar', ax=ax[1][2], color='teal', title='most_recent_sales_range');

x2['most_recent_purchases_range'].value_counts().sort_index().plot(kind='bar', ax=ax[1][3], color='teal', title='most_recent_purchases_range');



y3['most_recent_sales_range'].value_counts().sort_index().plot(kind='bar', ax=ax[2][0], color='teal', title='most_recent_sales_range');

y3['most_recent_purchases_range'].value_counts().sort_index().plot(kind='bar', ax=ax[2][1], color='teal', title='most_recent_purchases_range');

x3['most_recent_sales_range'].value_counts().sort_index().plot(kind='bar', ax=ax[2][2], color='teal', title='most_recent_sales_range');

x3['most_recent_purchases_range'].value_counts().sort_index().plot(kind='bar', ax=ax[2][3], color='teal', title='most_recent_purchases_range');
x3_1 = merchants.loc[merchants['active_months_lag12']==1]

x3_2 = merchants.loc[merchants['active_months_lag12']==2]

x3_3 = merchants.loc[merchants['active_months_lag12']==3]

x3_4 = merchants.loc[merchants['active_months_lag12']==4]

x3_5 = merchants.loc[merchants['active_months_lag12']==5]

x3_6 = merchants.loc[merchants['active_months_lag12']==6]

x3_7 = merchants.loc[merchants['active_months_lag12']==7]

x3_8 = merchants.loc[merchants['active_months_lag12']==8]

x3_9 = merchants.loc[merchants['active_months_lag12']==9]

x3_10 = merchants.loc[merchants['active_months_lag12']==10]

x3_11 = merchants.loc[merchants['active_months_lag12']==11]
fig, ax = plt.subplots(11, 2, figsize = (16, 40));

x3_1['most_recent_sales_range'].value_counts().sort_index().plot(kind='bar', ax=ax[0][0], color='teal', title='most_recent_sales_range');

x3_1['most_recent_purchases_range'].value_counts().sort_index().plot(kind='bar', ax=ax[0][1], color='teal', title='most_recent_purchases_range');

x3_2['most_recent_sales_range'].value_counts().sort_index().plot(kind='bar', ax=ax[1][0], color='teal', title='most_recent_sales_range');

x3_2['most_recent_purchases_range'].value_counts().sort_index().plot(kind='bar', ax=ax[1][1], color='teal', title='most_recent_purchases_range');

x3_3['most_recent_sales_range'].value_counts().sort_index().plot(kind='bar', ax=ax[2][0], color='teal', title='most_recent_sales_range');

x3_3['most_recent_purchases_range'].value_counts().sort_index().plot(kind='bar', ax=ax[2][1], color='teal', title='most_recent_purchases_range');

x3_4['most_recent_sales_range'].value_counts().sort_index().plot(kind='bar', ax=ax[3][0], color='teal', title='most_recent_sales_range');

x3_4['most_recent_purchases_range'].value_counts().sort_index().plot(kind='bar', ax=ax[3][1], color='teal', title='most_recent_purchases_range');

x3_5['most_recent_sales_range'].value_counts().sort_index().plot(kind='bar', ax=ax[4][0], color='teal', title='most_recent_sales_range');

x3_5['most_recent_purchases_range'].value_counts().sort_index().plot(kind='bar', ax=ax[4][1], color='teal', title='most_recent_purchases_range');

x3_6['most_recent_sales_range'].value_counts().sort_index().plot(kind='bar', ax=ax[5][0], color='teal', title='most_recent_sales_range');

x3_6['most_recent_purchases_range'].value_counts().sort_index().plot(kind='bar', ax=ax[5][1], color='teal', title='most_recent_purchases_range');

x3_7['most_recent_sales_range'].value_counts().sort_index().plot(kind='bar', ax=ax[6][0], color='teal', title='most_recent_sales_range');

x3_7['most_recent_purchases_range'].value_counts().sort_index().plot(kind='bar', ax=ax[6][1], color='teal', title='most_recent_purchases_range');

x3_8['most_recent_sales_range'].value_counts().sort_index().plot(kind='bar', ax=ax[7][0], color='teal', title='most_recent_sales_range');

x3_8['most_recent_purchases_range'].value_counts().sort_index().plot(kind='bar', ax=ax[7][1], color='teal', title='most_recent_purchases_range');

x3_9['most_recent_sales_range'].value_counts().sort_index().plot(kind='bar', ax=ax[8][0], color='teal', title='most_recent_sales_range');

x3_9['most_recent_purchases_range'].value_counts().sort_index().plot(kind='bar', ax=ax[8][1], color='teal', title='most_recent_purchases_range');

x3_10['most_recent_sales_range'].value_counts().sort_index().plot(kind='bar', ax=ax[9][0], color='teal', title='most_recent_sales_range');

x3_10['most_recent_purchases_range'].value_counts().sort_index().plot(kind='bar', ax=ax[9][1], color='teal', title='most_recent_purchases_range');

x3_11['most_recent_sales_range'].value_counts().sort_index().plot(kind='bar', ax=ax[10][0], color='teal', title='most_recent_sales_range');

x3_11['most_recent_purchases_range'].value_counts().sort_index().plot(kind='bar', ax=ax[10][1], color='teal', title='most_recent_purchases_range');
x4_1 = merchants.loc[merchants['active_months_lag12']==12]

x4_2 = merchants.loc[merchants['active_months_lag6']==6]

x4_2 = x4_2.loc[merchants['active_months_lag12']!=12]

x4_3 = merchants.loc[merchants['active_months_lag3']==3]

x4_3 = x4_3.loc[merchants['active_months_lag6']!=6]

x4_4 = merchants.loc[merchants['active_months_lag3']!=3]
fig, ax = plt.subplots(2, 4, figsize = (16, 16));

x4_1['most_recent_sales_range'].value_counts().sort_index().plot(kind='bar', ax=ax[0][0], color='teal', title='most_recent_sales_range');

x4_1['most_recent_purchases_range'].value_counts().sort_index().plot(kind='bar', ax=ax[1][0], color='teal', title='most_recent_purchases_range');

x4_2['most_recent_sales_range'].value_counts().sort_index().plot(kind='bar', ax=ax[0][1], color='teal', title='most_recent_sales_range');

x4_2['most_recent_purchases_range'].value_counts().sort_index().plot(kind='bar', ax=ax[1][1], color='teal', title='most_recent_purchases_range');

x4_3['most_recent_sales_range'].value_counts().sort_index().plot(kind='bar', ax=ax[0][2], color='teal', title='most_recent_sales_range');

x4_3['most_recent_purchases_range'].value_counts().sort_index().plot(kind='bar', ax=ax[1][2], color='teal', title='most_recent_purchases_range');

x4_4['most_recent_sales_range'].value_counts().sort_index().plot(kind='bar', ax=ax[0][3], color='teal', title='most_recent_sales_range');

x4_4['most_recent_purchases_range'].value_counts().sort_index().plot(kind='bar', ax=ax[1][3], color='teal', title='most_recent_purchases_range');
merchants["consistency"] = "Y"

merchants.loc[merchants["active_months_lag12"]!=12,"consistency"]= "N"



merchants.head()
print(merchants[merchants.avg_sales_lag3>10].shape[0])

print(merchants[merchants.avg_sales_lag6>10].shape[0])

print(merchants[merchants.avg_sales_lag12>10].shape[0])
# 최근

print((merchants[(merchants.avg_sales_lag3>10)&(merchants.most_recent_sales_range!="E")].shape)[0])

print((merchants[(merchants.avg_sales_lag6>10)&(merchants.most_recent_sales_range!="E")].shape)[0])

print((merchants[(merchants.avg_sales_lag12>10)&(merchants.most_recent_sales_range!="E")].shape)[0])
f, ax = plt.subplots(figsize=(8, 6))

plt.scatter(x='most_recent_sales_range', y="avg_sales_lag3", data=merchants)

plt.xlabel("most_recent_sales_range")

plt.ylabel("avg_sales_lag3")

plt.show
f, ax = plt.subplots(figsize=(8, 6))

plt.scatter(x='most_recent_sales_range', y="avg_sales_lag6", data=merchants)

plt.xlabel("most_recent_sales_range")

plt.ylabel("avg_sales_lag3")

plt.show
merchants_no = merchants.loc[merchants['avg_sales_lag3']<100]



f, ax = plt.subplots(figsize=(8, 6))

plt.scatter(x='most_recent_sales_range', y="avg_sales_lag3", data=merchants_no)

plt.xlabel("most_recent_sales_range")

plt.ylabel("avg_sales_lag3")

plt.show
merchants[merchants['avg_sales_lag3']<0]
merchants_no = merchants.loc[merchants['avg_sales_lag3']<1]

merchants_no = merchants_no.loc[merchants_no['avg_sales_lag3']>=0]



f, ax = plt.subplots(figsize=(8, 6))

plt.scatter(x='most_recent_sales_range', y="avg_sales_lag3", data=merchants_no)

plt.xlabel("most_recent_sales_range")

plt.ylabel("avg_sales_lag3")

plt.show
merchants_no = merchants.loc[merchants['avg_sales_lag3']>1]

merchants_no = merchants_no.loc[merchants_no['avg_sales_lag3']<12.5]



f, ax = plt.subplots(figsize=(8, 6))

plt.scatter(x='most_recent_sales_range', y="avg_sales_lag3", data=merchants_no)

plt.xlabel("most_recent_sales_range")

plt.ylabel("avg_sales_lag3")

plt.show
merchants[merchants==np.inf].count()
#data2 : not infinite variables in avg_purchases_lag

data2 = merchants.loc[merchants["avg_purchases_lag3"]!=np.inf,]

print(data2['avg_purchases_lag3'].describe())

print(data2['avg_purchases_lag6'].describe())

print(data2['avg_purchases_lag12'].describe())        
f, ax = plt.subplots(figsize=(8, 6))

plt.scatter(x='most_recent_purchases_range', y="avg_purchases_lag3", data=data2)

plt.xlabel("most_recent_purchases_range")

plt.ylabel("avg_purchases_lag3")

plt.show
merchants[merchants['avg_purchases_lag3']>50000]
data2 = merchants.loc[merchants["avg_purchases_lag3"]<10000]

print(data2['avg_purchases_lag3'].describe())

print(data2['avg_purchases_lag6'].describe())

print(data2['avg_purchases_lag12'].describe())        
f, ax = plt.subplots(figsize=(8, 6))

plt.scatter(x='most_recent_purchases_range', y="avg_purchases_lag3", data=data2)

plt.xlabel("most_recent_purchases_range")

plt.ylabel("avg_purchases_lag3")

plt.show
data2_no = data2.loc[merchants['avg_purchases_lag3']<1]



f, ax = plt.subplots(figsize=(8, 6))

plt.scatter(x='most_recent_purchases_range', y="avg_purchases_lag3", data=data2_no)

plt.xlabel("most_recent_purchases_range")

plt.ylabel("avg_purchases_lag3")

plt.show