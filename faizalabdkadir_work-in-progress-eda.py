import pandas as pd  

import numpy as np   

import re           

import datetime, time      

import calendar 

from pandas.plotting import table



import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

from matplotlib_venn import venn2

import seaborn as sns

color = sns.color_palette()

import plotly.tools as tls



import warnings

warnings.filterwarnings("ignore")



import nltk

from nltk.corpus import stopwords



%matplotlib inline



train_df = pd.read_csv('../input/train.tsv', sep='\t')

test_df = pd.read_csv('../input/test.tsv', sep='\t')

data = pd.concat([train_df,test_df], axis=0)



nrow = train_df.shape[0]

ncol = train_df.shape[1]



start = time.time()

# Extract 3 category related features 

def cat_split(row):

    try:

        text = row

        txt1, txt2, txt3 = text.split('/')

        return txt1, txt2, txt3

    except:

        return np.nan, np.nan, np.nan



train_df["cat_1"], train_df["cat_2"], train_df["cat_3"] = zip(*train_df.category_name.apply(lambda val: cat_split(val)))

test_df["cat_1"], test_df["cat_2"], test_df["cat_3"] = zip(*test_df.category_name.apply(lambda val: cat_split(val)))

data["cat_1"], data["cat_2"], data["cat_3"] = zip(*data.category_name.apply(lambda val: cat_split(val)))

print('shape')

print('merged data - ',data.shape)

print('train_df - ',train_df.shape)

print('test_df - ',test_df.shape)
fig, ax = plt.subplots(figsize=(12, 5), nrows=1, ncols=2, sharex=False, sharey=False)

sns.distplot(train_df['price'].values, axlabel = 'Price', ax=ax[0], label = 'Histogram-Price bins - 50', bins = 50, color="black")

table(ax[0], np.round(train_df['price'].to_frame().describe(),2),loc=6, colWidths=[0.2, 0.2, 0.2])

sns.distplot(np.log1p(train_df['price'].values), axlabel = 'Log(1+price)',ax=ax[1], label = 'Histogram-log(1+price) bins - 50', bins = 50, color="limegreen")

ax[0].legend(loc=0)

ax[1].legend(loc=0)

ax[0].set_title('Price')

ax[1].set_title('Log(1+price)')

plt.show()
# Prices with and without shipping 

fig, ax = plt.subplots(figsize=(12, 5), nrows=1, ncols=2, sharex=False, sharey=False)

sns.distplot((train_df.loc[train_df['shipping']==1]['price'].values), ax=ax[0], color='b', label='Price with shipping')

sns.distplot((train_df.loc[train_df['shipping']==0]['price'].values), ax=ax[0], color='m', label='Price with no shipping')

sns.distplot(np.log1p(train_df.loc[train_df['shipping']==1]['price'].values), ax=ax[1], color='b', label='Log(Price with shipping)')

sns.distplot(np.log1p(train_df.loc[train_df['shipping']==0]['price'].values), ax=ax[1], color='m', label='Log(Price with no shipping)')

ax[0].legend(loc=0)

ax[1].legend(loc=0)

ax[0].set_title('Price with/without shipping')

ax[1].set_title('Log(1+price) with/without shipping')

plt.show()

print("There are",len(data.category_name.unique()),"unique category_name")

print("There are",len(data.cat_1.unique()),"unique cat_1")

print("There are",len(data.cat_2.unique()),"unique cat_2")

print("There are",len(data.cat_3.unique()),"unique cat_3")

print("There are",len(data.brand_name.unique()),"unique brand_names")

print("There are",len(data['item_description'].unique()),"unique item_description")

print("There are",len(data['name'].unique()),"unique name")
df = train_df[train_df['price']<100]

df =df[df['cat_1'].notnull()]



plot1 = []

for i in df['cat_1'].unique():

    plot1.append(df[df['cat_1']==i]['price'])



    x = range(1,11)

labels = df['cat_1'].unique()



fig, axes = plt.subplots(figsize=(12, 6))

bp = axes.boxplot(plot1,vert=True,patch_artist=True,labels=range(1,11)) 

colors = ['orangered', 'gold', 'springgreen', 'coral', 'limegreen','cyan','lightgreen','orchid','olivedrab','lightcoral']

for patch, color in zip(bp['boxes'], colors):

    patch.set_facecolor(color)



#ax.yaxis.grid(False)

plt.xticks(x, labels, rotation=90)

plt.title('Price by cat_1', fontsize=15)

plt.xlabel('cat_1', fontsize=15)

plt.ylabel('Price', fontsize=15)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.show()



del df, plot1, x
df = train_df[train_df['price']<100]



plot1 = []

for i in df['item_condition_id'].unique():

    plot1.append(df[df['item_condition_id']==i]['price'])



fig, axes = plt.subplots(figsize=(10, 6))

bp = axes.boxplot(plot1,vert=True,patch_artist=True,labels=range(1,6)) 



colors = ['thistle', 'powderblue', 'palegreen', 'pink', 'tan']

for patch, color in zip(bp['boxes'], colors):

    patch.set_facecolor(color)



plt.title('Price by item_condition_id', fontsize=15)

plt.xlabel('item_condition_id', fontsize=15)

plt.ylabel('Price', fontsize=15)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.show()



del df
group_cat_1=train_df.groupby(['cat_1']).size().sort_values(ascending=False)#.reset_index()

group_cat_1=round(group_cat_1/sum(group_cat_1)*100,2)

group_cat_2=train_df.groupby(['cat_2']).size().sort_values(ascending=False)#.reset_index()

group_cat_2=round(group_cat_2/sum(group_cat_2)*100,2)

group_cat_3=train_df.groupby(['cat_3']).size().sort_values(ascending=False)#.reset_index()

group_cat_3=round(group_cat_3/sum(group_cat_3)*100,2)



group_cat_1b=train_df.groupby(['cat_1']).size().sort_values(ascending=False)

group_cat_1b=round(np.cumsum(group_cat_1)/sum(group_cat_1)*100,2)

group_cat_2b=train_df.groupby(['cat_2']).size().sort_values(ascending=False)

group_cat_2b=round(np.cumsum(group_cat_2)/sum(group_cat_2)*100,2)

group_cat_3b=train_df.groupby(['cat_3']).size().sort_values(ascending=False)

group_cat_3b=round(np.cumsum(group_cat_3)/sum(group_cat_3)*100,2)



group_cat_1=pd.concat([group_cat_1,group_cat_1b], axis=1)

group_cat_2=pd.concat([group_cat_2,group_cat_2b], axis=1)

group_cat_3=pd.concat([group_cat_3,group_cat_3b], axis=1)

lbl =['Percentage %','Cumulative %']

group_cat_1.columns=lbl

group_cat_2.columns=lbl

group_cat_3.columns=lbl

group_cat_1=group_cat_1.reset_index()

group_cat_2=group_cat_2.reset_index()[:30]

group_cat_3=group_cat_3.reset_index()[:100]

group_cat_1
group_cat_2.head(10)
group_cat_3.head(10)
fig, ax = plt.subplots(figsize=(20, 6), nrows=1, ncols=2, sharex=False, sharey=False)

hist = train_df.groupby(['cat_1'],as_index=False).count().sort_values(by='train_id',ascending=False)

sns.barplot(x=hist['cat_1'],y=hist['train_id'],ax=ax[0],palette='YlGnBu')

sns.barplot(x=group_cat_1['cat_1'],y=group_cat_1['Cumulative %'],ax=ax[1],palette='YlGnBu')#orient='h'

ax[0].set_title(" cat_1 by count", fontsize=30)

ax[1].set_title(" cat_1 by cumulative% count", fontsize=30)

#ax.set_xticklabels(ax.get_xticklabels(),rotation=45)

ax[0].set_xticklabels(labels=hist['cat_1'],rotation=30)

ax[1].set_xticklabels(labels=hist['cat_1'],rotation=30)

plt.rcParams.update({'font.size':12})

plt.show()
fig, ax = plt.subplots(figsize=(20, 7), nrows=1, ncols=2, sharex=False, sharey=False)

hist = train_df.groupby(['cat_3'],as_index=False).count().sort_values(by='train_id',ascending=False)[:100]

sns.barplot(x=hist['cat_3'],y=hist['train_id'],ax=ax[0],palette='YlGnBu')

sns.barplot(x=group_cat_3['cat_3'],y=group_cat_3['Cumulative %'], ax=ax[1],palette='YlGnBu')

ax[0].set_title(" cat_3 by count", fontsize=30)

ax[1].set_title(" cat_3 by cumulative% count", fontsize=30)

ax[0].set_xticklabels(labels=hist['cat_1'],rotation=90)

ax[1].set_xticklabels(labels=hist['cat_1'],rotation=90)

plt.rcParams.update({'font.size':12})

plt.show()
fig, ax = plt.subplots(figsize=(10, 5))

plt.title('cat_1 - Count vs Mean Price')

#style must be one of white, dark, whitegrid, darkgrid, ticks

sns.set_style("ticks")

ax2 = ax.twinx() #This allows the common axes to be shared

ax.set_xticklabels(ax.get_xticklabels(),rotation=45)

sns.countplot(x="cat_1",  data=train_df,  ax=ax, palette="Paired")

sns.set_style("dark")

sns.factorplot(x="cat_1", y="price", data=train_df,  ax=ax2, scale = 0.7,color='black')

ax2.set_ylim(0,40)

sns.despine(ax=ax)

plt.close(2)

plt.show()
fig, ax = plt.subplots(figsize=(4, 5))

plt.title('shipping - Count vs Mean Price')

sns.set_style("ticks")

ax2 = ax.twinx() #This allows the common axes to be shared

#ax.set_xticklabels(ax.get_xticklabels(),rotation=45)

sns.countplot(x="shipping",  data=train_df,  ax=ax, palette="Set2")

sns.set_style("dark")

sns.factorplot(x="shipping", y="price", data=train_df,  ax=ax2, scale = 0.7,color='black')

ax2.set_ylim(0,40)

sns.despine(ax=ax)

plt.close(2)

plt.show()
fig, ax = plt.subplots(figsize=(6, 5))

plt.title('item_condition_id - Count vs Mean Price')

#style must be one of white, dark, whitegrid, darkgrid, ticks

sns.set_style("ticks")

ax2 = ax.twinx() #This allows the common axes to be shared

ax.set_xticklabels(ax.get_xticklabels(),rotation=0)

sns.countplot(x="item_condition_id",  data=train_df,  ax=ax, palette="Set1")

#ax.set_ylim(0,50000)

sns.set_style("dark")

sns.factorplot(x="item_condition_id", y="price", data=train_df,  ax=ax2, scale = 0.7,color='black')

ax2.set_ylim(0,40)

sns.despine(ax=ax)

plt.close(2)

plt.show()
pd.pivot_table(train_df, index=['cat_1'], columns='shipping',values='price',aggfunc=[np.size,np.mean,np.sum],

                      fill_value=0,margins=True)
# start https://www.kaggle.com/golubev/naive-xgboost-v2

c_texts = ['name', 'item_description']

def count_words(key):

    return len(str(key).split())

def count_numbers(key):

    return sum(c.isalpha() for c in str(key))

def count_upper(key):

    return sum(c.isupper() for c in str(key))

for c in c_texts:

    data[c + '_c_words'] = data[c].apply(count_words)

    data[c + '_c_upper'] = data[c].apply(count_upper)

    data[c + '_c_numbers'] = data[c].apply(count_numbers)

    data[c + '_len'] = data[c].str.len()

    data[c + '_mean_len_words'] = data[c + '_len'] / data[c + '_c_words']

    data[c + '_mean_upper'] = data[c + '_len'] / data[c + '_c_upper']

    data[c + '_mean_numbers'] = data[c + '_len'] / data[c + '_c_numbers']

plt1=data.groupby(['name_c_words'],as_index=False)['price'].mean()

plt1['price']=plt1['price'].apply(lambda x: np.log(1+x))

plt2=data.groupby(['name_c_upper'],as_index=False)['price'].mean()

plt2['price']=plt2['price'].apply(lambda x: np.log(1+x))

plt3=data.groupby(['name_c_numbers'],as_index=False)['price'].mean()

plt3['price']=plt3['price'].apply(lambda x: np.log(1+x))

plt4=data.groupby(['name_len'],as_index=False)['price'].mean()

plt4['price']=plt4['price'].apply(lambda x: np.log(1+x))

plt5=data.groupby(['name_mean_len_words'],as_index=False)['price'].mean()

plt5['price']=plt5['price'].apply(lambda x: np.log(1+x))

plt6=data.groupby(['name_mean_upper'],as_index=False)['price'].mean()

plt6['price']=plt6['price'].apply(lambda x: np.log(1+x))

plt7=data.groupby(['name_mean_numbers'],as_index=False)['price'].mean()

plt7['price']=plt7['price'].apply(lambda x: np.log(1+x))



f, ax=plt.subplots(4,2, figsize=(15,16), sharey=True)

ax[0,0].plot(plt1['name_c_words'] ,plt1['price'],color='darkturquoise')

ax[0,1].plot(plt2['name_c_upper'] ,plt2['price'],color='y')

ax[1,0].plot(plt3['name_c_numbers'] ,plt3['price'],color='g')

ax[1,1].plot(plt4['name_len'] ,plt4['price'],color='m')

ax[2,0].plot(plt5['name_mean_len_words'] ,plt5['price'],color='limegreen')

ax[2,1].plot(plt6['name_mean_upper'] ,plt6['price'],color='blueviolet')

ax[3,0].plot(plt7['name_mean_numbers'] ,plt7['price'],color='tomato')



ax[0,0].set_ylim(0,6)



ax[0,0].set_ylabel('log(1+price)')

ax[1,0].set_ylabel('log(1+price)')

ax[2,0].set_ylabel('log(1+price)')

ax[3,0].set_ylabel('log(1+price)')



ax[0,0].set_title('name_c_words')

ax[0,1].set_title('name_c_upper')

ax[1,0].set_title('name_c_numbers')

ax[1,1].set_title('name_mean_upper')

ax[2,0].set_title('name_len')

ax[2,1].set_title('name_mean_upper')
plt1=data.groupby(['item_description_c_words'],as_index=False)['price'].mean()

plt1['price']=plt1['price'].apply(lambda x: np.log(1+x))

plt2=data.groupby(['item_description_c_upper'],as_index=False)['price'].mean()

plt2['price']=plt2['price'].apply(lambda x: np.log(1+x))

plt3=data.groupby(['item_description_c_numbers'],as_index=False)['price'].mean()

plt3['price']=plt3['price'].apply(lambda x: np.log(1+x))

plt4=data.groupby(['item_description_len'],as_index=False)['price'].mean()

plt4['price']=plt4['price'].apply(lambda x: np.log(1+x))

plt5=data.groupby(['item_description_mean_len_words'],as_index=False)['price'].mean()

plt5['price']=plt5['price'].apply(lambda x: np.log(1+x))

plt6=data.groupby(['item_description_mean_upper'],as_index=False)['price'].mean()

plt6['price']=plt6['price'].apply(lambda x: np.log(1+x))

plt7=data.groupby(['item_description_mean_numbers'],as_index=False)['price'].mean()

plt7['price']=plt7['price'].apply(lambda x: np.log(1+x))



f, ax=plt.subplots(4,2, figsize=(15,16), sharey=True)

ax[0,0].plot(plt1['item_description_c_words'] ,plt1['price'],color='darkturquoise')

ax[0,1].plot(plt2['item_description_c_upper'] ,plt2['price'],color='y')

ax[1,0].plot(plt3['item_description_c_numbers'] ,plt3['price'],color='g')

ax[1,1].plot(plt4['item_description_len'] ,plt4['price'],color='m')

ax[2,0].plot(plt5['item_description_mean_len_words'] ,plt5['price'],color='limegreen')

ax[2,1].plot(plt6['item_description_mean_upper'] ,plt6['price'],color='blueviolet')

ax[3,0].plot(plt7['item_description_mean_numbers'] ,plt7['price'],color='tomato')





ax[0,0].set_ylim(0,8)



ax[0,0].set_ylabel('1+price')

ax[1,0].set_ylabel('1+price')

ax[2,0].set_ylabel('1+price')

ax[3,0].set_ylabel('1+price')



ax[0,0].set_title('item_description_c_words')

ax[0,1].set_title('item_description_c_upper')

ax[1,0].set_title('item_description_c_numbers')

ax[1,1].set_title('item_description_mean_upper')

ax[2,0].set_title('item_description_len')

ax[2,1].set_title('item_description_mean_upper')
data.transpose()