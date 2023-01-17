import pandas as pd

import numpy as np

import os

import re

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/ramen-ratings/ramen-ratings.csv') # Loading the dataset

df.head() # Printing the fist 5 rows to see the available features
shape = df.shape

print("Rows :",shape[0])

print("Columns :",shape[1])
df.info(verbose=True)
#Although stars is numberic but it is stored as string in the dataframe

#Let's convert it into numeric value.

df['Stars']=pd.to_numeric(df['Stars'], errors='coerce')
df.describe()
df.describe(include ="all")
#different types of Ramen styles

df['Style'].value_counts()
#what columns have NaN values

df.isna().any()
# Sum of NaNs in each column

df.isna().sum()
# For all the NaNs in "Top Ten" column, we have assigned a temporary value 0.

# we will deal with "Top Ten" column later.

df['Top Ten'].fillna(0, inplace=True)



#section of dataframe with NaN values

df[df.isnull().any(axis=1)]
#Imputing style is not relevant in this case

#espescially when only two rows have missing style values

#so dropping the two rows with NaN in style

df.drop(2152, axis=0,inplace=True)

df.drop(2442, axis=0,inplace=True)
#storing data with NaN in seperate dataframe

df_with_Nan = df[df.isnull().any(axis=1)]

df_with_Nan
for i in df_with_Nan.index:

    subDf = df.loc[(df['Brand']==df_with_Nan.loc[i,'Brand']) & (df['Country']==df_with_Nan.loc[i,'Country'])]

    mean = subDf['Stars'].mean()

    df.loc[i,'Stars'] = round(mean,2)
df.isna().sum() # Checking again for NaNs
df[df['Top Ten'] != 0]
# we can we still '\n' in our data

# considering it as missing value let's fill it with 0 temporarily

top_ten_with_n=df[df['Top Ten'] == '\n']

for i in top_ten_with_n.index:

    df.loc[i,'Top Ten']=0
#creating seperate columns for each year

#and fill with NaN

years=['2012','2013','2014','2015','2016']

for y in years:

    df[y+'_rank']=np.nan
# for specific year with the help of regex.

#Stroring the row number for each value.

#Extracting the rank which is at the end of the string.

#Storing the rank at the specific year column and specific row index we stored at step2.

#Dropping the "Top Ten" column.

for rank in df['Top Ten'].values:

    for y in years:

        if re.search('^'+y,str(rank)):

            index = df[df['Top Ten']==rank].index.values

            rank_number = str(rank).split()[-1]

            df.loc[index,y+'_rank'] = int(''.join([i for i in rank_number if i.isdigit()]))

df.drop('Top Ten', axis=1, inplace=True)
#it very clean that no rank is equal to 0 rank

df.fillna(0, inplace=True)

df.head()
df.isna().sum()
df["2012_rank"].value_counts()
df["2013_rank"].value_counts()
df["2014_rank"].value_counts()
df["2015_rank"].value_counts()
df["2016_rank"].value_counts()
v = df.Country.value_counts()

v=v.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(12,8))

v.plot(kind='barh')

plt.show()
brands_name = df.Brand.value_counts()[:10].index

brand_size = df.Brand.value_counts()[:10].values



fig,ax=plt.subplots(figsize=(15,4))

ax.bar(brands_name, brand_size, data=df)

ax.set_ylabel('Number of products')

for p in ax.patches:

    an=ax.annotate(str(p.get_height()), xy=(p.get_x(),p.get_height()))

    an.set_size(12)
style=df.Style.value_counts()

style
plt.pie(style[0:4],pctdistance=1.5,autopct="%2.01f%%",radius=1.7,labels=['Pack','Bowl','Cup','Tray'],

        explode=[0,0,0,0.3],

       textprops={'fontsize': 14})

plt.show()
a4_dims = (4, 10)

fig, ax = plt.subplots(figsize=a4_dims)

sns.scatterplot(ax=ax, data=df,y='Country',x='Style')

plt.show()
sns.distplot(df['Stars'],hist=True,kde=True,bins=5)

plt.show()
v = df.Country.value_counts()

v=v.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(12,8))

v.plot(kind='barh')

plt.show()
japan=df.loc[(df['Country'] == 'Japan') & (df['Stars'] >= 4.5)]

usa=df.loc[(df['Country'] == 'USA') & (df['Stars'] >= 4.5)]

south_korea=df.loc[(df['Country'] == 'South Korea') & (df['Stars'] >= 4.5)]

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)



# TITLE

fig.suptitle('TOP RAMEN BRANDS')

axes[0].set_title('TOP 3 MOST PREFFERED RAMEN BRANDS IN JAPAN',fontsize=10)

axes[1].set_title('TOP 3 MOST PREFFERED RAMEN BRANDS IN USA',fontsize=10)

axes[2].set_title('TOP 3 MOST PREFFERED RAMEN BRANDS IN SOUTH KOREA',fontsize=10)

axes[0].set_ylabel('PREFFERED TIMES', fontsize=10)

axes[0].set_xlabel('BRANDS', fontsize=10)

axes[1].set_ylabel('PREFFERED TIMES', fontsize=10)

axes[1].set_xlabel('BRANDS', fontsize=10)

axes[2].set_ylabel('PREFFERED TIMES', fontsize=10)

axes[2].set_xlabel('BRANDS', fontsize=10)



# JAPAN

x_jp= japan['Brand'].value_counts()

x_jp= x_jp[:3,]

sns.barplot(ax=axes[0],x=x_jp.index, y=x_jp.values,palette="Paired")



# USA

x_usa= usa['Brand'].value_counts()

x_usa= x_usa[:3,]

sns.barplot(ax=axes[1],x=x_usa.index,y= x_usa.values,palette="hls")



# SOUTH KOREA

x_sk= south_korea['Brand'].value_counts()

x_sk= x_sk[:3,]

sns.barplot(ax=axes[2],x=x_sk.index,y= x_sk.values,palette="Paired")

plt.show()
# NISSIN

jn1=df.loc[(df['Country'] == 'Japan') & (df['Stars'] >= 4)& (df['Brand']=='Nissin')]

jn2=df.loc[(df['Country'] == 'Japan') & (df['Stars'] < 4)& (df['Brand']=='Nissin')]

totjnp=jn1['Review #'].sum()

totjnn=jn2['Review #'].sum()

rev1 = totjnp,totjnn



# MYOJO

jmy1=df.loc[(df['Country'] == 'Japan') & (df['Stars'] >= 4)& (df['Brand']=='Myojo')]

jmy2=df.loc[(df['Country'] == 'Japan') & (df['Stars'] < 4)& (df['Brand']=='Myojo')]

totmyp=jmy1['Review #'].sum()

totmyn=jmy2['Review #'].sum()

rev2 = totmyp,totmyn



# MARUCHAN

jma1=df.loc[(df['Country'] == 'Japan') & (df['Stars'] >= 4)& (df['Brand']=='Maruchan')]

jma2=df.loc[(df['Country'] == 'Japan') & (df['Stars'] < 4)& (df['Brand']=='Maruchan')]

totmap=jma1['Review #'].sum()

totman=jma2['Review #'].sum()

rev3 = totmap,totman

labels = 'Above 4.0','Below 4.0'

colors = ['yellowgreen', 'lightskyblue']

explode = (0.1, 0) 

fig ,ax=plt.subplots(1,3,figsize=(15,15))

ax[0].pie(rev1, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%')

ax[1].pie(rev2, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%')

ax[2].pie(rev3, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%')

ax[0].set_title('NISSIN',fontsize=10)

ax[1].set_title('MYOJO',fontsize=10)

ax[2].set_title('MARUCHAN',fontsize=10)

plt.show()

japan_b1=df.loc[(df['Country'] == 'Japan') & (df['Brand']=='Nissin')]

japan_b2=df.loc[(df['Country'] == 'Japan') & (df['Brand']=='Myojo')]

japan_b3=df.loc[(df['Country'] == 'Japan') & (df['Brand']=='Maruchan')]

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

fig.suptitle('Ratings Comparision')

axes[0].set_title('NISSIN',fontsize=10)

axes[1].set_title('MYOJO',fontsize=10)

axes[2].set_title('MARUCHAN',fontsize=10)

sns.countplot(ax=axes[0],x="Stars", data=japan_b1, palette="muted")

sns.countplot(ax=axes[1],x="Stars", data=japan_b2, palette="muted")

sns.countplot(ax=axes[2],x="Stars", data=japan_b3, palette="muted")

plt.show()
#Nongshim 

ub1=df.loc[(df['Country'] == 'USA') & (df['Stars'] >= 4)& (df['Brand']=='Nongshim')]

usb1=df.loc[(df['Country'] == 'USA') & (df['Stars'] < 4)& (df['Brand']=='Nongshim')]

totub1=ub1['Review #'].sum()

totusb1=usb1['Review #'].sum()

us1 = totub1,totusb1



# NISSIN

ub2=df.loc[(df['Country'] == 'USA') & (df['Stars'] >= 4)& (df['Brand']=='Nissin')]

usb2=df.loc[(df['Country'] == 'USA') & (df['Stars'] < 4)& (df['Brand']=='Nissin')]

totub2=ub2['Review #'].sum()

totusb2=usb2['Review #'].sum()

us2 = totub2,totusb2



#Yamachan

ub3=df.loc[(df['Country'] == 'USA') & (df['Stars'] >= 4.5)& (df['Brand']=='Yamachan')]

usb3=df.loc[(df['Country'] == 'USA') & (df['Stars'] < 4.5)& (df['Brand']=='Yamachan')]

totub3=ub3['Review #'].sum()

totusb3=usb3['Review #'].sum()

us3 = totub3,totusb3

labels = 'Above 4.0','Below 4.0'

label1="Above 4.5","Below 4.5"

colors = ['yellowgreen', 'lightskyblue']

explode = (0.1, 0) 

fig ,ax=plt.subplots(1,3,figsize=(15,15))

ax[0].pie(us1, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%')

ax[1].pie(us2, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%')

ax[2].pie(us3, explode=explode, labels=label1, colors=colors,autopct='%1.1f%%')

ax[0].set_title('Nongshim',fontsize=15)

ax[1].set_title('NISSIN',fontsize=15)

ax[2].set_title('Yamachan',fontsize=15)

plt.show()

usa_b1=df.loc[(df['Country'] == 'USA') &  (df['Brand']=='Nongshim')]

usa_b2=df.loc[(df['Country'] == 'USA') &  (df['Brand']=='Nissin')]

usa_b3=df.loc[(df['Country'] == 'USA') &  (df['Brand']=='Yamachan')]

fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)

fig.suptitle('Ratings Comparision')

axes[0].set_title('NONGSHIM',fontsize=10)

axes[1].set_title('NISSIN',fontsize=10)

axes[2].set_title('YAMACHAN',fontsize=10)

sns.countplot(ax=axes[0],x="Stars", data=usa_b1, palette="muted")

sns.countplot(ax=axes[1],x="Stars", data=usa_b2, palette="muted")

sns.countplot(ax=axes[2],x="Stars", data=usa_b3, palette="muted")

plt.show()
# PALDO

sk1=df.loc[(df['Country'] == 'South Korea') & (df['Stars'] >= 4)& (df['Brand']=='Paldo')]

skb1=df.loc[(df['Country'] == 'South Korea') & (df['Stars'] < 4)& (df['Brand']=='Paldo')]

totsk1=sk1['Review #'].sum()

totskb1=skb1['Review #'].sum()

rev1 = totsk1,totskb1



# NONGSHIM

sk2=df.loc[(df['Country'] == 'South Korea') & (df['Stars'] >= 4)& (df['Brand']=='Nongshim')]

skb2=df.loc[(df['Country'] == 'South Korea') & (df['Stars'] < 4)& (df['Brand']=='Nongshim')]

totsk2=sk2['Review #'].sum()

totskb2=skb2['Review #'].sum()

rev2 = totsk2,totskb2



# SAMYANG FOODS

sk3=df.loc[(df['Country'] == 'South Korea') & (df['Stars'] >= 4)& (df['Brand']=='Samyang Foods')]

skb3=df.loc[(df['Country'] == 'South Korea') & (df['Stars'] < 4)& (df['Brand']=='Samyang Foods')]

totsk3=sk3['Review #'].sum()

totskb3=skb3['Review #'].sum()

rev3 = totsk3,totskb3

labels = 'Above 4.0','Below 4.0'

colors = ['yellowgreen', 'lightskyblue']

explode = (0.1, 0) 

fig ,ax=plt.subplots(1,3,figsize=(15,15))

ax[0].pie(rev1, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%')

ax[1].pie(rev2, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%')

ax[2].pie(rev3, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%')

ax[0].set_title('PALDO',fontsize=15)

ax[1].set_title('NONGSHIM',fontsize=15)

ax[2].set_title('SAMYANG FOODS',fontsize=15)

plt.show()

south_k_b1=df.loc[(df['Country'] == 'South Korea') & (df['Stars'] >= 4)& (df['Brand']=='Paldo')]

south_k_b2=df.loc[(df['Country'] == 'South Korea') & (df['Stars'] >= 4)& (df['Brand']=='Nongshim')]

south_k_b3=df.loc[(df['Country'] == 'South Korea') & (df['Stars'] >= 4)& (df['Brand']=='Samyang Foods')]

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

fig.suptitle('Ratings Comparision')

axes[0].set_title('PALDO',fontsize=10)

axes[1].set_title('NONGSHIM',fontsize=10)

axes[2].set_title('SAMYANG FOODS',fontsize=10)

sns.countplot(ax=axes[0],x="Stars", data=south_k_b1, palette="muted")

sns.countplot(ax=axes[1],x="Stars", data=south_k_b2, palette="muted")

sns.countplot(ax=axes[2],x="Stars", data=south_k_b3, palette="muted")

plt.show()