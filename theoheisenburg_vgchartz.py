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
import numpy as np

import pandas as pd

import  matplotlib.pyplot as plt

import seaborn as sns

import datetime

from pyecharts import charts

from pyecharts import options as opts

%matplotlib inline
df = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv', header=0)

df.head(10)
df.info()
df.describe().T
df.describe(include='object').T
df.shape[0]
df[df['Publisher'].isnull()].shape[0]
df.dropna(how='any', inplace=True)

df.info()
df.describe().T
df.describe(include='object').T
FGG = pd.pivot_table(df, index='Year', columns='Genre', values='Global_Sales', aggfunc=np.sum).sum().sort_values(ascending=False)

FGG
FGG_5 = pd.pivot_table(df, index='Year', columns='Genre', values='Global_Sales', aggfunc=np.sum).iloc[-5:,:].sum().sort_values(ascending=False)

FGG_5
FGG = pd.pivot_table(df, index='Year', columns='Genre', values='Global_Sales', aggfunc=np.sum).sum().sort_values(ascending=False)

FGG = pd.DataFrame(data=FGG, columns=['Genre_Sales'])

FGG_5 = pd.pivot_table(df, index='Year', columns='Genre', values='Global_Sales', aggfunc=np.sum).iloc[-5:,:].sum().sort_values(ascending=False)

FGG_5 = pd.DataFrame(data=FGG_5,columns=['Genre_Sales'])
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,6))

sns.barplot(x=FGG.index,y='Genre_Sales',data=FGG,ax=ax1)

sns.barplot(x=FGG_5.index,y='Genre_Sales',data=FGG_5,ax=ax2)
FGP = pd.pivot_table(df, index='Year', columns='Platform', values='Global_Sales', aggfunc=np.sum).sum().sort_values(ascending=False)

FGP = pd.DataFrame(data=FGP,columns=['Global_Sales'])

FGP_5 = pd.pivot_table(df, index='Year', columns='Platform',values='Global_Sales',aggfunc=np.sum).iloc[-5:,:].sum().sort_values(ascending=False)

FGP_5 = pd.DataFrame(data=FGP_5, columns=['Global_Sales'])
fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(12,6))

sns.barplot(x=FGP.index, y='Global_Sales', data=FGP, ax=ax1)

sns.barplot(x=FGP_5.index, y='Global_Sales', data=FGP_5, ax=ax2)
PS = pd.pivot_table(data=df,index='Publisher', values='Global_Sales', aggfunc=np.sum).sort_values(by='Global_Sales',ascending=False)

df_5 = df[df['Year'] > 2015]

PS_5 = pd.pivot_table(data=df_5,index='Publisher', values='Global_Sales', aggfunc=np.sum).sort_values(by='Global_Sales',ascending=False)
demo = [list(z) for z in zip(PS.head(5).index, PS.head(5).values)]

list(demo)



list(PS.head(5).values[:, 0].astype('int'))

[list(z) for z in zip(PS.head(5).index, list(PS.head(5).values[:,0].astype('int')))]
pie = charts.Pie(opts.InitOpts(width="1800px", height="800px"))

pie.add(series_name="40年间", data_pair=[list(z) for z in zip(PS.head(5).index, list(PS.head(5).values[:,0]))], 

        center=["30%", "50%"], radius="40%")

pie.add(series_name="近5年间", data_pair=[list(z) for z in zip(PS_5.head(5).index, list(PS_5.head(5).values[:,0]))], 

        center=["78%", "50%"], radius="40%")

pie.set_series_opts(label_opts=opts.LabelOpts(is_show=True, font_size=18, font_weight='bold', formatter='{b}:{d}%'))

pie.set_global_opts(title_opts=opts.TitleOpts(title="发行商对比图", title_textstyle_opts=opts.TextStyleOpts(font_size=30, font_weight='bold'), pos_left='center'), legend_opts=opts.LegendOpts(is_show=False))



pie.render('前5名的发行商变化.html')
districts = ['NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales']

Market = pd.pivot_table(df, index='Year', values=districts, aggfunc=np.sum)
fig=plt.figure(figsize=(10,6))

sns.lineplot(data=Market)

plt.title('The development of all markets')

plt.show()
Plateform = ['Nintendo','Electronic Arts','Activision','Sony Computer Entertainment','Ubisoft']
PHS = df[df['Publisher'].isin(Plateform)]

PHS = pd.pivot_table(data=PHS, index='Year', columns='Publisher', values='Global_Sales', aggfunc=np.sum)

PHS.plot(title='Global Sales with year',figsize=(12,6))
PHS = df[df['Publisher'].isin(Plateform)]

Plateform_G = pd.pivot_table(data=PHS, index=['Genre','Publisher'], values=Market, aggfunc=np.sum).sort_values(by=['Genre','Global_Sales'],ascending=False) #这里必须对Genre排序不然会乱，只能对Genre分组

Plateform_G.head()
Plateform_G_PCT = Plateform_G.div(Plateform_G.groupby(level=0).sum()).round(2)

Plateform_G_PCT=Plateform_G_PCT.sort_values(by=['Genre','Global_Sales'],ascending=False)

Plateform_G_PCT
df_Ubisoft = df[df['Publisher']=='Ubisoft']

df_Ubisoft.sort_values(by='Year')

df_Ubisoft.isnull().sum()
df_Ubisoft.dropna(axis=0,how='any', inplace=True)

df_Ubisoft.isnull().sum()

df_Ubisoft
df_Ubisoft.sort_values(by='Year', inplace=True)

df_Ubisoft
len(df_Ubisoft['Name'].unique())
Ubisoft_G_N = pd.pivot_table(data=df_Ubisoft, index='Year', values='Name', aggfunc='count')

Ubisoft_G_N = pd.DataFrame(data=Ubisoft_G_N.values, index=Ubisoft_G_N.index, columns=['Count'])

Ubisoft_G_N
sns.lineplot(x=Ubisoft_G_N.index, y='Count', data=Ubisoft_G_N)
Ubisoft_G = df_Ubisoft.groupby('Genre')['Name'].agg('count')

Ubisoft_G = pd.DataFrame(data=Ubisoft_G.values, index=Ubisoft_G.index, columns=['Count'])

Ubisoft_G.sort_values(by='Count', ascending=False, inplace=True)
fig = plt.figure(figsize=(15, 5))

sns.barplot(x=Ubisoft_G.index, y='Count', data=Ubisoft_G)
Ubisoft_GLSales_genre = pd.pivot_table(data=df_Ubisoft, index='Genre', values='Global_Sales', aggfunc=np.sum)

Ubisoft_GLSales_genre = pd.DataFrame(data=Ubisoft_GLSales_genre.values, index=Ubisoft_GLSales_genre.index, columns=['Global Sales'])

Ubisoft_GLSales_genre.sort_values(by='Global Sales', ascending=False, inplace=True)



Ubisoft_GLSales_year = pd.pivot_table(data=df_Ubisoft, index='Year', values='Global_Sales', aggfunc=np.sum)

Ubisoft_GLSales_year = pd.DataFrame(data=Ubisoft_GLSales_year.values, index=Ubisoft_GLSales_year.index, columns=['Global Sales'])

Ubisoft_GLSales_year.sort_values(by='Year', ascending=True, inplace=True)
Ubisoft_GLSales_genre
Ubisoft_GLSales_year
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

sns.barplot(x=Ubisoft_GLSales_genre.index, y='Global Sales', data=Ubisoft_GLSales_genre, ax=axes[0])

sns.lineplot(x=Ubisoft_GLSales_year.index, y='Global Sales', data=Ubisoft_GLSales_year, ax=axes[1])
Ubisoft_NASales_genre = pd.pivot_table(data=df_Ubisoft, index='Genre', values='NA_Sales', aggfunc=np.sum)

Ubisoft_NASales_genre = pd.DataFrame(data=Ubisoft_NASales_genre.values, index=Ubisoft_NASales_genre.index, columns=['NA Sales'])

Ubisoft_NASales_genre.sort_values(by='NA Sales', ascending=False, inplace=True)



Ubisoft_NASales_year = pd.pivot_table(data=df_Ubisoft, index='Year', values='NA_Sales', aggfunc=np.sum)

Ubisoft_NASales_year = pd.DataFrame(data=Ubisoft_NASales_year.values, index=Ubisoft_NASales_year.index, columns=['NA Sales'])

Ubisoft_NASales_year.sort_values(by='Year', ascending=True, inplace=True)
Ubisoft_NASales_genre
Ubisoft_NASales_year
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

sns.barplot(x=Ubisoft_NASales_genre.index, y='NA Sales', data=Ubisoft_NASales_genre, ax=axes[0])

sns.lineplot(x=Ubisoft_NASales_year.index, y='NA Sales', data=Ubisoft_NASales_year, ax=axes[1])
Ubisoft_EUSales_genre = pd.pivot_table(data=df_Ubisoft, index='Genre', values='EU_Sales', aggfunc=np.sum)

Ubisoft_EUSales_genre = pd.DataFrame(data=Ubisoft_EUSales_genre.values, index=Ubisoft_EUSales_genre.index, columns=['EU Sales'])

Ubisoft_EUSales_genre.sort_values(by='EU Sales', ascending=False, inplace=True)



Ubisoft_EUSales_year = pd.pivot_table(data=df_Ubisoft, index='Year', values='EU_Sales', aggfunc=np.sum)

Ubisoft_EUSales_year = pd.DataFrame(data=Ubisoft_EUSales_year.values, index=Ubisoft_EUSales_year.index, columns=['EU Sales'])

Ubisoft_EUSales_year.sort_values(by='Year', ascending=True, inplace=True)
Ubisoft_EUSales_genre
Ubisoft_EUSales_year
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

sns.barplot(x=Ubisoft_EUSales_genre.index, y='EU Sales', data=Ubisoft_EUSales_genre, ax=axes[0])

sns.lineplot(x=Ubisoft_EUSales_year.index, y='EU Sales', data=Ubisoft_EUSales_year, ax=axes[1])
Ubisoft_JPSales_genre = pd.pivot_table(data=df_Ubisoft, index='Genre', values='JP_Sales', aggfunc=np.sum)

Ubisoft_JPSales_genre = pd.DataFrame(data=Ubisoft_JPSales_genre.values, index=Ubisoft_JPSales_genre.index, columns=['JP Sales'])

Ubisoft_JPSales_genre.sort_values(by='JP Sales', ascending=False, inplace=True)



Ubisoft_JPSales_year = pd.pivot_table(data=df_Ubisoft, index='Year', values='JP_Sales', aggfunc=np.sum)

Ubisoft_JPSales_year = pd.DataFrame(data=Ubisoft_JPSales_year.values, index=Ubisoft_JPSales_year.index, columns=['JP Sales'])

Ubisoft_JPSales_year.sort_values(by='Year', ascending=True, inplace=True)
Ubisoft_JPSales_genre
Ubisoft_JPSales_year
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

sns.barplot(x=Ubisoft_JPSales_genre.index, y='JP Sales', data=Ubisoft_JPSales_genre, ax=axes[0])

sns.lineplot(x=Ubisoft_JPSales_year.index, y='JP Sales', data=Ubisoft_JPSales_year, ax=axes[1])
Sales_Contrast = pd.concat([Ubisoft_GLSales_year, Ubisoft_NASales_year, Ubisoft_EUSales_year, Ubisoft_JPSales_year], axis=1)

Sales_Contrast
fig = plt.figure(figsize=(15, 5))

sns.lineplot(data=Sales_Contrast,markers=True, dashes=False)