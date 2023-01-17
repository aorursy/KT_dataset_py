

import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.simplefilter("ignore")



df= pd.read_csv("/kaggle/input/videogamesales/vgsales.csv", index_col=0)

df.head()

df.describe()
n=df.loc[df['Year'].isnull()]



df.loc[df['Year'].isnull()].count()
d_grouped=df.groupby(df['Platform']).count()

d_grouped.sort_values(by=('Global_Sales'),ascending=False,inplace=True)

d_grouped.reset_index(inplace=True)

d_grouped=d_grouped.head(15)

#we take the 15 most popular platforms only



plt.subplots(figsize=(20,8))

g=sns.barplot(x=d_grouped['Platform'],y=d_grouped['Name'])

g.set(xlabel='Platform', ylabel='Number of games')

var=g.set_title('Most popular platforms by number of games',size=30)

n.loc[n['Platform']=='DS','Platform'].count()

#DS release date 2011
n.loc[n['Platform']=='PS2','Platform'].count()

#PS2 release date 2000
n.loc[n['Platform']=='PS3','Platform'].count()

#PS3 release date 2006
n.loc[n['Platform']=='Wii','Platform'].count()

#Wii release date 2006
n.loc[n['Platform']=='X360','Platform'].count()

#X360 release date 2009
n.loc[n['Platform']=='PSP','Platform'].count()

#PSP release date 2004
n.loc[n['Platform']=='PS','Platform'].count()

#PS release date 1994
n.loc[n['Platform']=='PC','Platform'].count()
n.loc[n['Platform']=='XB','Platform'].count()

#XB release date 2001
n.loc[n['Platform']=='GBA','Platform'].count()

#GBA release date 2001
n.loc[n['Platform']=='GC','Platform'].count()

#GC release date 2000
n.loc[n['Platform']=='3DS','Platform'].count()

#3DS release date 2011
n.loc[n['Platform']=='PSV','Platform'].count()

#PSV release date 2011
n.loc[n['Platform']=='PS4','Platform'].count()

#PS4 release date 2013
n.loc[n['Platform']=='N64','Platform'].count()

#N64 release date 1996
bins=[1980,2000,2005,2011,2020]

labels=['Gen1','Gen2','Gen3','Gen4']

df['Year']= pd.cut(df["Year"], bins , labels=labels)

#dividing the years into bins





df.loc[(df['Year'].isnull())&((df['Platform']=='GBA')|(df['Platform']=='GC')|(df['Platform']=='XB')|(df['Platform']=='PS2')),'Year']='Gen2'

df.loc[(df['Year'].isnull())&((df['Platform']=='PSP')|(df['Platform']=='PS3')|(df['Platform']=='Wii')|(df['Platform']=='DS')|(df['Platform']=='X360')|(df['Platform']=='PC')),'Year']='Gen3'

df.loc[(df['Year'].isnull())&((df['Platform']=='XOne')|(df['Platform']=='PS4')|(df['Platform']=='3DS')|(df['Platform']=='WiiU')|(df['Platform']=='PSV')),'Year']='Gen4'

df.loc[df['Year'].isnull(),'Year']='Gen1'

#filling the missing values

df.loc[df['Publisher'].isnull()].count()
df=df.dropna()
c_grouped=df.groupby('Year',sort=True).count()

c_grouped.reset_index(inplace=True)



f,ax= plt.subplots(figsize=(12,5))

g=sns.barplot(x=c_grouped['Year'],y=c_grouped['Name'])

g.set_xticklabels( labels=c_grouped['Year'],size=14)

g.set_title('Number of games by generations',size=30)

var=g.set(xlabel='Generation', ylabel='Number of games')
gen_grouped=df.groupby('Year',sort=True).sum()

gen_grouped.reset_index(inplace=True)



f,ax= plt.subplots(figsize=(12,5))

g=sns.barplot(x=gen_grouped['Year'],y=gen_grouped['Global_Sales'])

g.set_xticklabels( labels=gen_grouped['Year'],size=14)

g.set_title('Sales by generation',size=30)

var=g.set(xlabel='Generation', ylabel='Sales')
df2=df

df2=df2.drop(columns='Genre')

df2=df2.drop(columns='Platform')

df2=df2.drop(columns='Publisher')



df2=df2.groupby(df2['Year']).sum()

df2['NA_Sales']=df2['NA_Sales']/df2['Global_Sales']

df2['JP_Sales']=df2['JP_Sales']/df2['Global_Sales']

df2['EU_Sales']=df2['EU_Sales']/df2['Global_Sales']

df2['Other_Sales']=df2['Other_Sales']/df2['Global_Sales']

df2.drop(columns='Global_Sales', inplace=True)

f=df2.plot.bar(stacked=True, alpha=0.6,title='Regions/generation comparison',figsize=(10,5))

f.set_yticklabels(labels=['0','20%','40%','60%','80%','100%'],size=14)

var=f.set(xlabel='Generation',ylabel= 'Sales percentage'  )
EU=df.sort_values(by='EU_Sales',ascending=False).head(5)

NA=df.sort_values(by='NA_Sales',ascending=False).head(5)

JP=df.sort_values(by='JP_Sales',ascending=False).head(5)

Other=df.sort_values(by='Other_Sales',ascending=False).head(5)



f = plt.figure(figsize=(25,10))

ax1 = f.add_subplot(1,6,1)

ax1=sns.barplot(EU['Name'],EU['EU_Sales'],palette='cubehelix_r')

ax1.set_xticklabels(labels=EU['Name'],rotation=90, size=14)

ax1.set(xlabel='', ylabel='Europe')

ax1.set_title('Sales in EU')

ax2 = f.add_subplot(1,6,2)

ax2=sns.barplot(NA['Name'],NA['NA_Sales'],palette='cubehelix_r')

ax2.set_xticklabels(labels=NA['Name'],rotation=90, size=14)

ax2.set(xlabel='', ylabel='North America')

ax2.set_title('Sales in NA')

ax3 = f.add_subplot(1,6,3)

ax3=sns.barplot(JP['Name'],JP['JP_Sales'],palette='cubehelix_r')

ax3.set_xticklabels(labels=JP['Name'],rotation=90, size=14)

ax3.set(xlabel='', ylabel='Japan')

ax3.set_title('Sales in JP')

ax4 = f.add_subplot(1,6,4)

ax4=sns.barplot(Other['Name'],Other['Other_Sales'],palette='cubehelix_r')

ax4.set_xticklabels(labels=Other['Name'],rotation=90, size=14)

ax4.set(xlabel='', ylabel='Other regions')

var=ax4.set_title('Sales in Other regions')
a_group=df.groupby(df['Platform']).count()

a_group.sort_values(by=('Global_Sales'),ascending=False,inplace=True)

a_group.reset_index(inplace=True)

a_group=a_group.head(15)

#we take the 15 most popular platforms only



plt.subplots(figsize=(20,8))

g=sns.barplot(x=a_group['Platform'],y=a_group['Name'])

g.set(xlabel='Platform', ylabel='Number of games')

var=g.set_title('Most popular platforms by number of games',size=30)

plt.subplots(figsize=(20,8))

g=sns.barplot(x=df['Platform'],y=df.groupby(df['Platform']).cumsum()['Global_Sales'].sort_values())

g.set(xlabel='Platform', ylabel='Sales')

var=g.set_title('Most popular platforms by number of games',size=30)

b_grouped=df.groupby(['Year','Platform'],sort=True).count()



b_grouped.reset_index(inplace=True)



f,ax= plt.subplots(figsize=(10,13))

g=sns.scatterplot(x=b_grouped['Year'],y=b_grouped['Platform'],size=b_grouped['Name'],sizes=(1,1000),legend=False)

g.set_title('Popular platform by generation by number of products',size=30)

var=g.set(xlabel='Generation', ylabel='Platform')



f,ax= plt.subplots(figsize=(10,13))

df=df.sort_values(by='Year')

p_grouped=df.groupby(['Year','Platform']).sum()

p_grouped.reset_index(inplace=True)



g=sns.scatterplot(x=p_grouped['Year'],y=p_grouped['Platform'],size=p_grouped['Global_Sales'],sizes=(1,1000),legend=False)

g.set_title('Popular platform by generation by sales',size=30)

var=g.set(xlabel='Generation', ylabel='Platform')


df1=df

df1=df1.drop(columns='Publisher')

df1=df1.drop(columns='Genre')



df1=df1.groupby(df1['Platform']).sum()



df1['NA_Sales']=df1['NA_Sales']/df1['Global_Sales']

df1['JP_Sales']=df1['JP_Sales']/df1['Global_Sales']

df1['EU_Sales']=df1['EU_Sales']/df1['Global_Sales']

df1['Other_Sales']=df1['Other_Sales']/df1['Global_Sales']

df1.drop(columns='Global_Sales', inplace=True)





df1.drop(['2600','3DO','DC','GB','GG','NG','NES','PCFX','SAT','SCD','SNES','TG16','WS'],inplace=True)

#We take only the most popular platform for a clearer view

f=df1.plot.bar(stacked=True, alpha=0.6,title='Regions/platform comparison',figsize=(10,5))

f.set_yticklabels(labels=['0','20%','40%','60%','80%','100%'],size=14)

var=f.set(xlabel='Platform',ylabel= 'Sales percentage'  )
pub_grouped=df.groupby(df['Publisher']).count()

pub_grouped.sort_values(by=('Global_Sales'),ascending=False,inplace=True)

pub_grouped.reset_index(inplace=True)

pub_grouped=pub_grouped.head(15)

#we take the 15 most popular platforms only



plt.subplots(figsize=(20,8))

g=sns.barplot(x=pub_grouped['Publisher'],y=pub_grouped['Name'])

g.set(xlabel='Publisher', ylabel='Number of games')

g.set_title('Most prolific Publisher by number of games',size=30)

var=g.set_xticklabels( labels=pub_grouped['Publisher'],size=8,rotation=10)

df1=df.loc[df['Publisher'].isin(pub_grouped['Publisher'])]

plt.subplots(figsize=(20,8))

g=sns.barplot(x=df1['Publisher'],y=df1.groupby(df['Publisher']).cumsum()['Global_Sales'].sort_values())

#Replace the y label with Number of published games

g.set_title('Most popular Publisher by number of sales',size=30)

g.set(xlabel='Publisher', ylabel='Sales')

var=g.set_xticklabels( labels=pub_grouped['Publisher'],size=8,rotation=10)

e_grouped=df1.groupby(['Year','Publisher'],sort=True).count()

e_grouped.reset_index(inplace=True)



f,ax= plt.subplots(figsize=(10,13))

g=sns.scatterplot(x=e_grouped['Year'],y=e_grouped['Publisher'],size=e_grouped['Name'],sizes=(1,1000),legend=False)

g.set_title('Popular publisher by generation by number of products',size=30)

var=g.set(xlabel='Generation', ylabel='Publisher')

f,ax= plt.subplots(figsize=(10,13))

df1=df1.sort_values(by='Year')

r_grouped=df1.groupby(['Year','Publisher']).sum()

r_grouped.reset_index(inplace=True)



g=sns.scatterplot(x=r_grouped['Year'],y=r_grouped['Publisher'],size=r_grouped['Global_Sales'],sizes=(1,1000),legend=False)

g.set_title('Popular publisher by generation by sales',size=30)

var=g.set(xlabel='Generation', ylabel='Publisher')



df2=df1

df2=df2.drop(columns='Genre')

df2=df2.drop(columns='Platform')



df2=df2.groupby(df2['Publisher']).sum()

df2['NA_Sales']=df2['NA_Sales']/df2['Global_Sales']

df2['JP_Sales']=df2['JP_Sales']/df2['Global_Sales']

df2['EU_Sales']=df2['EU_Sales']/df2['Global_Sales']

df2['Other_Sales']=df2['Other_Sales']/df2['Global_Sales']

df2.drop(columns='Global_Sales', inplace=True)

f=df2.plot.bar(stacked=True, alpha=0.6,title='Regions/publisher comparison',figsize=(10,5))

var=f.set_yticklabels(labels=['0','20%','40%','60%','80%','100%'],size=14)

var=f.set(xlabel='Publisher',ylabel= 'Sales percentage'  )
f_grouped= df.groupby(df['Genre']).count()

f_grouped.sort_values(by=('Global_Sales'),ascending=False,inplace=True)

f_grouped.reset_index(inplace=True)



plt.subplots(figsize=(20,8))

g=sns.barplot(x=f_grouped['Genre'],y=f_grouped['Name'])

g.set(xlabel='Genre', ylabel='Number of games')

var=g.set_title('Most popular Genre by number of games',size=30)

plt.subplots(figsize=(20,8))

g=sns.barplot(x=df['Genre'],y=df.groupby(df['Genre']).cumsum()['Global_Sales'].sort_values())

g.set(xlabel='Genre', ylabel='Sales')

var=g.set_title('Most popular Genre by number of games',size=30)
g_grouped=df.groupby(['Year','Genre'],sort=True).count()

g_grouped.reset_index(inplace=True)



f,ax= plt.subplots(figsize=(10,13))

g=sns.scatterplot(x=g_grouped['Year'],y=g_grouped['Genre'],size=g_grouped['Name'],sizes=(1,1000),legend=False)

g.set_title('Popular genre by generation by number of products',size=30)

var=g.set(xlabel='Generation', ylabel='Genre')
f,ax= plt.subplots(figsize=(10,13))

df=df.sort_values(by='Year')

h_grouped=df.groupby(['Year','Genre']).sum()

h_grouped.reset_index(inplace=True)



g=sns.scatterplot(x=h_grouped['Year'],y=h_grouped['Genre'],size=h_grouped['Global_Sales'],sizes=(1,1000),legend=False)

g.set_title('Popular genre by generation by sales',size=30)

var=g.set(xlabel='Generation', ylabel='Genre')
df2=df

df2=df2.drop(columns='Publisher')

df2=df2.drop(columns='Platform')



df2=df2.groupby(df2['Genre']).sum()

df2['NA_Sales']=df2['NA_Sales']/df2['Global_Sales']

df2['JP_Sales']=df2['JP_Sales']/df2['Global_Sales']

df2['EU_Sales']=df2['EU_Sales']/df2['Global_Sales']

df2['Other_Sales']=df2['Other_Sales']/df2['Global_Sales']

df2.drop(columns='Global_Sales', inplace=True)

f=df2.plot.bar(stacked=True, alpha=0.6,title=' Regions/Genre comparison',figsize=(10,5))

var=f.set_yticklabels(labels=['0','20%','40%','60%','80%','100%'],size=14)

var=f.set(xlabel='Genre',ylabel= 'Sales percentage'  )
j_grouped=df.groupby(['Genre','Platform'],sort=True).count()

j_grouped.reset_index(inplace=True)



f,ax= plt.subplots(figsize=(20,10))

g=sns.scatterplot(x=j_grouped['Platform'],y=j_grouped['Genre'],size=j_grouped['Name'],sizes=(1,1000),legend=False)

var=g.set_title('Popular genre by platform by number of products',size=30)
f,ax= plt.subplots(figsize=(20,10))

df=df.sort_values(by='Year')

i_grouped=df.groupby(['Platform','Genre']).sum()

i_grouped.reset_index(inplace=True)



g=sns.scatterplot(x=i_grouped['Platform'],y=i_grouped['Genre'],size=i_grouped['Global_Sales'],sizes=(1,1000),legend=False)

var=g.set_title('Popular genre by platform by sales',size=30)
m_grouped=df1.groupby(['Publisher','Platform'],sort=True).count()

m_grouped.reset_index(inplace=True)



f,ax= plt.subplots(figsize=(20,10))

g=sns.scatterplot(x=m_grouped['Platform'],y=m_grouped['Publisher'],size=m_grouped['Name'],sizes=(1,1000),legend=False)

var=g.set_title('Popular publisher by platform by number of products',size=30)
k_grouped=df.groupby(df['Publisher']).count()

k_grouped.sort_values(by=('Global_Sales'),ascending=False,inplace=True)

k_grouped.reset_index(inplace=True)

k_grouped=k_grouped.head(15)

df1=df.loc[df['Publisher'].isin(k_grouped['Publisher'])]







f,ax= plt.subplots(figsize=(20,10))

df1=df1.sort_values(by='Year')

l_grouped=df1.groupby(['Platform','Publisher']).sum()

l_grouped.reset_index(inplace=True)



g=sns.scatterplot(x=l_grouped['Platform'],y=l_grouped['Publisher'],size=l_grouped['Global_Sales'],sizes=(1,1000),legend=False)

var=g.set_title('Popular publisher by platform by sales',size=30)

  
o_grouped=df1.groupby(['Publisher','Genre'],sort=True).count()



o_grouped.reset_index(inplace=True)



f,ax= plt.subplots(figsize=(20,10))

g=sns.scatterplot(x=o_grouped['Genre'],y=o_grouped['Publisher'],size=o_grouped['Name'],sizes=(1,2000),legend=False)

var=g.set_title('Popular publisher by genre by number of products',size=30)


f,ax= plt.subplots(figsize=(20,10))

n_grouped=df1.groupby(['Genre','Publisher']).sum()

n_grouped.reset_index(inplace=True)



g=sns.scatterplot(x=n_grouped['Genre'],y=n_grouped['Publisher'],size=n_grouped['Global_Sales'],sizes=(1,2000),legend=False)

var=g.set_title('Popular publisher by genre by sales',size=30)

  
print("Skewness: %f" % df['Global_Sales'].skew())

print("Kurtosis: %f" % df['Global_Sales'].kurt())
quant = df['Global_Sales'].quantile(0.90)

df2 = df[df['Global_Sales'] < quant]



print("Skewness: %f" % df2['Global_Sales'].skew())

print("Kurtosis: %f" % df2['Global_Sales'].kurt())
var=sns.distplot(df2['Global_Sales'])


#mapping genre

Genre_dict = {'Misc':0,'Fighting':1,'Action':2,'Shooter':3,'Sports':4,'Platform':5,'Simulation':6,'Racing':7,'Puzzle':8,'Adventure':9,

             'Role-Playing':10,'Strategy':11}

df2['Genre']=df2['Genre'].map(Genre_dict)



#mapping publisher

j=0

a=(df2["Publisher"].unique())

dict_Pub={}

while j<len(a):

    dict_Pub[j]=(a[j])

    j+=1

dict_Pub = {v: k for k, v in dict_Pub.items()}

df2['Publisher']=df2['Publisher'].map(dict_Pub)



#mapping platform

j=0

dict_plat={}

a=df2["Platform"].unique()



while j < len(a):

    dict_plat[j]=a[j]

    j+=1



dict_plat = {v:k for k, v in dict_plat.items()}

df2["Platform"]=df2["Platform"].map(dict_plat)
sns.set()

cols = ['Platform','Publisher','Genre','NA_Sales', 'JP_Sales', 'EU_Sales', 'Other_Sales']

g=sns.pairplot(df2[cols], size = 2.5)

plt.show()
corr=df2.corr()

f,ax=plt.subplots(figsize=(10,8))

var=sns.heatmap(corr,annot=True,cmap='cubehelix_r')