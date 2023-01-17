import pandas as pd

from pandas import Series,DataFrame

import numpy as np



# For Visualization

import matplotlib.pyplot as plt

import matplotlib

#%matplotlib inline



matplotlib.style.use('ggplot')

df=pd.read_csv('../input/indicators_by_company.csv')
#number of indicators by company

df_ind_count = pd.concat([ df[['company_id', 'indicator_id', '2010']].dropna().groupby('company_id')['indicator_id'].count()

,df[['company_id', 'indicator_id', '2011']].dropna().groupby('company_id')['indicator_id'].count()

,df[['company_id', 'indicator_id', '2012']].dropna().groupby('company_id')['indicator_id'].count()

,df[['company_id', 'indicator_id', '2013']].dropna().groupby('company_id')['indicator_id'].count()

,df[['company_id', 'indicator_id', '2014']].dropna().groupby('company_id')['indicator_id'].count()

,df[['company_id', 'indicator_id', '2015']].dropna().groupby('company_id')['indicator_id'].count()

,df[['company_id', 'indicator_id', '2016']].dropna().groupby('company_id')['indicator_id'].count()

], axis=1)

df_ind_count.columns=['2010','2011','2012','2013','2014','2015','2016']

df_ind_count.head()
#distribution of the indicators number per company in the data set per year

l_df=[]

for c in df_ind_count.columns:

    l_df.append(list(df_ind_count[c].dropna().describe()))

df_ind_count_desc=DataFrame(l_df,columns=['count','mean','std','min','25%','50%','75%','max'],index= df_ind_count.columns)

df_ind_count_desc


df_ind_count.plot.box()
df_ind_count['2010'].hist(bins=15)
df_ind_count['2011'].hist(bins=20)
df_ind_count['2012'].hist(bins=20)
df_ind_count['2013'].hist(bins=20)
df_ind_count['2014'].hist(bins=20)
df_ind_count['2015'].hist(bins=20)
df_ind_count['2016'].hist(bins=20)
#first 20 indicators which have maximum number of companies

#each cell contains the num of companies with not empty indicator 

#(one and only one indicator without taking into account any other indicators )

#in this year

df_comp_count = pd.concat([

df[['company_id', 'indicator_id', '2010']].dropna().groupby('indicator_id')['company_id'].count().sort_values(ascending=False).head(200),

df[['company_id', 'indicator_id', '2011']].dropna().groupby('indicator_id')['company_id'].count().sort_values(ascending=False).head(200),

df[['company_id', 'indicator_id', '2012']].dropna().groupby('indicator_id')['company_id'].count().sort_values(ascending=False).head(200),

df[['company_id', 'indicator_id', '2013']].dropna().groupby('indicator_id')['company_id'].count().sort_values(ascending=False).head(200),

df[['company_id', 'indicator_id', '2014']].dropna().groupby('indicator_id')['company_id'].count().sort_values(ascending=False).head(200),

df[['company_id', 'indicator_id', '2015']].dropna().groupby('indicator_id')['company_id'].count().sort_values(ascending=False).head(200),

df[['company_id', 'indicator_id', '2016']].dropna().groupby('indicator_id')['company_id'].count().sort_values(ascending=False).head(200)

], axis=1)



df_comp_count.columns=['2010','2011','2012','2013','2014','2015','2016']



df_comp_count.head()
df_comp_count.dropna().describe()
#indicators common for as many companies as possible

#The script below calculates the number of companies (intersect)  for each next indicator 

#e.g. how many companies have not null indicator 1 and 2 and 3 and so on..)

#it's done separately in each year

#an indicator with the max companies is the first, then the second



list_s_int=[] 

for c in df_comp_count.columns:

 df_comp_count.sort_values(c, axis=0, ascending=False ,inplace=True)

 li=df_comp_count.index

 s_int = pd.Series(np.zeros(len(li)), index=li)

 s1=df.loc[((df['indicator_id']==li[0]) & (df[c].notnull())),'company_id'].unique()

 s_int[li[0]]=len(s1)

 for i in range(1,len(li)):

  s2=df.loc[((df['indicator_id']==li[i]) & (df[c].notnull())),'company_id'].unique()

  s1=pd.Series(np.intersect1d(s1, s2))

  s_int[li[i]]=len(s1)

 list_s_int.append(s_int)

 

df_comp_int_count = pd.concat(list_s_int, axis=1) 

df_comp_int_count.columns=['2010','2011','2012','2013','2014','2015','2016']
#it makes sense to review dscening sorted data in each column separately

df_comp_int_count.head()
#The number of companies decreasies with adding each new indicator into the intersection

df_comp_int_count['2010'].sort_values(ascending=False).head(20).plot('bar')
df_comp_int_count['2011'].sort_values(ascending=False).head(20).plot('bar')
df_comp_int_count['2012'].sort_values(ascending=False).head(20).plot('bar')
df_comp_int_count['2013'].sort_values(ascending=False).head(20).plot('bar')
df_comp_int_count['2014'].sort_values(ascending=False).head(20).plot('bar')
df_comp_int_count['2015'].sort_values(ascending=False).head(20).plot('bar')
df_comp_int_count['2016'].sort_values(ascending=False).head(20).plot('bar')
df_comp_int_count['2011'].sort_values(ascending=False).head(20)
df_comp_int_count['2012'].sort_values(ascending=False).head(20)
df_comp_int_count['2013'].sort_values(ascending=False).head(20)
df_comp_int_count['2014'].sort_values(ascending=False).head(20)
df_comp_int_count['2015'].sort_values(ascending=False).head(20)
#indicators common in first 25 for '2012', '2013', '2014', '2015'

indicators=df_comp_int_count['2011'].sort_values(ascending=False).index[0:25]

for c in (['2012', '2013', '2014', '2015']):

    indicators2=df_comp_int_count[c].sort_values(ascending=False).index[0:25]

    indicators=pd.Series(np.intersect1d(pd.Series(indicators), pd.Series(indicators2)))

indicators
#now let's build the list of companies based on the intersection of 20 indicators with not null values

list_indicators_int_comp=[] 

li=indicators

for c in (['2011', '2012', '2013', '2014', '2015']):

 s1=df.loc[((df['indicator_id']==li[0]) & (df[c].notnull())),'company_id'].unique()

 for i in range(1,len(li)):

  s2=df.loc[((df['indicator_id']==li[i]) & (df[c].notnull())),'company_id'].unique()

  s1=pd.Series(np.intersect1d(s1, s2))

 list_indicators_int_comp.append(s1)
#df ready to be analyzed in the next part

#it can be easily separated for each year

#I exclude 2010 and 2016 because of the lack of the data

list_df_rtba=[]

start_year=2011

for i in range(0,len(list_indicators_int_comp)):

 year=start_year+i

 if year<2016:

     df_rtba_i=df.loc[((df['indicator_id'].isin(indicators)) & (df['company_id'].isin(list_indicators_int_comp[i]))),['company_id','indicator_id',str(year)]]

     df_rtba_piv_i=df_rtba_i.pivot(index='company_id', columns='indicator_id', values=str(year))

     df_rtba_piv_i['year']=year

     list_df_rtba.append(df_rtba_piv_i)



df_rtba=pd.concat(list_df_rtba)
#starting 2011

df_rtba.head(10)
#and end 2015

df_rtba.tail(10)