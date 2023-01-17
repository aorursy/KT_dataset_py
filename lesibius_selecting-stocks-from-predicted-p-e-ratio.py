%matplotlib inline



#Basic data manipulation

import numpy as np

import pandas as pd



#Tools

import datetime as dt



#IPython manager

from IPython.display import display



#Graphs

import seaborn as sns

import matplotlib.pyplot as plt



#Machine learning

from sklearn.neighbors import NearestNeighbors
fundamentals = pd.read_csv('../input/fundamentals.csv',index_col = 0)

prices = pd.read_csv('../input/prices.csv')

securities = pd.read_csv('../input/securities.csv',index_col=0)

sectors = securities['GICS Sector'].unique()

sub_industry = securities['GICS Sub Industry'].unique()

accounts = fundamentals.columns.values[2:]
def set_year(period):

    """

    Take a date as a string formatted such as in the fundamentals dataframe and returns its year as an integer

    

    Parameters

    ----------

    period: Date as '%Y-%m-%d'

    

    Returns

    -------

    year: Year in the 'Period Ending' column

    """

    x = dt.datetime.strptime(period,'%Y-%m-%d')

    if x.month >= 6:        #If the reporting month is after June, return the reported year

        return x.year

    else:                   #Else, return the year before

        return x.year - 1



def get_publication_date(period):

    """

    Returns the estimated publication date as a datetime object, assuming a 90 days delay from the reporting day

    

    Parameters

    ----------

    period: Reporting date string as '%Y-%m-%d'

    

    Returns

    --------

    published: Publication date as datetime object

    

    """

    x = dt.datetime.strptime(period,'%Y-%m-%d') + dt.timedelta(days=90)

    

    return x
df_eps = fundamentals[['Ticker Symbol','Period Ending','Earnings Per Share']]

df_eps['Year'] = df_eps['Period Ending'].map(set_year)   #This replace values in Period Ending by their closest year

df_eps = df_eps[df_eps['Year'] > 2011] #Just keep year from 2012



df_eps[[col for col in df_eps.columns if col not in ['Period Ending','Year']]].describe() #I do not need stats about 'Period Ending'
_ = [col for col in df_eps.columns if col != 'Period Ending']

sns.violinplot(data=df_eps[_],x='Year',y='Earnings Per Share')
df_eps = df_eps[df_eps['Earnings Per Share'] > 0.10] #Remove any data where EPS is below 1cts
_ = np.array(fundamentals['Ticker Symbol'].unique())

companies = [symb for symb in prices['symbol'].unique() if symb in fundamentals['Ticker Symbol'].unique()]



df_prices = prices[prices['symbol'].isin(companies)]

df_eps = df_eps[df_eps['Ticker Symbol'].isin(companies)]



#This is a quick summary of the 447 remaining stocks

securities.loc[companies].head()
def clean_prices_date(d):

    try:

        return dt.datetime.strptime(d,'%Y-%m-%d 00:00:00')

    except:

        return dt.datetime.strptime(d,'%Y-%m-%d')



def find_nearest(group, match, groupname):

    match = match[match[groupname] == group.name]

    nbrs = NearestNeighbors(1).fit(match['date'].values[:, None])

    dist, ind = nbrs.kneighbors(group['Published'].values[:, None])



    group['KeyDate'] = group['Published']

    group['ActualDate'] = match['date'].values[ind.ravel()]

    return group



df_eps['Published'] = df_eps['Period Ending'].map(get_publication_date)

df_prices['date'] = df_prices['date'].map(clean_prices_date)



df_eps_mod = df_eps.groupby('Ticker Symbol').apply(find_nearest, df_prices, 'symbol')

df_prices_mod = df_prices[[col for col in df_prices.columns]]

df_prices_mod.rename(columns={'symbol':'Ticker Symbol','date':'ActualDate'},inplace=True)

df_merged = pd.merge(df_eps_mod,df_prices_mod,on=['ActualDate','Ticker Symbol'])

df_pe = df_merged[['Ticker Symbol','Year','Earnings Per Share','close']]

df_pe['PE'] = df_pe['close'] / df_pe['Earnings Per Share']
_ = securities.loc[df_pe['Ticker Symbol'],'GICS Sector']

_.index = df_pe.index

df_pe_sector = pd.concat([df_pe,_],axis=1)
yearsplt = [2014,2015]

g = sns.violinplot(data=df_pe_sector[df_pe_sector['Year'].isin(yearsplt)][['PE','GICS Sector','Year']],

                   x='GICS Sector',y='PE',hue='Year',split=True)

plt.xticks(rotation=90)
_ = securities.loc[df_pe['Ticker Symbol'],'GICS Sub Industry']

_.index = df_pe.index

df_pe_subindustry = pd.concat([df_pe,_],axis=1)
pd.pivot_table(data=df_pe_subindustry[['GICS Sub Industry','PE','Year']],

               index='GICS Sub Industry',columns='Year',values='PE').sort_values(2015,ascending=False)
#df_eps[['Stock Symbol','Year','Retained Earnings']]

df_re_ni = fundamentals[['Ticker Symbol','Period Ending','Net Income','Retained Earnings']]

df_re_ni['Year'] = df_re_ni['Period Ending'].map(set_year)

df_re_ni = df_re_ni[df_re_ni['Year']>2011]

_ = df_re_ni[['Ticker Symbol','Year','Retained Earnings']]

x = pd.pivot_table(data=_,values='Retained Earnings',columns='Year',index='Ticker Symbol')

re0 = x[[x for x in range(2012,2016)]]/1000000

re1 = x[[x for x in range(2013,2017)]]/1000000

re0.columns = re1.columns

delta_re = re0 - re1

_ = df_re_ni[['Ticker Symbol','Year','Net Income']]

ni = pd.pivot_table(data=_,values='Net Income',columns='Year',index='Ticker Symbol')/1000000

df_b = 1+delta_re.divide(ni[delta_re.columns])# - re

df_b.columns = ['b_'+str(x) for x in df_b.columns]

pt_pe = pd.pivot_table(data=df_pe,columns='Year',values='PE',index='Ticker Symbol')

pt_pe.columns = ['pe_'+str(x) for x in pt_pe.columns]

_ = pd.concat([pt_pe,df_b],axis=1)[['pe_2014','b_2014']]

_ = pd.concat([_,securities.loc[_.index][['GICS Sector']]],axis=1)

_ = _[(_['pe_2014'] < 60) & (_['b_2014'] < 1) & (_['b_2014'] > 0)]



sns.lmplot(data=_,x='b_2014',y='pe_2014',

           hue='GICS Sector',col='GICS Sector',col_wrap=3, ci=None, truncate=True,

           sharex = False)

#sns.lmplot(data=df_b,x='',y='')