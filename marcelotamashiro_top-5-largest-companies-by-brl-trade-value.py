import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import dates

import os
data = pd.read_csv('../input/COTAHIST_A2009_to_A2018P.csv',parse_dates=['DATPRE'])
data = data[['TIPREG', 'DATPRE', 'CODBDI', 'CODNEG', 'TPMERC',
       'NOMRES', 'ESPECI','TOTNEG', 'QUATOT', 'VOLTOT']]
data['year'] = pd.to_datetime(data.DATPRE).dt.year
# Average daily trading 
years = np.sort(data['year'].unique())
graph = pd.DataFrame([])
for y in years:
    tot_vol = (data[(data.year==y)].groupby(by='DATPRE')['VOLTOT'].sum()/(10**11)).mean()
    graph = graph.append({'Total_Volume': tot_vol,'Year': y},ignore_index=True)
graph['Year'] = graph['Year'].astype('uint16')
pos = np.arange(len(graph))
fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(pos, graph['Total_Volume'].values,color='grey')
ax.set_yticks(pos)
ax.set_yticklabels(graph['Year'].values,fontsize=15)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('BRL Billion',fontsize=16)
ax.set_title('Average daily trading value in Bovespa per year',fontsize=20)
sns.despine(bottom=True, left=True)

plt.show()
del graph
def make_dataframe(df,company_name):
    df['TOTAL_VOL'] = (data[['CODNEG','NOMRES','TOTNEG','VOLTOT','DATPRE','QUATOT']][(data.NOMRES==company_name)].groupby(by='DATPRE')['VOLTOT'].sum())/(100)
    df['NUM_NEG'] = (data[['CODNEG','NOMRES','TOTNEG','VOLTOT','DATPRE','QUATOT']][(data.NOMRES==company_name)].groupby(by='DATPRE')['TOTNEG'].sum())
    df['NUM_TIT'] = (data[['CODNEG','NOMRES','TOTNEG','VOLTOT','DATPRE','QUATOT']][(data.NOMRES==company_name)].groupby(by='DATPRE')['QUATOT'].sum())
    df['ticket'] = df['TOTAL_VOL']/df['NUM_NEG']
    df['price'] = df['TOTAL_VOL']/df['NUM_TIT']
    return df
#Top 10 Companies by trading value between 2009 and 2018
top10 = pd.DataFrame([]) ; company_list = [] 
top10['TOTAL_VOL'] = (data[['CODNEG','NOMRES','TOTNEG','VOLTOT','DATPRE']].groupby(by='NOMRES')['VOLTOT'].sum())
company_list = top10.nlargest(12, 'TOTAL_VOL').index.values
company_list= np.delete(company_list, [8,9])  #ETF's not companies

graph = pd.DataFrame([])
graph['DATPRE'] = data['DATPRE'].unique()
graph.set_index('DATPRE',inplace=True)
for company in company_list:
   df = pd.DataFrame([])
   grp = make_dataframe(df,company)[['TOTAL_VOL']].rename(index=str, columns={'TOTAL_VOL': company})
   graph = pd.merge(graph,grp,right_index=True,left_index=True,how='left')
   del df ; del grp

graph.fillna(0, inplace=True)
graph.sort_index(inplace=True)
fig, ax = plt.subplots(5,1,figsize=(40,48))
ax[0].set_title("Top 5 companies by trade value in Bovespa", fontsize=30)
colors = [(0, 158/255, 111/255),'blue',(246/255,94/255,1/255),'red',(249/255,221/255,22/255)]
plt.title('Top 5 companies with largest average daily trade')
for i in range(5):
    ax[i].bar(graph.index, graph[company_list[i]], width=1.2, color=colors[i])
    ax[i].set_ylabel(r"Billion BRL")
    months = dates.MonthLocator(range(1, 13), bymonthday=1, interval=6)
    monthsFmt = dates.DateFormatter("%b '%y")
    ax[i].xaxis.set_major_locator(months)
    ax[i].xaxis.set_major_formatter(monthsFmt)
    ax[i].tick_params(axis='both', which='major', labelsize=16)
    ax[i].tick_params(axis='x' ,rotation=10)
    ax[i].legend([company_list[i]],loc=2, fontsize=25)
    ax[i].set_ylim((0,4*10**9))
    ax[i].set_xlim(pd.Timestamp('2009-02-15'), pd.Timestamp('2018-02-15'))
    ax[i].axhline(graph[company_list[i]].mean(), color='k', linestyle='dashed', linewidth=1)
plt.subplots_adjust(bottom=0.5)
del graph
plt.show()
def make_dataframe1(df,stock_code):
    df['TOTAL_VOL'] = (data[['CODNEG','NOMRES','TOTNEG','VOLTOT','DATPRE','QUATOT']][(data.CODNEG.isin(stock_code))].groupby(by='DATPRE')['VOLTOT'].sum())/(100)
    df['NUM_NEG'] = (data[['CODNEG','NOMRES','TOTNEG','VOLTOT','DATPRE','QUATOT']][(data.CODNEG.isin(stock_code))].groupby(by='DATPRE')['TOTNEG'].sum())
    df['NUM_TIT'] = (data[['CODNEG','NOMRES','TOTNEG','VOLTOT','DATPRE','QUATOT']][(data.CODNEG.isin(stock_code))].groupby(by='DATPRE')['QUATOT'].sum())
    df['ticket'] = df['TOTAL_VOL']/df['NUM_NEG']
    df['price'] = df['TOTAL_VOL']/df['NUM_TIT']
    return df
graph = pd.DataFrame([])
graph['DATPRE'] = data['DATPRE'].unique()
graph.set_index('DATPRE',inplace=True)
stock_list=[['VALE3','VALE3F'],['PETR4','PETR4F'],['ITUB4','ITUB4F'],['BBDC4','BBDC4F'],['BBAS3','BBAS3F']]
for stock in stock_list:
   df = pd.DataFrame([])
   grp = make_dataframe1(df,stock)[['TOTAL_VOL']].rename(index=str, columns={'TOTAL_VOL': stock[0]})
   graph = pd.merge(graph,grp,right_index=True,left_index=True,how='left')
   del df ; del grp

price = pd.DataFrame([])
price['DATPRE'] = data['DATPRE'].unique()
price.set_index('DATPRE',inplace=True)
for stock in stock_list:
   df = pd.DataFrame([])
   grp = make_dataframe1(df,stock)[['price']].rename(index=str, columns={'price': stock[0]})
   price = pd.merge(price,grp,right_index=True,left_index=True,how='left')
   del df ; del grp

graph.fillna(0, inplace=True)
graph.sort_index(inplace=True)
fig, ax = plt.subplots(5,1,figsize=(40,48))

colors = [(0, 158/255, 111/255),'blue',(246/255,94/255,1/255),'red',(249/255,221/255,22/255)]
plt.title('Top 5 companies with largest average daily trade')
for i in range(5):
    ax[i].bar(graph.index, graph[stock_list[i][0]], width=1.2, color=colors[i])
    ax[i].set_ylabel(r"Billion BRL", fontsize=16)
    ax2 = ax[i].twinx()
    ax2.scatter(price.index, price[stock_list[i][0]],color='black', s=1.2)
    ax2.set_ylabel(r"Stock Price (BRL)", fontsize=16)
    months = dates.MonthLocator(range(1, 13), bymonthday=1, interval=6)
    monthsFmt = dates.DateFormatter("%b '%y")
    ax[i].xaxis.set_major_locator(months)
    ax[i].xaxis.set_major_formatter(monthsFmt)
    ax[i].tick_params(axis='both', which='major', labelsize=16)
    ax[i].tick_params(axis='x' ,rotation=10)
    ax[i].legend([stock_list[i][0]],loc=2, fontsize=25)
    #ax[i].set_ylim((0,4*10**9))
    ax[i].set_xlim(pd.Timestamp('2009-02-15'), pd.Timestamp('2018-02-15'))
    ax[i].axhline(graph[stock_list[i][0]].mean(), color='k', linestyle='dashed', linewidth=1)
plt.subplots_adjust(bottom=0.5)
del graph
plt.show()