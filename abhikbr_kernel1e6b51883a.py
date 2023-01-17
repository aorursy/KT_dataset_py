# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



#Plotly

import plotly.offline as py

from plotly.offline import init_notebook_mode,iplot

import plotly.graph_objects as go

py.init_notebook_mode(connected=True)



#Cufflinks

import cufflinks as cf

from cufflinks.offline import go_offline
amazon=pd.read_csv('../input/brazilian-amazon-rainforest-degradation/inpe_brazilian_amazon_fires_1999_2019.csv')

amazon.head()
amazon.info()
trace1=go.Bar(x=amazon['state'],y=amazon['firespots'])

layout=go.Layout(title="State having Firespots",legend=dict(x=0.1,y=1.1),template="plotly_dark")

data1=[trace1]

fig=go.Figure(data1,layout)

fig.show()
year = amazon[['year','firespots']].groupby('year').sum().reset_index()

year.head()
trace1=go.Scatter(x=year['year'],y=year['firespots'])

data1=[trace1]

layout=go.Layout(title="Firespots according to the years",legend=dict(x=0.1,y=1.1,orientation='h',) )

fig=go.Figure(data1,layout=layout)

fig.show()
import calendar
month=amazon[['month','firespots']].groupby('month').sum().reset_index()

month['month'] = month['month'].apply(lambda x: calendar.month_abbr[x])

month.tail()
trace1=go.Bar(x=month['month'],y=month['firespots'])

layout=go.Layout(title="Firespots according to the month",legend=dict(x=0.1,y=1.1))

data1=[trace1]

fig=go.Figure(data1,layout)

fig.show()
plt.figure(figsize=(18,6))

sns.heatmap(amazon.corr(),annot=True)
legal_amazon =amazon[['state','firespots']].groupby('state',as_index=False).sum().sort_values('firespots',ascending=False)['state'].values





fig, ax = plt.subplots(3, 3, figsize=(18, 14), sharex=True)

sns.set_style("whitegrid")

ax = ax.flat



i=0

for x in legal_amazon:

     sns.lineplot(data=amazon[amazon['state']==x], x='year',

                 y='firespots', estimator='sum', ax=ax[i], color='firebrick', ci=None)

     ax[i].set_xticks([2000, 2005, 2010, 2015, 2020])

     ax[i].set_title(x, size='large')

     if i==0 or i==3 or i==6:

        ax[i].set_ylabel("Firespots",size='large')

     else:

        ax[i].set_ylabel("")    

     i += 1
legal_amazon
fig, ax = plt.subplots(3, 3, figsize=(18, 14), sharex=True)

sns.set_style("dark")

ax = ax.flat



i=0

for x in legal_amazon:

     sns.barplot(data=amazon[amazon['state']==x], x='month',

                 y='firespots', ax=ax[i], color='firebrick', ci=None)

     ax[i].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])

     ax[i].set_title(x, size='large')

     if i==0 or i==3 or i==6:

        ax[i].set_ylabel("Firespots",size='large')

     else:

        ax[i].set_ylabel("")

     i += 1
deforest=pd.read_csv('../input/brazilian-amazon-rainforest-degradation/def_area_2004_2019.csv')

deforest.head()
col=['years','ACRE','AMAZONAS','AMAPA','MARANHAO','MATO GROSSO','PARA','RONDONIA','RORAIMA','TOCANTINS','Deforest_Area']

deforest.columns=col

deforest.head()

#'PARA', 'MATO GROSSO', 'RONDONIA', 'AMAZONAS', 'MARANHAO', 'ACRE',

 #      'RORAIMA', 'AMAPA', 'TOCANTINS'
deforest['ACRE']
trace3=go.Scatter(x=deforest['years'],y=deforest['Deforest_Area'])

layout=go.Layout(title="Deforested area of all states over years",template="plotly_dark")

deforest_p=[trace3]

fig=go.Figure(deforest_p,layout=layout)

fig.show()
fig,ax=plt.subplots(3,3,figsize=(18,14),sharex=True)

sns.set_style("dark")

ax=ax.flat



i=0

m=1

for x in legal_amazon:

     dataf=deforest[['years',deforest.columns[m],'Deforest_Area']]

     sns.lineplot(data=dataf, x='years',

                 y=deforest[x], ax=ax[i], color='firebrick')

     ax[i].set_xticks([2004,2006, 2008,2010, 2012,2014, 2016,2018, 2020])

     ax[i].set_title(deforest.columns[m], size='large')

     if i==0 or i==3 or i==6:

        ax[i].set_ylabel("Deforested Area",size='large')

     else:

        ax[i].set_ylabel("")

     m +=1

     i += 1
amazon_def_area = deforest.melt(id_vars=['years'], var_name='state', value_name='defarea')

amazon_def_area.head(20)

plt.figure(figsize=(12,5), dpi=85)

sns.set_style("whitegrid")

sns.barplot(data=amazon_def_area, x='state', y='defarea', estimator=sum, color='red',order=legal_amazon,  ci=None)    

plt.ylabel("State")

plt.title("Total Deforested Area by State (km²)")

plt.xlabel("Total deforested area (km²)")
el=pd.read_csv('../input/brazilian-amazon-rainforest-degradation/el_nino_la_nina_1999_2019.csv')

el.head()