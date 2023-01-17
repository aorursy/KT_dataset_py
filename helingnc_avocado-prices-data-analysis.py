# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.colors import n_colors

from plotly.subplots import make_subplots

# Minmax scaler

from sklearn.preprocessing import MinMaxScaler



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
avocado=pd.read_csv('../input/avocado-prices/avocado.csv')
avocado.info()
avocado.head(10)
avocado.tail()
avocado['Date']
avocado['region'][:5]
avocado[['Date','AveragePrice']][:5]
avocado[(avocado['type']=='organic') & (avocado['AveragePrice']<1) & (avocado['region']=='West')].sort_values('AveragePrice', axis=0, ascending=False)

avocado.sort_values('Date', axis = 0, ascending = False)

avocado.groupby('type').size()

avocado.corr()
f,ax=plt.subplots(figsize=(10,9))

sns.heatmap(avocado.corr(),annot=True,fmt='.2f',ax=ax,vmin=-1, vmax=1, center= 0, cmap= 'coolwarm',linewidths=3, linecolor='black')

plt.show()
f,ax=plt.subplots(figsize=(10,9))

d = avocado.loc[lambda avocado: avocado['region'] == 'Seattle']

matrix = np.triu(d.corr())

sns.heatmap(d.corr(), annot=True, mask=matrix)

plt.show()
fig = px.scatter(avocado, x='AveragePrice', y='Total Volume',

                 color='type') # Added color to previous basic 

fig.update_layout(title='Average Price Vs Volume with Avocado Type ',xaxis_title="Price",yaxis_title="Volume")

fig.show()
total_confirmed=avocado[['Date','Total Volume']].groupby('Date').sum().reset_index()
fig = go.Figure(data=go.Scatter(x=total_confirmed['Date'],

                                y=total_confirmed['Total Volume'],

                                mode='lines')) 

fig.update_layout(title='Total Volume Changes Over Time',xaxis_title="Date",yaxis_title="Volume")

fig.show()
avocado_albany=avocado[avocado['region']=="Albany"][['year','Total Volume']].groupby('year').sum().reset_index()

avocado_washington=avocado[avocado['region']=="BaltimoreWashington"][['year','Total Volume']].groupby('year').sum().reset_index()

covid_boston=avocado[avocado['region']=="Boston"][['year','Total Volume']].groupby('year').sum().reset_index()
fig = go.Figure()



fig.add_trace(go.Scatter(x=avocado_albany['year'], y=avocado_albany['Total Volume'], name = 'Albany',

                         line=dict(color='royalblue', width=4,dash="dot")))



fig.add_trace(go.Scatter(x=avocado_washington['year'], y=avocado_washington['Total Volume'], name = 'Washington',

                         line=dict(color='green', width=4,dash="dashdot")))



fig.add_trace(go.Scatter(x=covid_boston['year'], y=covid_boston['Total Volume'], name = 'Boston',

                         line=dict(color='brown', width=4,dash="dash")))

fig.update_layout(title='Total Volume over time for different countries',xaxis_title="Date",yaxis_title="Volume")

fig.show()
fig = go.Figure(go.Bar(

    x=avocado['type'],y=avocado['Total Volume'],

))

fig.update_layout(title_text='Total Volume vs Avocado Type',xaxis_title="Type",yaxis_title="Volume")

fig.show()
avocado.plot(x='Total Volume', y='AveragePrice', style='*')

plt.show()

disp = lambda str: print('Output: ' + str)

disp("Hello World!")
name="conventional"

it=iter(name)

print(next(it))
print(*it)
a= avocado.AveragePrice.mean()

print(a)
avocado["Price_analysis"]=["High" if i>=a else "Low" for i in avocado.AveragePrice]

avocado.loc[:10,["Price_analysis","AveragePrice"]] 
data_new=avocado.head()

data_new
melted=pd.melt(frame=data_new,id_vars='Date',value_vars=['Total Volume','4046'])

melted
first5=avocado.head()

last5=avocado.tail()

avocado_conc=pd.concat([first5,last5],axis=0,ignore_index=False)

avocado_conc
Pricefirst5=avocado['AveragePrice'].head()

TotalVolumefirst5=avocado['Total Volume'].head()

Regionfirs5=avocado['region'].head()

avocado_conc1=pd.concat([Pricefirst5,TotalVolumefirst5,Regionfirs5],axis=1)

avocado_conc1
avocado.dtypes
avocado['Total Volume']=avocado['Total Volume'].astype('int64')

                                                       

avocado.dtypes                                                     
avocado.head()
avocado.info()
assert avocado.columns[5]=='Date'
assert avocado.columns[1]=='Date'
print(avocado['type'].value_counts(dropna=False))
avocado.describe()
avocado1=avocado[(avocado['type']=='organic')& (avocado['region']=='Albany')|(avocado['region']=='West')&(avocado['year']==2015)]

avocado1
avocado1.boxplot(column='AveragePrice',by='region',figsize=(10,20))

plt.title('Average Price Box Plot Group by West and Albany')

plt.xlabel('Region')

plt.ylabel('Average Price')

plt.show()
avocado.head()
data1=avocado['Date']= pd.to_datetime(avocado['Date']) 

data1

data2=avocado.head()

data2["Date"] = data1

data2= data2.set_index("Date")

data2 

print(data2.loc["2015-11-29"])

data2.resample("A").mean()

data2.resample("M").mean()

avocado1 = avocado.loc[:,["Small Bags","Large Bags"]]

avocado1.plot()

plt.show()
avocado1.plot(subplots = True)

plt.show()
Name = ['Ronan', 'Brayden', 'Marco', 'Aaron']  

    

 

Age = [15, 15, 16, 15]  





ID = [555,798,156,652]

    

 

list_of_tuples = list(zip(ID,Name, Age))  

    

 

list_of_tuples   

  

  



df = pd.DataFrame(list_of_tuples, columns = ['ID','Name', 'Age'])  

     

  

df  
df["Grades"] = ["100","50","75","82"]

df