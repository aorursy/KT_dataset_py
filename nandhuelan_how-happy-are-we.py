# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.style.use('fivethirtyeight')
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
Hp15=pd.read_csv('../input/2015.csv')
Hp16=pd.read_csv('../input/2016.csv')
Hp17=pd.read_csv('../input/2017.csv')
Hp15['Year']='2015'
Hp16['Year']='2016'
Hp17['Year']='2017'
Hp15.columns
Hp16.columns
Hp17.columns
Hp17.columns=['Country','Happiness Rank','Happiness Score','Whisker high','Whisker low','Economy (GDP per Capita)','Family','Health (Life Expectancy)','Freedom','Generosity','Trust (Government Corruption)','Dystopia Residual','Year']
twentyfiften = dict(zip(list(Hp15['Country']), list(Hp15['Region'])))
twentysixten = dict(zip(list(Hp16['Country']), list(Hp16['Region'])))
regions = dict(twentyfiften, **twentysixten)

def find_region(row):
    return regions.get(row['Country'])


Hp17['Region'] = Hp17.apply(lambda row: find_region(row), axis=1)
Hp17[Hp17['Region'].isnull()]['Country']
Hp17 = Hp17.fillna(value = {'Region': regions['China']})
hreport=pd.concat([Hp15,Hp16,Hp17])
hreport.head()
hreport.fillna(0,inplace=True)
avghappscore=hreport.groupby(['Year','Country'])['Happiness Score'].mean().reset_index().sort_values(by='Happiness Score',ascending=False)
avghappscore=avghappscore.pivot('Country','Year','Happiness Score').fillna(0)
hscore=avghappscore.sort_values(by='2017',ascending=False)[:11]
hscore.plot.barh(width=0.8,figsize=(10,10))
groupcountryandyear=hreport.groupby(['Year','Country']).sum()
a=groupcountryandyear['Happiness Rank'].groupby(level=0, group_keys=False)
top10=a.nsmallest(10).reset_index()
yearwisetop10=pd.pivot_table(index='Country',columns='Year',data=top10,values='Happiness Rank')
ax=plt.figure(figsize=(10,10))
fig=ax.add_axes([1,1,1,1])
fig.set_xlabel('Country')
fig.set_ylabel('Rank')
yearwisetop10.plot.bar(ax=fig,cmap='YlOrRd')
sns.heatmap(hreport.drop(['Whisker high','Whisker low','Upper Confidence Interval','Lower Confidence Interval'],axis=1).corr(),annot=True,cmap='RdYlGn')
twentyseventen=hreport[hreport['Year']=='2017']
data = dict(type = 'choropleth', 
           locations = twentyseventen['Country'],
           locationmode = 'country names',
           z = twentyseventen['Happiness Rank'], 
           text = twentyseventen['Country'],
           colorbar = {'title':'Happiness'})
layout = dict(title = 'Global Happiness', 
             geo = dict(showframe = False, 
                       projection = {'type': 'Mercator'}))
choromap3 = go.Figure(data = [data], layout=layout)
iplot(choromap3)
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score,confusion_matrix
hreport.head()
x=hreport[['Economy (GDP per Capita)','Trust (Government Corruption)','Freedom','Health (Life Expectancy)','Family','Dystopia Residual']]
y=hreport['Happiness Rank']
lm=LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=101)
lm.fit(X_train,y_train)
ypred=lm.predict(X_test)
plt.scatter(y_test,ypred)
coef = zip(x.columns, lm.coef_)
coef_df = pd.DataFrame(list(zip(x.columns, lm.coef_)), columns=['features', 'coefficients'])
coef_df
r2_score(y_test,ypred)
