import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
df=pd.read_csv('../input/edu.csv')
corr = df.corr()
fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(corr, annot=True,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
import seaborn as sns
df['date1']=pd.to_numeric(df.date)
df.drop(['date','country','c_codes'],axis=1).dtypes
sns.pairplot(df.drop(['date','country','c_codes'],axis=1).fillna(0))
df.loc[df.Edu_prime>100,'Edu_prime']=100
sns.pairplot(df.drop(['date','country','c_codes'],axis=1).fillna(0))
df.isna().sum()/df.shape[0]
df1=df.drop(['Edu_lower_second','Edu_post_second','Edu_second','Unemp'],axis=1)
df1=df1.dropna()
mem=0
mem_date=0
for date in range (1960,2018):
    length=len(df1[df1.date==date].country.unique())
    if length>mem:
        mem=length
        mem_date=date
print('best date ',mem_date)
df1_2011=df1[df1.date==2011]
data=dict(type='choropleth',
         locations=df1_2011['c_codes'],
         z=df1_2011['Edu_prime'],
         text=df1_2011['country'],
         colorbar={'title':'Edu_prime'})
layout=dict(title='Edu_prime',
           geo=dict(showframe=False,
                   projection={'type':'mercator'}))
choromap=go.Figure(data=[data],layout=layout)
iplot(choromap)
df2=df.drop(['Edu_lower_second','Edu_post_second','Edu_prime','Unemp'],axis=1)
df2=df2.dropna()
mem=0
mem_date=0
for date in range (1960,2018):
    length=len(df1[df1.date==date].country.unique())
    if length>mem:
        mem=length
        mem_date=date
print('best date ',mem_date)
df2_2012=df2[df2.date==2012]
data=dict(type='choropleth',
         locations=df2_2012['c_codes'],
         z=df2_2012['Edu_second'],
         text=df2_2012['country'],
         colorbar={'title':'Edu_second'})
layout=dict(title='Edu_second',
           geo=dict(showframe=False,
                   projection={'type':'mercator'}))
choromap=go.Figure(data=[data],layout=layout)
iplot(choromap)
df3=df.drop(['Edu_lower_second','Edu_post_second','Edu_prime','Edu_second'],axis=1)
df3=df3.dropna()
mem=0
mem_date=0
for date in range (1960,2018):
    length=len(df1[df1.date==date].country.unique())
    if length>mem:
        mem=length
        mem_date=date
print('best date ',mem_date)
df3_2012=df3[df3.date==2012]
data=dict(type='choropleth',
         locations=df3_2012['c_codes'],
         z=df3_2012['Unemp'],
         text=df3_2012['country'],
         colorbar={'title':'Unemp'})
layout=dict(title='Unemployment',
           geo=dict(showframe=False,
                   projection={'type':'mercator'}))
choromap=go.Figure(data=[data],layout=layout)
iplot(choromap)