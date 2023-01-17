import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
df=pd.read_csv('../input/world-happiness/2019.csv')

df.head()

original=df.copy()

def highlight_max(s):    

    is_max = s == s.max()

    return ['background-color: limegreen' if v else '' for v in is_max]

 

df.style.apply(highlight_max, subset=['Score','GDP per capita','Social support','Healthy life expectancy','Freedom to make life choices','Generosity','Perceptions of corruption'])
df.shape
corrmat = data.corr()

f, ax = plt.subplots()

sns.heatmap(corrmat, square=True)
sns.pairplot(df)
fig, axes = plt.subplots(nrows=2, ncols=2,constrained_layout=True,figsize=(12,8))



sns.barplot(x='GDP per capita',y='Country or region',data=df.nlargest(10,'GDP per capita'),ax=axes[0,0],palette="Blues_d")



sns.barplot(x='Social support' ,y='Country or region',data=df.nlargest(10,'Social support'),ax=axes[0,1],palette="YlGn")



sns.barplot(x='Healthy life expectancy' ,y='Country or region',data=df.nlargest(10,'Healthy life expectancy'),ax=axes[1,0],palette='OrRd')



sns.barplot(x='Freedom to make life choices' ,y='Country or region',data=df.nlargest(10,'Freedom to make life choices'),ax=axes[1,1],palette='YlOrBr')
fig, axes = plt.subplots(nrows=1, ncols=2,constrained_layout=True,figsize=(10,4))



sns.barplot(x='Generosity' ,y='Country or region',data=df.nlargest(10,'Generosity'),ax=axes[0],palette='Spectral')

sns.barplot(x='Perceptions of corruption' ,y='Country or region',data=df.nlargest(10,'Perceptions of corruption'),ax=axes[1],palette='RdYlGn')
print('max:',df['Score'].max())

print('min:',df['Score'].min())

add=df['Score'].max()-df['Score'].min()

grp=round(add/3,3)

print('range difference:',(grp))
low=df['Score'].min()+grp

mid=low+grp



print('upper bound of Low grp',low)

print('upper bound of Mid grp',mid)

print('upper bound of High grp','max:',df['Score'].max())
df.info()
cat=[]

for i in df.Score:

    if(i>0 and i<low):

        cat.append('Low')

        

        

    elif(i>low and i<mid):

         cat.append('Mid')

    else:

         cat.append('High')



df['Category']=cat  
color = (df.Category == 'High' ).map({True: 'background-color: blue',False:'background-color: violet'})

df.style.apply(lambda s: color)
df.loc[df['Country or region']=='United States']
df.loc[df['Country or region']=='India']
df.loc[df['Country or region']=='United Kingdom']
df.loc[df['Country or region']=='Canada']
d= df[(df['Country or region'].isin(['India','Canada','United Kingdom', 'United States']))]

d
ax = d.plot(y="Social support", x="Country or region", kind="bar",color='C1')

d.plot(y="GDP per capita", x="Country or region", kind="bar", ax=ax, color="C2")

d.plot(y="Healthy life expectancy", x="Country or region", kind="bar", ax=ax, color="C5")



plt.show()
ax = d.plot(y="Freedom to make life choices", x="Country or region", kind="bar",color='C4')

d.plot(y="Generosity", x="Country or region", kind="bar", ax=ax, color="C7",)

d.plot(y="Perceptions of corruption", x="Country or region", kind="bar", ax=ax, color="C8",)



plt.show()
import plotly.graph_objs as go

from plotly.offline import iplot



data = dict(type = 'choropleth', 

           locations = df['Country or region'],

           locationmode = 'country names',

           colorscale='RdYlGn',

           z = df['Score'], 

           text = df['Country or region'],

           colorbar = {'title':'Happiness Score'})



layout = dict(title = 'Geographical Visualization of Happiness Score', 

              geo = dict(showframe = True, projection = {'type': 'azimuthal equal area'}))



choromap3 = go.Figure(data = [data], layout=layout)

iplot(choromap3)
df15=pd.read_csv('../input/world-happiness/2015.csv')

df16=pd.read_csv('../input/world-happiness/2016.csv')

df17=pd.read_csv('../input/world-happiness/2017.csv')

df18=pd.read_csv('../input/world-happiness/2018.csv')
plt.figure(figsize=(10,5))

sns.kdeplot(df15['Health (Life Expectancy)'],color='blue')

sns.kdeplot(df16['Health (Life Expectancy)'],color='brown')

sns.kdeplot(df17['Health..Life.Expectancy.'],color='green')

sns.kdeplot(df18['Healthy life expectancy'],color='violet')

sns.kdeplot(df['Healthy life expectancy'],color='red')

plt.title('Health over the Years',size=20)

plt.show()
plt.figure(figsize=(10,5))

sns.kdeplot(df15['Economy (GDP per Capita)'],color='red')

sns.kdeplot(df16['Economy (GDP per Capita)'],color='blue')

sns.kdeplot(df17['Economy..GDP.per.Capita.'],color='limegreen')

sns.kdeplot(df18['GDP per capita'],color='orange')

sns.kdeplot(df['GDP per capita'],color='pink')

plt.title('Economy over the Years',size=20)

plt.show()
plt.figure(figsize=(10,5))

sns.kdeplot(df15['Family'],color='red')

sns.kdeplot(df16['Family'],color='blue')

sns.kdeplot(df17['Family'],color='limegreen')

sns.kdeplot(df18['Social support'],color='orange')

sns.kdeplot(df['Social support'],color='pink')

plt.title('Family over the Years',size=20)

plt.show()
df18['Year']='2018'

df['Year']='2019'

df15['Year']='2015'

df16['Year']='2016'

df17['Year']='2017'
df.rename(columns={'Country or region':'Country'},inplace=True)

data1=df.filter(['Country','GDP per capita','Year'],axis=1)



df15.rename(columns={'Economy (GDP per Capita)':'GDP per capita'},inplace=True)

data2=df15.filter(['Country','GDP per capita',"Year"],axis=1)



df16.rename(columns={'Economy (GDP per Capita)':'GDP per capita'},inplace=True)

data3=df16.filter(['Country','GDP per capita',"Year"],axis=1)



df17.rename(columns={'Economy..GDP.per.Capita.':'GDP per capita'},inplace=True)

data4=df17.filter(['Country','GDP per capita','Year'],axis=1)



df18.rename(columns={'Country or region':'Country'},inplace=True)

data5=df18.filter(['Country','GDP per capita',"Year"],axis=1)



data2=data2.append([data3,data4,data5,data1])
plt.figure(figsize=(10,8))

df = data2[data2['Country']=='India']

sns.lineplot(x="Year", y="GDP per capita",data=df,label='India')



df = data2[data2['Country']=='United States']

sns.lineplot(x="Year", y="GDP per capita",data=df,label='US')



df = data2[data2['Country']=='Finland']

sns.lineplot(x="Year", y="GDP per capita",data=df,label='Finland')



df = data2[data2['Country']=='United Kingdom']

sns.lineplot(x="Year", y="GDP per capita",data=df,label="UK")



df = data2[data2['Country']=='Canada']

sns.lineplot(x="Year", y="GDP per capita",data=df,label='Canada')



plt.title("GDP per capita 2015-2019")