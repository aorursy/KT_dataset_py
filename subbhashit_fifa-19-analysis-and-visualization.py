import numpy as np

import pandas as pd

data=pd.read_csv("/kaggle/input/fifa19/data.csv")

data.head()
data.describe()
import wordcloud as wc

text=np.array(data['Nationality'])

cloud=wc.WordCloud()

cloud.generate(" ".join(text))

cloud.to_image()
import plotly.graph_objects as go

from plotly.offline import init_notebook_mode,iplot

import plotly.express as px

countries=data.Nationality.value_counts()

f= go.Figure(data=go.Choropleth(

    locations=countries.index,

    z =countries, 

    locationmode = 'country names', 

    colorscale =px.colors.sequential.Plasma,

    colorbar_title = "NO. of players",

))



f.update_layout(

    title_text = 'Number of players from each country',

)

iplot(f)
import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(10,10))

sns.scatterplot(data.Composure,data.Potential,color='r')

sns.scatterplot(data.Composure,data.Overall,color='g')
fig=plt.figure(figsize=(15,10))

ax=fig.add_subplot(121)

ax.plot(data.StandingTackle,'.',color='r')

bx=fig.add_subplot(122)

bx.plot(data.SlidingTackle,'.',color='g')
fig=plt.figure(figsize=(15,10))

ax=fig.add_subplot(321)

ax.plot(data.GKDiving,'.',color='y')

bx=fig.add_subplot(322)

bx.plot(data.GKHandling,'.',color='c')

cx=fig.add_subplot(323)

cx.plot(data.GKKicking,'.',color='r')

dx=fig.add_subplot(324)

dx.plot(data.GKPositioning,'.',color='b')
Clubs=data.Club.value_counts()

plt.plot(np.unique(Clubs),'.',color='r')
def Money(x):

    if type(x)==float:

        pass

    else:

        m=x[1:]

        x=m[:-1]

        return round(float(x))

data['Release Clause']=data['Release Clause'].apply(Money)
data.head()
plt.figure(figsize=(10,10))

plt.plot(data['Release Clause'],'.',color='c')
sns.scatterplot(x='International Reputation',y='Release Clause',data=data,color='y')
plt.figure(figsize=(15,15))

sns.heatmap(data.corr(),annot=True,cmap='inferno')
sns.scatterplot(data.Age,data['Release Clause'])
plt.figure(figsize=(15,10))

sns.distplot(data['Release Clause'],color='g')
def age_d(x):

    if x>30:

        return 'Above 30'

    if x<=30 and x>25:

        return 'Between 25-30'

    if x<=25 and x>20:

        return 'Between 20-25'

    else:

        return 'Below 20'

        

data['Age_dist']=data.Age.apply(age_d)
plt.figure(figsize=(15,10))

sns.countplot(data['Age_dist'])
def overall_d(x):

    if x>90:

        return 'Above 90'

    if x<=90 and x>80:

        return 'Between 90-80'

    if x<=80 and x>70:

        return 'Between 80-70'

    if x<=70 and x>60:

        return 'Between 70-60'

    if x<=60 and x>50:

        return 'Between 60-50'

    else:

        return 'Below 50'

    

data.Overall=data.Overall.apply(overall_d)
plt.figure(figsize=(15,10))

sns.countplot(data['Overall'])
crosstab=pd.crosstab(data['Age_dist'],data['Overall'])

crosstab.plot.bar(stacked=True,figsize=(15,10))
plt.figure(figsize=(15,10))

sns.lineplot(x='Jersey Number',y='Skill Moves',data=data,hue='Overall')
plt.figure(figsize=(15,10))

sns.lineplot(x='Jersey Number',y='Potential',data=data,hue='Overall')