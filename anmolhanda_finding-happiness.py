import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#importing dataset
df = pd.read_csv('../input/2017.csv')
df.head()
df.info()
#data is already cleaned and in proper formats, so we don't need to perform data cleaning and refining, anyway its a small dataset
#our main aim is to get high impact parameters affecting happiness score
#visualizing countries geographically based on their happiness score
import plotly.plotly as py 

import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
data = dict(type='choropleth',

           locations=df['Country'],

           locationmode="country names",

           z=df['Happiness.Score'],

           text=df['Country'],

           colorbar={'title':'Happiness Score out of 10 for 2017'})
layout= dict(title= "Happiness Score for 2017",

            geo=dict(showframe= False,

                    projection = {'type':'Mercator'}))
choromap3=go.Figure(data=[data], layout=layout)
iplot(choromap3)
#from the visualization, we see that happier people are found in North America, South America, North Western Europe, Russia and Australia
#the plot is interactive, feel free to play around with it
#data is skewed because we couldnt statewise happiness score which could have helped us more
#Lets try to develop a correlation between happiness score and other attributes
df.head()
#dependent attributes comprises of - Health Life Expectancy, Freedom
#corr matrix

plt.figure(figsize=(12,8))

sns.heatmap(df.corr(),annot=True)
#We will not consider Happiness rank here as its obviously directly linked with Happiness Score. 
#Important correlated fields with Happiness Score can be find out by looking at Happiness Score row in heatmap
#it shows Whisker(confidence of estomates) decides happiness score mostly
#also, other factors except generosity affects happiness to an high extent
#we will get rid of generosity and happiness rank as they are not useful here now
df_new= df.drop(['Happiness.Rank','Generosity'],axis=1)
df_new.head()
#checking linearity between whisker high, low and Happiness Score as correlation shows they are highly linked
sns.lmplot(x='Whisker.high', y='Happiness.Score', data=df_new)
#repeating it for whisker low score too
sns.lmplot(x='Whisker.low', y='Happiness.Score', data=df_new)
#The plot shows that whisker high and low are directly link with happiness score and hence won't be useful in analysis, we will remove them too.
df_new.drop(['Whisker.low','Whisker.high'],axis=1, inplace=True)
df_new.head()
sns.jointplot(x='Economy..GDP.per.Capita.', y='Happiness.Score', data=df_new , kind='scatter')  #for bi variate data for two variables
#jointplot shows Happiness Score is highly linked with Per Capita GDP
sns.pairplot(df_new)
#pairplot helped me understand that Countries with higher freedom still have low Happiness score for many countries
#Inversely , countries with low trust and more Govt Corruption have High Happiness Score.
#This shows countries doesn't count govt corruption and freedom as their main source of happiness, we need to think more than that
#From this viz, i noticed even GDP is highly linked with better family rating and better life expectancy and health which is true.
sns.jointplot(x='Happiness.Score', y='Dystopia.Residual', data=df_new , kind='reg') 
#its seen here that with increasing happiness score, there is slight increase in Dystopia.Residual where people are slightly dishappy but not able to figure out the reasons
#this shows with increasing happiness score, countries couldn't explain the reason of being unhappy as they have all the other factors fulfilled
#take top 5 countris
df_top=df_new.head()

df_top=df_top.drop(['Country'], axis=1)

df_top
#taking bottom 5 countries

df_bottom=df_new.tail()

df_bottom=df_bottom.drop(['Country'], axis=1)

df_bottom.reset_index(inplace=True)

df_bottom
#calculating difference between there values

df_top.subtract(df_bottom, axis='index')
#it Shows dystopia residual is varying between top and lowest "happy" countries
#Economy- GDP, Family and Health Life Expectancy are major sources of happiness for any country
#Pay people better, they will take better care of their health and family and hence be more happy. That's the conclusion :D