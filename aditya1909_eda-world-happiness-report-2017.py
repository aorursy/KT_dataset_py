import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set(style="white", color_codes=True)

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.plotly as py

import plotly.graph_objs as go 

print(__version__)

import cufflinks as cf

init_notebook_mode(connected=True)

# For offline use

cf.go_offline()
#Importing file of Year 2017

h2017 =  pd.read_csv('2017.csv')

h2017.head()

#Checking Data Type

h2017.info()

#No Data Cleaning was done as it was already done. 
#Doing Geographical Plotting



data = dict(type = 'choropleth',

             locations = h2017['Country'],

            locationmode = "country names",

            colorscale= 'YIOrRd',

            text= h2017['Country'],

            z=h2017['Happiness.Score'],

            colorbar = {'title':'Happiness Score'})



layout = dict(title = 'Happiness Score',

                geo = dict(showframe = False,projection = {'type':'Mercator'}))
choromap = go.Figure(data = [data],layout = layout)

iplot(choromap,validate=False)
# It can be inferred from the Plot above that people in North America are the happiest  

# People in Africa are not happy enough , this is evident for developing nations

# All the Developed Nations seems to be Happiest, which makes sense.
# Checking the correlation of the Happiness Score with the over variables such as Family, Freedom, Generosity,etc.
h2017.info()
correlation = h2017.corr()

plt.figure(figsize=(10,6))

sns.heatmap(correlation,annot=True)
#This shows that Happiness Score is dependent on Economy and Per Capita Income , Family, Life Expectency, Whisker High, and Whisker Low 
sns.lmplot(data=h2017,x='Happiness.Score',y='Whisker.high')
sns.lmplot(data=h2017,x='Happiness.Score',y='Whisker.low')
#Whisker.high and Whisker.low denotes the confidence and support level of the ratings done.  

#Whisker.high shows the highest confidence level whereas Whisker.Low shows the lowest Whisker Value

#It is evident that Happiness.Score is linked to Whisker Value. Higher the Whisker Value, higher the Happiness Score
#Removing the columns Happiness Rank, Whisker.High and Whisker Low
h2017_new = h2017.drop(columns=['Happiness.Rank','Whisker.high', 'Whisker.low'])
h2017_new.info()
#As inferred , Economy and Per Capita Income , Family, Life Expectency,are important. Hence, I shall be plotting them against Happiness Score
#Checking Happiness Score vs Economy

sns.jointplot(x='Happiness.Score', y='Economy..GDP.per.Capita.', data=h2017_new,dropna=True,color='r', kind='scatter') 
sns.lmplot(x='Happiness.Score', y='Economy..GDP.per.Capita.', data=h2017_new)
#Checking Happiness Score vs Family

sns.jointplot(x='Happiness.Score', y='Family', data=h2017_new,dropna=True,color='y', kind='kde') 
#Checking Happiness Score vs Health Life Expectancy

sns.jointplot(x='Happiness.Score', y='Health..Life.Expectancy.', data=h2017_new,dropna=True,color='g', kind='hex') 
sns.pairplot(h2017_new)
#The correlation matrix shows that GDP is dependent on Life Expectancy

sns.lmplot(x='Economy..GDP.per.Capita.',y='Health..Life.Expectancy.', data=h2017_new, hue='round_family')
h2017_new['round_family'] = round(h2017_new['Family'])
round(h2017_new['Family'])