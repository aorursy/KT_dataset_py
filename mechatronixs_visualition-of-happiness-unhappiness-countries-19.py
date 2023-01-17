# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #plot

import seaborn as sns #visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/world-happiness/2019.csv')

data.info()
data['Generosity'].value_counts
data.head()
data.corr()
f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(data.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

plt.show()
fig = plt.figure(figsize=(20,20))

sns.set(style="white",font_scale=1);

sns.pairplot(data[['Score','GDP per capita', 'Social support', 'Healthy life expectancy', \

    'Freedom to make life choices', 'Generosity' , 'Perceptions of corruption']]);
data.describe()
# we create 2 data frames.

data1 = data.head()

data2= data.tail()

data_new= pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row

data_new
best_worst_five_countries = data_new['Country or region'] # The 5 Happiest Countries and Unhappiest Countries as this report

heal_life_expect = data_new['Healthy life expectancy'] # Healthy Life Expectancy

life_choices = data_new['Freedom to make life choices'] # Freedom to make life choices

generosity = data_new['Generosity'] # Generosity
f,ax1 = plt.subplots(figsize =(20,10))

sns.pointplot(x=best_worst_five_countries,y=heal_life_expect,color = 'blue')

sns.pointplot(x=best_worst_five_countries,y=life_choices,color = 'red')

sns.pointplot(x=best_worst_five_countries,y=generosity,color = 'green')

plt.text(7,0.6,'*healthy life expect',color='blue',fontsize = 17,style = 'italic')

plt.text(7,0.55,'*freedom to make life choices',color='red',fontsize = 18,style = 'italic')

plt.text(7,0.5,'*generosity',color='green',fontsize = 18,style = 'italic')

plt.title('Healthy Life , Freedom to Make Life Choices & Generosity' ,fontsize = 20)

plt.xlabel("Happies and Unhappiess Countries",fontsize = 16 , color = 'blue')

plt.ylabel("Values",fontsize = 16 , color = 'red')
import plotly.graph_objects as go

import plotly.express as px





fig = go.Figure(data=[go.Scatter(

    x = data_new['Country or region'],

    y = data_new['GDP per capita'],

    mode ='markers',

    marker=dict(

        color=[135, 145, 160, 145, 150, 70,80,70,40,60],

        size=[60, 80, 100, 80, 90,20,30,20,5,15 ],

        showscale=True

)

    )])



fig.update_layout(

    title='Country v. Per Capita GDP,Economy Rate 2019',

     xaxis=dict(

        title='Happiness-Unhappiness Countries 2019',

        gridcolor='white'),

    yaxis=dict(

        title='Economy Rate',

        gridcolor='blue')    

)

    



fig.show()







# Top 10 Generosity Countries

x = data.sort_values(by='Generosity', ascending=False).head(15)

data2= x.drop("Score", axis=1)

data2
plt.figure(figsize=(20,10))

sns.barplot(data2['Country or region'], data2['Generosity'])
#Scatter Plot

# a = Health Life Expectancy , b = Generosity

data.plot(kind = 'scatter', x='Healthy life expectancy' , y='Generosity', alpha=0.5 , color= 'red')