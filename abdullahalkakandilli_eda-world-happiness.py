# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns



from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go



# word cloud library

from wordcloud import WordCloud



import matplotlib.pyplot as plt

from collections import Counter

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
happiness_2019 = pd.read_csv("/kaggle/input/world-happiness/2019.csv")

happiness_2018 = pd.read_csv("/kaggle/input/world-happiness/2018.csv")

happiness_2017 = pd.read_csv("/kaggle/input/world-happiness/2017.csv")

happiness_2016 = pd.read_csv("/kaggle/input/world-happiness/2016.csv")

happiness_2015 = pd.read_csv("/kaggle/input/world-happiness/2015.csv")
happiness_2015.head()
happiness_2015.info()
happiness_2015.rename(columns= {'Happiness Score': 'happiness_score'})
list(happiness_2015['Region'].unique())
region_list = list(happiness_2015['Region'].unique())

region_score_ratio = []

economy_ratio = []

family_ratio = []

health_ratio = []

freedom_ratio = []

trust_ratio = []

generosity_ratio = []

dystopia_ratio = []



for i in region_list:

    x = happiness_2015[happiness_2015['Region'] == i]

    region_score_rate = sum(x['Happiness Score'])/len(x)

    economy_rate = sum(x['Economy (GDP per Capita)'])/len(x)

    family_rate = sum(x['Family'])/len(x)

    health_rate = sum(x['Health (Life Expectancy)'])/len(x)

    freedom_rate = sum(x['Freedom'])/len(x)

    trust_rate = sum(x['Trust (Government Corruption)'])/len(x)

    generosity_rate = sum(x['Generosity'])/len(x)

    dystopia_rate = sum(x['Dystopia Residual'])/len(x)

    

    region_score_ratio.append(region_score_rate)

    economy_ratio.append(economy_rate)

    family_ratio.append(family_rate)

    health_ratio.append(health_rate)

    freedom_ratio.append(freedom_rate)

    trust_ratio.append(trust_rate)

    generosity_ratio.append(generosity_rate)

    dystopia_ratio.append(dystopia_rate)

    

    

data1 = pd.DataFrame({'region_list': region_list, 'region_score_ratio': region_score_ratio, 'economy_ratio': economy_ratio, 'family_ratio': family_ratio, 

                      'health_ratio':health_ratio, 'freedom_ratio': freedom_ratio, 'trust_ratio': trust_ratio, 'generosity_ratio': generosity_ratio, 'dystopia_ratio': dystopia_ratio})

new_index = (data1['region_score_ratio'].sort_values(ascending=False)).index.values

sorted_data1 = data1.reindex(new_index)

sorted_data1.head()

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data1['region_list'], y=sorted_data1['region_score_ratio'])

plt.xticks(rotation= 45)

plt.xlabel('Regions')

plt.ylabel('Region Score')

plt.title('Region Score Given Regions')

plt.show()
sorted_data1.head()
# simple normalization for graphs

sorted_data1['region_score_ratio'] = sorted_data1['region_score_ratio']/max( sorted_data1['region_score_ratio'])

sorted_data1['dystopia_ratio'] = sorted_data1['dystopia_ratio']/max( sorted_data1['dystopia_ratio'])

sorted_data1['economy_ratio'] = sorted_data1['economy_ratio']/max( sorted_data1['economy_ratio'])

sorted_data1['health_ratio'] = sorted_data1['health_ratio']/max( sorted_data1['health_ratio'])

sorted_data1['freedom_ratio'] = sorted_data1['freedom_ratio']/max( sorted_data1['freedom_ratio'])

sorted_data1['trust_ratio'] = sorted_data1['trust_ratio']/max( sorted_data1['trust_ratio'])

sorted_data1['generosity_ratio'] = sorted_data1['generosity_ratio']/max( sorted_data1['generosity_ratio'])
sorted_data1.head()
f, ax1 = plt.subplots(figsize=(20,10))





sns.pointplot(x = sorted_data1.region_list	, y = sorted_data1.economy_ratio, color='blue', alpha = 0.8)



sns.pointplot(x = sorted_data1.region_list	, y = sorted_data1.health_ratio, color='lime', alpha = 0.8)





plt.xlabel('Coutry or region',fontsize = 15,color='blue')

plt.ylabel('Healthy life expectancy vs Economy ratio',fontsize = 15,color='blue')

plt.xticks(rotation = 90)

plt.title('Values by Regions',fontsize = 20,color='blue')

plt.grid()
# Show the joint distribution using kernel density estimation 

j = sns.jointplot(sorted_data1.health_ratio, sorted_data1.economy_ratio, kind = 'reg', size = 7)

plt.show()

# Show the results of a linear regression within each dataset

sns.lmplot(x="health_ratio", y="economy_ratio", data=sorted_data1)

plt.show()
# Use cubehelix to get a custom sequential palette

pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)

sns.violinplot(data=sorted_data1, palette=pal, inner="points")

plt.xticks(rotation = 90)

plt.show()
data1.corr()
# correlation map

f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(data1.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

plt.show()
import plotly.graph_objs as go



trace1 = go.Scatter(

                    x = data1.region_list,

                    y = data1.economy_ratio,

                    mode = "lines",

                    name = "economy ratio",

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

                    text = data1.region_list)

trace2 = go.Scatter(

                    x = data1.region_list,

                    y = data1.health_ratio,

                    mode = "lines",

                    name = "health ratio",

                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),

                    text = data1.region_list)



data = [trace1, trace2]

layout = dict(title = 'health and economy',

              xaxis= dict(title= 'Table 1.1',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)