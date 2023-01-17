# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#matplotlib

import matplotlib.pyplot as plt

%matplotlib inline



#seaborn

import seaborn as sns



# plotly

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go



import warnings

warnings.filterwarnings('ignore') 



#World Cloud

from wordcloud import WordCloud



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/fifa19/data.csv')
data.info()
data.head(10)
data.shape
data.columns
sample_cols = ['ID', 'Name', 'Age', 'Nationality',

       'Overall', 'Potential', 'Club', 'Value', 'Wage',

       'Preferred Foot', 'International Reputation', 'Weak Foot',

       'Skill Moves', 'Work Rate', 'Position', 'Contract Valid Until',

       'Height', 'Weight',]



sample_data = data[sample_cols]



sample_data.head()
sample_data.info()
sample_data.isnull().sum() / len(sample_data)
sample_data = sample_data.dropna()
sample_data.shape
fig,(ax1,ax2) = plt.subplots(1,2 , figsize=(15,5))



sns.countplot('Preferred Foot' , data = sample_data , ax = ax1).set_title('Preferred Foot Dist.')

sns.distplot(sample_data['Overall'] , ax=ax2).set_title('Overall Score')

fig,(ax3,ax4) = plt.subplots(1,2 , figsize=(25,10))





sns.countplot(y = 'Nationality' , data = sample_data.loc[:100,:] , ax = ax3).set_title('Nationality of Players')

# Second Graph was sorted by the value counts 

sns.countplot(y = 'Club' , data = sample_data.loc[:100,:] , order = sample_data.loc[:100,'Club'].value_counts().index, palette = 'PuBuGn_d', ax = ax4).set_title('Club of Footballers')



plt.savefig('horizontal.png')
#Top100 Players Country distribution



dftop = sample_data.loc[:100,:]



plt.subplots(figsize=(12,12))



wordcloud = WordCloud(background_color = 'white',

                     width = 512,

                     height = 384).generate(' '.join(dftop['Nationality']))



plt.imshow(wordcloud)



plt.axis('off')





plt.show()



plt.savefig('wordcloud.png')
datatop300 = sample_data.loc[:300,:]

datatop300['Value'] = pd.Series(float(i[1:-1]) for i in datatop300['Value'])



data1 = go.Box(

    y=datatop300.loc[:100,'Value'],

    name = 'Value of Top100 Players',

    marker = dict(

        color = 'rgb(12, 12, 140)',

    )

)



data2 = go.Box(

    y=datatop300.loc[100:200,'Value'],

    name = 'Value of Top 100~200 Players',

    marker = dict(

        color = 'rgb(12, 140, 12)',

    )

)



data3 = go.Box(

    y=datatop300.loc[200:300,'Value'],

    name = 'Value of top 200~300 Players',

    marker = dict(

        color = 'rgb(140, 12, 12)',

    )

)



data = [data1,data2,data3]

iplot(data)



plt.savefig('box.png')

plt.subplots(figsize=(15,5))

sns.countplot('Position', data=datatop300 ,hue = 'Preferred Foot', order = datatop300['Position'].value_counts().index , palette = 'winter_r')

plt.xticks(rotation=90)



plt.savefig('vertical.png')
fig = plt.subplots(figsize=(15,10))

sns.violinplot(x = 'Preferred Foot' , y = 'Value', data = datatop300 ,inner = 'quartile' , palette ='Set3')



plt.savefig('violin.png')
plt.subplots(figsize=(10,5))

sns.scatterplot (x = 'Age' , y = 'Value' , data = datatop300, color = 'red' , marker = '+')
labels= dftop['Nationality'].value_counts().index



sizes = dftop['Nationality'].value_counts().values

plt.figure(figsize = (15,15))

plt.pie(sizes,  labels=labels, autopct='%1.1f%%')

plt.show()