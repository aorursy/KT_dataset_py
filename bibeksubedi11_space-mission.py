# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        



# Any results you write to the current directory are saved as output.

%time df = pd.read_csv("/kaggle/input/all-space-missions-from-1957/Space_Corrected.csv")

print(df.shape)
df.head()
df.isnull().sum()
df.drop(['Unnamed: 0','Unnamed: 0.1'],axis=1,inplace=True)
df.info()
df['Status Mission'].value_counts()
df['Company Name'].value_counts()[:15]
plt.figure(figsize = (20,8))

plt.style.use('fivethirtyeight')

ax = sns.countplot(df['Status Mission'],palette = 'Set2')

ax.set_xlabel(xlabel="Status of mission ", fontsize=18)

ax.set_ylabel(ylabel = 'Total count', fontsize= 18)

ax.set_title(label = "Outcome of mission  ", fontsize = 20)

plt.show()
labels = ['Success 89.7%', 'Failure 7.8', 'Partial Failure 2.4%', 'Prelaunch Failure 0.1%']

size = df['Status Mission'].value_counts()

colors = plt.cm.Dark2(np.linspace(0,1,5))



plt.rcParams['figure.figsize']= (30,9)

plt.pie(size, labels = labels,  colors= colors)

plt.title('Outcome of mission Piechart')

plt.legend()

plt.show()

plt.figure(figsize =(16,8))

plt.style.use('classic')

df['Company Name'].value_counts()[:15].plot(kind='bar')

plt.title('Company name count in data set', fontsize = 20)

plt.xlabel("Company name", fontsize= 16)

plt.ylabel(" count", fontsize = 16)

plt.show()

df['Status Rocket'].value_counts()
plt.figure(figsize = (20,8))

sns.countplot(df['Status Rocket'])
import plotly.express as px

import plotly

from plotly.subplots import make_subplots

import plotly.graph_objects as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)




status = df['Status Rocket'].value_counts()



fig = make_subplots(rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "domain"}]])

fig.add_trace(go.Bar( x=status.keys(), y=status.values, text=status.values.tolist(), textposition='auto',marker_color='#003786',name='Status'), row=1, col=1)

fig.add_trace(go.Pie(labels=status.keys(),values=status.values,textposition='inside', textinfo='percent+label',marker={'colors':['rgb(178,24,43)','rgb(253,219,199)']}), row=1, col=2)

fig.update_layout(title_text='Status of Rockets', font_size=10, autosize=False, width=800, height=400)

fig.show()
df['Country'] = df['Location'].apply(lambda x:x.split(',')[-1])

country = df.groupby('Country').count()['Detail'].sort_values(ascending=False).reset_index()
country.rename(columns={"Detail":"No of Launches"},inplace=True)

country.head(10).style.background_gradient(cmap='Blues').hide_index()