# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import seaborn as sn

import plotly.express as px

from plotly.offline import iplot

from plotly.subplots import make_subplots

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('/kaggle/input/co2-ghg-emissionsdata/co2_emission.csv')

df.head()
df = df.dropna()

print(f' shape of the dataframe after dropping null values: {df.shape}')
df.sort_values(by = 'Annual CO₂ emissions (tonnes )',axis = 0,ascending=False,inplace=True)

df.head(3)
df_17 = df[(df['Year']>=1751)&(df['Year']<=1800)]

df_18 = df[(df['Year']>=1801)&(df['Year']<=1900)]

df_19 = df[(df['Year']>=1901)&(df['Year']<=2000)]

df_20 = df[(df['Year']>=2005)]

print(f' shape of the df_17 dataframe is <{df_17.shape}>\n\n shape of the df_18 dataframe is <{df_18.shape}>\n\n shape of the df_19 dataframe is <{df_19.shape}>\n\n shape of the df_20 dataframe is <{df_20.shape}>')
df_17 = df[(df['Entity']!='World')]

df_18 = df[(df['Entity']!='World')]

df_19 = df[(df['Entity']!='World')]

df_20 = df[(df['Entity']!='World')]
def namereplace(df):

    df.columns = ['Country','Code','Year','emission']

    return df

df_20 = namereplace(df_20)

df_17 = namereplace(df_17)

df_18 = namereplace(df_18)

df_19 = namereplace(df_19)

df_20.head(2)
df_18 = df_18[(df_18['Year']>=1801)&(df_18['Year']<=1900)]

df_19 = df_19[(df_19['Year']>=1901)&(df_19['Year']<=2000)]

df_20 = df_20[(df_20['Year']>=2005)&(df_20['Year']<=2017)]

df_20 = df_20[(df_20['Year']>=2005)]

df_20.shape
px.bar(data_frame=df_18.head(50),x = 'Country',y = 'emission',color = 'Year',labels={'x':'Country','y':'emission'})
px.bar(data_frame=df_19.head(50),x = 'Country',y = 'emission',color = 'Year',labels={'x':'Country','y':'emission'})
px.bar(data_frame=df_20.head(50),x = 'Country',y = 'emission',color = 'Year',labels={'x':'Country','y':'emission'})
def areaplot(df):

    fig = px.area(data_frame=df,x = 'Country',y = 'emission',color='Year',height=700,title = 'Co2 emission stats')

    fig.update_layout(xaxis_rangeslider_visible = True)

    fig.show()
df_list = [df_17,df_18,df_19,df_20]

title_list  =['18th century stats','19th century stats','20th century stats','21st century stats']

for i,j in zip(df_list,title_list):

    print(f' Area plot for {j} is displayed below ↓')

    areaplot(i)

    print("="*50)
def treemaps(df,i):

        fig = px.treemap(data_frame=df,path=['Country'],values='emission',color_discrete_sequence=px.colors.qualitative.Plotly_r,

                        height=700,title=i)

        fig.data[0].textinfo = 'label+text+value'

        fig.show()
df_list = [df_17,df_18,df_19,df_20]

title_list  =['18th century stats','19th century stats','20th century stats','21st century stats']

for i,j in zip(df_list,title_list):

    treemaps(i,j)

    print("="*50)