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
import numpy as np 

import pandas as pd

import seaborn as sns

import os

import datetime



from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



import plotly.graph_objects as go

import matplotlib.pyplot as plt

import plotly.figure_factory as ff

from plotly.subplots import make_subplots



import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv('/kaggle/input/data-police-shootings/fatal-police-shootings-data.csv')

df.head()
df.isnull().sum()
df['age'].describe()
df['race'].value_counts()
df['armed'].value_counts()
#feature generation

df['date'] = pd.to_datetime(df['date'])

df['year'] = pd.to_datetime(df['date']).dt.year

df['month'] = pd.to_datetime(df['date']).dt.month

df['month_name'] = df['date'].dt.strftime('$B')

df['month_num'] = df['date'].dt.strftime('%m')

df['weekdays'] = df['date'].dt.strftime('%A')

df['date_num'] = df['date'].dt.strftime('%d').astype(int)

df['date_categ'] = np.where(df['date_num']<16, "First Half", "Second Half")

df['date_mon'] = df.date.dt.to_period('M')



df['age_freq']=np.where(df['age']<18,'<18',np.where((df['age']>17)&(df['age']<=30),'18-30',

np.where((df['age']>30)&(df['age']<=40),'31-40',np.where(df['age']>50,'50+',

np.where((df['age']>40)&(df['age']<=50),'41-50',"Not Specified")))))



df['race_name']=np.where(df['race']=='W','White',np.where(df['race']=='B','Black',

np.where(df['race']=='N','Native American',np.where(df['race']=='H','Hispanic',

np.where(df['race']=='A','Asian',np.where(df['race']=='O','Others','Not Specified'))))))
monthly_df=df['date'].groupby(df.date.dt.to_period("M")).agg('count').to_frame(name="count").reset_index()

month_year=[]

for i in monthly_df['date']:

    month_year.append(str(i))



monthly_df.plot()

plt.show()
monthly_df.boxplot(vert=False)

plt.show()
#fluctuation on average of 80 deaths per month

#25% and 75% interquartile show that the number of kills are between 76 and 89 per month
monthly_df['year']=monthly_df['date'].dt.strftime('%Y') 

def plot_month(year,color):

    temp_month=[]

    for i in monthly_df.loc[monthly_df['year']==year]['date']:

        temp_month.append(str(i))

    trace=go.Bar(x=temp_month, y=monthly_df.loc[monthly_df['year']==year]['count'],

           name=year,marker_color=color)

    return trace



print(monthly_df)



monthly_df.groupby(['year']).plot.bar()

plt.show()
only_month=df.groupby(['month_name','date_categ'])['id'].agg('count').reset_index().rename(columns={'id':'count'})

only_month['month_name'] = pd.Categorical(only_month['month_name'],categories=['January','February','March','April','May','June','July','August','September','October','November','December'],ordered=True)

only_month = only_month.sort_values('month_name')



print(only_month)



only_month.groupby(['date_categ']).plot.bar()

plt.show()