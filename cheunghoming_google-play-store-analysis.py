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
import string # library used to deal with some text data

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualization library

pd.set_option('display.max_columns', 100) # Setting pandas to display a N number of columns

pd.set_option('display.max_rows', 10) # Setting pandas to display a N number rows

pd.set_option('display.width', 1000) # Setting pandas dataframe display width to N

from scipy import stats # statistical library

from statsmodels.stats.weightstats import ztest # statistical library for hypothesis testing

import plotly.graph_objs as go # interactive plotting library

import plotly.express as px # interactive plotting library

from itertools import cycle # used for cycling colors at plotly graphs

import matplotlib.pyplot as plt # plotting library

import pandas_profiling # library for automatic EDA

%pip install autoviz # installing and importing autoviz, another library for automatic data visualization

from autoviz.AutoViz_Class import AutoViz_Class

from IPython.display import display # display from IPython.display

from itertools import cycle # function used for cycling over values

%pip install ppscore # installing ppscore, library used to check non-linear relationships between our variables

import ppscore as pps # importing ppscore

import datetime #importing datetime
df = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')
df
df.info()
df.drop('Current Ver', axis = 1, inplace = True)

df.drop('Android Ver', axis = 1, inplace = True)
df['Reviews'].unique()
df['Size'].unique()
df['Installs'].unique()
df[df['Installs']=='Free']
df.drop(10472,inplace=True)
df['Size'] = df['Size'].apply(lambda x: str(x).replace('Varies with device', 'NaN') if 'Varies with device' in str(x) else x)

df['Size'] = df['Size'].apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)

df['Size'] = df['Size'].apply(lambda x: str(x).replace(',', '') if 'M' in str(x) else x)

df['Size'] = df['Size'].apply(lambda x: float(str(x).replace('k', '')) / 1000 if 'k' in str(x) else x)

df['Size'] = df['Size'].apply(lambda x: float(x))



df['Installs'] = df['Installs'].apply(lambda x: x.replace('+', '') if '+' in str(x) else x)

df['Installs'] = df['Installs'].apply(lambda x: x.replace(',', '') if ',' in str(x) else x)

df['Installs'] = df['Installs'].apply(lambda x: int(x))



df['Reviews'] = df['Reviews'].apply(lambda x: int(x))
df
df['Year_Updated'] = df['Last Updated'].apply(lambda x: str(x).replace(',','').split()[2])

df['Year_Updated'] = df['Year_Updated'].apply(lambda x: int(x))



df['Month_Updated'] = df['Last Updated'].apply(lambda x: str(x).replace(',','').split()[0])



df['Day_Updated'] = df['Last Updated'].apply(lambda x: datetime.datetime.strptime(x, '%B %d, %Y').strftime('%a'))
df.drop('Last Updated', axis = 1, inplace = True)
df
df[df['App'].duplicated() == True]
df.drop_duplicates(subset=['App'],keep='last',inplace=True)
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df['Miss_Rating'] = df['Rating'].isnull()

df['Miss_Size'] = df['Size'].isnull()
df
matrix_df = pps.matrix(df)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')

matrix_df = matrix_df.apply(lambda x: round(x, 2))



sns.heatmap(matrix_df, vmin=0, vmax=1, cmap="Blues", linewidths=0.75, annot=True)
df.drop('Genres', axis = 1, inplace = True)

df.drop('Miss_Rating', axis = 1, inplace = True)

df.drop('Miss_Size', axis = 1, inplace = True)

df['Size'].fillna(value=df['Size'].mean(),inplace=True)

df.dropna(inplace=True)
df=pd.get_dummies(df, columns=['Type'])

df.drop('Price', axis = 1, inplace = True)

df.drop('Type_Paid', axis = 1, inplace = True)
df
df.set_index('App',inplace=True)

report = pandas_profiling.ProfileReport(df)

display(report)
plt.figure(figsize=(20,10))

sns.countplot(x='Installs',data=df)

plt.title("Distribution of Install", size=18)

plt.xticks(rotation=90)
df['Install_r'] = pd.cut(df.Installs,bins=[0,10000,100000,1000000,10000000,1000000000],labels=['1-10000','10001-100000','100001-1000000','1000001-10000000','>100000000'])

df.drop('Installs', axis = 1, inplace = True)
df
report = pandas_profiling.ProfileReport(df)

display(report)
col=list(df.columns)

col.remove('Reviews')

col.remove('Size')



for col_name in col:

   

    if col_name == 'Month_Updated':

        plt.figure(figsize=(7,4))

        order_x = ['January','February','March','April','May','June','July','August','September','October','November','December' ]

        plt.title("Distribution of "+col_name, size=18)

        sns.countplot(df[col_name],order=order_x)

        plt.xticks(rotation=90)

        fig.show()

    elif col_name == 'Day_Updated':

        plt.figure(figsize=(7,4))

        order_x = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun' ]

        plt.title("Distribution of "+col_name, size=18)

        sns.countplot(df[col_name],order=order_x)

        plt.xticks(rotation=90)

        fig.show()

    else:

        plt.figure(figsize=(7,4))

        plt.title("Distribution of "+col_name, size=18)

        sns.countplot(df[col_name])

        plt.xticks(rotation=90)

        fig.show()

        
plt.figure(figsize=(50,20))

sns.catplot(x='Category',y='Rating',data=df,kind='box',height=10,showmeans=True)

plt.title("Rating of each category", size=18)

plt.xticks(rotation=90)
sns.barplot(x='Type_Free',y='Rating',data=df)

plt.title("Average Rating by Charging Scheme", size=18)



dist_a = df[['Rating','Type_Free']][df['Type_Free']==1]

dist_b = df[['Rating','Type_Free']][df['Type_Free']==0]



# Z-test: Checking if the distribution means (Rating of free app vs Rating of paid app) are statistically different

t_stat, p_value = ztest(dist_a, dist_b)

print("----- Z Test Results -----")

print("T stat. = " + str(t_stat))

print("P value = " + str(p_value)) # P-value is less than 0.05



print("")



# T-test: Checking if the distribution means (Rating of free app vs Rating of paid app) are statistically different

t_stat_2, p_value_2 = stats.ttest_ind(dist_a, dist_b)

print("----- T Test Results -----")

print("T stat. = " + str(t_stat_2))

print("P value = " + str(p_value_2)) # P-value is less than 0.05
cat_con_rate = df.pivot_table(values='Rating',index='Category',columns='Content Rating')

plt.figure(figsize=(20,10))

sns.heatmap(cat_con_rate)