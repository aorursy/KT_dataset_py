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
df=pd.read_csv('/kaggle/input/world-happiness-report/2020.csv')
df
import matplotlib.pyplot as plt

import seaborn as sns
df.isnull().sum()
plt.figure(figsize=(12,6))

sns.barplot(x=df['Country name'].head(10),y='Social support',data=df,palette='viridis')

plt.title('A plot of the top ten happiest countries to live versus their ladder score')

plt.tight_layout()
plt.figure(figsize=(12,6))

sns.barplot(x=df['Country name'].head(10),y='Generosity',data=df,palette='viridis')

plt.title('A plot of the top ten happiest countries to live versus their ladder score')

plt.tight_layout()
plt.figure(figsize=(12,6))

sns.barplot(x=df['Country name'].head(10),y='Perceptions of corruption',data=df,palette='viridis')

plt.title('A plot of the top ten happiest countries to live versus their ladder score')

plt.tight_layout()
plt.figure(figsize=(12,6))

sns.barplot(x=df['Country name'].head(10),y='Healthy life expectancy',data=df,palette='viridis')

plt.title('A plot of the top ten happiest countries to live versus their ladder score')

plt.tight_layout()
plt.figure(figsize=(12,6))

sns.barplot(x=df['Country name'].head(10),y='Freedom to make life choices',data=df,palette='viridis')

plt.title('A plot of the top ten happiest countries to live versus their ladder score')

plt.tight_layout()
plt.figure(figsize=(12,6))

sns.barplot(x=df['Country name'].head(10),y='Logged GDP per capita',data=df,palette='viridis')

plt.title('A plot of the top ten happiest countries to live versus their ladder score')

plt.tight_layout()
# i am considering the countries to be top in happiness if they are present in all the lists below

# here all the comparisons are done with the average value of particular feature.

#features to be considered are ['Generosity','Social support','Health Life Expactancy','Logged GDP Per Capita','Freedom to make life choices','Perceptions of corruption']
list_gen=[]

list_gen=list(df[df['Generosity']>0]['Country name'])
list_gen
list_corrupt=[]

list_corrupt=list(df[df['Perceptions of corruption']<0.73]['Country name'])
list_corrupt
df
list_social=[]

list_social=list(df[df['Social support']>0.8]['Country name'])
list_social
list_health=[]

list_health=list(df[df['Healthy life expectancy']>64.44]['Country name'])
list_health
list_choices=[]

list_choices=list(df[df['Freedom to make life choices']>0.78]['Country name'])
list_choices
list_gdp=[]

list_gdp=list(df[df['Logged GDP per capita']>9.29]['Country name'])
list_gdp
final_list=[]

for i in df['Country name']:

    if i in list_corrupt and i in list_social and i in list_health and i in list_choices and i in list_gdp and i in list_gen:

        final_list.append(i)
final_list
#these are top happiest countires in the world 