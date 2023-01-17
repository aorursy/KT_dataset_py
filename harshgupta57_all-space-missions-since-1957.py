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
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/all-space-missions-from-1957/Space_Corrected.csv')
df.info()
df.head()
df.shape
df.keys()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')
df.isnull().sum()
columns = [' Rocket', 'Unnamed: 0', 'Unnamed: 0.1']
df = df.drop(columns, axis =1)
df.shape
df['Country']=df['Location'].apply(lambda x:x.split(',')[-1])
df['Country']=df['Country'].str.strip()

df['Year']=df['Datum'].apply(lambda x:x.split(',')[1].split(' ')[1])
df['Year'] = pd.to_datetime(df['Year'])

df.drop('Datum', axis = 1)
df['Country'].value_counts().head(9).plot.pie(figsize=(10,10),autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2,
                                              title='% Distribution Country-wise')
sns.countplot(x='Status Mission', data=df,palette='viridis')
data2 = df['Country'].unique()
data2
df['Country'].value_counts().plot(kind='bar'
                                         )
grouped = df.groupby(['Country','Status Mission'])['Country'].count().unstack()
grouped.sort_index(ascending=False)
grouped.plot(kind='bar', stacked=True, figsize=(16,16),title='Mission Status Country-wise')
from wordcloud import WordCloud, STOPWORDS

df2 = df.query('Year > 2010')
company = " ".join(df2['Company Name'])
country = " ".join(df2['Country'])
#Defining a function to plot wordclouds
def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(15, 10))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off");
    
wordcloud = WordCloud(width = 1600, height = 1200, random_state=1, background_color='black', colormap='Pastel1', collocations=False,
                      stopwords = STOPWORDS).generate(company)
    
    
plot_cloud(wordcloud)
