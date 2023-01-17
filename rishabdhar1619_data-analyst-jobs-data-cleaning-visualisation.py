# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

pd.set_option('display.max_columns',None)

plt.style.use('fivethirtyeight')



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
df=pd.read_csv('/kaggle/input/data-analyst-jobs/DataAnalyst.csv')

df.shape
df.head()
df.info()
df=df.replace(-1.0,np.NaN)

df=df.replace(-1,np.NaN)

df=df.replace('-1',np.NaN)
df.isnull().sum()
df.drop('Unnamed: 0',axis=1,inplace=True)
df['Job Title'],df['Department']=df['Job Title'].str.split(', ',1).str
df['Company Name'],_=df['Company Name'].str.split('\n',1).str
df['City'],df['State']=df['Location'].str.split(', ',1).str

df.drop('Location',axis=1,inplace=True)
df['Easy Apply']=df['Easy Apply'].fillna(False).astype('bool')
# Remove the the texts within brackets

df['Salary Estimate'],_=df['Salary Estimate'].str.split('(',1).str



#Replacing K with ""

df['Salary Estimate']=df['Salary Estimate'].replace('K','',regex=True)



# Creating two columns Maximum and Minimum salaries by splitting the "-" 

df['Minimum Salary'],df['Maximum Salary']=df['Salary Estimate'].str.split('-',1).str



# Removing the $ sign

df['Minimum Salary']=df['Minimum Salary'].str.lstrip('$').fillna(0).astype('int')

df['Maximum Salary']=df['Maximum Salary'].str.lstrip('$').fillna(0).astype('int')



# Dropping the salary estimate column

df.drop('Salary Estimate',axis=1,inplace=True)
df['Revenue'].value_counts()
fig,ax=plt.subplots(1,2,figsize=(15,5))

_=sns.distplot(df['Minimum Salary'],ax=ax[0])

_=sns.distplot(df['Maximum Salary'],ax=ax[1])
plt.figure(figsize=(8,5))

sns.distplot(df['Rating']);
df['Job Title'].value_counts().head(10).plot.bar();
df_min=df.groupby(['Minimum Salary'])[['Job Title','Minimum Salary']]

df_min.head().nsmallest(10,'Minimum Salary')
df_max=df.groupby(['Maximum Salary'])[['Job Title','Maximum Salary']]

df_max.head().nlargest(10,'Maximum Salary')
data=df[['Job Title','Company Name']]

data[data['Easy Apply']==True].sort_values(by='Easy Apply',ascending=False).head(50)
df['Company Name'].value_counts().head(10).plot.bar();
df_rating=df.groupby(['Rating'])['Company Name','Type of ownership','Rating']

df_rating.head().nlargest(10,'Rating')
df[df['Easy Apply']==True][['Company Name','Easy Apply','Rating']].head(10).nlargest(10,'Rating')
df['Headquarters'].value_counts().head(10).plot.bar();
plt.figure(figsize=(8,8))

df['Headquarters'].value_counts().head(10).plot.pie(autopct='%.2f%%')

plt.axis('off');
df['Founded'].value_counts().head(10).plot.bar();
plt.figure(figsize=(12,6))

df['Founded'].value_counts().head(10).plot.pie(autopct='%.2f%%')

plt.axis('off');
plt.figure(figsize=(8,5))

df['Industry'].value_counts().head(10).plot.bar();
plt.figure(figsize=(8,8))

df['Industry'].value_counts().head(10).plot.pie(autopct='%.2f%%')

plt.axis('off');
plt.figure(figsize=(8,5))

df['Sector'].value_counts().head(10).plot.bar();
plt.figure(figsize=(8,8))

df['Sector'].value_counts().head(10).plot.pie(autopct='%.2f%%')

plt.axis('off');
plt.figure(figsize=(8,5))

df['Type of ownership'].value_counts().head(10).plot.bar();
plt.figure(figsize=(9,9))

df['Type of ownership'].value_counts().head(10).plot.pie(autopct='%.2f%%')

plt.axis('off');
from wordcloud import WordCloud

from wordcloud import STOPWORDS

stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color = 'black',

                     height=1500,

                     width=1500).generate(str(df["Job Title"]))

plt.rcParams['figure.figsize'] = (12,12)

plt.axis("off")

plt.imshow(wordcloud)

plt.title("Most available Job Title")

plt.show()
from wordcloud import WordCloud

from wordcloud import STOPWORDS

stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color = 'black',

                     height=1500,

                     width=1500).generate(str(df["Company Name"]))

plt.rcParams['figure.figsize'] = (12,12)

plt.axis("off")

plt.imshow(wordcloud)

plt.title("Most available Companies")

plt.show()