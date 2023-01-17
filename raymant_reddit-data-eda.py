# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

reddit_data = pd.read_csv('/kaggle/input/dataisbeautiful/r_dataisbeautiful_posts.csv')

reddit_data.head()
reddit_data1 = reddit_data.drop(['author_flair_text', 'removed_by', 'total_awards_received', 'awarders'], axis=1)
reddit_data1 = reddit_data1.dropna()
reddit_data2 = reddit_data1.drop(reddit_data.index[reddit_data['author']=='[deleted]'])
plt.figure(figsize = (10,6))

chart = sns.countplot(x = 'author', data = reddit_data2, 

                      order = reddit_data2.author.value_counts().iloc[:10].index)

chart.set_xticklabels(chart.get_xticklabels(), rotation = 45)

plt.title('Top 10 authors having more number of titles')

plt.xlabel('Author')

plt.ylabel('Number of titles')
reddit_data3 = pd.DataFrame(reddit_data1.groupby('title').agg({'score':['sum']}))

reddit_data3.columns = ['sum']



reddit_data3 = reddit_data3.reset_index()
reddit_data3_sorted_desc = reddit_data3.sort_values('sum',ascending = False)

reddit_data3_sorted_desc.head(10)
plt.figure(figsize = (10,6))



chart1 = sns.barplot(x = 'title', y = 'sum', data = reddit_data3_sorted_desc[:10])

chart1.set_xticklabels(chart1.get_xticklabels(), rotation = 45)

plt.title('Top 10 titles with most scores')

plt.xlabel('Title')

plt.ylabel('Scores')
reddit_data4 = pd.DataFrame(reddit_data1.groupby('title').agg({'num_comments':['sum']}))

reddit_data4.columns = ['sum']



reddit_data4 = reddit_data4.reset_index()
reddit_data4_sorted_desc = reddit_data4.sort_values('sum',ascending = False)

reddit_data4_sorted_desc.head(10)
plt.figure(figsize = (10,6))



chart1 = sns.barplot(x = 'title', y = 'sum', data = reddit_data4_sorted_desc[:10])

chart1.set_xticklabels(chart1.get_xticklabels(), rotation = 45)

plt.title('Top 10 titles with most number of comments')

plt.xlabel('Title')

plt.ylabel('Number of Comments')
reddit_data1['over_18'].value_counts()
print(reddit_data1[reddit_data1['over_18'] == True]['score'].mean())

print(reddit_data1[reddit_data1['over_18'] == False]['score'].mean())
print(reddit_data1[reddit_data1['over_18'] == True]['num_comments'].mean())

print(reddit_data1[reddit_data1['over_18'] == False]['num_comments'].mean())
plt.figure(figsize = (10,8))



splot = reddit_data1.groupby('over_18').agg({'score':['sum']}).plot.bar()

for p in splot.patches:

    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., 

                                                   p.get_height()), ha = 'center', va = 'center', 

                   xytext = (0, 10), textcoords = 'offset points')

plt.ylabel('Total score')
plt.figure(figsize = (10,8))



splot = reddit_data1.groupby('over_18').agg({'num_comments':['sum']}).plot.bar()

for p in splot.patches:

    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., 

                                                   p.get_height()), ha = 'center', va = 'center', 

                   xytext = (0, 10), textcoords = 'offset points')

plt.ylabel('Total number of comments')
sns.regplot(x = 'num_comments', y = 'score', data = reddit_data1[reddit_data1['over_18'] == True])

plt.title('Number of Comments vs Score')

plt.xlabel('Number of Comments')

plt.ylabel('Score')
sns.residplot(x = 'num_comments', y = 'score', data = reddit_data1[reddit_data1['over_18'] == True])

plt.title('Residual plot')

plt.xlabel('Number of Comments')

plt.ylabel('Score')
sns.countplot(reddit_data[reddit_data['over_18'] == True]['removed_by'].dropna(),)



plt.title('NSFW content')

plt.xlabel('Removed by')

plt.ylabel('Count')
sns.countplot(reddit_data[reddit_data['over_18'] == False]['removed_by'].dropna())



plt.title('Mass Orinted content')

plt.xlabel('Removed by')

plt.ylabel('Count')