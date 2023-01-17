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
import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('/kaggle/input/fifa19/data.csv', index_col=0)

df.head()
df.info()
df.drop('Loaned From', axis=1, inplace=True)

df.dropna(inplace=True)
df1 = pd.DataFrame(df['Preferred Foot'].value_counts())

display(df1)

plt.pie(df1['Preferred Foot'], labels=df1.index, autopct='%1.1f%%')

plt.title('Right Foot & Left Foot Players', size=16)

plt.show()
df2 = df.groupby('Nationality').agg({'ID':'count', 'Age':'mean'}).sort_values('ID', ascending=False)

df2.columns = ['Number of Players', 'Average age of Players']

df2.head(10)
df3 = df2.head(20)

plt.subplots(figsize=(16, 8))

sns.barplot(df3.index, df3['Number of Players'], palette='RdBu')

plt.title('Top 20 Countries - Number of Players', fontsize=20)

plt.xticks(rotation=45)

plt.show()
def get_value(x):

    if '€' in str(x):

        x = x.replace('€', '')

    if 'M' in x:

        return float(x.replace('M', '')) * 10**6

    elif 'K' in x:

        return float(x.replace('K', '')) * 10**3

    else:

        return float(x)



df['Value'] = df['Value'].apply(get_value)
df4 = df.groupby('Club').agg({'Value':'sum'}).rename(columns={'Value':'Value Sum'})

df4.sort_values('Value Sum', ascending=False, inplace=True)

df4.head(10)
plt.subplots(figsize=(16,8))

sns.barplot(df4.head(20).index, df4.head(20)['Value Sum'], palette='RdBu')

plt.xticks(rotation=45)

plt.ylim(2e8, 8e8)

plt.ylabel('Value Sum (€)')

plt.title('Top 20 Clubs - Value Sum', fontsize=20)

plt.show()