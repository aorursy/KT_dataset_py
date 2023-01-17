# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# plt.style.use('fivethirtyeight')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%time data = pd.read_csv('/kaggle/input/despesas-atividade-parlamentar/Ano-2020.csv', skiprows=[1,2,3,6], header=0, sep=None)

print(data.shape)
data.head()
data.describe()
data['party_name'] = data[data.columns[0]] +" ("+ data[data.columns[6]] +")" 

data
data['sgPartido'].value_counts().sort_values()
party_occur = data['sgPartido'].value_counts(sort=True).reset_index()
party_occur = party_occur.rename(columns={'index': 'name', 'sgPartido': 'count'})

plt.figure(figsize=(20, 10))
chart = sns.barplot(y='name', x='count', data=party_occur)
chart.set_title(label='Spending occurrences by party', fontsize=16)
chart.set_xlabel(xlabel='Occurrences', fontsize=13)
chart.set_ylabel(ylabel='Party', fontsize=13)
plt.show()
politican_occur = data['party_name'].value_counts(sort=True).reset_index()
politican_occur = politican_occur.rename(columns={'index': 'name', 'party_name': 'count'})

politican_occur = politican_occur.head(25)

plt.figure(figsize=(20, 10))
chart = sns.barplot(data = politican_occur, x = 'count', y = 'name', palette='Reds_r')
chart.set_title(label='Spending occurrences by politician', fontsize=16)
chart.set_xlabel(xlabel='Occurrences', fontsize=13)
chart.set_ylabel(ylabel='Politician', fontsize=13)
plt.show()
expensive_party = data.groupby('sgPartido')['vlrLiquido'].sum().sort_values(ascending=False).reset_index()
expensive_party['vlrLiquido'] = expensive_party['vlrLiquido'].apply(lambda x: "{:,.2f}".format((x/1000000)))

expensive_party

plt.figure(figsize=(20, 10))
chart = sns.barplot(y='sgPartido', x='vlrLiquido', data=expensive_party)
chart.set_title(label='Party spending (in millions R$)', fontsize=16)
chart.set_xlabel(xlabel='millions', fontsize=13)
chart.set_ylabel(ylabel='Party', fontsize=13)
plt.show()