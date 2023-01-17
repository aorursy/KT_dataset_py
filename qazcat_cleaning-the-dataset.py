# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/bgg_db_2017_04.csv', encoding='latin1')

df.head()
df.columns
df[['min_players','max_players','avg_time','min_time','max_time']].describe()
missing = df[(df['min_players'] < 1) | (df['max_players'] < 1) | (df['avg_time'] < 1) | (df['min_time'] < 1) | (df['max_time'] < 1)]

missing.shape[0]
df = df[(df['min_players'] >= 1) & (df['max_players'] >= 1) & (df['avg_time'] >= 1) & (df['min_time'] >= 1) & (df['max_time'] >= 1)]

df.shape[0]


import matplotlib.pyplot as plt

import seaborn as sns



sns.countplot(df['max_players'])

plt.title('Maximum Players')
df[df['max_players'] > 12]['max_players'].value_counts().sort_index()
df[df['max_players'] >= 99]['names']
df = df[df['max_players'] < 99]

df.shape[0]
sns.distplot(df['max_time'],kde=False)

plt.title('Maximum Playing Time')
df[df['max_time'] >= 1440][['names','max_time']]
df[['min_players','max_players','avg_time','min_time','max_time']].describe()
def get_cat_data(series):#Fuction for extracting set of categorical data labels

    cat_names = series.apply(lambda s:s.split(','))

    cat_names = cat_names.tolist()

    full_cats = []

    for lst in cat_names:

        for item in lst:

            full_cats.append(item.lstrip())

    return set(full_cats)

        

cat_data = {'Categories':get_cat_data(df['category']),'Mechanics':get_cat_data(df['mechanic']),'Designers':get_cat_data(df['designer'])}

cat_data