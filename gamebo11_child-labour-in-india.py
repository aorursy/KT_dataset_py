# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/child-labour-in-inida/Child Labour in India.csv')
df.head()
df.info()
print('The number of rows are {}' '\n' 'The number of columns are {}'.format(df.shape[0], df.shape[1]) )
df.drop('Total', axis = 1, inplace = True)
All_india = df[df['States'] == 'All India']

All_india
df.drop(20, axis = 0, inplace = True)
df.columns = ['Hospitality' if 'Restaurants' in col_name else 'Services' if 'Services' in col_name else col_name for col_name in df.columns]
df['Category of States'].value_counts().plot(kind = 'bar')
df['States'].unique()
df['States'].nunique()
df['Manufacturing'] = [9.9 if x == '9. 9' else float(x) for x in df['Manufacturing']]
num_cols = np.array([col for col in df.columns if df[col].dtype == 'float64'])
fig, ax = plt.subplots(3,2, figsize = (45, 24))

for x in range(3):

    for y in range(2):

        sns.barplot(x = 'States', y = num_cols.reshape(3,2)[x][y], data = df, ax = ax[x][y])

        plt.setp(ax[x][y].get_xticklabels(), rotation = 'vertical', fontsize=11)

        ax[x][y].set_ylabel(num_cols.reshape(3, 2)[x][y], fontsize = 15)
States = df['States'].values.reshape(4,5)
df.tail()
fig, ax = plt.subplots(4, 5, figsize = (35, 25))

explode = [0, 0.2, 0, 0, 0.2, 0]

for x in range(States.shape[0]):

    for y in range(States.shape[1]):

        ax[x][y].pie(x = df[df['States'] == States[x][y]][num_cols].values.tolist()[0] , labels = df[df['States'] == States[x][y]][num_cols].columns

                     , explode=explode, autopct ='%.0f%%', wedgeprops = {'linewidth': 2.0}, pctdistance = 0.8, shadow = True, startangle = 90.5)

        ax[x][y].set_xlabel(States.reshape(4,5)[x][y], fontsize = 14)

        fig.show()
fig, ax = plt.subplots(nrows = len(num_cols), figsize = (25, 35))

for col in range(len(num_cols)):

    dat = df.sort_values(by = num_cols[col], ascending = False)

    sns.barplot(x = 'States', y = num_cols[col], data = dat, ax = ax[col])

    plt.setp(ax[col].get_xticklabels(), rotation = 'vertical')
fig, ax = plt.subplots(ncols = len(num_cols), figsize = (30, 3))

sns.set_style('darkgrid')

colors = ['blue', 'red', 'yellow', 'green', 'orange', 'gold']

for col in range(len(num_cols)):

    sns.distplot(df[num_cols[col]], ax = ax[col], color = colors[col], kde = False, bins = 31)
df.head()
Mean_of_all = [round(df[x].mean()) for x in num_cols]
Mean_of_all_df = pd.DataFrame(Mean_of_all, index=num_cols)
Mean_of_all_df.plot.bar(figsize = (12, 5))

plt.title("Mean of Child Labour in various sectors across all states")