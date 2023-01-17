# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import seaborn as sns

import numpy as np

from IPython.display import display

import matplotlib.pyplot as plt
top2018 = pd.read_csv('/kaggle/input/top-spotify-tracks-of-2018/top2018.csv')

print('Print top 5 rows of the dataframe\n')

display(top2018.head())

print('\n\nDescriptive column view for the dataframe\n')

top2018.describe()
for i, col in enumerate(top2018._get_numeric_data()):

    m = top2018[col].mean()

    st = top2018[col].std()

    plt.figure(i)

    sns.scatterplot(data=top2018[col]).set_title('Distribution for Item {}'.format(col))

    plt.axhline(y=m, linewidth=2)

    plt.axhline(y=m+st, color = 'orange', linewidth=2)

    plt.axhline(y=m-st, color = 'orange', linewidth=2)
#Get number of songs falling within 1 standard deviation for each feature



cat_distribution = {}

for i, col in enumerate(top2018._get_numeric_data()):

    m = top2018[col].mean()

    st = top2018[col].std()

    cat_distribution[col] = [np.where((top2018[col] <= m+st) & (top2018[col] >= m-st))[0].size, m+st, m-st]
def Correlation_heat_map(df):

    sns.set(rc={'figure.figsize':(7,7)}, font_scale=1)

    ax = sns.heatmap(df.corr(), vmin = -1, vmax = 1)

    labels = [t.get_text() for t in ax.get_xticklabels()]

    ax.set_xticklabels(labels, rotation=30, horizontalalignment="right")

    



Correlation_heat_map(top2018)
def corr_feature(df, threshold):

    corr_data = df.corr().abs()

    upper = corr_data.where(np.triu(np.ones(corr_data.shape), k=1).astype(np.bool))

    cor_col = {}

    for i in upper.columns:

        for j in upper.index:

            if upper[i][j] > threshold:

                cor_col[i] = j

    return cor_col



corr_feature(top2018, 0.5)
#Get list of top performing artist by number of songs

songsperartist = top2018[['id','artists']].groupby(['artists']).agg('count').reset_index().sort_values('id', ascending = False)

songsperartist.head()
for k,v in cat_distribution.items():

    print("{}% of most heard track has {} in the range of {} to {}".format(v[0], k,  v[2], v[1]))