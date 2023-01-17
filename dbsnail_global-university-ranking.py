import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

data = pd.read_csv('../input/cwurData.csv')
data.head(3)
data.shape
data.info()
dt_country = data.country.value_counts()



fig = plt.figure(figsize=(6, 14))

dt_country.plot(kind='barh')

plt.title('Total Institute by Country')

plt.xlabel('Counts')

plt.show()
def plot_correlation(df,fg_width=9, fg_height=9):

    

    corr = df.corr()

    fig = plt.figure(figsize=(fg_width, fg_height))

    ax = fig.add_subplot(211)

    cax = ax.matshow(corr, cmap='Blues', interpolation='nearest')

    

    fig.colorbar(cax)

    plt.xticks(range(len(corr.columns)), corr.columns)

    plt.yticks(range(len(corr.columns)), corr.columns)



plot_correlation(data.drop(['score', 'national_rank', 'year'], axis=1))
dt_sub2 = data.ix[:,["world_rank", "publications","influence","citations", "broad_impact"]]

pd.scatter_matrix(dt_sub2, alpha=0.3, diagonal='kde', figsize=(9,9))

plt.show()
data[['world_rank','institution', 'country', 'year']][(data['world_rank']<=10)]
data[['world_rank','institution', 'country', 'year']][(data['country']=='China')&(data['world_rank']<=500)]