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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



color = sns.color_palette()



%matplotlib inline
artists = pd.read_csv('../input/artists.csv',encoding='utf-8')

artworks = pd.read_csv('../input/artworks.csv',encoding='utf-8')
def sample_df(df):

    return pd.DataFrame(np.concatenate([df.dtypes.T.values.reshape(1,-1),df.sample(1),df.columns.T.values.reshape(1,-1)]).T,columns=['dtypes','sample','columns'])
sample_df(artists)
sample_df(artworks)
artists.shape,artworks.shape
joint_df = pd.concat([artists,artworks],join='inner',keys='Artist ID')
joint_df.Name.value_counts()[:20]
def plot_val_counts(df,column='Name',figsize=(10,8),title=None):

    counts = df[column].value_counts()[:30]

    plt.figure(figsize=figsize)

    sns.barplot(counts.values,counts.index)

#     plt.xlabel(column)

    plt.xlabel('Number of artworks')

    plt.xticks(rotation=0)

    plt.title(title)

    plt.show()



plot_val_counts(joint_df,title='Top 20 artists with greatest number of artworks on display at MoMa')
print('{} pieces of art were made in 1979'.format(artworks[artworks['Date'] == '1979'].shape[0]))



#This number could be greater, as there are a lot of rows unique descriptive text in the Date column:

print('{} rows with uncleaned text in the \'Date\' column.'.format(sum(artworks['Date'].value_counts()==1)))



print('\nValue counts:')

artworks['Date'].value_counts()[:20]
print('Gift or donation responsible for most artwork in the collection: \n{0}'.format(artworks['Credit'].value_counts()[:1].index[0]))
artworks['Credit'].value_counts()[:1].index[0]