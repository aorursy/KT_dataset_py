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

import matplotlib.pyplot as plt          

import numpy as np                      

from scipy.sparse import csr_matrix      

from scipy import stats

import seaborn as sns

import missingno as msno

import string

from pandas.api.types import CategoricalDtype 

sample_submission = pd.read_csv("../input/categorical-feature-encoding-challenge-ii/sample_submission.csv")

df_test = pd.read_csv("../input/categorical-feature-encoding-challenge-ii/test.csv")

df_train = pd.read_csv("../input/categorical-feature-encoding-challenge-ii/train.csv")
sm_test = df_test[['bin_1', 'ord_1', 'ord_3', 'ord_5', 'nom_9']]

sm_train = df_train[['bin_1', 'ord_1', 'ord_3','ord_5', 'nom_9', 'target']] 
def resumetable(df):

    print(f"Dataset Shape: {df.shape}")

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.loc[0].values

    summary['Second Value'] = df.loc[1].values

    summary['Third Value'] = df.loc[2].values



    for name in summary['Name'].value_counts().index:

        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 



    return summary
summary_train = resumetable(sm_train)

summary_train
summary_test = resumetable(sm_test)

summary_test
sns.heatmap(sm_train.isnull(), cbar=False)
msno.matrix(sm_train, figsize=(10,5), fontsize=10)
msno.heatmap(sm_train, figsize=(10,5), fontsize=10)
plt.figure(figsize=(10,5))

num_cols = sm_train.select_dtypes(exclude=['object']).columns

corr = sm_train[num_cols].corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
total = len(sm_train)

plt.figure(figsize=(10,5))



g = sns.countplot(x='target', data=sm_train)

g.set_title("TARGET DISTRIBUTION", fontsize = 20)

g.set_xlabel("Target Vaues", fontsize = 15)

g.set_ylabel("Count", fontsize = 15)

sizes=[] # Get highest values in y

for p in g.patches:

    height = p.get_height()

    sizes.append(height)

    g.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format(height/total*100),

            ha="center", fontsize=14) 

g.set_ylim(0, max(sizes) * 1.15) # set y limit based on highest heights



plt.show()
bin_cols = ['bin_1']
#Looking the V's features

import matplotlib.gridspec as gridspec # to do the grid of plots

grid = gridspec.GridSpec(3, 2) # The grid of chart

plt.figure(figsize=(16,20)) # size of figure



# loop to get column and the count of plots

for n, col in enumerate(sm_train[bin_cols]): 

    ax = plt.subplot(grid[n]) # feeding the figure of grid

    sns.countplot(x=col, data=sm_train, hue='target', palette='hls') 

    ax.set_ylabel('Count', fontsize=15) # y axis label

    ax.set_title(f'{col} Distribution by Target', fontsize=18) # title label

    ax.set_xlabel(f'{col} values', fontsize=15) # x axis label

    sizes=[] # Get highest values in y

    for p in ax.patches: # loop to all objects

        height = p.get_height()

        sizes.append(height)

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(height/total*100),

                ha="center", fontsize=14) 

    ax.set_ylim(0, max(sizes) * 1.15) # set y limit based on highest heights

    

plt.show()
sm_train['bin_1'] = sm_train['bin_1'].replace(np.nan, 0)

sm_test['bin_1'] = sm_test['bin_1'].replace(np.nan, 0)
sm_train['bin_1'].head(10)
sm_test['bin_1'].head(10)
# nom_cols = ['nom_9']
sns.countplot('nom_9', hue='target', data= sm_train)

plt.show()
# ord_cols = ['ord_1', 'ord_3']
# ord_1 = CategoricalDtype(categories=['Novice', 'Contributor','Expert', 

                                    #'Master', 'Grandmaster'], ordered=True)

# ord_3 = CategoricalDtype(categories=['a', 'b', 'c', 'd', 'e', 'f', 'g',

                                     #'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'], ordered=True)
# sm_train.ord_1 = sm_train.ord_1.astype(ord_1)

# sm_train.ord_3 = sm_train.ord_3.astype(ord_3)
# sm_train.ord_1.head()
# sm_train.ord_3.head()
tmp = pd.crosstab(sm_train['ord_5'], sm_train['target'], normalize='index') * 100

tmp = tmp.reset_index()

tmp.rename(columns={0:'No',1:'Yes'}, inplace=True)

plt.figure(figsize=(10,5))



plt.subplot()

ax = sns.countplot(x='ord_5', data=sm_train, order=list(tmp['ord_5'].values) , color='chocolate') 

ax.set_ylabel('Count', fontsize=17) # y axis label

ax.set_title('ord_5 Distribution', fontsize=20) # title label

ax.set_xlabel('ord_5 values', fontsize=17) # x axis label
ord_5_count = sm_train['ord_5'].value_counts().reset_index()['ord_5'].values
sm_train_split= sm_train.select_dtypes(include='object').fillna(\

sm_train.select_dtypes(include='object').mode().iloc[0])
sm_train = sm_train.apply(lambda x:x.fillna(x.value_counts().index[0]))
sm_train_split.head()
# sm_train['ord_5'].iloc[0].apply(lambda x:[(string.ascii_letters.find(letter)+1) for letter in x])
sm_train_split['ord_5_add'] = sm_train_split['ord_5'].apply(lambda x:sum([(string.ascii_letters.find(letter)+1) for letter in x]))
sm_train_split['ord_5_add'].head()
ord_1 = CategoricalDtype(categories=['Novice', 'Contributor','Expert', 

                                     'Master', 'Grandmaster'], ordered=True)

ord_3 = CategoricalDtype(categories=['a', 'b', 'c', 'd', 'e', 'f', 'g',

                                     'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'], ordered=True)
sm_train.ord_1 = sm_train.ord_1.astype(ord_1)

sm_train.ord_3 = sm_train.ord_3.astype(ord_3)
sm_train.ord_1.head()
sm_train.ord_3.head()