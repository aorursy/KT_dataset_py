# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline

import plotly.express as px

import plotly.graph_objects as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_excel('/kaggle/input/serratus-ultrahigh-throughput-viral-discovery/notebook/200411/div_v_alignment_test1.xlsx')

df.head()
# checking dataset



print ("Rows     : " ,df.shape[0])

print ("Columns  : " ,df.shape[1])

print ("\nFeatures : \n" ,df.columns.tolist())

print ("\nMissing values :  ", df.isnull().sum().values.sum())

print ("\nUnique values :  \n",df.nunique())
# Distribution of different type of amount

fig , ax = plt.subplots(1,3,figsize = (12,5))



bowtie2 = df.bowtie2.values

urmap= df.urmap.values

mu = df.mu.values



sns.distplot(bowtie2 , ax = ax[0] , color = 'blue').set_title('Bowtie2' , fontsize = 14)

sns.distplot(urmap , ax = ax[1] , color = 'cyan').set_title('URMAP' , fontsize = 14)

sns.distplot(mu , ax = ax[2] , color = 'purple').set_title('MU' , fontsize = 14)





plt.show()
import matplotlib.gridspec as gridspec

from scipy.stats import skew

from sklearn.preprocessing import RobustScaler,MinMaxScaler

from scipy import stats

import matplotlib.style as style

style.use('seaborn-colorblind')
def plotting_3_chart(df, feature): 

    ## Creating a customized chart. and giving in figsize and everything. 

    fig = plt.figure(constrained_layout=True, figsize=(10,6))

    ## crea,ting a grid of 3 cols and 3 rows. 

    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

    #gs = fig3.add_gridspec(3, 3)



    ## Customizing the histogram grid. 

    ax1 = fig.add_subplot(grid[0, :2])

    ## Set the title. 

    ax1.set_title('Histogram')

    ## plot the histogram. 

    sns.distplot(df.loc[:,feature], norm_hist=True, ax = ax1)



    # customizing the QQ_plot. 

    ax2 = fig.add_subplot(grid[1, :2])

    ## Set the title. 

    ax2.set_title('QQ_plot')

    ## Plotting the QQ_Plot. 

    stats.probplot(df.loc[:,feature], plot = ax2)



    ## Customizing the Box Plot. 

    ax3 = fig.add_subplot(grid[:, 2])

    ## Set title. 

    ax3.set_title('Box Plot')

    ## Plotting the box plot. 

    sns.boxplot(df.loc[:,feature], orient='v', ax = ax3 );

 



print('Skewness: '+ str(df['bowtie2'].skew())) 

print("Kurtosis: " + str(df['bowtie2'].kurt()))

plotting_3_chart(df, 'bowtie2')
stats.probplot(df['urmap'].values, dist="norm", plot=plt)

plt.show()
stats.probplot(df['mu'].values, dist="norm", plot=plt)

plt.show()
sns.heatmap(df.corr(), annot = True, linewidths=.5, cmap='cubehelix')

plt.title('Correlations', fontsize = 20)

plt.show()
f, (ax1, ax2) = plt.subplots(1, 2, sharey = True)



ax1.plot(df.bowtie2, df.urmap, c = 'green')

ax1.set_title('Bowtie2 vs. Urmap', c = 'green')

ax2.scatter(df.urmap, df.bowtie2, c='red')

ax2.set_title('Urmap vs. Bowtie2', c ='red')



plt.ylabel('Urmap', fontsize = 20)



plt.show()
sns.boxplot(x=df['urmap'], color = 'cyan')

plt.title('Urmap Boxplot', fontsize = 20)

plt.show()
sns.boxplot(x=df['bowtie2'], color = 'magenta')

plt.title('Bowtie2 Boxplot', fontsize = 20)

plt.show()
#Code from Mario Filho

from category_encoders import OneHotEncoder

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler



cols_selected = ['urmap']

ohe = OneHotEncoder(cols=cols_selected, use_cat_names=True)

df_t = ohe.fit_transform(df[cols_selected+['bowtie2']])



#scaler = MaxAbsScaler()

X = df_t.iloc[:,:-1]

y = df_t.iloc[:, -1].fillna(df_t.iloc[:, -1].mean()) / df_t.iloc[:, -1].max()



mdl = Ridge(alpha=0.1)

mdl.fit(X,y)



pd.Series(mdl.coef_, index=X.columns).sort_values().head(10).plot.barh()
ax = df.groupby('urmap')['bowtie2'].mean().plot(kind='barh', figsize=(12,8),

                                                           title='Mean estimated Urmap')

plt.xlabel('Mean estimated Urmap')

plt.ylabel('Bowtie2')

plt.show()
ax = df.groupby('bowtie2.1')['urmap.1'].min().sort_values(ascending=True).plot(kind='barh', figsize=(12,8), color='r',

                                                                                  title='Min.estimated Bowtie2.1')

plt.xlabel('Min.estimated Bowtie2.1')

plt.ylabel('Urmap.1')

plt.show()
ax = df.groupby('bowtie2')['urmap', 'bowtie2.1'].sum().plot(kind='bar', rot=45, figsize=(12,6), logy=True,

                                                                 title='Rate of Read Aligners')

plt.xlabel('Rate of Read Aligners')

plt.ylabel('Log')



plt.show()
ax = df.groupby('mu_count')['bowtie2', 'urmap'].sum().plot(kind='bar', rot=45, figsize=(12,6), logy=True,

                                                                 title='Read Aligners')

plt.xlabel('Read Aligners')

plt.ylabel('Log')



plt.show()
ax = df.groupby('mu')['urmap', 'bowtie2'].sum().plot(kind='barh', figsize=(14,8),

                                                                 title='', logx=True, linewidth=3)

plt.xlabel('Log')

plt.ylabel('Read Aligners Rate')

plt.show()
ax = df.groupby('bowtie2')['urmap'].mean().sort_values(ascending=True).plot(kind='barh', figsize=(20,6), color='orange',

                                                                                    title='Read Aligners')

plt.xlabel('Count')

plt.ylabel('Bowtie2')

plt.show()
fig=sns.lmplot(x='bowtie2', y="urmap",data=df)
import matplotlib.ticker as ticker

ax = sns.distplot(df['bowtie2'])

plt.xticks(rotation=45)

ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

figsize=(10, 4)
plt.style.use('fivethirtyeight')

df.plot(subplots=True, figsize=(4, 4), sharex=False, sharey=False)

plt.show()
def plot_dist_col(column):

    pos__df = df[df['bowtie2'] ==1]

    neg__df = df[df['bowtie2'] ==0]



    '''plot dist curves for train and test weather data for the given column name'''

    fig, ax = plt.subplots(figsize=(8, 8))

    sns.distplot(pos__df[column].dropna(), color='green', ax=ax).set_title(column, fontsize=16)

    sns.distplot(neg__df[column].dropna(), color='purple', ax=ax).set_title(column, fontsize=16)

    plt.xlabel(column, fontsize=15)

    plt.legend(['Bowtie2', 'Urmap'])

    plt.show()

plot_dist_col('urmap')
from scipy import stats



plt.figure(figsize=(8,6))

fig,ax = plt.subplots(2,2,figsize=(10,8))

sns.distplot(df['bowtie2'], fit = stats.norm,color='coral',ax=ax[0][0])

sns.distplot(df['mu'], fit = stats.norm,color='coral',ax=ax[0][1])

sns.distplot(df['urmap'], fit = stats.norm,color='coral',ax=ax[1][0])

sns.distplot(df['bowtie2.1'], fit = stats.norm,color='coral',ax=ax[1][1])



plt.tight_layout()

plt.show()