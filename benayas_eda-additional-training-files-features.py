# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import warnings

warnings.filterwarnings("ignore")

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats.stats import pearsonr

%matplotlib inline

sns.set()

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

PATH = '/kaggle/input/jigsaw-multilingual-toxic-comment-classification/'
# Function to get a summary table for numeric columns and another one for object columns

def eda(df): 

    eda = df.describe().T

    eda['null_sum'] = df.isnull().sum()

    eda['null_pct'] = df.isnull().mean()

    eda['dtypes'] = df.dtypes

    

    objects = df[[ x for x in df.columns if not x in eda.index]]

    eda_objects = objects.describe().T

    eda_objects['null_sum'] = df.isnull().sum()

    eda_objects['null_pct'] = df.isnull().mean()

    eda_objects['dtypes'] = df.dtypes

    return eda, eda_objects
train1 = pd.read_csv(PATH+'jigsaw-unintended-bias-train.csv')

train1.head()
train1_eda, train1_eda_objects = eda(train1)

train1_eda
train1_eda_objects
fig, ax = plt.subplots(figsize=(10,6), nrows=1, ncols=2)

fig.suptitle("Distribution of Target Variable", size=25)

sns.distplot(train1['toxic'], kde=False, bins=20, ax=ax[0])

ax[0].set(xlabel='Distribution')

sns.distplot(train1['toxic'], kde=False, bins=[0,0.5,1], ax=ax[1])

ax[1].set(xlabel='Treshold = 0.5')

plt.show()
# Features 

toxic_ratios = ['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'sexual_explicit']

fig = plt.figure(figsize=(8,8))

train1_ratios = train1[toxic_ratios]

sns.pairplot(train1_ratios)

plt.show()
fig, ax = plt.subplots(figsize=(15,10), nrows=2,ncols=3)

for i,t in enumerate(toxic_ratios):

    r,c = int(i/3),int(i%3)

    sns.scatterplot(x=t, y="toxic", data=train1, ax=ax[r][c])

    ax[r][c].set(xlabel=t)

    ax[r][c].plot([0,1], color='red')

plt.show()
exceptions = []

for i,t in enumerate(toxic_ratios):

    c = len(train1[train1[t]>train1['toxic']])

    exceptions.append({'feature':t,'count': c, 'pct': c/len(train1)})

pd.DataFrame(exceptions).set_index('feature')
train1['ratios'] = train1[toxic_ratios].sum(axis=1)

fig = plt.figure(figsize=(8,8))

sns.scatterplot(x='ratios', y="toxic", data=train1)

plt.show()
feature_ratios = list(train1_eda[train1_eda['null_sum']>1000000].index) 

train1_ratios = train1[feature_ratios + ['toxic']].dropna()

train1_ratios.head()
fig, ax = plt.subplots(figsize=(21,14), nrows=4,ncols=6)

for i,t in enumerate(feature_ratios):

    r,c = int(i/6),int(i%6)

    sns.scatterplot(x=t, y="toxic", data=train1_ratios, ax=ax[r][c])

    ax[r][c].set(xlabel=t)

    ax[r][c].plot([0,1], color='red')

    plt.subplots_adjust(hspace=0.5, wspace= 0.5)

plt.show()
correlations = train1_ratios.corrwith(train1_ratios['toxic']).iloc[:-1].to_frame()

sorted_correlations = correlations[0].sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(5,10))

sns.heatmap(sorted_correlations.to_frame(), cmap='coolwarm', annot=True, vmin=-1, vmax=1, ax=ax)
correlations = []

for i,t in enumerate(feature_ratios):

    r,c = int(i/6),int(i%6)

    corr = {'feature':t}

    corr['original'] = pearsonr(train1_ratios['toxic'], train1_ratios[t])[0]

    df = train1_ratios[train1_ratios[t]>0]

    corr['filtered'] = pearsonr(df['toxic'], df[t])[0]

    correlations.append(corr)

    

correlations = pd.DataFrame(correlations).set_index('feature')

correlations['original'] = correlations['original']

correlations['filtered'] = correlations['filtered']

correlations
fig = plt.figure(figsize=(8,10))

fig.suptitle('Pearson Correlation vs target BEFORE vs AFTER filtering zeros', size=25)

for t in correlations.index:

    plt.plot([correlations.loc[t,'original'],correlations.loc[t,'filtered']], label=t)

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()
ids = ['id','publication_id', 'parent_id', 'article_id']

train1[ids].nunique()
plt.figure(figsize=(50,10)) # adjust the fig size to see everything

sns.barplot(x=train1['publication_id'].value_counts().index, y=train1['publication_id'].value_counts())

plt.show()
reactions = ['funny', 'wow', 'sad', 'likes' , 'disagree']

train1_reaction = train1[reactions]

train1_reaction['nreactions'] = train1_reaction.sum(axis=1)

n = len(train1_reaction[train1_reaction['nreactions']==0])

print('Number of comments without reaction: {}'.format(n))

print('Pctg of comments without reaction: {s:.3f}'.format(s=n/len(train1)))
train1_reaction = train1[reactions]

train1_reaction['nreactions'] = train1_reaction.sum(axis=1)

sns.pairplot(train1_reaction[train1_reaction['nreactions']!=0].drop('nreactions',axis=1))

plt.show()
train1_reaction = train1[reactions+['toxic']]

train1_reaction['nreactions'] = train1_reaction.drop('toxic',axis=1).sum(axis=1)

train1_reaction = train1_reaction[train1_reaction['nreactions']>0].drop('nreactions', axis=1)

train1_reaction['reaction'] = train1_reaction.drop('toxic',axis=1).apply(lambda x: x.argmax(), axis=1 )

train1_reaction['toxic_tr'] = train1_reaction['toxic'].apply(lambda x: int(1) if x>=0.5 else int(0) )

grouped = train1_reaction[['reaction','toxic_tr', 'wow']].groupby(['reaction', 'toxic_tr']).count().reset_index(drop=False).pivot(index='reaction', columns='toxic_tr', values='wow')

grouped['sum'] = grouped.sum(axis=1)

grouped[0] = 100*grouped[0]/grouped['sum']

grouped[1] = 100*grouped[1]/grouped['sum']

grouped = grouped.drop('sum', axis=1)

grouped
fig = plt.figure()

cm = plt.get_cmap('viridis')

ax = fig.add_axes([0,0,1,1])

ax.set_title('$P( toxic | reaction=x)$', size=23)

colors = [ cm(i/(len(grouped.index))) for i in range(len(grouped.index))]

labels = grouped.index

values = grouped[1]

rects = ax.bar(labels, values, color=colors)

for p in rects:

    ax.text( p.get_x() + p.get_width() / 2., p.get_height()* 1.05, s=str('{0:.2f}'.format(p.get_height())), ha = 'center', va = 'center')

plt.show()
train1['rating'].value_counts()
train1['rating_binary'] = train1['rating'].apply(lambda x: 1 if x == 'approved' else 0)

sns.boxplot(train1['rating_binary'], train1['toxic']).set_title('Toxic comments by Rating')

plt.show()
print("Pearson correlation between rating and target variable {s:.2f}".format(s=pearsonr(train1['rating_binary'], train1['toxic'])[0]))
train1['created_date_date'] = pd.to_datetime(train1['created_date']).dt.date

grouped = train1.groupby('created_date_date').count()[['id']]

fig = plt.figure(figsize=(20,5))

ax = sns.lineplot(x=grouped.index, y= grouped.id)

ax.set_title('Number of comments by date', size=23)

plt.show()
train1['created_date_date'] = pd.to_datetime(train1['created_date']).dt.date

grouped = train1[['created_date_date','toxic']].groupby('created_date_date').mean()

fig = plt.figure(figsize=(20,6))

ax = sns.lineplot(x=grouped.index, y= grouped.toxic)

ax.set_title('Average Toxic by Date', size=23)

plt.show()
from statsmodels.tsa.seasonal import seasonal_decompose

from pylab import rcParams

rcParams['figure.figsize'] = 20, 8

result = seasonal_decompose(grouped, model='additive', freq=1)

fig = result.plot()

plt.show()
from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.stattools import kpss



def adf_test(timeseries):

    print ('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

       dfoutput['Critical Value (%s)'%key] = value

    print (dfoutput)

    

def kpss_test(timeseries):

    print ('Results of KPSS Test:')

    kpsstest = kpss(timeseries, regression='c', nlags=None)

    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])

    for key,value in kpsstest[3].items():

        kpss_output['Critical Value (%s)'%key] = value

    print (kpss_output)

    

adf_test(grouped)

print('*'*20)

kpss_test(grouped)
grouped['toxic_diff'] = grouped['toxic'] - grouped['toxic'].shift(1)

grouped['toxic_diff'].dropna().plot(figsize=(12,8))
adf_test(grouped[['toxic_diff']].dropna())

print('*'*20)

kpss_test(grouped[['toxic_diff']].dropna())
train2 = pd.read_csv(PATH+'jigsaw-toxic-comment-train.csv')

train2.head()
train2_eda, train2_eda_objects = eda(train2)

train2_eda
train2_eda_objects
toxic_ratios = ['severe_toxic', 'obscene', 'identity_hate', 'insult', 'threat']

fig, ax = plt.subplots(figsize=(25,3), nrows=1,ncols=5)

for i,t in enumerate(toxic_ratios):

    df = train2[['toxic',t,'id']].groupby(['toxic',t]).count().reset_index()

    df = df.pivot(index='toxic', columns=t, values='id')

    sns.heatmap( data=df, ax=ax[i], annot=True, fmt='.0f')

    ax[i].set(xlabel=t)

    plt.subplots_adjust( wspace= 0.5)

plt.show()
correlations = []

for i,t in enumerate(toxic_ratios):

    corr = {'feature':t}

    corr['correlation'] = pearsonr(train2['toxic'], train2[t])[0]

    df = train2[train2[t]>0]

    correlations.append(corr)

    

correlations = pd.DataFrame(correlations).set_index('feature')

correlations