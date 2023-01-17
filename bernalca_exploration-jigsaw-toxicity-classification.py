# Exploration derived from https://www.kaggle.com/kabure/simple-eda-hard-views-w-easy-code



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Standard plotly imports

import plotly.offline as py 

import plotly.graph_objs as go

import plotly.tools as tls

from plotly.offline import iplot, init_notebook_mode

import cufflinks



# Using plotly + cufflinks in offline mode

init_notebook_mode(connected=True)

cufflinks.go_offline(connected=True)



import warnings

warnings.filterwarnings("ignore")
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.sort_values(by=['severe_toxicity'])
df_train.isnull().sum() / len(df_train) * 100
count_target_zero = round(df_train[df_train['target'] == 0]['target'].count() / len(df_train['target']) * 100,2)

print(f'Total of zero values in Toxic rate: {count_target_zero}%')
df_train.info()
df_train.shape
plt.figure(figsize=(13,6))



# Plot samples with > 0 toxicity 

g = sns.distplot(df_train[df_train['target'] > 0]['target'])

plt.title('Toxic Distribuition', fontsize=22)

plt.xlabel("Toxic Rate", fontsize=18)

plt.ylabel("Distribuition", fontsize=18) 



plt.show()
comment_adj = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']



plt.figure(figsize=(14,6))



for col in comment_adj[1:]:

    g = sns.distplot(df_train[df_train[col] > 0][col], label=col, hist=False)

    plt.xlabel("Rate", fontsize=18)

    plt.ylabel("Distribuition", fontsize=18)

    plt.legend(loc=1, prop={'size': 14})



plt.show()
# Label samples Toxic vs Non-Toxic relative to a target toxicity threshold of 0.5

df_train['toxic'] = np.where(df_train['target'] >= .5, 'Toxic', 'Non-Toxic')



# Plot

df_train['toxic'].value_counts().iplot(kind='bar', xTitle='Toxic or Non-Toxic', yTitle="Count", 

                                       title='Distribuition of Toxicity of comments')