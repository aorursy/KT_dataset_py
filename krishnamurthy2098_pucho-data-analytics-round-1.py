# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import cufflinks as cf
import seaborn as sns
import nltk
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
from nltk.tokenize import word_tokenize
from nltk.corpus import opinion_lexicon
import matplotlib.pyplot as plt
%matplotlib inline
init_notebook_mode(connected=True)
cf.go_offline()
#print(os.listdir("../input"))
data_folder = '../input'
# prepare for calculate word frequencies
positive = set(opinion_lexicon.positive())
negative = set(opinion_lexicon.negative())
fileNameList = [os.path.join(data_folder, fn) for fn in os.listdir(data_folder)]
#print(len(fileNameList))
df = pd.DataFrame(fileNameList, columns=['path'])
df['file'] = df.path.apply(lambda x: x.split('/')[-1])
df['year'] = df.file.apply(lambda x: x.split('_')[-1].split('.')[0]).astype(int)
df['president'] = df.file.apply(lambda x: x.split('_')[0])
df['tokens'] = df.path.apply(lambda x: word_tokenize(open(x).read()))
df['positive'] = df.tokens.apply(lambda x: len(positive.intersection(x)))
df['negative'] = df.tokens.apply(lambda x: len(negative.intersection(x)))
df['sentiment'] = df.positive - df.negative
df = df[['file', 'positive', 'negative', 'sentiment', 'year', 'president']]
df.loc[(df.year >= 2000) & (df.president == 'Bush'), 'president'] = 'Bush Sr.'
df.sort_values(by='year', inplace=True)
df.describe()


#print(df)


# Any results you write to the current directory are saved as output.
df.iloc[0,]
df.iplot(kind='bar', x='president', y='sentiment',color='blue')
df.iplot(kind='box')
df.iplot(kind='surface',colorscale='rdylbu')
df.iplot(kind='hist',bins=50)

b1=df.iplot(kind='bubble',x='positive',y='negative', size='sentiment')

a1=df['sentiment'].hist(bins=10)
a1.set_xlabel("positive")
a1.set_ylabel("sentiment")
a2=df.plot.bar(stacked=True,figsize=(12,3))
a2.set_ylabel("year")
a2.set_xlabel("sentiment")
a3=df.plot.line(x='president',y='sentiment',figsize=(12,3),lw=7)
a3.set_ylabel("sentiment")
sns.jointplot(x='positive',y='sentiment',data=df,kind='reg',color='orange')
sns.jointplot(x='negative',y='sentiment',data=df,kind='hex')
sns.jointplot(x='positive',y='sentiment',data=df,kind='kde',color='red')
sns.pairplot(df,hue='president',palette='magma')
sns.boxplot(x='president',y='sentiment',data=df)
sns.violinplot(x='president',y='sentiment',data=df,split=True)
df['party'] = df.president.apply(lambda x: 'D' if x in ['Clinton', 'Obama'] else 'R')
g = sns.boxplot(x="party", y="sentiment", hue="party", data=df)
sns.lmplot(x="year", y="sentiment", hue="president", truncate=True, size=5, data=df)
