# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import json



from IPython.display import display



#local script

from tfutils_py import get_answer, read_sample



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

%matplotlib inline
df = read_sample(n=3)

df.head()
df = read_sample(n=10)

df.loc[0,'document_text'][:50]
df = read_sample(n=1000)

doc_text_words = df['document_text'].apply(lambda x: len(x.split(' ')))

plt.figure(figsize=(12,6))

sns.distplot(doc_text_words.values,kde=True,hist=False).set_title('Distribution of text word count of 1000 docs')
df = read_sample(n=3)

df.long_answer_candidates[0][:5]
# sample answer

sample = df.iloc[0]

get_answer(sample.document_text, sample.long_answer_candidates[0])
def preprocess(n=10):

    df = read_sample(n=n,ignore_doc_text=True)

    df['yes_no'] = df.annotations.apply(lambda x: x[0]['yes_no_answer'])

    df['long'] = df.annotations.apply(lambda x: [x[0]['long_answer']['start_token'], x[0]['long_answer']['end_token']])

    df['short'] = df.annotations.apply(lambda x: x[0]['short_answers'])

    return df

df = preprocess(5000)
display(df.long.apply(lambda x: "Answer Doesn't exist" if x[0] == -1 else "Answer Exists").value_counts(normalize=True))
# let's keep a mask of the answers that exist

mask_answer_exists = df.long.apply(lambda x: "Answer Doesn't exist" if x == -1 else "Answer Exists") == "Answer Exists"
yes_no_dist = df[mask_answer_exists].yes_no.value_counts(normalize=True)

yes_no_dist
short_dist = df[mask_answer_exists].short.apply(lambda x: "Short answer exists" if len(x) > 0 else "Short answer doesn't exist").value_counts(normalize=True)

plt.figure(figsize=(8,6))

sns.barplot(x=short_dist.index,y=short_dist.values).set_title("Distribution of short answers in answerable questions")
short_size_dist = df[mask_answer_exists].short.apply(len).value_counts(normalize=True)

short_size_dist_pretty = pd.concat([short_size_dist.loc[[0,1,],], pd.Series(short_size_dist.loc[2:].sum(),index=['>=2'])])

short_size_dist_pretty = short_size_dist_pretty.rename(index={0: 'No Short answer',1:"1 Short answer",">=2":"More than 1 short answers"})

plt.figure(figsize=(12,6))

sns.barplot(x=short_size_dist_pretty.index,y=short_size_dist_pretty.values).set_title("Distribution of Number of Short Answers in answerable questions")