# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import warnings

warnings.filterwarnings("ignore")
plt.rcParams['figure.figsize'] = (12,7)
df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

df.head()
sns.countplot(df['target'])

sns.despine()
df['keyword'].value_counts().head(20).plot.barh()

sns.despine()
df['location'].value_counts().head(20).plot.barh()

sns.despine()
fig, ax = plt.subplots(figsize=(12,7))

for label, group in df.groupby('target'):

    sns.distplot(group['text'].str.len(), label=str(label), ax=ax)

plt.xlabel('# of characters')

plt.ylabel('density')

plt.legend()

sns.despine()
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.pipeline import make_pipeline

from umap import UMAP



features = []

for i in range(1, 11):

    X_dim = CountVectorizer(min_df=i, stop_words='english').fit_transform(df['text'])

    features.append(X_dim.shape[1])

plt.plot(range(1, 11), features)

plt.xlabel('min df')

plt.ylabel('# of features')

sns.despine()
from sklearn.decomposition import TruncatedSVD

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.pipeline import make_pipeline

from umap import UMAP



dim_red = make_pipeline(

    CountVectorizer(min_df=2, stop_words='english'),

    UMAP()

)

X_dim = dim_red.fit_transform(df['text'])
%%capture

!pip install altair notebook vega # needs internet in settings (right panel)

import altair as alt

alt.renderers.enable('kaggle')
alt.Chart(pd.DataFrame({

    'x0': X_dim[:,0],

    'x1': X_dim[:,1],

    'text': df['text'],

    'keyword': df['keyword'],

    'location': df['location'],

    'target': df['target']

}).sample(5000, random_state=42)).mark_point().encode(

    x='x0',

    y='x1',

    color='target:N',

    tooltip='keyword'

).properties(

    title='Based on text',

    width=500,

    height=500

).interactive()
from sklearn.preprocessing import OneHotEncoder



enc = make_pipeline(

    OneHotEncoder(),

    UMAP(metric='cosine', random_state=42)

)

X_onehot = enc.fit_transform(df[['keyword']].fillna(''))
alt.Chart(pd.DataFrame({

    'x0': X_onehot[:,0],

    'x1': X_onehot[:,1],

    'text': df['text'],

    'keyword': df['keyword'],

    'location': df['location'],

    'target': df['target']

}).sample(5000, random_state=42)).mark_point().encode(

    x='x0',

    y='x1',

    color='target:N',

    tooltip='text'

).properties(

    title='Based on keywords',

    width=500,

    height=500

).interactive()
keywords = df.groupby('keyword').agg({

    'target': 'mean'

})
keywords['target'].sort_values(ascending=False).head(10).plot.barh()

plt.xlabel('p(target=1)')

sns.despine()
keywords['target'].sort_values().head(10).plot.barh()

plt.xlabel('p(target=1)')

sns.despine()
for index, row in df[df['keyword'] == 'body%20bags'].sample(10).iterrows():

    print('Label: {} | {}'.format(row.target, row.text))
keywords.query('target > .45 and target < .55').sort_values('target').plot.barh()

plt.xlabel('p(target=1)')

sns.despine()
for index, row in df[df['keyword'] == 'hail'].sample(10).iterrows():

    print('Label: {} | {}'.format(row.target, row.text))
for index, row in df[df['keyword'] == 'bombed'].sample(10).iterrows():

    print('Label: {} | {}'.format(row.target, row.text))