# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px

import nltk

import string

from nltk.corpus import stopwords

import re



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/hackathon/task_2-BCG_world_atlas_data-bcg_strain-7July2020.csv', encoding='utf8')

df.head()
df.isnull().sum()
# categorical features with missing values

categorical_nan = [feature for feature in df.columns if df[feature].isna().sum()>1 and df[feature].dtypes=='O']

print(categorical_nan)
# replacing missing values in categorical features

for feature in categorical_nan:

    df[feature] = df[feature].fillna('None')
df[categorical_nan].isna().sum()
# Lets first handle numerical features with nan value

#numerical_nan = [feature for feature in df.columns if df[feature].isna().sum()>1 and df[feature].dtypes!='O']

#numerical_nan
#df[numerical_nan].isna().sum()
## Replacing the numerical Missing Values



#for feature in numerical_nan:

    ## We will replace by using median since there are outliers

   # median_value=df[feature].median()

    

   # df[feature].fillna(median_value,inplace=True)

    

#df[numerical_nan].isnull().sum()
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='black',

        stopwords=stopwords,

        max_words=200,

        colormap='Set2',

        max_font_size=40, 

        scale=3,

        random_state=1 # chosen at random by flipping a coin; it was heads

).generate(str(data))



    fig = plt.figure(1, figsize=(15, 15))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()



show_wordcloud(df['bcg_strain_original'])
cnt_srs = df['vaccination_timing'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Blues',

        reversescale = True

    ),

)



layout = dict(

    title='Vaccination Timing Distribution',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="vaccination_timing")
df['vaccination_timing_length']=df['vaccination_timing'].apply(len)
sns.set(font_scale=2.0)



g = sns.FacetGrid(df,col='bcg_strain_id',height=5)

g.map(plt.hist,'vaccination_timing_length')
plt.figure(figsize=(10,8))

ax=sns.countplot(df['bcg_strain_t_cell_grp_3'])

ax.set_xlabel(xlabel="bcg_strain_t_cell_grp_3",fontsize=17)

ax.set_ylabel(ylabel='count',fontsize=17)

ax.axes.set_title('BCG Strain Lymphocyte Group 3',fontsize=17)

ax.tick_params(labelsize=13)
sns.set(font_scale=1.4)

plt.figure(figsize = (10,5))

sns.heatmap(df.corr(),cmap='summer',annot=True,linewidths=.5)
from sklearn.model_selection import cross_val_score

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer



all_text=df['vaccination_timing']

train_text=df['vaccination_timing']

y=df['is_bcg_mandatory_for_all_children']
word_vectorizer = TfidfVectorizer(

    sublinear_tf=True,

    strip_accents='unicode',

    analyzer='word',

    token_pattern=r'\w{1,}',

    stop_words='english',

    ngram_range=(1, 1),

    max_features=10000)

word_vectorizer.fit(all_text)

train_word_features = word_vectorizer.transform(train_text)
char_vectorizer = TfidfVectorizer(

    sublinear_tf=True,

    strip_accents='unicode',

    analyzer='char',

    stop_words='english',

    ngram_range=(2, 6),

    max_features=50000)

char_vectorizer.fit(all_text)

train_char_features = char_vectorizer.transform(train_text)



train_features = hstack([train_char_features, train_word_features])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_features, y,test_size=0.3,random_state=101)
import xgboost as xgb

xgb=xgb.XGBClassifier()

##xgb.fit(X_train,y_train)