import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from nltk.tokenize import TweetTokenizer

import datetime

import lightgbm as lgb

from scipy import stats

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn import metrics

from wordcloud import WordCloud

from collections import Counter

import gensim

from gensim.utils import simple_preprocess

from gensim.parsing.preprocessing import STOPWORDS

from nltk.stem import WordNetLemmatizer, SnowballStemmer

#from nltk.corpus import stopwords

from nltk.util import ngrams

from nltk.stem.porter import *

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn.multiclass import OneVsRestClassifier

pd.set_option('max_colwidth',400)

pd.set_option('max_columns', 50)

import json

import altair as alt

from  altair.vega import v3

from IPython.display import HTML

import gc

import os
# Preparing altair. I use code from this great kernel: https://www.kaggle.com/notslush/altair-visualization-2018-stackoverflow-survey



vega_url = 'https://cdn.jsdelivr.net/npm/vega@' + v3.SCHEMA_VERSION

vega_lib_url = 'https://cdn.jsdelivr.net/npm/vega-lib'

vega_lite_url = 'https://cdn.jsdelivr.net/npm/vega-lite@' + alt.SCHEMA_VERSION

vega_embed_url = 'https://cdn.jsdelivr.net/npm/vega-embed@3'

noext = "?noext"



paths = {

    'vega': vega_url + noext,

    'vega-lib': vega_lib_url + noext,

    'vega-lite': vega_lite_url + noext,

    'vega-embed': vega_embed_url + noext

}



workaround = """

requirejs.config({{

    baseUrl: 'https://cdn.jsdelivr.net/npm/',

    paths: {}

}});

"""



#------------------------------------------------ Defs for future rendering

def add_autoincrement(render_func):

    # Keep track of unique <div/> IDs

    cache = {}

    def wrapped(chart, id="vega-chart", autoincrement=True):

        if autoincrement:

            if id in cache:

                counter = 1 + cache[id]

                cache[id] = counter

            else:

                cache[id] = 0

            actual_id = id if cache[id] == 0 else id + '-' + str(cache[id])

        else:

            if id not in cache:

                cache[id] = 0

            actual_id = id

        return render_func(chart, id=actual_id)

    # Cache will stay outside and 

    return wrapped

            

@add_autoincrement

def render(chart, id="vega-chart"):

    chart_str = """

    <div id="{id}"></div><script>

    require(["vega-embed"], function(vg_embed) {{

        const spec = {chart};     

        vg_embed("#{id}", spec, {{defaultStyle: true}}).catch(console.warn);

        console.log("anything?");

    }});

    console.log("really...anything?");

    </script>

    """

    return HTML(

        chart_str.format(

            id=id,

            chart=json.dumps(chart) if isinstance(chart, dict) else chart.to_json(indent=None)

        )

    )



HTML("".join((

    "<script>",

    workaround.format(json.dumps(paths)),

    "</script>",

)))
print(os.listdir("../input"))
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sub = pd.read_csv('../input/sample_submission.csv')
train.head()
test.head()
#We may need Time series analisys because train-data has created_date.

#What is "intellectual_or_learning_disability","rating","sexual_explicit","identity_annotator_count","toxicity_annotator_count","publication_id"? I have to research them.
train.shape
test.shape
train.describe()
test.describe()
train["id"].value_counts().head()
train_id = train["id"].astype(int)

plt.subplots(figsize=(10, 7)) 

sns.set_context("poster")

sns.distplot(train_id,kde =False)

plt.title('Train ID Histogram')
train["id"].min()
train["id"].max()
test_id = test["id"].astype(int)

plt.subplots(figsize=(10, 7)) 

sns.set_context("poster")

sns.distplot(test_id,kde =False)

plt.title('Test ID Histogram');
sns.set_context("poster")

train['created_date'] = pd.to_datetime(train['created_date']).values.astype('datetime64[M]')

plt.subplots(figsize=(15, 6))

ax1 = sns.lineplot(x=train['created_date'], y=train['id'],label='Train ID',color = "red");

ax2 = ax1.twinx()

sns.lineplot(x=train['created_date'], y=train['target'], label='target', ax=ax2);

h1, l1 = ax1.get_legend_handles_labels()

h2, l2 = ax2.get_legend_handles_labels()

ax1.legend(h1+h2, l1+l2, loc='lower right');
hist_df = pd.cut(train['target'], 20).value_counts().sort_index().reset_index().rename(columns={'index': 'bins'})

hist_df['bins'] = hist_df['bins'].astype(str)

render(alt.Chart(hist_df).mark_bar().encode(

    x=alt.X("bins:O", axis=alt.Axis(title='Target')),

    y=alt.Y('target:Q', axis=alt.Axis(title='Count')),

    tooltip=['target', 'bins']

).properties(title="Counts of target bins", width=1000, height=500).interactive())

print("(target=0)/all_target = ",(train['target'] >= 0).sum() / train.shape[0])

print("(target>0)/all_target = ",(train['target'] > 0).sum() / train.shape[0])

print("(target>0.25)/all_target = ",(train['target'] >= 0.25).sum() / train.shape[0])

print("(target>=0.5)/all_target ",(train['target'] >= 0.5).sum() / train.shape[0])
train['target'].value_counts().head(20)
train['comment_text'].value_counts().head(50)
test['comment_text'].value_counts().head(50)
train_plus = train

train_plus['train_comment_length'] = train.comment_text.apply(len)

test_plus = test

test_plus['test_comment_length'] = test.comment_text.apply(len)
sns.set_context("poster");

plt.figure(figsize=(15, 8));

sns.distplot(train_plus['train_comment_length'] ,kde=False, rug=False,bins=100,norm_hist=True,label = "train");

sns.distplot(test_plus['test_comment_length'] ,kde=False, rug=False,bins=100,norm_hist=True,label = "test",axlabel = "comment_length");

plt.legend();

plt.title("Histgram of train adn test comment_length with normalize");
plt.figure(figsize=(8, 8));

sns.scatterplot(x=train_plus['train_comment_length'], y=train_plus['target'],alpha = 0.5);

plt.title('Target & Train_comment_length');
print("correlation coefficients is ",np.corrcoef(train_plus['train_comment_length'],train_plus['target'])[1,0])
plt.figure(figsize=(3, 8));

sns.boxplot(y=train_plus['train_comment_length'] )

plt.title("Boxplot of train_comment_length");
#Check the Number of 75% point

q25,q50,q75= np.percentile(train_plus['train_comment_length'], [25,50,75])

iqr = q75- q25

q100 = q75 + iqr*1.5

print("Nuber of 25% point = ",q25)

print("Nuber of 50% point = ",q50)

print("Nuber of 75% point = ",q75)

print("Nuber of 100% point = ",q100)
train_plus[train_plus['train_comment_length'] > 894].head()
train_plus['comment_text'][train_plus['train_comment_length'] > 414].value_counts().head()
plt.figure(figsize=(8, 8));

sns.scatterplot(x=train_plus['train_comment_length'][train_plus['train_comment_length'] > 894], y=train_plus['target'][train_plus['train_comment_length'] > 894],alpha = 0.5);

plt.title('Target & Train_comment_length');
print("correlation coefficients is ",np.corrcoef(train_plus['train_comment_length'][train_plus['train_comment_length'] > 894],train_plus['target'][train_plus['train_comment_length'] > 894])[1,0])
train_plus['num_words'] = train['comment_text'].apply(lambda comment: len(comment.split()))
plt.figure(figsize=(8, 8));

sns.scatterplot(x=train_plus['num_words'], y=train_plus['target'],alpha = 0.5);

plt.title('Target & Number of words');
print("correlation coefficients is ",np.corrcoef(train_plus['num_words'],train_plus['target'])[1,0])
plt.figure(figsize=(3, 8));

sns.boxplot(y=train_plus['num_words'] )

plt.title("Boxplot of num_words");
#Check the Number of 100% point

q25,q50,q75= np.percentile(train_plus['num_words'], [25,50,75])

iqr = q75- q25

q100 = q75 + iqr*1.5

print("Nuber of 25% point = ",q25)

print("Nuber of 50% point = ",q50)

print("Nuber of 75% point = ",q75)

print("Nuber of 100% point = ",q100)
print("correlation coefficients is ",np.corrcoef(train_plus['num_words'][train_plus['num_words'] > 156],train_plus['target'][train_plus['num_words'] > 156])[1,0])
train_plus['capitals'] = train['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
plt.figure(figsize=(8, 8));

sns.scatterplot(x=train_plus['capitals'], y=train_plus['target'],alpha = 0.5);

plt.title('Target & Number of capitals');
print("correlation coefficients is ",np.corrcoef(train_plus['capitals'],train_plus['target'])[1,0])
plt.figure(figsize=(3, 8));

sns.boxplot(y=train_plus['capitals'] )

plt.title("Boxplot of capitals");
#Check the Number of 100% point

q25,q50,q75= np.percentile(train_plus['capitals'], [25,50,75])

iqr = q75- q25

q100 = q75 + iqr*1.5

print("Nuber of 25% point = ",q25)

print("Nuber of 50% point = ",q50)

print("Nuber of 75% point = ",q75)

print("Nuber of 100% point = ",q100)
print("correlation coefficients is ",np.corrcoef(train_plus['capitals'][train_plus['capitals'] > 23],train_plus['target'][train_plus['capitals'] > 23])[1,0])
train_plus['num_exclamation_marks'] = train['comment_text'].apply(lambda comment: comment.count('!'))
plt.figure(figsize=(8, 8));

sns.scatterplot(x=train_plus['num_exclamation_marks'], y=train_plus['target'],alpha = 0.5);

plt.title('Target & Number of num_exclamation_marks');
print("correlation coefficients is ",np.corrcoef(train_plus['num_exclamation_marks'],train_plus['target'])[1,0])
plt.figure(figsize=(3, 8));

sns.boxplot(y=train_plus['num_exclamation_marks'] )

plt.title("Boxplot of num_exclamation_marks");
#Check the Number of 100% point

q25,q50,q75= np.percentile(train_plus['num_exclamation_marks'], [25,50,75])

iqr = q75- q25

q100 = q75 + iqr*1.5

print("Nuber of 25% point = ",q25)

print("Nuber of 50% point = ",q50)

print("Nuber of 75% point = ",q75)

print("Nuber of 100% point = ",q100)
stopwords = set(STOPWORDS)



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='white',

        stopwords=stopwords,

        max_words=50,

        max_font_size=40, 

        scale=5,

        random_state=1

    ).generate(str(data))



    fig = plt.figure(1, figsize=(10,10))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()
show_wordcloud(train['comment_text'].sample(20000), title = 'Prevalent words in comments - train data')
show_wordcloud(test['comment_text'].sample(20000), title = 'Prevalent words in comments - test data')
show_wordcloud(train.loc[train['insult'] < 0.25]['comment_text'].sample(20000), 

               title = 'Prevalent comments with insult score < 0.25')
show_wordcloud(train.loc[train['insult'] > 0.75]['comment_text'].sample(20000), 

               title = 'Prevalent comments with insult score > 0.75')
show_wordcloud(train.loc[train['threat'] < 0.25]['comment_text'], 

               title = 'Prevalent words in comments with threat score < 0.25')
show_wordcloud(train.loc[train['threat'] > 0.75]['comment_text'], 

               title = 'Prevalent words in comments with threat score > 0.75')
show_wordcloud(train.loc[train['obscene']< 0.25]['comment_text'], 

               title = 'Prevalent words in comments with obscene score < 0.25')
show_wordcloud(train.loc[train['obscene'] > 0.75]['comment_text'], 

               title = 'Prevalent words in comments with obscene score > 0.75')
show_wordcloud(train.loc[train['target'] < 0.25]['comment_text'], 

               title = 'Prevalent words in comments with target score < 0.25')
show_wordcloud(train.loc[train['target'] > 0.75]['comment_text'], 

               title = 'Prevalent words in comments with target score > 0.75')