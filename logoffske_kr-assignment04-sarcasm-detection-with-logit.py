!cd ../input/sarcasm
# some necessary imports

import os

import numpy as np

import pandas as pd

import json

import string



from IPython.display import Image



from nltk.util import ngrams

import re



from scipy.sparse import hstack

from sklearn import preprocessing, metrics, ensemble, naive_bayes, linear_model, model_selection

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix



import seaborn as sns

from matplotlib import pyplot as plt



from wordcloud import WordCloud, STOPWORDS

from collections import defaultdict



import lightgbm as lgb



import warnings

warnings.filterwarnings('ignore')



color = sns.color_palette()



%matplotlib inline

pd.options.mode.chained_assignment = None

pd.options.display.max_rows = 100
train_df = pd.read_csv('../input/sarcasm/train-balanced-sarcasm.csv')
train_df.head()
train_df.info()
train_df.dropna(subset=['comment'], inplace=True)
train_df.info()
train_df['label'].value_counts()
training_df, testing_df = train_test_split(train_df, random_state=17)
train_texts = training_df['comment'] 

valid_texts = testing_df['comment']

y_train = training_df['label']

y_valid = testing_df['label']
def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), 

                   title = None, title_size=40, image_color=False):

    stopwords = set(STOPWORDS)

    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}

    stopwords = stopwords.union(more_stopwords)



    wordcloud = WordCloud(background_color='black',

                    stopwords = stopwords,

                    max_words = max_words,

                    max_font_size = max_font_size, 

                    random_state = 42,

                    width=800, 

                    height=400,

                    mask = mask)

    wordcloud.generate(str(text))

    

    plt.figure(figsize=figure_size)

    if image_color:

        image_colors = ImageColorGenerator(mask);

        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");

        plt.title(title, fontdict={'size': title_size,  

                                  'verticalalignment': 'bottom'})

    else:

        plt.imshow(wordcloud);

        plt.title(title, fontdict={'size': title_size, 'color': 'black', 

                                  'verticalalignment': 'bottom'})

    plt.axis('off');

    plt.tight_layout()  

    

plot_wordcloud(train_df["comment"], max_words=800, title="Word Cloud of Comments")
train1_df = train_df[train_df["label"]==1]

train0_df = train_df[train_df["label"]==0]



def generate_ngrams(s, n_gram:int):

    #Generate a list of n-grams from the input data 

    

    s = s.lower()

    s = re.sub(r'[0-9-,.$"!?+\s]', ' ', s)

    tokens = [token for token in s.split(" ") if token != "" if token not in STOPWORDS if len(token) > 1]

    output = list(ngrams(tokens, n_gram))

    

    #clean duplicate n-grams in case dimension is > 1 e,g: (fake, fake)

    if n_gram > 1:

        full_output = [i for i in output if i[0] != i[1]]

    else:

        return output

    return full_output



def count_ngrams(train_df, n_gram:int):

    #We count the n-grams repetitions in specific feature

    #train_df - DataFrame

    

    freq_dict = defaultdict(int)

    for sent in train_df['comment']:

        for word in generate_ngrams(sent, n_gram):

            freq_dict[word] += 1

    

    #check commutations (donald, trump) <-> (trump, donald) if n-gram > 1

    final_dict = defaultdict(int)

    if n_gram > 1:

        for key, value in freq_dict.items():

            if (key[1], key[0]) in final_dict.keys() or (key[0], key[1]) in final_dict.keys():

                pass

            else:

                final_dict[key] = value 

    else:

        final_dict = freq_dict

    

    fd_sorted = pd.DataFrame(sorted(final_dict.items(), key=lambda x: x[1])[::-1])

    fd_sorted.columns = ["word", "wordcount"]

    return fd_sorted



def ngram_compare_plot(x1, x2, y1, y2, barcolor:str):

    #Create a plot to compare words used in sarcastic comments and non-saracstic comments

    

    fig, axes = plt.subplots(ncols=2, figsize=(16,26), sharey=False)

    

    axes[0].xaxis.tick_top()   

    axes[0].invert_yaxis()

    axes[0].patch.set_facecolor('black')

    axes[0].barh([' '.join(list(y)) for y in y1], x1, align = 'center', zorder=10, color = barcolor)

    axes[0].set(title='Frequent words in sarcastic comments')

    

    axes[1].xaxis.tick_top() 

    axes[1].invert_yaxis()

    axes[1].patch.set_facecolor('black')

    axes[1].barh([' '.join(list(y)) for y in y2], x2, align = 'center', zorder=10, color = barcolor)

    axes[1].set(title='Frequent words in non-sarcastic comments')

 

    

    for ax in axes.flat:

        ax.margins(0.01)

        ax.grid(True)

    

    plt.show()
train_ngrams1_sarcasm = count_ngrams(train1_df, n_gram=1)

train_ngrams1_nosarcasm = count_ngrams(train0_df, n_gram=1)
ngram_compare_plot(list(train_ngrams1_sarcasm['wordcount'][:50].values), \

                   list(train_ngrams1_nosarcasm['wordcount'][:50].values), \

                   list(train_ngrams1_sarcasm['word'][:50].values), \

                   list(train_ngrams1_nosarcasm['word'][:50].values), 'teal')
train_ngrams2_sarcasm = count_ngrams(train1_df, n_gram=2)

train_ngrams2_nosarcasm = count_ngrams(train0_df, n_gram=2)
ngram_compare_plot(list(train_ngrams2_sarcasm['wordcount'][:50].values), \

                   list(train_ngrams2_nosarcasm['wordcount'][:50].values), \

                   list(train_ngrams2_sarcasm['word'][:50].values), \

                   list(train_ngrams2_nosarcasm['word'][:50].values), 'salmon')
train_df['has_downvote'] = train_df['downs'].apply(lambda x: 'Yes' if x == -1 else 'No')
plt.figure(figsize=(15,8))

scores = sns.countplot(x='has_downvote', hue='label', data=train_df)



plt.legend(title='Downvote influence on label', loc='upper right', labels=['Not sarcasm', 'Sarcasm'])

plt.xlabel('Has a downvote')

plt.show(scores)
train_df['score_negative'] = train_df['score'].apply(lambda x: 'Yes' if x < 0 else 'No')
plt.figure(figsize=(15,8))

scores = sns.countplot(x='score_negative', hue='label', data=train_df)



plt.legend(title='Score influence on label', loc='upper right', labels=['Not sarcasm', 'Sarcasm'])

plt.xlabel('Is the comment score negative?')

plt.show(scores)
train_df['downvotes_more'] = (train_df['downs'].abs() >= train_df['ups'].abs())
train_df['downvotes_more'] = train_df['downvotes_more'].apply(lambda x: 'Yes' if x == True else 'No')
plt.figure(figsize=(15,8))

upvotes = sns.countplot(x='downvotes_more', hue='label', data=train_df)





plt.legend(title='Downvotes influence on label', loc='upper left', labels=['Not sarcasm', 'Sarcasm'])

plt.xlabel('More downvotes than upvotes')

plt.show(scores)
train_df['upvotes_more3'] = train_df['ups'].apply(lambda x: 'Yes' if x > 0 else 'No')
plt.figure(figsize=(15,8))

scores = sns.countplot(x='upvotes_more3', hue='label', data=train_df)



plt.legend(title='Upvote influence on label', loc='upper right', labels=['Not sarcasm', 'Sarcasm'])

plt.xlabel('More than 3 upvotes')

plt.show(scores)
train_df['score_negative'] = train_df['score'].apply(lambda x: '1' if x < 0 else '0')

training_df['score_negative'] = training_df['score'].apply(lambda x: '1' if x < 0 else '0')

testing_df['score_negative'] = testing_df['score'].apply(lambda x: '1' if x < 0 else '0')
train_df.head()
train_df['author'].value_counts()
train_df[train_df['author'] == 'Biffingston']['label'].value_counts()
mlist = ['author', 'score', 'ups', 'downs', 'date', \

         'created_utc', 'downvotes_more', 'upvotes_more3', 'has_downvote', 'parent_comment']

train_df.drop(mlist, 1, inplace=True)
train_df.head()
def text_format(s):

    s = s.lower()

    s = re.sub(r'[0-9-,.$"!?+\s]', ' ', s)

    output = str(s)

    return output
train_texts.apply(lambda x: text_format(x));

valid_texts.apply(lambda x: text_format(x));
misc = ['author', 'score', 'ups', 'downs', 'date', 'created_utc']

training_df.drop(misc, 1, inplace=True)

testing_df.drop(misc, 1, inplace=True)

score_negative = train_df['score_negative']

score_negative_train, score_negative_test = train_test_split(score_negative, random_state=17) 
score_negative.shape, score_negative_test.shape, score_negative_train.shape
subreddits = train_df['subreddit']

train_subreddits, valid_subreddits = train_test_split(subreddits, random_state=17)
tf_idf_texts  = TfidfVectorizer(max_features=50000, min_df=2, ngram_range=(1,2))

tf_idf_subreddits = TfidfVectorizer(ngram_range=(1, 1))
%%time

X_train_texts = tf_idf_texts.fit_transform(train_texts)

X_valid_texts = tf_idf_texts.transform(valid_texts)
X_train_texts.shape, X_valid_texts.shape
%%time

X_train_subreddits = tf_idf_subreddits.fit_transform(train_subreddits)

X_valid_subreddits = tf_idf_subreddits.transform(valid_subreddits)
X_train = hstack([X_train_texts, X_train_subreddits])

X_valid = hstack([X_valid_texts, X_valid_subreddits])
logit = LogisticRegression(C=1, n_jobs=4, solver='lbfgs', 

                           random_state=17, verbose=1)
score_train_reshaped = score_negative_train.values.reshape((len(score_negative_train.values), 1))

score_test_reshaped = score_negative_test.values.reshape((len(score_negative_test.values), 1))
X_training = hstack([X_train, score_train_reshaped.astype(float)])

X_validation = hstack([X_valid, score_test_reshaped.astype(float)])
logit.fit(X_training, y_train)
%%time

valid_pred = logit.predict(X_validation)
accuracy_score(y_valid, valid_pred)
logit.fit(X_train, y_train)
%%time

valid_pred = logit.predict(X_valid)
accuracy_score(y_valid, valid_pred)