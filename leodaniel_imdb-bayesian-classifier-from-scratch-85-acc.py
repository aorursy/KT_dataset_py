# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



pd.options.display.max_colwidth = 150
# graphics imports

import plotly

import plotly.graph_objs as go

import matplotlib.pyplot as plt

from wordcloud import WordCloud,STOPWORDS



# Natural language tool kits

import nltk

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

# download stopwords

nltk.download('stopwords')



# string operations

import string 

import re



# general imports

import math
# load data

df = pd.read_csv('/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')

df.head()


lens = df['review'].str.len()



fig = go.Figure()

fig.add_trace(

    go.Histogram(x=lens, xbins=dict(size=200))

    )

fig.update_layout(title='Length of reviews', 

                    xaxis_title="Length",

                    yaxis_title="# of reviews")

plotly.offline.iplot(fig)
poslens = df[df['sentiment']=='positive']['review'].str.len()

neglens = df[df['sentiment']=='negative']['review'].str.len()

fig = go.Figure()

fig.add_trace(

    go.Histogram(x=poslens, xbins=dict(size=200), name='positive'),

    )

fig.add_trace(

    go.Histogram(x=neglens, xbins=dict(size=200), name='negative'),

    )

fig.update_layout(title='Length of reviews', 

                    xaxis_title="Length",

                    yaxis_title="# of reviews",)

plotly.offline.iplot(fig)
df_pos = df[df['sentiment']=='positive']['review']



wordcloud1 = WordCloud(stopwords=STOPWORDS,

                      background_color='white',

                      width=2500,

                      height=2000

                      ).generate(" ".join(df_pos))



plt.figure(1,figsize=(15, 15))

plt.imshow(wordcloud1)

plt.axis('off')

plt.show()
df_neg = df[df['sentiment']=='negative']['review']



wordcloud1 = WordCloud(stopwords=STOPWORDS,

                      background_color='white',

                      width=2500,

                      height=2000

                      ).generate(" ".join(df_neg))



plt.figure(1,figsize=(15, 15))

plt.imshow(wordcloud1)

plt.axis('off')

plt.show()
# the text mode is enough...

df['sentiment'].value_counts()
# show the reviews again... 

df[['review']].head(20)
df['review_lw'] = df['review'].str.lower()

df[['review','review_lw']].head(10)
sw = stopwords.words('english')



print(f'Stopwords sample: {sw[0:10]}')

print(f'Number of stopwords: {len(sw)}')
print(f'Punctuation {string.punctuation}')
def transform_text(s):

    

    # remove html

    html=re.compile(r'<.*?>')

    s = html.sub(r'',s)

    

    # remove numbers

    s = re.sub(r'\d+', '', s)

    

    # remove punctuation

    # remove stopwords

    tokens = nltk.word_tokenize(s)

    

    new_string = []

    for w in tokens:

        # remove words with len = 2 AND stopwords

        if len(w) > 2 and w not in sw:

            new_string.append(w)

    

    

    

    s = ' '.join(new_string)

    s = s.strip()



    exclude = set(string.punctuation)

    s = ''.join(ch for ch in s if ch not in exclude)

    

    return s.strip()
transform_text('there is a tree near <br/> the river 123! see')
df['review_sw'] = df['review_lw'].apply(transform_text)

df[['review','review_lw', 'review_sw']].head(20)
lemmatizer = WordNetLemmatizer() 



print(lemmatizer.lemmatize("rocks", pos="v"))

print(lemmatizer.lemmatize("gone", pos="v"))
def lemmatizer_text(s):

    tokens = nltk.word_tokenize(s)

    

    new_string = []

    for w in tokens:

        lem = lemmatizer.lemmatize(w, pos="v")

        # exclude if lenght of lemma is smaller than 2

        if len(lem) > 2:

            new_string.append(lem)

    

    s = ' '.join(new_string)

    return s.strip()
df['review_lm'] = df['review_sw'].apply(lemmatizer_text)

df[['review','review_lw', 'review_sw', 'review_lm']].head(20)
df_pos = df[df['sentiment']=='positive']['review_lm']



wordcloud1 = WordCloud(stopwords=STOPWORDS,

                      background_color='white',

                      width=2500,

                      height=2000

                      ).generate(" ".join(df_pos))



plt.figure(1,figsize=(15, 15))

plt.imshow(wordcloud1)

plt.axis('off')

plt.show()
df_neg = df[df['sentiment']=='negative']['review_lm']



wordcloud1 = WordCloud(stopwords=STOPWORDS,

                      background_color='white',

                      width=2500,

                      height=2000

                      ).generate(" ".join(df_neg))



plt.figure(1,figsize=(15, 15))

plt.imshow(wordcloud1)

plt.axis('off')

plt.show()
# There are 25.000 reviews for each outcome, so we can use the first 17.500 (70%) for training and 7.500 (30%) remaining for testing



# Train dataset (first 17.500 rows)

pos_train = df[df['sentiment']=='positive'][['review_lm', 'sentiment']].head(17500)

neg_train = df[df['sentiment']=='negative'][['review_lm', 'sentiment']].head(17500)





# Test dataset (last 7.500 rows)

pos_test = df[df['sentiment']=='positive'][['review_lm', 'sentiment']].tail(7500)

neg_test = df[df['sentiment']=='negative'][['review_lm', 'sentiment']].tail(7500)



# put all toghether again...

train_df = pd.concat([pos_train, neg_train]).sample(frac = 1).reset_index(drop=True)

test_df = pd.concat([pos_test, neg_test]).sample(frac = 1).reset_index(drop=True)

train_df.head()
test_df.head()
def get_word_counts(words):

    word_counts = {}

    for word in words:

        word_counts[word] = word_counts.get(word, 0.0) + 1.0

    return word_counts



def fit(df_fit):

    num_messages = {}

    log_class_priors = {}

    word_counts = {}

    vocab = set()

 

    n = df_fit.shape[0]

    num_messages['positive'] = df_fit[df_fit['sentiment']=='positive'].shape[0]

    num_messages['negative'] = df_fit[df_fit['sentiment']=='negative'].shape[0]

    log_class_priors['positive'] = math.log(num_messages['positive'] / n)

    log_class_priors['negative'] = math.log(num_messages['negative'] / n)

    word_counts['positive'] = {}

    word_counts['negative'] = {}

 

    for x, y in zip(df_fit['review_lm'], df_fit['sentiment']):

        

        counts = get_word_counts(nltk.word_tokenize(x))

        for word, count in counts.items():

            if word not in vocab:

                vocab.add(word)

            if word not in word_counts[y]:

                word_counts[y][word] = 0.0

 

            word_counts[y][word] += count

    

    return word_counts, log_class_priors, vocab, num_messages
word_counts, log_class_priors, vocab, num_messages = fit(train_df)
word_count_df = pd.DataFrame(word_counts).fillna(0).sort_values(by='positive', ascending=False).reset_index()

word_count_df
# Let's see how some words are distributed

word_count_sample_df = word_count_df.head(5000)

fig = go.Figure(go.Scatter(

    x = word_count_sample_df['positive'],

    y = word_count_sample_df['negative'],

    text = word_count_sample_df['index'],

    mode='markers'

))

fig.update_layout(title='Word distribution sample', 

                xaxis_title="Positive word count",

                yaxis_title="Negative word count",)



plotly.offline.iplot(fig)         
def predict(df_predict, vocab, word_counts, num_messages, log_class_priors):

    result = []

    for x in df_predict:

        counts = get_word_counts(nltk.word_tokenize(x))

        positive_score = 0

        negative_score = 0

        for word, _ in counts.items():

            if word not in vocab: continue

            

            # add Laplace smoothing

            log_w_given_positive = math.log((word_counts['positive'].get(word, 0.0) + 1) / (num_messages['positive'] + len(vocab)) )

            log_w_given_negative= math.log((word_counts['negative'].get(word, 0.0) + 1) / (num_messages['negative'] + len(vocab)) )

 

            positive_score += log_w_given_positive

            negative_score += log_w_given_negative

 

        positive_score += log_class_priors['positive']

        negative_score += log_class_priors['negative']

 

        if positive_score > negative_score:

            result.append('positive')

        else:

            result.append('negative')

    return result
result = predict(test_df['review_lm'], vocab, word_counts, num_messages, log_class_priors)

result[0:10] # result sample...
y_true = test_df['sentiment'].tolist()



acc = sum(1 for i in range(len(y_true)) if result[i] == y_true[i]) / float(len(y_true))

print("{0:.4f}".format(acc))
y_actu = pd.Series(y_true, name='Real')

y_pred = pd.Series(result, name='Predicted')

df_confusion = pd.crosstab(y_actu, y_pred)

df_confusion = df_confusion / df_confusion.sum(axis=1) * 100

df_confusion.round(2)
def plot_confusion_matrix(df_confusion, title='Confusion matrix'):

    plt.matshow(df_confusion) # imshow

    #plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(df_confusion.columns))

    plt.xticks(tick_marks, df_confusion.columns, rotation=45)

    plt.yticks(tick_marks, df_confusion.index)

    #plt.tight_layout()

    plt.ylabel(df_confusion.index.name)

    plt.xlabel(df_confusion.columns.name)    



plot_confusion_matrix(df_confusion)