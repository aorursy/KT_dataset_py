import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk # natural language toolkit - http://www.nltk.org/
# We have a train.csv and test.csv available

df = pd.read_csv('../input/train.csv')



df.head()
# I want to remove punctuation from the text for word counting purposes

import string

no_punct_translator=str.maketrans('','',string.punctuation)



# tokenize each sentence and remove punctuation

df['words'] = df['text'].apply(lambda t: nltk.word_tokenize(t.translate(no_punct_translator).lower()))
# create a new column with the count of all words

df['word_count'] = df['words'].apply(lambda words: len(words))



# for normalization, how many characters per sentence w/o punctuation

df['sentence_length'] = df['words'].apply(lambda w: sum(map(len, w)))



# for future calculations, let's keep around the full text length, including punctuation

df['text_length'] = df['text'].apply(lambda t: len(t))
df.head()
# import some graphing libraries

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')
# Let's plot how many words per sentence each author uses

sns.boxplot(x = "author", y = "word_count", data=df, color = "red")
# Here's the same thing in text form.  

# Not a huge distinction but EAP seems to average less words per sentence

df.groupby(['author'])['word_count'].describe()
# here's the number of characters in each sentence, we do get a little separation here

df.groupby(['author'])['sentence_length'].describe()
# the string library defines `string.punctuation` which is all the punctuation chars

df['punctuation_count'] = df['text'].apply(lambda t: len(list(filter(lambda c: c in t, string.punctuation))))



df['punctuation_per_char'] = df['punctuation_count'] / df['text_length'] 
df.groupby(['author'])['punctuation_per_char'].describe()
def unique_words(words):

    word_count = len(words)

    unique_count = len(set(words)) # creating a set from the list 'words' removes duplicates

    return unique_count / word_count



df['unique_ratio'] = df['words'].apply(unique_words)

df.groupby(['author'])['unique_ratio'].describe()
authors = ['MWS', 'HPL', 'EAP']



for author in authors:

    sns.distplot(df[df['author'] == author]['unique_ratio'], label = author, hist=False)



plt.legend();
# add up the length of each words and devide by the total number of words

avg_length = lambda words: sum(map(len, words)) / len(words)



df['avg_word_length'] = df['words'].apply(avg_length)

df.groupby(['author'])['avg_word_length'].describe()
for author in authors:

    sns.distplot(df[df['author'] == author]['avg_word_length'], label = author, hist=False)



plt.legend();
df.head(2)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()



# Let's test how this works

print(sid.polarity_scores('Vader text analysis is my favorite thing ever'))

print(sid.polarity_scores('I hate vader and everything it stands for'))
df['sentiment'] = df['text'].apply(lambda t: sid.polarity_scores(t)['compound'])

df.groupby('author')['sentiment'].describe()
for author in authors:

    sns.distplot(df[df['author'] == author]['sentiment'], label = author, hist=False)



plt.legend();
sns.boxplot(x="author", y="sentiment", data=df);
# # TODO: do this later after we have more data



# # let's create a correlation matrix

# corr = df.corr()



# # make 

# plt.subplots(figsize=(11, 9))



# # Generate a custom diverging colormap

# cmap = sns.diverging_palette(220, 10, as_cmap=True)



# # Draw the heatmap with the mask and correct aspect ratio

# sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,

#             square=True, linewidths=.5, cbar_kws={"shrink": .5})
# Let's start by figuring out the most common words in our word dataset



# iterate all rows and create a new dataframe with author->word (single word)

df_words = pd.concat([pd.DataFrame(data={'author': [row['author'] for _ in row['words']], 'word': row['words']})

           for _, row in df.iterrows()], ignore_index=True)



# use NLTK to remove all rows with simple stop words

df_words = df_words[~df_words['word'].isin(nltk.corpus.stopwords.words('english'))]



df_words.shape
# let's use wordclouds to see which words each author likes to use

from wordcloud import WordCloud, STOPWORDS



def authorWordcloud(author):

    # lower max_font_size

    wordcloud = WordCloud(max_font_size=40,background_color="black", max_words=10000).generate(" ".join(df_words[df_words['author'] == author]['word'].values))

    plt.figure(figsize=(11,13))

    plt.title(author, fontsize=16)

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis("off")

    plt.show()

    

authorWordcloud('HPL')

authorWordcloud('EAP')

authorWordcloud('MWS')
# function for a specific author to count occurances of each word

def authorCommonWords(author, numWords):

    authorWords = df_words[df_words['author'] == author].groupby('word').size().reset_index().rename(columns={0:'count'})

    authorWords.sort_values('count', inplace=True)

    return authorWords[-numWords:]



# for example, here's how we get the 10 most common EAP words.

authorCommonWords('EAP', 10)
# get all top words from our authors.

# this will represent our top words "vocabulary list"

authors_top_words = []

for author in authors:

    authors_top_words.extend(authorCommonWords(author, 10)['word'].values)



# use a set to remove duplicates

authors_top_words = list(set(authors_top_words))
# put all the top words used in each example into a new column    

df['top_words'] = df['words'].apply(lambda w: list(set(filter(set(w).__contains__, authors_top_words))))

df[['author','top_words', 'words']].head()
# First, let's just pull out the columns we need

# feature_columns = ['author', 'word_count', 'text_length', 'punctuation_per_char', 'unique_ratio', 'avg_word_length', 'sentiment', 'top_words']

# TODO: put back in top_words once we figure it out

feature_columns = ['author', 'word_count', 'text_length', 'punctuation_per_char', 'unique_ratio', 'avg_word_length', 'sentiment']

df_features = df[feature_columns]



# Now let's split into a train and dev set

# use random_state seed so we get the same split each time

df_train=df_features.sample(frac=0.8,random_state=1)

df_dev=df_features.drop(df_train.index)



df_train.head()
import tensorflow as tf



# continual numeric features

feature_word_count = tf.feature_column.numeric_column("word_count")

feature_text_length = tf.feature_column.numeric_column("text_length")

feature_punctuation_per_char = tf.feature_column.numeric_column("punctuation_per_char")

feature_unique_ratio = tf.feature_column.numeric_column("unique_ratio")

feature_avg_word_length = tf.feature_column.numeric_column("avg_word_length")

feature_sentiment = tf.feature_column.numeric_column("sentiment")



# if we just used the single top word we could do it this way (single-hot)

# feature_top_words = tf.feature_column.categorical_column_with_vocabulary_list(

#    "top_words", vocabulary_list=authors_top_words)



# feature_top_words = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(

#     "top_words_test", vocabulary_list=authors_top_words))



base_columns = [

    feature_word_count, feature_text_length, feature_punctuation_per_char, feature_unique_ratio, feature_avg_word_length, feature_sentiment

]
import tempfile



model_dir = tempfile.mkdtemp() # base temp directory for running models



# our Y value labels, i.e. the thing we are classifying

labels_train = df_train['author']



# let's make a training function we can use with our estimators

train_fn = tf.estimator.inputs.pandas_input_fn(

    x=df_train,

    y=labels_train,

    batch_size=100,

    num_epochs=None, # unlimited

    shuffle=True, # shuffle the training data around

    num_threads=5)



# let's try a simple linear classifier

linear_model = tf.estimator.LinearClassifier(

    model_dir=model_dir, 

    feature_columns=base_columns,

    n_classes=len(authors),

    label_vocabulary=authors)
train_steps = 5000



# now let's train that model!

linear_model.train(input_fn=train_fn, steps=train_steps)
# let's see how well we did on our training set

dev_test_fn = tf.estimator.inputs.pandas_input_fn(

    x=df_dev,

    y=df_dev['author'],

    batch_size=100,

    num_epochs=1, # just one run

    shuffle=False, # don't shuffle test here

    num_threads=5)



linear_model.evaluate(input_fn=dev_test_fn)["accuracy"]