!pip install contractions
import numpy as np

import pandas as pd

import sklearn as sk

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sbn

import sys

import os

import re

import string

import scipy

from scipy.stats import chi2_contingency

from scipy.interpolate import interp1d

from statsmodels.stats.multitest import fdrcorrection, multipletests

from sklearn.feature_extraction.text import CountVectorizer

from tqdm import tqdm

import contractions

import nltk

from nltk.corpus import stopwords, wordnet

from nltk.tokenize import word_tokenize 

from nltk.tag import pos_tag
nltk.download("stopwords")

nltk.download('punkt')

nltk.download('averaged_perceptron_tagger')
print("python", sys.version)

for module in np, pd, mpl, sbn, nltk, sk, nltk, re, scipy:

    print(module.__name__, module.__version__)
np.random.seed(0)
train = pd.read_csv('../input/nlp-getting-started/train.csv')

test = pd.read_csv('../input/nlp-getting-started/test.csv')

sample_submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
pd.set_option('display.max_colwidth', None)



train.head()
test.head()
print("The training set has {} rows and {} columns.".format(train.shape[0], train.shape[1]))
print("The test set has {} rows and {} columns.".format(test.shape[0], test.shape[1]))
print("The training set has {} duplicated rows.".format(train.drop('id',axis=1).duplicated(keep=False).sum()))
print("The test set has {} duplicated rows.".format(test.drop('id',axis=1).duplicated(keep=False).sum()))
train[(train.drop(['id', 'target'], axis=1).duplicated(keep=False)) & ~(train.drop(['id'], axis=1).duplicated(keep=False))]
train[train.text.str.contains('CLEARED:incident with injury:I-495')]
test[(test.drop(['id'], axis=1).duplicated(keep=False))].iloc[:10,:]
train[(train['text'].duplicated(keep=False)) & ~(train.drop(['id','target'], axis=1).duplicated(keep=False))].sort_values(by="text")[10:20]
print("There are {} such tweets.".format(train[(train['text'].duplicated(keep=False)) & ~(train.drop(['id','target'], axis=1).duplicated(keep=False))].shape[0]))
print("{}% of the locations are missing in the training set and {}% in the test set".format(round(train.location.isnull().sum()/train.shape[0]*100, 1), round(test.location.isnull().sum()/test.shape[0]*100, 1)))

print("{}% of the keywords are missing in the training set and {}% in the test set".format(round(train.keyword.isnull().sum()/train.shape[0]*100, 2), round(test.keyword.isnull().sum()/test.shape[0]*100, 2)))
zeros = round((train.target==0).sum()/train.shape[0], 2)

ones = round(1-zeros, 2)



sbn.barplot(x=["Non disasters","Disasters"], y= [zeros,ones], color='gray')



plt.gca().set_ybound(0, 0.7)

plt.gca().set_ylabel('Proportion of tweets')

plt.gca().set_yticklabels([])



plt.gca().tick_params(axis='x')



plt.annotate(str(zeros)+'%', xy=(-0.1,zeros+0.01), size=15)

plt.annotate(str(ones)+'%', xy=(0.9,ones+0.01), size=15)

plt.suptitle('Distribution of disasters', size=15)

plt.show()
remove_url = lambda x:re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", x)



train['original_text'] = train.text.copy()

train['text'] = train.text.apply(remove_url)



test['original_text'] = test.text.copy()

test['text'] = test.text.apply(remove_url)
train.loc[train.text!=train.original_text, ['original_text', 'text']].head()
def cleaning(data):

  data = data.apply(lambda x: re.sub(r'\x89Ûª', '\'', x))

  data = data.apply(lambda x: re.sub(r'&;amp;', '&', x))

  data = data.apply(lambda x: re.sub(r'&amp;', '&', x))

  data = data.apply(lambda x: re.sub(r'&amp', '&', x))

  data = data.apply(lambda x: re.sub(r'Û¢åÊ', '', x))

  data = data.apply(lambda x: re.sub(r'ÛÒåÊ', '', x))

  data = data.apply(lambda x: re.sub(r'Û_', '', x))

  data = data.apply(lambda x: re.sub(r'ÛÒ', '', x))

  data = data.apply(lambda x: re.sub(r'ÛÓ', '', x))

  data = data.apply(lambda x: re.sub(r'ÛÏ', '', x))

  data = data.apply(lambda x: re.sub(r'Û÷', '', x))

  data = data.apply(lambda x: re.sub(r'Ûª', '', x))

  data = data.apply(lambda x: re.sub(r'\x89Û\x9d', '', x))

  data = data.apply(lambda x: re.sub(r'Û¢', '', x))

  data = data.apply(lambda x: re.sub(r'åÈ', '', x))

  data = data.apply(lambda x: re.sub(r'åÊ', ' ', x))

  data = data.apply(lambda x: re.sub(r'å¨', '', x))

  data = data.apply(lambda x: re.sub(r'åÇ', '', x))

  data = data.apply(lambda x: re.sub(r'å_', '', x))

  return data



train.text = cleaning(train.text)

test.text = cleaning(test.text)
train.loc[train.original_text.str.contains('&amp'), ['original_text', 'text']].head()
vocabulary = set(train['text'].apply(lambda x: re.sub(r'[0-9]', '', x)).apply(lambda x:x.split()).sum())

vocabulary_test = set(test['text'].apply(lambda x: re.sub(r'[0-9]', '', x)).apply(lambda x:x.split()).sum())



print("The tweets from the traing set contain {} unique words (after removing the URL and the figures).".format(len(vocabulary)))

print("The tweets from the test set contain {} unique words (after removing the URL and the figures).".format(len(vocabulary_test)))
contractions_detected = pd.DataFrame({word:[contractions.fix(word)] for word in vocabulary if word!=contractions.fix(word)}, index=['Corrections']).T

print("We detected {} differents contractions in the tweets from the traing set.".format(contractions_detected.shape[0]))
contractions_detected[:10]
def check_contractions(w):

  if w in contractions_detected.index:

    return contractions_detected.loc[w, 'Corrections']

  else:

    return w



train.text = train.text.apply(lambda x: ' '.join([check_contractions(w) for w in x.split()]))

test.text = test.text.apply(lambda x: ' '.join([check_contractions(w) for w in x.split()]))
train.loc[train.original_text.str.contains('theres'), ['original_text', 'text']]
test.loc[test.original_text.str.contains('what\'s'), ['original_text', 'text']].head()
train.loc[~train.keyword.isna(), 'keyword'] = train.loc[~train.keyword.isna(), 'keyword'].apply(lambda x: re.sub(r'%20', ' ', str(x)))

test.loc[~test.keyword.isna(), 'keyword'] = test.loc[~test.keyword.isna(), 'keyword'].apply(lambda x: re.sub(r'%20', ' ', str(x)))
print("The training set contains {} keywords. The rarest appears in {} tweets and the least rare in {} tweets. The keyword variable also contains {} missing values.".format(

    train.keyword.value_counts().shape[0], train.keyword.value_counts().min(), train.keyword.value_counts().max(), train.keyword.isna().sum()))
groupby_keyword = train[['keyword', 'target']].groupby('keyword')['target'].agg(frequencies= 'mean', count = 'size').reset_index().sort_values(by='count', ascending=False)



sbn.barplot(y='keyword', x='count', data=groupby_keyword.iloc[:20], color='gray')

plt.gca().set_xlabel('Count')

plt.gca().set_ylabel('Keywords')

plt.suptitle("20 most occurring Keywords", size=15)

plt.show()
groupby_keyword.sort_values(by='frequencies', ascending=False, inplace=True)

sbn.barplot(y='keyword', x='frequencies', data=groupby_keyword.iloc[:20], color='gray')

plt.gca().set_xlabel('Frequency of disasters')

plt.gca().set_ylabel('Keywords')

plt.suptitle("20 highest frequencies of disaster by keyword", size=15)

plt.show()
print("The 20 keywords associated with the highest frequencies of disaster occur between {} and {} times with an average of {}.".format(groupby_keyword[:20]['count'].min(), 

                                                                                                                                        groupby_keyword[:20]['count'].max(), 

                                                                                                                                        groupby_keyword[:20]['count'].mean()))
print("{} samples have 'debris', 'wreckage' or 'derailment' for keyword.".format(train.keyword.isin(['debris','wreckage', 'derailment']).sum()))
sbn.barplot(y='keyword', x='frequencies', data=groupby_keyword.iloc[-20:], color='gray')

plt.gca().set_xlabel('Frequency of disasters')

plt.gca().set_ylabel('Keywords')

plt.suptitle("20 lowest frequencies of disaster by keyword", size=15)

plt.show()
print("The 20 keywords associated with the lowest frequencies of disaster occur between {} and {} times with an average of {}.".format(groupby_keyword[-20:]['count'].min(), 

                                                                                                                                       groupby_keyword[-20:]['count'].max(), 

                                                                                                                                       groupby_keyword[-20:]['count'].mean()))
groupby_location = train[['location', 'target']].groupby('location')['target'].agg(frequencies= 'mean', count = 'size').reset_index().sort_values(by='frequencies', ascending=False)

x = groupby_location.frequencies.apply(lambda x:np.round(x,1)).value_counts().index

y = groupby_location.frequencies.apply(lambda x:np.round(x,1)).value_counts()

y = y/np.sum(y)

y*=100



sbn.barplot(x=x, y=y, color='gray')

plt.gca().set_xlabel('Frequency of disasters')

plt.gca().set_ylabel('Proportion of locations')

plt.gca().set_yticklabels([])

plt.gca().set_xbound(-0.5, 11)

plt.gca().set_ybound(0, 65)



for i in range(11):

  plt.annotate(str(round(y.loc[i/10], 1))+'%', xy=(i-0.4,y.loc[i/10]+1), size=10)



plt.suptitle("Proportion of locations by frequency of disasters", size=15)

plt.show()
x = groupby_location['count'].value_counts()[:10].index

y = groupby_location['count'].value_counts().to_numpy()

y = y[:10]/np.sum(y)

y *= 100



sbn.barplot(x=x, y=y, color='gray')

plt.gca().set_ylim([0, 100])

plt.gca().set_xlabel('Number of occurrences')

plt.gca().set_ylabel('Proportion of locations')



for i in range(10):

  plt.annotate(str(round(y[i], 1))+'%', xy=(i-0.3,y[i]+2), size=10)



plt.suptitle("Proportion of locations by number of occurrences in the training set", size=15)

plt.show()
print("Besides, {}% of the locations appear more than 11 times in the training set.".format(round(100-np.sum(y), 1)))
groupby_location[groupby_location['count']>=20]
keyword_location = train.copy()

keyword_location['keyword_location'] = keyword_location.keyword + '_' + keyword_location.location

pd.DataFrame(keyword_location.keyword_location.value_counts().reset_index().to_numpy(), columns=['(keyword, location) pairs', 'occurrences'])
keyword_location[keyword_location.keyword_location=="sandstorm_USA"]
def ngram_occurrences(corpus=train.text, i=0, j=20, stop_words=None, ngram_range=(1, 1), tokenizer=None):

  

  """ Function to return a dataframe containing some chosen n-grams and, for each n-grams, the number of times it appear in the corpus. 

      The defaults values return a the barplot for the 20 most frequent n-grams.



        Parameters

        ----------

        corpus : Series (default=train.text)

            A Series containing the tweets.

        i : Integer (default=0)

            The minimum index of the n-grams to draw. The n-grams are sorted by number of occurrences (with the index starting at 0). So 0 stands for the most frequent n-gram, 

            1 for the second most frequent n-gram, etc.

        j : Integer (default=20)

            1 + the maximum index of the n-grams to draw. The n-grams are sorted by number of occurrences (with the index starting at 0). So j=20 stands for the 19th index, wich stands for the

             20th most frequent n-gram.

        stop_words : Iterable (default=None)

            If not None, the stop words to remove from the tokens.

        ngram_range : Tuple (min_n, max_n) (default=(1, 1))

            The lower and upper boundary of the range of n-values for different word n-grams or char n-grams to be extracted. All values of n such such that min_n <= n <= max_n will be used.

            For example an ngram_range of (1, 1) means only unigrams, (1, 2) means unigrams and bigrams, and (2, 2) means only bigrams. Only applies if analyzer is not callable.

        tokenizer : Tokenizer (default=None)

            If not None, a tokenizer to use instead of the default tokenizer.



        """  

  

  vectorizer = CountVectorizer(stop_words=stop_words, ngram_range=ngram_range, tokenizer=tokenizer).fit(corpus)

  document_term_matrix = vectorizer.transform(corpus)

  count = document_term_matrix.sum(axis=0) 

  ngram_count = [(word, count[0, id]) for word, id in vectorizer.vocabulary_.items()]

  ngram_count = sorted(ngram_count, key = lambda x: x[1], reverse=True)

  return pd.DataFrame(ngram_count[i:j], columns=['ngram', 'count'])
def plot_ngram_occurrences(corpus=train.text, i=0, j=20, stop_words=None, n=1, tokenizer=None, add_title= ""):



  """ Function to display a barplot depicting the number of occurences for some chosen n-grams. The defaults values return a the barplot for the 20 most frequent n-grams.



        Parameters

        ----------

        corpus : Series (default=train.text)

            A Series containing the tweets.

        i : Integer (default=0)

            The minimum index of the n-grams to draw. The n-grams are sorted by number of occurrences (with the index starting at 0). So 0 stands for the most frequent n-gram, 

            1 for the second most frequent n-gram, etc.

        j : Integer (default=20)

            1 + the maximum index of the n-grams to draw. The n-grams are sorted by number of occurrences (with the index starting at 0). So j=20 stands for the 19th index, wich stands for the

             20th most frequent n-gram.

        stop_words : Iterable (default=None)

            If not None, the stop words to remove from the tokens.

        n : Integer (default=1)

            The kind of n-grams to consider. For example, 1 stands for unigrams, 2 for bigrams and 3 for trigrams.

        tokenizer : Tokenizer (default=None)

            If not None, a tokenizer to use instead of the default tokenizer.

        min_df : float in range [0.0, 1.0] or int (default=50)

            When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is also called cut-off in the literature. 

            If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.

        add_title : String(default="")

            If not the empty string "", some string to add before the predefined title.

        

        """    

  

  ngram_count = ngram_occurrences(corpus, i, j, stop_words, (n, n), tokenizer)



  ylabel = 'Words'*int(n==1) + 'Bigrams'*(n==2) + 'Trigrams'*(n==3)



  sbn.barplot(x='count', y = 'ngram', data=ngram_count, color='gray')



  plt.yticks(size=12)

  plt.gca().set_xlabel('Count')

  plt.gca().set_ylabel(ylabel)



  plt.suptitle("{} occurrences ".format(ylabel) + add_title, size=15)

  plt.show()
plot_ngram_occurrences(add_title= "in the training set before removing stopwords")
stop_words = set(stopwords.words('english'))

plot_ngram_occurrences(train.text, stop_words=stop_words, add_title="in the training set after removing stopwords")
plot_ngram_occurrences(test.text, stop_words=stop_words, add_title="in the test set after removing stopwords")
plot_ngram_occurrences(train.loc[train.target==1 ,'text'], stop_words=stop_words, add_title="in the disaster tweets")
plot_ngram_occurrences(train.loc[train.target==0,'text'], i=2, stop_words=stop_words, add_title="in the non-disaster tweets")
plot_ngram_occurrences(train.loc[train.target==1 ,'text'], stop_words=stop_words, n=2, add_title="in the disaster tweets")
plot_ngram_occurrences(train.loc[train.target==0,'text'], stop_words=stop_words, n=2, add_title="in the non-disaster tweets")
print("As an illustration, {} bigrams appear more than 25 times in disaster tweets, against only {} in non-disaster tweets.".format(

    (ngram_occurrences(train.loc[train.target==1,'text'], stop_words=stop_words, ngram_range=(2,2))['count']>25).sum(),

    (ngram_occurrences(train.loc[train.target==0,'text'], stop_words=stop_words, ngram_range=(2,2))['count']>25).sum()))
plot_ngram_occurrences(train.loc[train.target==1,'text'], stop_words=stop_words, n=3, add_title="in the disaster tweets")
plot_ngram_occurrences(train.loc[train.target==0,'text'], stop_words=stop_words, n=3, add_title="in the non-disaster tweets")
print("As an illustration, {} trigrams appear more than 25 times in disaster tweets, against only {} in non-disaster tweets.".format(

    (ngram_occurrences(train.loc[train.target==1,'text'], j=None, stop_words=stop_words, ngram_range=(3,3))['count']>20).sum(),

    (ngram_occurrences(train.loc[train.target==0,'text'], j=None, stop_words=stop_words, ngram_range=(3,3))['count']>20).sum()))
def disaster_frequency_by_ngram(corpus=train.text, i=0, j=20, stop_words=None, ngram_range=(1, 1), tokenizer=None, min_df=50):



  """ Function to return a dataframe containing some chosen n-grams and, for each n-grams, the frequency of disaster among the tweets containing this n-gram. The defaults values return a dataframe 

      containing the n-grams associated to the 20 highest frequencies of disaster and their frequencies.



        Parameters

        ----------

        corpus : Series (default=train.text)

            A Series containing the tweets.

        i : Integer (default=0)

            The minimum index of the n-grams to draw. The n-grams are sorted by frequency of disaster (with the index starting at 0). So 0 stands for the n-gram with the highest frequency, 

            1 for one with the second highest frequency, etc.

        j : Integer (default=20)

            1 + the maximum index of the n-grams to draw. The n-grams are sorted by frequency of disaster (with the index starting at 0). So j=20 stands for the 19th index, wich stands for the

             n-gram with the 20th highest frequency.

        stop_words : Iterable (default=None)

            If not None, the stop words to remove from the tokens.

        ngram_range : Tuple (min_n, max_n) (default=(1, 1))

            The lower and upper boundary of the range of n-values for different word n-grams or char n-grams to be extracted. All values of n such such that min_n <= n <= max_n will be used.

            For example an ngram_range of (1, 1) means only unigrams, (1, 2) means unigrams and bigrams, and (2, 2) means only bigrams. Only applies if analyzer is not callable.

        tokenizer : Tokenizer (default=None)

            If not None, a tokenizer to use instead of the default tokenizer.

        min_df : float in range [0.0, 1.0] or int (default=50)

            When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is also called cut-off in the literature. 

            If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.

        

        """



  vectorizer = CountVectorizer(stop_words=stop_words, ngram_range=ngram_range, tokenizer=tokenizer, min_df=min_df, binary=True).fit(corpus)

  document_term_matrix = vectorizer.transform(corpus)

  count = document_term_matrix.sum(axis=0) 

  frequency_by_ngram = document_term_matrix[train.target.to_numpy()==1,:].sum(axis=0)/count 

  ngram_freq_count = [(word, frequency_by_ngram[0, id], count[0, id]) for word, id in vectorizer.vocabulary_.items()]

  ngram_freq_count = sorted(ngram_freq_count, key = lambda x: x[1], reverse=True)

  return pd.DataFrame(ngram_freq_count[i:j], columns=['ngram', 'disaster_frequency', 'count'])
def plot_disaster_frequency_by_ngram(corpus=train.text, i=0, j=20, stop_words=None, n=1, tokenizer=None, min_df=50, add_title= ""):

  

  """ Function to display a barplot depicting the frequency of disaster for some chosen n-grams. The defaults values return a the barplot for

      the n-grams associated to the 20 highest frequencies of disaster.



        Parameters

        ----------

        corpus : Series (default=train.text)

            A Series containing the tweets.

        i : Integer (default=0)

            The minimum index of the n-grams to draw. The n-grams are sorted by frequency of disaster (starting at 0). So 0 stands for the n-gram with the highest frequency, 

            1 for one with the second highest frequency, etc.

        j : Integer (default=20)

            1 + the maximum index of the n-grams to draw. The n-grams are sorted by frequency of disaster (starting at 0). So j=20 stands for the 19th index, wich stands for the

            n-gram with the 20th highest frequency.

        stop_words : Iterable (default=None)

            If not None, the stop words to remove from the tokens.

        n : Integer (default=1)

            The kind of n-grams to consider. For example, 1 stands for unigrams, 2 for bigrams and 3 for trigrams.

        tokenizer : Tokenizer (default=None)

            If not None, a tokenizer to use instead of the default tokenizer.

        min_df : float in range [0.0, 1.0] or int (default=50)

            When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is also called cut-off in the literature. 

            If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.

        add_title : String(default="")

            If not the empty string "", some string to add before the predefined title.

        

        """  



  ngram_count = disaster_frequency_by_ngram(corpus, i, j, stop_words, (n, n), tokenizer, min_df=min_df)



  ylabel = 'Words'*int(n==1) + 'Bigrams'*(n==2) + 'Trigrams'*(n==3)



  sbn.barplot(x='disaster_frequency', y = 'ngram', data=ngram_count, color='gray')

  plt.yticks(size=12)

  plt.gca().set_xlabel('Frequency of disaster tweets')

  plt.gca().set_ylabel(ylabel)



  plt.suptitle(add_title + " frequencies of disaster by {} ".format(ylabel.lower()), size=15)

  plt.show()
plot_disaster_frequency_by_ngram(stop_words=stop_words, add_title="Highest")
plot_disaster_frequency_by_ngram(i=-20, j=None, stop_words=stop_words, add_title="Lowest")
plot_disaster_frequency_by_ngram(n=2, stop_words=stop_words, min_df=25, add_title="Highest")
plot_disaster_frequency_by_ngram(i=20, j=40, n=2, stop_words=stop_words, min_df=25, add_title="From 20 to 40 highest")
plot_disaster_frequency_by_ngram(i=-20, j=None, n=2, stop_words=stop_words, min_df=25, add_title="Lowest")
plot_disaster_frequency_by_ngram(i=0, j=None, n=3, stop_words=stop_words, min_df=25, add_title="All")
train['tweet_lengths'] = train.text.apply(lambda x:x.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))) # Remove punctuation

train['tweet_lengths'] = train['tweet_lengths'].apply(lambda x: len(x.split()))
def print_quick_stats(x, units):

  """ Function to compute simple statistics on a feature x (given as the argument of the function). """

  

  print("In the training set, the tweets contain from {} to {} {}, with an average of {} and a standard deviation equal to {}.".format(train[x].min(), train[x].max(), units,

                                                                                                                                       round(train[x].mean(), 1), round(train[x].std(), 1)))
x = 'tweet_lengths'

units = "words"

print_quick_stats(x, units)
def groupby_x(x):

  """ Function that return a dataframe with two columns. The first one contains each unique value taken by a feature x (given as the argument of the function).

      The second one contains the average frequency of disaster for each unique value of x. """



  return train[[x,'target']].groupby(x)['target'].agg(frequency= 'mean').reset_index()
def plot_disaster_frequency_by_x(x, groupby_x, title="(with 95% confidence interval)", vline=True, annotate=False, xlim=None, data=train):



  """ Function to plot a linear interpolation of the frequencies of disaster for each value taken by a given feature x. 

      Besides, an horizontal line is drawn on top of this graph to indicate the average frequency of disaster through all the training set.



        Parameters

        ----------

        x : String

            The name of the feature according to which we compute the frequencies of disaster.

        groupby_x : DataFrame

            A dataframe containing the average frequency of disaster for each value of x.

        title : String (default="(with 95% confidence interval)")

            A title that is used in addition to the suptitle of the graph.

        vline : Boolean (default=True)

            Whether to plot a vertical line at each intersection point between the interpolation curve of frequencies and the horizontal line equal to the average frequency among all the training set.

        annotate : Boolean (default=False)

            Whether to display the abscissa of the intersection point where the vertical lines are plotted.

        xlim : Tuple (default=None)

            The limits of the x-axis.

        data : DataFrame (default=train)

            The dataframe to which the feature x belongs.

        

        Display

        -------

        

        - A linear interpolation of the frequencies of disaster for each value taken by a given feature x.

        - An horizontal line indicating the average frequency of disaster through all the training set.

        - Optionally : a vertical line at each intersection point between the interpolation curve of frequencies and the horizontal line

        

        """



  sbn.lineplot(x=x, y='target', data=data, color='gray')



  z = groupby_x[x].to_numpy()

  y = groupby_x.frequency.to_numpy()

  args = np.linspace(z.min(), z.max(), 2000)

  f = interp1d(z, y).__call__(args)

  g = interp1d(z, data.target.mean()*np.ones(shape=z.shape)).__call__(args)

  idx = np.argwhere(np.diff(np.sign(f - g))).flatten()



  plt.axhline(y=data.target.mean(), color='red')

  plt.plot(args[idx], f[idx], 'ro', markersize=4)



  plt.gca().set_xlim(xlim)



  locs, labels = plt.yticks()

  norm_cst = min(np.max(locs), 1)



  if vline:

    if annotate:  

      for i in idx:

        plt.axvline(x=args[i], ymax=f[i]/norm_cst, color='red', linestyle='dashed')

        plt.annotate(str(round(args[i],1)), xy=(args[i]+0.05, -0.03), color='red')

    else:

      for i in idx:

        plt.axvline(x=args[i], ymax=f[i]/norm_cst, color='red', linestyle='dashed')



  plt.gca().set_xlabel(" ".join(x.split(sep='_')).capitalize())

  plt.gca().set_ylabel('Frequency of disasters')

  plt.suptitle('Frequency of disasters by ' + " ".join(x.split(sep='_')), size=15)

  plt.title(title)

  plt.show()
plot_disaster_frequency_by_x(x=x, groupby_x=groupby_x(x))
def distribution(x, bins=None, data=train):



  """ Function to plot an histogram depicting the distribution of a feature x of the training set (with a gaussian kernel density estimate).



        Parameters

        ----------

        x : String

            The name of the feature whose the distribution will be plotted.

        bins : Integer (default=None)

            The number of bins we want the histogram to have.

        data : DataFrame (default=train)

            The dataframe to which the feature x belongs.            

        """  



  sbn.distplot(data[x], color='gray', kde=True, bins=bins)

  plt.gca().set_xlabel(" ".join(x.split(sep='_')).capitalize())

  plt.gca().set_ylabel('Proportion')

  plt.suptitle("Distribution of " + " ".join(x.split(sep='_')), size=15)

  plt.title("(with a gaussian kernel density estimate)")

  plt.show()
distribution(x, bins=31)
print("{}% of tweets contain less than 7 words or more than 22.".format(round(100*train[(train.tweet_lengths<7) | (train.tweet_lengths>22)].shape[0]/train.shape[0], 1)))
train[train.tweet_lengths<7][:20]
train[train.tweet_lengths>22][:15]
train[(train.tweet_lengths>7) & (train.tweet_lengths<22)][:15]
x = 'tweet_lengths_without_stopwords'

stop_words = set(stopwords.words('english'))

train[x] = train.text.apply(lambda x:x.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))) # Remove punctuation

train[x] = train[x].apply(lambda x: len(set(x.lower().split()).difference(stop_words)))



distribution(x)
print("The {} tweets with a length superior to 22 have an average length equal to {} once the stop words are removed.".format((train.tweet_lengths>22).sum(), train.loc[train.tweet_lengths>22, 

                                                                                                                              'tweet_lengths_without_stopwords'].mean()))
plot_disaster_frequency_by_x(x, groupby_x(x))
x = 'stopwords_count'

units = 'stop words'

stop_words = set(stopwords.words('english'))

train[x] = train.original_text.apply(lambda x: len(set(x.lower().split()).intersection(stop_words)))



print_quick_stats(x, units)
plot_disaster_frequency_by_x(x=x, groupby_x=groupby_x(x), annotate=True)
def marginal_tweets_count(x, units, level):

  print('Only {} tweets contain more than {} {}.'.format((train[x]>level).sum(), level, units))



marginal_tweets_count(x, units, 14)
distribution(x)
x = 'URL_count'

units = 'URL'

train[x] = train.original_text.apply(lambda x : len(re.findall("https?:\/\/t.co\/[A-Za-z0-9]+", x)))



print_quick_stats(x, units)
def distribution_annotated(x, xaxis=range(train[x].max()+1), data=train):



  """ Function to plot an histogram depicting the distribution of a feature x of the training set, with the values directly annotated on the bars.



        Parameters

        ----------

        x : String

            The name of the feature whose the distribution will be plotted.

        xaxis : Iterable (default=train[x].max()+1)

            The values of the x-axis.

        data : DataFrame (default=train)

            The dataframe to which the feature x belongs.            

        """    



  groupby = data[[x, 'target']].groupby(x)['target'].agg(count= 'size').reset_index()

  z = groupby.loc[groupby[x].isin(xaxis), x]

  y = groupby.loc[groupby[x].isin(xaxis), 'count']/groupby['count'].sum()

  y*=100



  sbn.barplot(x=z, y=y, color='gray')

  plt.gca().set_yticklabels([])

  plt.gca().tick_params(axis='x', labelsize=12)

  plt.gca().set_ylim((0, max(y)+10))

  plt.gca().set_xlabel(" ".join(x.split(sep='_')).capitalize())

  plt.gca().set_ylabel('Proportion of tweets')



  for i in xaxis:

    plt.annotate(str(round(y.loc[i], 1))+'%', xy=(i-0.4, y.loc[i]+1), size=10)

  

  plt.suptitle('Distribution of '+ " ".join(x.split(sep='_')), size=15)

  plt.show()
distribution_annotated(x)
plot_disaster_frequency_by_x(x=x, groupby_x=groupby_x(x))
def cochran_criterion(crosstab):

  """ Function taking as argument a contingency table and returning a boolean indicating whether the Cochran's criterion (which checks the validity for a chi-squared test) 

      is satisfied for this table. """

  N = crosstab.sum().sum()

  E = [O_i*O_j/N > 4 for O_i in crosstab.sum(axis=1) for O_j in crosstab.sum()]

  criterion = (0 not in set(crosstab.to_numpy().reshape(-1))) & (np.mean(E)>=0.8)

  return criterion



def chi2_test(X,Y):

  """ Function taking two variables as arguments and returning a list of results about a chi-squared test of independence between these two variables. The list of results

      contains a boolean indicating whether the Cochran's criterion is satisfied, the test statistic, the p-value of the test, and a boolean indicating whether the

      test is statistically significant (i.e the Cochran's critetion is satisfied and the p-value is under 5%). """

  crosstab = pd.crosstab(X, Y)

  criterion = cochran_criterion(crosstab)

  chi2, p = chi2_contingency(crosstab)[:2]

  statistically_significative = criterion & (p<0.5)

  return [criterion, chi2, p, statistically_significative]



def cohen_d(x,y):

    """ Function taking two variables as arguments and returning the Cohen's d computed between them as well as a string indicating the nature of the measured effect

        ("Positive" or "Negative"). This string indicates which of the two variables x and y has the greatest average ("Negative" means it is x and "Positive" means it is y). """

    nx = len(x)

    ny = len(y)

    dof = nx + ny - 2

    value = (np.mean(y) - np.mean(x)) / np.sqrt(((ny-1) * np.std(y, ddof=1)**2 + (nx-1) * np.std(x, ddof=1)**2) / dof)

    effect = "Positive"*(np.sign(value)==1) + "Negative"*(np.sign(value)==-1)

    return np.abs(value), effect



def chi2_test_with_effect_size(x, threshold=0):

  """ Function taking as argument the name of a feature of the training set and a threshold. It returns the list of results provided by the cohen_d function appended to

      the one provided by the chi2_test function, where both of these functions are feeded with the target variable and an indicator variable taking value 0 or 1

      according to the exceeding of the given threshold by the given feature. """

  values = chi2_test(X = (train[x]>threshold).apply(int), Y = train.target)

  values.extend(cohen_d(train[train[x]<=threshold].target, train[train[x]>threshold].target))

  return values



chi2_tests = pd.DataFrame(columns=['Cochran_criterion', 'Chi2', 'p_value', 'Statistically_significative', 'Cohen_d', 'Effect' ])

chi2_tests.loc[units + ' presence'] = chi2_test_with_effect_size(x)

chi2_tests 
x = 'punctuation_marks_count'

units = 'punctuation marks'

train[x] = train['text'].apply(lambda x: len([w for w in x if w in string.punctuation]))



print_quick_stats(x, units)
distribution(x, bins=62)
marginal_tweets_count(x, units, 25)
plot_disaster_frequency_by_x(x=x, groupby_x=groupby_x(x), xlim=(0,25), vline=False)
punctuation_marks_count = {}

for p in string.punctuation:

  punctuation_marks_count[p] = [train['text'].apply(lambda x: len([w for w in x if w in p])).sum(), train['text'].apply(lambda x: int(len([w for w in x if w in p])>0)).sum()]



punctuation_marks_count = pd.DataFrame(punctuation_marks_count, index=['count', 'tweets_concerned']).sort_values(by='count', axis=1)

punctuation_marks_count
x = 'dots_count'

units = 'dots'

train[x] = train['text'].apply(lambda x: len([w for w in x if w in '.']))



print_quick_stats(x, units)
distribution_annotated(x, xaxis=range(10))
marginal_tweets_count(x, units, 9)
plot_disaster_frequency_by_x(x=x, groupby_x=groupby_x(x), xlim=(0,9), annotate=True)
train[train[x]==3].head()
x = 'at_least_3_dots'

units = 'at least 3 dots'

train[x] = (train['dots_count']>=3).apply(int)
chi2_tests.loc[units] = chi2_test_with_effect_size(x)

pd.DataFrame(chi2_tests.iloc[-1,:]).T
x = 'exclamation_marks_count'

units = 'exclamation marks'

train[x] = train['text'].apply(lambda x: len([w for w in x if w in '!']))



print_quick_stats(x, units)
distribution_annotated(x, xaxis=range(7))
marginal_tweets_count(x, units, 6)
def frequencies_by_x_presence(x, units, xticklabels='', threshold=0, data=train):



  """ Function to display a barplot depicting the frequencies of disaster according to the exceeding of a threshold by a feature x from the train set. With the default

      setting, this function displays the frequencies of disaster among two kinds of tweets : those containing zero unit of the feature x (x=0), 

      and those with at least 1 unit of x (x>0). Hence the name of the function.



        Parameters

        ----------

        x : String

            The name of the feature whose the distribution will be plotted.

        units : String

            What we want to check the presence. This string is used in the title and the x-ticks name.

        threshold : Integer or Float (default=0)

            The threshold that we check if x exceeds.

        data : DataFrame (default=train)

            The dataframe to which the feature x belongs.            

        """    



  without_x = np.round_(100*np.mean(data[data[x]<=threshold].target))

  with_x = np.round_(100*np.mean(data[data[x]>threshold].target))



  sbn.barplot(x=['Tweets without {}'.format(units), 'Tweets with {} (at least one)'.format(units)], y=[without_x, with_x], color='gray')

  plt.gca().set_ylabel('Frequency of disasters')

  plt.gca().set_yticklabels([])  

  plt.gca().set_ylim((0, max(without_x, with_x)+10))

  plt.annotate(str(without_x)+'%', xy=(-0.1, without_x+1), size=15)

  plt.annotate(str(with_x)+'%', xy=(0.9, with_x+1), size=15)

  plt.suptitle('Frequencies of disasters by {} presence'.format(units), size=15)



  if xticklabels!='':

    plt.gca().set_xticklabels(xticklabels)



  plt.show()
frequencies_by_x_presence(x, units, xticklabels=['Witouht \"!\"', 'With \"!\"'])
chi2_tests.loc[units + ' presence'] = chi2_test_with_effect_size(x)

pd.DataFrame(chi2_tests.iloc[-1,:]).T
x = 'question_marks_count'

units = 'question marks'

train['question_marks_count'] = train['text'].apply(lambda x: len([w for w in x if w in '?']))

print_quick_stats(x, units)
distribution_annotated(x, xaxis=range(10))
marginal_tweets_count(x, units, 9)
frequencies_by_x_presence(x, units, xticklabels=['Tweets without \"?\"', 'Tweets with \"?\"'])
chi2_tests.loc[units + ' presence'] = chi2_test_with_effect_size(x)

pd.DataFrame(chi2_tests.iloc[-1,:]).T
x = 'colons_count'

units = 'colons'

train['colons_count'] = train['text'].apply(lambda x: len([w for w in x if w in ':']))

print_quick_stats(x, units)
distribution_annotated(x, xaxis=range(8))
marginal_tweets_count(x, units, 7)
frequencies_by_x_presence(x, units)
chi2_tests.loc[units + ' presence'] = chi2_test_with_effect_size(x)

pd.DataFrame(chi2_tests.iloc[-1,:]).T
x = 'ampersands_count'

units = 'ampersands'

train[x] = train['text'].apply(lambda x: len([w for w in x if w in '&']))

print_quick_stats(x, units)
distribution_annotated(x, xaxis=range(3))
marginal_tweets_count(x, units, 3)
frequencies_by_x_presence(x, units, xticklabels=['Tweets without \"&\"', 'Tweets with \"&\"'])
chi2_tests.loc[units + ' presence'] = chi2_test_with_effect_size(x)

pd.DataFrame(chi2_tests.iloc[-1,:]).T
x = 'hyphen_count'

units = 'hyphens'

train[x] = train['text'].apply(lambda x: len([w for w in x if w in '-']))

print_quick_stats(x, units)
distribution_annotated(x, xaxis=range(7))
marginal_tweets_count(x, units, 6)
frequencies_by_x_presence(x, units)
chi2_tests.loc[units + ' presence'] = chi2_test_with_effect_size(x)

pd.DataFrame(chi2_tests.iloc[-1,:]).T
x = 'slashes_count'

units = 'slashes'

train[x] = train['text'].apply(lambda x: len([w for w in x if w in '/']))

print_quick_stats(x, units)
distribution_annotated(x, xaxis=range(5))
marginal_tweets_count(x, units, 4)
frequencies_by_x_presence(x, units)
chi2_tests.loc[units + ' presence'] = chi2_test_with_effect_size(x)

pd.DataFrame(chi2_tests.iloc[-1,:]).T
x = 'asterisks_count'

units = 'asterisks'

train[x] = train['text'].apply(lambda x: len([w for w in x if w in '*']))
train.loc[train[x]>0][['text', 'asterisks_count']].head()
print_quick_stats(x, units)
distribution_annotated(x, xaxis=range(3))
marginal_tweets_count(x, units, 2)
frequencies_by_x_presence(x, units)
print('Only {} tweets contain at least one asterisk.'.format((train[x]>0).sum()))
chi2_tests.loc[units + ' presence'] = chi2_test_with_effect_size(x)

pd.DataFrame(chi2_tests.iloc[-1,:]).T
x = 'hashtags_count'

units = 'hashtags'

train['hashtags_count'] = train['text'].apply(lambda x: len([w for w in x if w in '#']))

print_quick_stats(x, units)
distribution_annotated(x, xaxis=range(9))
marginal_tweets_count(x, units, 8)
frequencies_by_x_presence(x, units, xticklabels=['Tweets without \"#\"', 'Tweets with \"#\"'])
chi2_tests.loc[units + ' presence'] = chi2_test_with_effect_size(x)

pd.DataFrame(chi2_tests.iloc[-1,:]).T
x = 'mentions_count'

units = 'mentions'

train[x] = train['text'].apply(lambda x: len([w for w in x if w in '@']))

print_quick_stats(x, units)
distribution_annotated(x, xaxis=range(9))
frequencies_by_x_presence(x, units)
chi2_tests.loc[units + ' presence'] = chi2_test_with_effect_size(x)

pd.DataFrame(chi2_tests.iloc[-1,:]).T
x = 'underscores_count'

units = 'underscores'

train[x] = train['text'].apply(lambda x: len([w for w in x if w in '_']))

print_quick_stats(x, units)
distribution_annotated(x, xaxis=range(5))
frequencies_by_x_presence(x, units)
chi2_tests.loc[units + ' presence'] = chi2_test_with_effect_size(x)

pd.DataFrame(chi2_tests.iloc[-1,:]).T
x = 'apostrophes_count'

units = 'apostrophes'

train[x] = train['text'].apply(lambda x: len([w for w in x if w in '\'']))

print_quick_stats(x, units)
distribution_annotated(x, xaxis=range(6))
marginal_tweets_count(x, units, 5)
frequencies_by_x_presence(x, units)
chi2_tests.loc[units + ' presence'] = chi2_test_with_effect_size(x)

pd.DataFrame(chi2_tests.iloc[-1,:]).T
chi2_tests.sort_values(by='Cohen_d', ascending=False)
x = 'pos_tags'

units = 'pos tags'

train[x] = train.text.apply(lambda x: pos_tag(word_tokenize(x)))
train[['text', x, 'target']].head()
x = 'NNS_count'

units = 'NNS'

train[x] = train.pos_tags.apply(lambda x : len([z for z in x if z[1] == 'NNS']))

print_quick_stats(x, units)
distribution_annotated(x, xaxis=range(6))
marginal_tweets_count(x, units, 5)
plot_disaster_frequency_by_x(x, groupby_x(x), xlim=(0,5), annotate=True)
frequencies_by_x_presence(x, units)
chi2_tests.loc[units + ' presence'] = chi2_test_with_effect_size(x)

pd.DataFrame(chi2_tests.iloc[-1,:]).T
x = 'NNP_count'

units = 'NNP'

train[x] = train.pos_tags.apply(lambda x : len([z for z in x if z[1] == 'NNP']))

print_quick_stats(x, units)
distribution(x)
marginal_tweets_count(x, units, 15)
plot_disaster_frequency_by_x(x, groupby_x(x), xlim=(0,15), annotate=True)
frequencies_by_x_presence(x, units)
chi2_tests.loc[units + ' presence'] = chi2_test_with_effect_size(x)

pd.DataFrame(chi2_tests.iloc[-1,:]).T
x = 'VBD_count'

units = 'VBD'

train[x] = train.pos_tags.apply(lambda x : len([z for z in x if z[1] == 'VBD']))

print_quick_stats(x, units)
distribution_annotated(x, xaxis=range(7))
marginal_tweets_count(x, units, 3)
plot_disaster_frequency_by_x(x, groupby_x(x), xlim=(0,3), annotate=True)
frequencies_by_x_presence(x, units)
chi2_tests.loc[units + ' presence'] = chi2_test_with_effect_size(x)

pd.DataFrame(chi2_tests.iloc[-1,:]).T
x = 'VBN_count'

units = 'VBN'

train[x] = train.pos_tags.apply(lambda x : len([z for z in x if z[1] == 'VBN']))

print_quick_stats(x, units)
distribution_annotated(x, xaxis=range(5))
plot_disaster_frequency_by_x(x, groupby_x(x), vline=False)
frequencies_by_x_presence(x, units)
chi2_tests.loc[units + ' presence'] = chi2_test_with_effect_size(x)

pd.DataFrame(chi2_tests.iloc[-1,:]).T
x = 'VBP_count'

units = 'VBP'

train[x] = train.pos_tags.apply(lambda x : len([z for z in x if z[1] == 'VBP']))

print_quick_stats(x, units)
distribution_annotated(x, xaxis=range(6))
plot_disaster_frequency_by_x(x, groupby_x(x), vline=False)
frequencies_by_x_presence(x, units)
chi2_tests.loc[units + ' presence'] = chi2_test_with_effect_size(x)

pd.DataFrame(chi2_tests.iloc[-1,:]).T
x = 'MD_count'

units = 'MD'

train[x] = train.pos_tags.apply(lambda x : len([z for z in x if z[1] == 'MD']))

print_quick_stats(x, units)
distribution_annotated(x, xaxis=range(5))
plot_disaster_frequency_by_x(x, groupby_x(x), annotate=True)
frequencies_by_x_presence(x, units)
chi2_tests.loc[units + ' presence'] = chi2_test_with_effect_size(x)

pd.DataFrame(chi2_tests.iloc[-1,:]).T
x = 'PRP_count'

units = 'PRP'

train[x] = train.pos_tags.apply(lambda x : len([z for z in x if z[1] == 'PRP']))

print_quick_stats(x, units)
distribution_annotated(x, xaxis=range(7))
marginal_tweets_count(x, units, 6)
plot_disaster_frequency_by_x(x, groupby_x(x), xlim=(0,6), annotate=True)
frequencies_by_x_presence(x, units)
chi2_tests.loc[units + ' presence'] = chi2_test_with_effect_size(x)

pd.DataFrame(chi2_tests.iloc[-1,:]).T
x = 'PRP$_count'

units = 'PRP$'

train[x] = train.pos_tags.apply(lambda x : len([z for z in x if z[1] == 'PRP$']))

print_quick_stats(x, units)
distribution_annotated(x, xaxis=range(7))
plot_disaster_frequency_by_x(x, groupby_x(x), vline=False)
frequencies_by_x_presence(x, units)
chi2_tests.loc[units + ' presence'] = chi2_test_with_effect_size(x)

pd.DataFrame(chi2_tests.iloc[-1,:]).T
x = 'RB_count'

units = 'RB'

train[x] = train.pos_tags.apply(lambda x : len([z for z in x if z[1] == 'RB']))

print_quick_stats(x, units)
distribution_annotated(x, xaxis=range(8))
marginal_tweets_count(x, units, 4)
plot_disaster_frequency_by_x(x, groupby_x(x), data=train, xlim=(0,4), annotate=True)
frequencies_by_x_presence(x, units)
chi2_tests.loc[units + ' presence'] = chi2_test_with_effect_size(x)

pd.DataFrame(chi2_tests.iloc[-1,:]).T
x = 'WRB_count'

units = 'WRB'

train[x] = train.pos_tags.apply(lambda x : len([z for z in x if z[1] == 'WRB']))

print_quick_stats(x, units)
distribution_annotated(x, xaxis=range(4))
marginal_tweets_count(x, units, 2)
plot_disaster_frequency_by_x(x, groupby_x(x), data=train, xlim=(0,2), annotate=True)
frequencies_by_x_presence(x, units)
chi2_tests.loc[units + ' presence'] = chi2_test_with_effect_size(x)

pd.DataFrame(chi2_tests.iloc[-1,:]).T
x = 'UH_count'

units = 'UH'

train[x] = train.pos_tags.apply(lambda x : len([z for z in x if z[1] == 'UH']))

print_quick_stats(x, units)
distribution_annotated(x, xaxis=range(3))
frequencies_by_x_presence(x, units)
chi2_tests.loc[units + ' presence'] = chi2_test_with_effect_size(x)

pd.DataFrame(chi2_tests.iloc[-1,:]).T
chi2_tests.iloc[13:,:].sort_values(by='Cohen_d', ascending=False)
x = 'uppercases_count'

units = 'capital letters'

remove_mentions = lambda x:re.sub(r"@[A-Za-z0-9]+", "", x)

train[x] = train.text.apply(remove_mentions).apply(lambda x : len([z for z in x if z.isupper()]))



print_quick_stats(x, units)
distribution(x)
plot_disaster_frequency_by_x(x, groupby_x(x), xlim=(0,20), vline=False)
x = 'more_than_3_uppercases'

train[x] = (train.uppercases_count>3).apply(int)
frequencies_by_x_presence(x, units='more than 3 capital letters', xticklabels=['< 3', '> 3'])
chi2_tests.loc[x] = chi2_test_with_effect_size(x)

pd.DataFrame(chi2_tests.iloc[-1,:]).T
x = 'lowercases_count'

units = 'lowercase characters'

train[x] = train.text.apply(remove_mentions).apply(lambda x : len([z for z in x if z.islower()]))

print_quick_stats(x, units)
distribution(x)
plot_disaster_frequency_by_x(x, groupby_x(x), vline=False)
x = 'more_than_40_lowercase characters'

train[x] = (train.lowercases_count>40).apply(int)
frequencies_by_x_presence(x, units='more than 40 lowercase characters', xticklabels=['< 40', '> 40'])
chi2_tests.loc[x] = chi2_test_with_effect_size(x)

pd.DataFrame(chi2_tests.iloc[-1,:]).T
x = 'characters_count'

units = 'characters'

train[x] = train.text.apply(remove_mentions).apply(lambda x : len([z for z in x if z.islower()|z.isupper()])) # We do not just compute len(x) because it would consider the whitespaces as characters.

print_quick_stats(x, units)
distribution(x)
plot_disaster_frequency_by_x(x, groupby_x(x), vline=False)
x = "more_than_40_characters"

units = "more than 40 characters"

train[x] = (train.characters_count>40).apply(int)
frequencies_by_x_presence(x, units='more than 40 characters', xticklabels=['< 40 characters', '> 40 characters'])
chi2_tests.loc[x] = chi2_test_with_effect_size(x)

pd.DataFrame(chi2_tests.iloc[-1,:]).T
x = "Start_with_capital_letter"

units = "Start with capital letter"

train[x] = train.text.apply(remove_mentions).apply(lambda x: x[0].isupper()).apply(int)
distribution_annotated(x, range(2))
frequencies_by_x_presence(x, units='first character in uppercase', xticklabels=[units, 'don\'t ' + units])
chi2_tests.loc[x] = chi2_test_with_effect_size(x)

pd.DataFrame(chi2_tests.iloc[-1,:]).T
chi2_tests.iloc[-4:,:].sort_values(by='Cohen_d', ascending=False)
f, ax = plt.subplots(figsize=(30, 15))

ax = sbn.heatmap(train.corr('spearman'), cmap=plt.cm.gray)
correlations = pd.DataFrame({}, columns=['correlations'])

for i in range(train.corr('spearman').shape[0]-1):

  for j in range(i+1, train.corr('spearman').shape[0]):

    if (np.abs(train.corr('spearman').iloc[i,j]) > 0.5):

      correlations.loc[train.corr('spearman').columns[i]+'_'+train.corr('spearman').columns[j]] = train.corr('spearman').iloc[i,j]
correlations.sort_values(by='correlations', ascending=False)
train.to_csv('modified_train', index=False)

test.to_csv('modified_test', index=False)

contractions_detected.reset_index().rename(columns={'index':'Contractions'}).to_csv('contractions_detected', index=False)

chi2_tests.reset_index().rename(columns={'index':'variables'}).sort_values(by='Cohen_d', ascending=False).to_csv('chi2_tests', index=False)

correlations.reset_index().rename(columns={'index':'pairs of variables'}).sort_values(by='correlations', ascending=False).to_csv('correlations', index=False)