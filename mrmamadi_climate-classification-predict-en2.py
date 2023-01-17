!pip3 install comet_ml

# Setting up experiments

from comet_ml import Experiment



# Linking to Comet

experiment = Experiment(api_key='tvx43aEjuXWoGnkxiQOaNjXD7',

                       project_name='classification-predict',

                       workspace='en2-jhb')
# Data wrangling libraries

import pandas as pd

import numpy as np

from collections import Counter

import re



# Data visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns



# NLP libraries

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import TweetTokenizer

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.stem.wordnet import WordNetLemmatizer

from spacy import load

from textblob import TextBlob



# Download NLTK Dependencies

# nltk.download('stopwords')

# nltk.download('vader_lexicon')



# Load spacy

# nlp = load('en_core_web_sm')



# Feature Preprocessing

from sklearn.utils import resample

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.pipeline import Pipeline



# Models

from sklearn.linear_model import LogisticRegression,LinearRegression, RidgeCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier,StackingClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier



# Evaluating Model Performance

from sklearn.metrics import f1_score, recall_score, precision_score



# Storing Models and Vectorizers

import pickle



# Settings

sns.set_style('whitegrid')

%matplotlib inline
# load the datasets

train_data = pd.read_csv('../input/climate-change-belief-analysis/train.csv')

test_data = pd.read_csv('../input/climate-change-belief-analysis/test.csv')

sample_data = pd.read_csv('../input/climate-change-belief-analysis/sample_submission.csv')



# check the number of rows and columns of the data

print (f"There are {train_data.shape[0]} rows and {train_data.shape[1]} columns in the training set.")

print (f"There are {test_data.shape[0]} rows and {test_data.shape[1]} columns in the test set.")
train_data.head(3)
train_data.info()
# here is an example tweet selected randomly

tweet = train_data.loc[771, 'message']

tweet
count = 0

for i,s,m,tid in train_data.itertuples():

    if type(m)==str:

        if str(m).isspace():

            count += 1



print (f"There are {train_data['message'].isnull().sum()} missing values and {count} empty strings in the training set.")
# create dictionary of target number with their actual string

target_map = {-1:'Anti', 0:'Neutral', 1:'Pro', 2:'News'}

# create 'target' column

train_data['target'] = train_data['sentiment'].map(target_map)
# Function to convert data types

def typeConvert(df):

    """

    Return converted data-types of selected columns from dataframe.

    

    Parameters

    ----------

        df (DataFrame): input dataframe.

        

    Returns

    -------

        clean_df (DataFrame): output dataframe with new converted data types.

        

    Examples

    --------

    >>> traun_data.dtypes

            

            sentiment     int64

            message      object

            tweetid       int64

            target       object

            dtype: object

            

    >>> typeConvert(train_data).dtypes

        

            sentiment    category

            message        object

            tweetid         int16

            target         object

            dtype: object

    """

    

    if 'sentiment' in df.columns:

        # convert 'sentiment' to category as data type

        df['sentiment'] = df['sentiment'].astype('category')

    # convert 'tweetid' to int16 as data type

    df['tweetid'] = df['tweetid'].astype('int16')

    # return new converted data type

    return df



# train_data = typeConvert(train_data)
# Function to extract urls

def findURLs(tweet):

    """

    Return a string of an url from a tweet.

    

    Parameters

    ----------

        tweet (str): tweet containing an url.

        

    Returns

    -------

        extractedUrl(str): an url from a tweet.

        

    Examples

    --------

    >>> findURLs("This is my kaggle link, https://www.kaggle.com/touch-hd")

        

        "https://www.kaggle.com/touch-hd"

        

    """

    # create string to extract an url

    pattern = r'ht[t]?[p]?[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'

    # extract an url

    extractedUrl = " ".join(re.findall(pattern,tweet))

    # return a string of an url

    return extractedUrl



# example implementation

findURLs(tweet)
# finally implement the function in the train dataset

train_data['urls'] = train_data['message'].map(findURLs)
#Function to replace an url

def strip_url(df):

    """

     Return removal of all urls from the DataFrame raw message and replaces them with "urlweb".

    

    Parameters

    ----------

        df (DataFrame): input dataframe.

        

    Returns

    -------

        clean_df (DataFrame): output dataframe with an "urlweb" string replacement to each url from raw message.

        

    Example

    --------

    >>> train_data[0]

    

        |index|sentiment|message|tweetid

        |0    |-1       |https..|13442

        

    >>> strip_url(train_data[0])

        

        |index|sentiment|message|tweetid

        |0    |-1       |urlweb |13442

    """

    # copy the original DataFrame

    clean_df = df.copy()

    # create string to remove an url

    pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-@.&+]|[!*(),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'

    # create new string

    subs_url = r'urlweb'

    # replace an url with 'urlweb' string

    clean_df['message'] = clean_df['message'].replace(to_replace = pattern_url, value = subs_url, regex = True)

    return clean_df



# transforming dataframe

train_data = strip_url(train_data)
# show an example of a tweet

tweet = train_data['message'][771]

tweet
# Function to extract twitter handles

def findHandles(tweet):

    """

    Return a list of all handles from a tweet.

     

    Parameters

    ----------

         tweet (str): tweet containing handles.

         

    Returns

    -------

         handles (list): list of all handles.

         

    Examples

    --------

    >>> findHandles("hi @SenBernieSanders, you will beat @realDonaldTrump")

    

        ['@SenBernieSanders','@realDonaldTrump']

    """

    # create an empty list

    handles = list()

    for token in tweet.split():

        if token.startswith('@'):

            handles.append(token) # or .replace('@', '')

    # return twitter handles from the tweet

    return handles



# example implementation

findHandles(tweet)
# finally implement the function across the training data

train_data['handles'] = train_data['message'].map(findHandles)
# Function to extract hash-tags

def findHashTags(tweet):

    """

     Return a list of all hashtags from a tweet.

     

     Parameters

     ----------

        tweet (str): text containing hashtags

        

     Returns

     -------

        hash_tags (list): list of all hashtags

        

     Examples

     --------

     >>> findHashTags("Oil is killing the world renewables and EVS are the 

                                way the go! #EVs #GlobalWarming #Fossilfuels")

                                

         ['#EVs', '#GlobalWarming', '#Fossilfuels']

     """

    # create an empty list

    hash_tags = list()

    for token in tweet.split():

        if token.startswith('#'):

            hash_tags.append(token)

    # return hash-tags from the tweet

    return hash_tags



# example implementation

findHashTags(tweet)
# finally implement the function across the training data

train_data['hash_tags'] = train_data['message'].map(findHashTags)
train_data.head(3)
train_data.info()
# for interest sake we print out the example tweet again

tweet
# Function to remove punctuation

def removePunctuation(tweet):

    """

    Return the removal of punctuation and other uncommon characters in the tweet.

    

    Parameters

    ----------

        tweet (str): string containing punctuation to be removed.

        

    Returns

    -------

        clean_tweet (str): string without punctuation.

        

    Examples

    --------

    >>> removePunctuation("Hey! Check out this story: urlweb. He doesn't seem impressed. :)")

            

        "Hey Check out this story urlweb He doesn't seem impressed"

    """    

    # first remove line spaces

    clean_tweet = tweet.replace('\n', ' ')

    

    # substitute digits within text with an empty strring

    clean_tweet = re.sub('\w*\d\w*', ' ', clean_tweet)

    

    # remove punctuation

    # some of the character removed here were determined by visually inspecting the text

    clean_tweet = re.sub(r'[:;.,_()/\{}"?\!&¬¦ãÃâÂ¢\d]', '', clean_tweet) 

    

    # return cleaner tweet

    return clean_tweet



# example implementation

tweet = removePunctuation(tweet)

tweet
# finally implement the function across the training data

train_data['tweets'] = train_data['message'].map(removePunctuation)
# Function to generate tweet tokenization

def tweetTokenizer(tweet):

    """

    This method tokenizes and strips handles from twitter data.

    

    Parameters

    ----------

        tweet (str): string to be tokenized.

    Returns

    -------

        tokenized_tweet (list): list of tokens in tweet.

    Examples

    --------

    >>> tweetTokenizer("Read @swrightwestoz's latest on climate change insurance amp lending 

                                       featuring APRA speech and @CentrePolicyDev work urlweb")

    

        ['read',

        'latest',

        'on',

        'climate',

        'change',

        'insurance',

        'amp',

        'lending',

        'featuring',

        'apra',

        'speech',

        'and',

        'work',

        'urlweb']

    """

    tokenizer = TweetTokenizer(preserve_case = False, strip_handles = True)

    tokenized_tweet = tokenizer.tokenize(tweet)

    return tokenized_tweet



# example implementation

tokenized_tweet = tweetTokenizer(tweet)

tokenized_tweet
# finally implement the function across the training data

train_data['tweets'] = train_data['tweets'].map(tweetTokenizer)
# Function to remove stop words

def removeStopWords(tokenized_tweet):

    """

    This method removes stop words and punctuation relics.

    

    Parameters

    ----------

        tokenized_tweet (list): list of tokens to be cleaned.

        

    Returns

    -------

        clean_tweet (list): list of tokens without stopwords.

        

    Examples

    --------

    >>> removeStopWords(['read',

                        'latest',

                        'on',

                        'climate',

                        'change',

                        'insurance',

                        'amp',

                        'lending',

                        'featuring',

                        'apra',

                        'speech',

                        'and',

                        'work',

                        'urlweb'])

                        

        ['read',

        'latest',

        'on',

        'climate',

        'change',

        'insurance',

        'amp',

        'lending',

        'featuring',

        'apra',

        'speech',

        'and',

        'work',

        'urlweb']

    """

    # initialising an empty list as container for the cleaned tweet

    clean_tweet = list()

    # iterating through all words in a list

    for token in tokenized_tweet:

        # checking if current word is not a stopword # also checking if the current word is a hash_tag # also checking if the current word has more than one character

        if token not in stopwords.words('english') + ['amp','rt'] and token.startswith('#') == False and len(token) > 1:

            clean_tweet.append(token)            

    # return the cleaner tweet

    return clean_tweet



# example implementation

clean_tweet = removeStopWords(tokenized_tweet)

clean_tweet
# finally implement the function across the training data

train_data['tweets'] = train_data['tweets'].map(removeStopWords)
# Function to generate tweet lemmatization

def lemmatizeTweet(tweet):

    """

    Return tweet lemmatizer.

    

    Parameters

    ----------

        tweet (list): tokens to be lemmatized.

        

    Returns

    -------

        lemmatized_tweet (list): lemmatized list of tokens.

        

    Examples

    --------

    >>> lemmatizeTweet(['read',

                        'latest',

                        'on',

                        'climate',

                        'change',

                        'insurance',

                        'amp',

                        'lending',

                        'featuring',

                        'apra',

                        'speech',

                        'and',

                        'work',

                        'urlweb'])

                        

        ['read',

        'latest',

        'climate',

        'change',

        'insurance',

        'lending',

        'featuring',

        'apra',

        'speech',

        'work',

        'urlweb']

    """

    lemmatized_tweet = list()

    lmtzr = WordNetLemmatizer()

    for token in tweet:

        lemmatized_tweet.append(lmtzr.lemmatize(token))

    return lemmatized_tweet



# example implementation

lemmatized_tweet = lemmatizeTweet(clean_tweet)

lemmatized_tweet
# finally implement the function across the training data

train_data['tweets'] = train_data['tweets'].map(lemmatizeTweet)
# check the dataframe

train_data.head(3)
# Function to generate vocabulary

def getVocab(df):

    """

    Return a list of vocabulary from 'tweets' column in dataframe.

    

    Parameters

    ----------

        df (DataFrame): input dataframe.

        

    Returns

    -------

        vocab (list): list of all words that occur atleast once in the tweets.

        

    Examples

    --------

    >>> df['tweets']

     

    0   ['terror', 'major']

    1   ['yes', 'taken']

    

    >>> getVocab(df['tweets'])

        

        ['terror',

         'major',

         'yes',

         'taken']

    """

    # create an empty list

    vocab = list()

    for tweet in df:

        for token in tweet:

            vocab.append(token)

    # return giant list of vocab

    return vocab
#  example implementation

vocab = getVocab(df = train_data['tweets'])
# check length and count unique words

len(vocab), pd.Series(vocab).nunique()
word_frequency_dict = {}

for label in train_data['target'].unique():

    data = train_data[train_data['target'] == label]

    class_vocab = getVocab(data['tweets'])

    length_of_vocab = len(class_vocab)

    ordered_class_words = Counter(class_vocab).most_common()

    ordered_class_words_freq = list()

    for paired_tuple in ordered_class_words:

        word, count = paired_tuple

        word_frequency = round((count / length_of_vocab) * 100, 3)

        ordered_class_words_freq.append((word, word_frequency))

        word_frequency_dict[label] = ordered_class_words_freq

        

word_frequency_dict['Anti'][-5:]
fig, axes = plt.subplots(1, 4, figsize = (16, 5))

for i, label in enumerate(train_data['target'].unique()):

    data = pd.DataFrame(word_frequency_dict[label])[1]

    g = sns.distplot(data, ax = axes[i])

    g.set_title(label)

    g.set_xlabel('Percentage (%)')

    if label == 'Pro':

        g.set_ylabel('Frequency')

    g.set_xlim(0, 1) # comment this

    g.set_ylim(0, 1) # comment this
frequency_threshold = 0.01

class_words = {}

for label, value in word_frequency_dict.items():

    words = list()

    for paired_tuple in value:

        word, freq = paired_tuple

        if freq > frequency_threshold:

            words.append(word)

    class_words[label] = words

    

class_words['News'][:5]
pro_spec_words = list(set(class_words['Pro']) - set(class_words['Neutral']).union(set(class_words['Anti'])).union(set(class_words['News'])))

neutral_spec_words = list(set(class_words['Neutral']) - set(class_words['Pro']).union(set(class_words['Anti'])).union(set(class_words['News'])))

anti_spec_words = list(set(class_words['Anti']) - set(class_words['Pro']).union(set(class_words['Neutral'])).union(set(class_words['News'])))

news_spec_words = list(set(class_words['News']) - set(class_words['Pro']).union(set(class_words['Neutral'])).union(set(class_words['Anti'])))



label_specific_words = dict(

    Pro = pro_spec_words, Neutral = neutral_spec_words, Anti = anti_spec_words, News = news_spec_words

    )

label_specific_words['Pro'][:5]
print(f" Pro: {len(pro_spec_words):{10}} \n Neutral: {len(neutral_spec_words):{6}} \n Anti: {len(anti_spec_words):{8}} \n News: {len(news_spec_words):{9}}")
class_specific_words = pro_spec_words + neutral_spec_words + anti_spec_words + news_spec_words

class_specific_words[:5], len(class_specific_words)
# example implementation

ordered_words = Counter(vocab).most_common()

ordered_words[:5]
# Note, we set default n = 5000

# function to generate the top N of words

def topNWords(ordered_words, n = 5000):

    """

    

    

    Parameters

    ----------

        ordered_words (): input dataframe.

        n (int):

        

    Returns

    -------

        most_common (list): list of all words that occur atleast once in the tweets.

        

    Examples

    --------

        

    """

    most_common = list()

    for word in ordered_words[:n]:

        most_common.append(word[0])

    return most_common



# implementing the function

top_n_words = topNWords(ordered_words)

top_n_words[:5]
with open("topnwords.txt", "w") as output:

    output.write(str(top_n_words))
# show the count of uniques

pd.Series(top_n_words).nunique()
tweet
def removeInfrequentWords(tweet, include):

    """

    This function goes through the words in a tweet,

    determines if there are any words that are not in

    the top n words and removes them from the tweet

    and return the filtered tweet.

    

    Parameters

    ----------

        tweet (list): list tokens to be flitered.

        top_n_words (int): number of tweets to keep.

        

    Returns

    -------

        filt_tweet (list): list of top n words.

        

    Examples

    --------

    >>> bag_of_words = [('change', 12634),

                        ('climate', 12609),

                        ('rt', 9720),

                        ('urlweb', 9656),

                        ('global', 3773)]

                        

    >>> removeInfrequentWords(['rt', 'climate', 'change', 'equation', 'screenshots', 'urlweb'],2)

    

        ['change', 'climate']    

    """

    

    filt_tweet = list()

    for token in tweet:

        if token in include:

            filt_tweet.append(token)

    return filt_tweet
# finally implement the function across the training data

train_data['tweets_clean'] = train_data['tweets'].map(lambda tweet: removeInfrequentWords(tweet, include = top_n_words + class_specific_words))
all_vocab = list()

for tweet in train_data['tweets_clean']:

    for token in tweet:

        all_vocab.append(token)

pd.Series(all_vocab).nunique()
train_data.head(2)
very_common_words = topNWords(ordered_words, n = 20)

very_common_words[:5]
def removeCommonWords(tweet):

    """

    This method removes the most common words from a list of given words.

    

    Parameters

    ----------

        tweet (list): list of words to be cleaned.

        

    Returns

    -------

        filt_tweet (list): list of cleaned words.

        

    Examples

    --------

    >>> very_common_words = ['change', 'climate', 'rt', 'urlweb', 'global']

    

    >>> removeCommonWords(['rt', 'climate', 'change', 'equation', 'screenshots', 'urlweb'])

        

        ['equation']

    """

    filt_tweet = list()

    for token in tweet:

        if token not in very_common_words:

            filt_tweet.append(token)

    return filt_tweet



# example implementation

# print(filtered_tweet)

# print(removeCommonWords(filtered_tweet))
# finally implement the function across the training data

train_data['tweets_clean'] = train_data['tweets_clean'].map(removeCommonWords)
train_data.head(3)
def lengthOfTweet(tweet):

    """

    Return the length of each tweet in the dataframe.

    

    Parameters

    ----------

        tweet (list): list of a tweet.

        

    Returns

    -------

        length (int): length of each tweet.

    

    """

    length = len(tweet)

    return length



# example implementation

lengthOfTweet(tweet)
# implement the function across the dataset and save the results in a column

train_data['len_of_tweet'] = train_data['tweets_clean'].map(lengthOfTweet)



# plot a distribution of the tweet lengths

sns.distplot(train_data['len_of_tweet'])

plt.show()
len(train_data[train_data['len_of_tweet'] == 0])
# plotting counter-plots

sns.countplot(data = train_data, x = 'target', palette = {'Pro':'#CCCC00', 'News':'teal', 'Neutral':'teal', 'Anti':'teal'})

plt.title('Count of Sentiments\n')

plt.xlabel('\nSentiment')

plt.ylabel('Count\n')

plt.show()
tweet = train_data.loc[0, 'tweets_clean']

tweet
# function for generating word cloud

def plotWordCloud(data, label, text_col = 'tweets_clean'):

    """

    the plot of the most common use of words that appear bigger than words that

    appear infrequent in a text document by each sentiment

    

    Parameters

    ----------

        data (DataFrame): input of dataframe

        label (int): sentiment variable from dataframe

        

    Returns

    -------

        fig (matplotlib.figure.Figure): Final plot to be displayed

    

    """

    words = list()

    for tweet in data[text_col]:

        for token in tweet:

            words.append(token)

    words = ' '.join(words)



    from wordcloud import WordCloud

    wordcloud = WordCloud(contour_width=3, contour_color='steelblue').generate(words)



    # Display the generated image:

    fig = plt.figure(figsize = (10, 6))

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.title(label, fontsize = 30)

    plt.axis("off")

    plt.margins(x=0, y=0)

    plt.show()
# plotting word cloud for 'Pro' as target

data = train_data[train_data['target'] == 'Pro']

plotWordCloud(data, label = 'Sentiment = Pro')
# plotting word cloud for 'Anti' as target

data = train_data[train_data['target'] == 'Anti']

plotWordCloud(data, label = 'Target = Anti')
# plotting word cloud for 'Neutral' as target

data = train_data[train_data['target'] == 'Neutral']

plotWordCloud(data, label = 'Target = Neutral')
# plotting word cloud for 'News' as target

data = train_data[train_data['target'] == 'News']

plotWordCloud(data, label = 'Target = News')
def getPolarityScores(tweet):

    """

    return the polarity score of each tweet in the dataset

    

    Parameters

    ----------

        tweet (list): list of a tweet

    Returns

    -------

        scores (dict): dictionary of polarity scores

    

    """

    tweet = ' '.join(tweet)

    # analyse the sentiment

    sid = SentimentIntensityAnalyzer()

    # get polarity score of each data

    scores = sid.polarity_scores(tweet) 

    # return the polarity scores

    return scores





# example implementation

print(tweet)

getPolarityScores(tweet)
# implement the function across the dataset

nltk_scores = dict(compound = list(), negative = list(), neutral = list(), positive = list())

for tweet in train_data['tweets_clean']:

    output = getPolarityScores(tweet)

    nltk_scores['compound'].append(output['compound'])

    nltk_scores['negative'].append(output['neg'])

    nltk_scores['neutral'].append(output['neu'])

    nltk_scores['positive'].append(output['pos'])

    

# concatenate the output from above into the main dataset

if 'compound' in train_data.columns:

    

    # drop the columns if this has been executed before

    train_data.drop(['compound', 'negative', 'neutral', 'positive'], axis = 1, inplace = True)

    

    # concatenate a DataFrame version of the nltk_scores dictionary

    train_data = pd.concat([train_data, pd.DataFrame(nltk_scores)], axis = 1)

else:

    # concatenate directly if this is the first execution

    train_data = pd.concat([train_data, pd.DataFrame(nltk_scores)], axis = 1)
train_data.head(3)
sentiment_scores = [TextBlob(' '.join(tweet)).sentiment for tweet in train_data['tweets_clean']]



# Add output to dataframe

pol = list()

subj = list()

for scores in sentiment_scores:

    pol.append(scores.polarity)

    subj.append(scores.subjectivity)



train_data['polarity'] = pol

train_data['subjectivity'] = subj
train_data.head(2)
# summary statistics

train_data[['compound', 'polarity', 'subjectivity']].describe()
# plotting violin-plots

fig, axes = plt.subplots(1, 3, figsize = (18, 5))

for i, column in enumerate(['compound', 'polarity', 'subjectivity']):

    g = sns.violinplot(data = train_data, x = 'target', y = column, ax = axes[i], palette = {'Pro':'#CCCC00', 'News':'teal', 'Neutral':'teal', 'Anti':'teal'})

    g.set_title(column)

    if column == "compound":

        g.set_ylabel('Scores')

    else:

        g.set_ylabel(' ')

    g.set_xlabel(' ')
data = train_data.groupby('target')[['negative', 'positive', 'neutral', 'compound', 'polarity', 'subjectivity']].mean().reset_index()

data
# function for generating scatter-plot

def plotScatter(x, y, df, title):

    """

    display the scatter plot

    

    Parameters

    ----------

        x (str): variable string from dataframe

        y (str): variable string from dataframe

        df (DataFrame): input of dataframe

        

    Returns

    -------

        g (plot): display a scatter plot with points of each labelled sentiments

    

    """

    fig = plt.figure(figsize = (8, 5))

    g = sns.scatterplot(data = df, x = x, y = y, hue = 'target', legend = False, palette = {'Pro':'#CCCC00', 'News':'teal', 'Neutral':'teal', 'Anti':'teal'})

    g.set_title(title, fontsize = 20)

    

    # add annotations one by one with a loop

    for line in range(0,data.shape[0]):

        g.text(data[x][line], data[y][line], data['target'][line], 

                horizontalalignment='left', size='large', color='black')

    

    return g
# plotting scatter-plot

plotScatter(x = 'compound', y = 'polarity', df = data, title = 'Compound Vs Polarity\n')

plt.xlabel('\nCompound Score')

plt.ylabel('Polarity\n')

plt.show()
# plotting scatter-plot

fig = plt.figure(figsize = (8, 5))

g = sns.scatterplot(data = train_data, x = 'subjectivity', y = 'polarity', color = 'teal', hue = 'target', alpha = 1/3)

g.arrow(0, 0.1, 0.99, 1, fc = 'black', ec = '#CCCC00')

g.arrow(0, 0.1, 0.99, -1, fc = 'black', ec = '#CCCC00')

plt.title('Subjectiviy vs Polarity\n')

plt.xlabel('\nSubjectivity')

plt.ylabel('Polarity')

plt.show()
number_of_handles_per_tweet = train_data['handles'].map(lengthOfTweet)

fig = plt.figure(figsize = (8, 5))

number_of_handles_per_tweet.value_counts().plot(kind = 'barh', color = 'teal')

plt.title('Number of Handles in tweets\n ', fontsize = 16)
# plotting word cloud for 'News' as target

data = train_data[train_data['target'] == 'Pro']

plotWordCloud(data, label = 'Handles \n Pro \n', text_col = 'handles')
# plotting word cloud for 'News' as target

data = train_data[train_data['target'] == 'Anti']

plotWordCloud(data, label = 'Handles \n Anti \n', text_col = 'handles')
# plotting word cloud for 'News' as target

data = train_data[train_data['target'] == 'Neutral']

plotWordCloud(data, label = 'Handles \n Neutral \n', text_col = 'handles')
# plotting word cloud for 'News' as target

data = train_data[train_data['target'] == 'News']

plotWordCloud(data, label = 'Handles \n News \n', text_col = 'handles')
#plotting word cloud

lower = -0.15

interval = 0.3

data = train_data[(train_data['compound'] < lower+interval) & (train_data['compound'] > lower)]

plotWordCloud(data, label = f'Neutral where: {lower} < Compound < {lower+interval}')
number_of_handles_per_tweet = train_data['hash_tags'].map(lengthOfTweet)

fig = plt.figure(figsize = (8, 5))

number_of_handles_per_tweet.value_counts().plot(kind = 'barh', color = 'teal')

plt.title('Number of HashTags in tweets\n ', fontsize = 16)
# plotting word cloud

lower = -0.8

data = train_data[train_data['compound'] < lower]

plotWordCloud(data, label = f'compound < {lower}')
# plotting word cloud

upper = 0.6

sent = 0

data = train_data[(train_data['sentiment']==sent)&(train_data['compound'] > upper)]

plotWordCloud(data, label = f'compound > {upper}')
# Setting the random_state to reproduce the same results

rs = 42



#Creating data subsets

pos = train_data[train_data['sentiment']==1]

neg = train_data[train_data['sentiment']==-1]

neu = train_data[train_data['sentiment']==0]

news = train_data[train_data['sentiment']==2]



# Class size

frac = 100

size = int(len(pos)* frac/100)



# Resampling

neg_upsampled = resample(neg,



                          replace=True, # sample with replacement (need to duplicate observations)

                          n_samples=size, # match number in half of majority class

                          random_state=rs) # reproducible results



news_upsampled = resample(news,

                          replace=True, # sample with replacement (need to duplicate observations)

                          n_samples=size, # match number in half of majority class

                          random_state=rs) # reproducible results



neu_upsampled = resample(neu,

                          replace=True, # sample with replacement (need to duplicate observations)

                          n_samples=size, # match number in half of majority class

                          random_state=rs) # reproducible results



pos_downsampled = resample(pos,

                          replace=False, # sample without replacement (no need to duplicate observations)

                          n_samples=size, # match number in half of majority class

                          random_state=rs) # reproducible results



sampled = pd.concat([neg_upsampled, news_upsampled, neu_upsampled, pos_downsampled])



# Display new class counts



sns.countplot(data = train_data, x = 'target', palette = 'gray')

sns.countplot(data = sampled, x = 'target', palette = "Blues", alpha=0.5)

plt.title('Count of Sentiments\n')

plt.xlabel('\nSentiment')

plt.ylabel('Count\n')

plt.show()
#consider also using 'compound' column

X = sampled['tweets_clean'].apply(lambda x:' '.join(x))

y = sampled['sentiment']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=rs)
def best_classifier(param_search,vectoriser, cv_folds=2):



    """

    Return a dataframe including, and ranking, the weighted f1 scores

    of the classifiers (from input), using the best parameters 

    (from GridSearch performed on input parameters) after being run through

    a pipeline with the input vectoriser.



    Creates a dictionary in the form:

    {'name of classifier':'classifier with best parameters'}



    Parameters:

    param_search -- a list of tuples, in the form

                  [(name of classifier,classifier,param_grid)]

    vectoriser -- word vectoriser

    """



  # Creates empty dictionary to store classifier with best parameters

    class_dict = {}

  # New list to append information from classifer gridsearch

    results = []



  # Gridsearching each classifier in provided list

    for i in param_search:

        pipe = Pipeline([('vectorizer', vectoriser),

                         ('classifier', i[1])])

    

        grid = GridSearchCV(pipe,i[2],scoring='f1_weighted',n_jobs=-1, cv=cv_folds)

        grid.fit(X_train,y_train)



  # Making prediction on test set data using classifier fitted with best parameters

        y_pred = grid.best_estimator_.predict(X_test)

  

  # Appending tuple of information gained from grid search to results list

        results.append([i[0],f1_score(y_test,y_pred, average='weighted'),

                        grid.best_params_])

        

# Converting name string for naming file

        key = i[0].lower().replace(' ','_')

  # Adding classifier, loaded with best parameters, to dictionary   

        class_dict[key] = [grid.best_estimator_,

                            f1_score(y_test,y_pred, average='weighted')]



  # Creating dataframe to display sorted f1_scores

    results = pd.DataFrame(results, columns=['classifier','f1_score',

                                         'best_params'])

    results.set_index(['classifier'], inplace=True)

    results = results.sort_values('f1_score', ascending=False)

    return class_dict, results
# Instantiate a vectorizer

vectoriser = TfidfVectorizer()



# Name all of the classifiers

names = ['Logistic Regression',

         'Nearest Neighbors',

         'AdaBoost',

         'Naive Bayes',

         'Decision Tree', 

         'Support Vector'

        ]



# Instatiate the classifiers

classifiers = [LogisticRegression(),

               KNeighborsClassifier(),

               AdaBoostClassifier(),

               MultinomialNB(),

               DecisionTreeClassifier(),

               SVC()

               ]



# List model parameters

params = [{'classifier__C':[0.001, 0.01, 0.1, 1, 10],

           'classifier__solver':['liblinear','sag','saga','lbfgs'],

           'classifier__max_iter':[500, 1000]},

          {'classifier__n_neighbors':[1, 5, 10],

           'classifier__weights':['uniform','distance'],

           'classifier__algorithm':['auto', 'brute']},

          {'classifier__n_estimators':[50, 100, 500, 1000],

           'classifier__algorithm':['SAMME','SAMME.R']},

          {'classifier__alpha':[0.01,0.5,1],

           'classifier__fit_prior':[True, False]},

          {'classifier__criterion':["gini", "entropy"],

           'classifier__splitter':['best','random']},

          {'classifier__C':[0.1, 1, 10],

           'classifier__kernel':['poly', 'rbf', 'sigmoid'],

           'classifier__gamma':[0.1, 1]}

         ]
class_dict, results = best_classifier(zip(names,classifiers,params),vectoriser)

results
# Sort dictiionary by weighted f1_score and convert to list

sub_comet = sorted(class_dict.items(), key=lambda x:x[1][1], reverse=True)



# Select 3 best performing pipelines

models = []

for terms in sub_comet[:3]:

    models.append((terms[0],terms[1][0]['classifier']))



    

    

# Initialise meta learner

meta_learner_reg = DecisionTreeClassifier()



# Initialise stacking classifier

s_reg = StackingClassifier(estimators=models, 

                          final_estimator=meta_learner_reg

                          )



# Train stacking classifier

s_reg.fit(vectoriser.fit_transform(X_train), y_train)



# Stacking classifier prediction and performance

stack_y_pred = s_reg.predict(vectoriser.transform(X_test))

f1_score(y_test,stack_y_pred, average='weighted')
class_dict['stacking_regressor'] = [Pipeline(steps=[('vectorizer', vectoriser),

                                                    ('classifier', s_reg)]),

                                   f1_score(y_test,stack_y_pred, average='weighted')]


# Submit the parameters and metrics of the best performing model



# Submitting all experiments' metrics and parameters to comet



for name, ppl in class_dict.items():

    y_pred = ppl[0].predict(X_test)

    f1 = f1_score(y_test,y_pred, average='weighted')

    precision = precision_score(y_test, y_pred, average='weighted')

    recall = recall_score(y_test, y_pred, average='weighted')

    

# Storing metrics

    metrics = {'f1':f1,

               'recall':recall,

               'precision':precision}

    

# Storing paramerters

    params = {"random_state":rs,

              "model_type":name,

              "param_grid":ppl[0].get_params()}  

    

    experiment.log_metrics(metrics)

    experiment.log_parameters(params)

    

    

experiment.end()

experiment.display()
for name, classifier in class_dict.items():

# Saving models with best parametres for use in streamlit app

    model_save_path = f"{name}.pkl"

    with open(model_save_path, 'wb') as file:

        pickle.dump(classifier[0],file)
test_data = test_data.copy()

test_data = typeConvert(test_data[['message', 'tweetid']])

test_data = strip_url(test_data)

test_data['tweets'] = test_data['message'].map(removePunctuation)

test_data['tweets'] = test_data['tweets'].map(tweetTokenizer)

test_data['tweets'] = test_data['tweets'].map(removeStopWords)

test_data['tweets'] = test_data['tweets'].map(lemmatizeTweet)

test_data['tweets_clean'] = test_data['tweets'].map(lambda tweet: removeInfrequentWords(tweet, include = top_n_words + class_specific_words))

test_data['tweets_clean'] = test_data['tweets_clean'].map(removeCommonWords)



test_data.head()
X_sub = test_data['tweets_clean'].apply(lambda x:' '.join(x))



sub_classifier = sorted(class_dict.items(), key=lambda x:x[1][1], reverse=True)

sub_classifier = sub_classifier[0]

sub_classifier = sub_classifier[1][0]



sample_data['sentiment'] = sub_classifier.predict(X_sub)

sample_data.set_index('tweetid', inplace=True)



sample_data.to_csv('submission.csv')



sample_data.head()
# Install required libraries

!pip install --upgrade pip

!pip install kaggle --upgrade
%mkdir --parents /root/.kaggle/

%cp /kaggle/input/kaggle-token/kaggle.json   /root/.kaggle/

!chmod 600 /root/.kaggle/kaggle.json
!kaggle competitions submit -c climate-change-belief-analysis -f submission.csv -m "new submission"