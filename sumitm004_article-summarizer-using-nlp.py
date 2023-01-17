import numpy as np

import pandas as pd

import warnings

import re

import nltk

from nltk import word_tokenize

from nltk.tokenize import sent_tokenize

from textblob import TextBlob

import string

from string import punctuation

from nltk.corpus import stopwords

from statistics import mean

from heapq import nlargest

from wordcloud import WordCloud

import seaborn as sns

import matplotlib.pyplot as plt



stop_words = set(stopwords.words('english'))

punctuation = punctuation + '\n' + '—' + '“' + ',' + '”' + '‘' + '-' + '’'

warnings.filterwarnings('ignore')
# Importing the dataset

df_1 = pd.read_csv("/kaggle/input/all-the-news/articles1.csv")

df_2 = pd.read_csv("/kaggle/input/all-the-news/articles2.csv")

df_3 = pd.read_csv("/kaggle/input/all-the-news/articles3.csv")
# Checking if the columns are same or not

df_1.columns == df_2.columns
# Checking if the columns are same or not

df_2.columns == df_3.columns
# Making one Dataframe by appending all of them for the further process

d = [df_1, df_2, df_3]

df = pd.concat(d, keys = ['x', 'y', 'z'])

df.rename(columns = {'content' : 'article'}, inplace = True);
df.head(10)
# Shape of the dataset

print ("The shape of the dataset : ", df.shape)
# Dropping the unnecessary columns

df.drop(columns = ['Unnamed: 0'], inplace = True)

df.head()
df.head()
# Replacing the unnecessary row value of year with it's actual values

df['year'] = df['year'].replace("https://www.washingtonpost.com/outlook/tale-of-a-woman-who-died-and-a-woman-who-killed-in-the-northern-ireland-conflict/2019/03/08/59e75dd4-2ecd-11e9-8ad3-9a5b113ecd3c_story.html", 2019)
# Years

df['year'].value_counts()
# Countplot shows the distribution of the articles according to the year

plt.rcParams['figure.figsize'] = [15, 8]

sns.set(font_scale = 1.2, style = 'whitegrid')

sns_year = sns.countplot(df['year'], color = 'darkcyan')

sns_year.set(xlabel = "Year", ylabel = "Count", title = "Distribution of the articles according to the year")
# Authors

df['author'].value_counts()
# Changing the value "The Associated Press" to "Associated Press"

df['author'] = df['author'].replace("The Associated Press", "Associated Press")
# Top 100 authors

df['author'].value_counts()[0:100]
# Barplot showing the top 50 Authors

plt.rcParams['figure.figsize'] = [20, 10]

author_count = df['author'].value_counts()[0:50]

author_top_50 = sns.barplot(x = list(author_count.index), y = (author_count.values), color = 'orange')

author_top_50.set(xlabel = "Authors", ylabel = "Published Articles", title = "Top 50 Authors")

plt.setp(author_top_50.get_xticklabels(), rotation = 90);
# Countplot shows the distribution of the Publications

plt.rcParams['figure.figsize'] = [15, 10]

sns.set(font_scale = 1.2, style = 'whitegrid')

sns_pub = sns.countplot(df['publication'], color = 'slategrey')

sns_pub.set(xlabel = "Publications", ylabel = "Count", title = "Distribution of the Publications across the dataset")

plt.setp(sns_pub.get_xticklabels(), rotation = 90);
contractions_dict = { 

"ain't": "am not",

"aren't": "are not",

"can't": "cannot",

"can't've": "cannot have",

"'cause": "because",

"could've": "could have",

"couldn't": "could not",

"couldn't've": "could not have",

"didn't": "did not",

"doesn't": "does not",

"doesn’t": "does not",

"don't": "do not",

"don’t": "do not",

"hadn't": "had not",

"hadn't've": "had not have",

"hasn't": "has not",

"haven't": "have not",

"he'd": "he had",

"he'd've": "he would have",

"he'll": "he will",

"he'll've": "he will have",

"he's": "he is",

"how'd": "how did",

"how'd'y": "how do you",

"how'll": "how will",

"how's": "how is",

"i'd": "i would",

"i'd've": "i would have",

"i'll": "i will",

"i'll've": "i will have",

"i'm": "i am",

"i've": "i have",

"isn't": "is not",

"it'd": "it would",

"it'd've": "it would have",

"it'll": "it will",

"it'll've": "it will have",

"it's": "it is",

"let's": "let us",

"ma'am": "madam",

"mayn't": "may not",

"might've": "might have",

"mightn't": "might not",

"mightn't've": "might not have",

"must've": "must have",

"mustn't": "must not",

"mustn't've": "must not have",

"needn't": "need not",

"needn't've": "need not have",

"o'clock": "of the clock",

"oughtn't": "ought not",

"oughtn't've": "ought not have",

"shan't": "shall not",

"sha'n't": "shall not",

"shan't've": "shall not have",

"she'd": "she would",

"she'd've": "she would have",

"she'll": "she will",

"she'll've": "she will have",

"she's": "she is",

"should've": "should have",

"shouldn't": "should not",

"shouldn't've": "should not have",

"so've": "so have",

"so's": "so is",

"that'd": "that would",

"that'd've": "that would have",

"that's": "that is",

"there'd": "there would",

"there'd've": "there would have",

"there's": "there is",

"they'd": "they would",

"they'd've": "they would have",

"they'll": "they will",

"they'll've": "they will have",

"they're": "they are",

"they've": "they have",

"to've": "to have",

"wasn't": "was not",

"we'd": "we would",

"we'd've": "we would have",

"we'll": "we will",

"we'll've": "we will have",

"we're": "we are",

"we've": "we have",

"weren't": "were not",

"what'll": "what will",

"what'll've": "what will have",

"what're": "what are",

"what's": "what is",

"what've": "what have",

"when's": "when is",

"when've": "when have",

"where'd": "where did",

"where's": "where is",

"where've": "where have",

"who'll": "who will",

"who'll've": "who will have",

"who's": "who is",

"who've": "who have",

"why's": "why is",

"why've": "why have",

"will've": "will have",

"won't": "will not",

"won't've": "will not have",

"would've": "would have",

"wouldn't": "would not",

"wouldn't've": "would not have",

"y'all": "you all",

"y’all": "you all",

"y'all'd": "you all would",

"y'all'd've": "you all would have",

"y'all're": "you all are",

"y'all've": "you all have",

"you'd": "you would",

"you'd've": "you would have",

"you'll": "you will",

"you'll've": "you will have",

"you're": "you are",

"you've": "you have",

"ain’t": "am not",

"aren’t": "are not",

"can’t": "cannot",

"can’t’ve": "cannot have",

"’cause": "because",

"could’ve": "could have",

"couldn’t": "could not",

"couldn’t’ve": "could not have",

"didn’t": "did not",

"doesn’t": "does not",

"don’t": "do not",

"don’t": "do not",

"hadn’t": "had not",

"hadn’t’ve": "had not have",

"hasn’t": "has not",

"haven’t": "have not",

"he’d": "he had",

"he’d’ve": "he would have",

"he’ll": "he will",

"he’ll’ve": "he will have",

"he’s": "he is",

"how’d": "how did",

"how’d’y": "how do you",

"how’ll": "how will",

"how’s": "how is",

"i’d": "i would",

"i’d’ve": "i would have",

"i’ll": "i will",

"i’ll’ve": "i will have",

"i’m": "i am",

"i’ve": "i have",

"isn’t": "is not",

"it’d": "it would",

"it’d’ve": "it would have",

"it’ll": "it will",

"it’ll’ve": "it will have",

"it’s": "it is",

"let’s": "let us",

"ma’am": "madam",

"mayn’t": "may not",

"might’ve": "might have",

"mightn’t": "might not",

"mightn’t’ve": "might not have",

"must’ve": "must have",

"mustn’t": "must not",

"mustn’t’ve": "must not have",

"needn’t": "need not",

"needn’t’ve": "need not have",

"o’clock": "of the clock",

"oughtn’t": "ought not",

"oughtn’t’ve": "ought not have",

"shan’t": "shall not",

"sha’n’t": "shall not",

"shan’t’ve": "shall not have",

"she’d": "she would",

"she’d’ve": "she would have",

"she’ll": "she will",

"she’ll’ve": "she will have",

"she’s": "she is",

"should’ve": "should have",

"shouldn’t": "should not",

"shouldn’t’ve": "should not have",

"so’ve": "so have",

"so’s": "so is",

"that’d": "that would",

"that’d’ve": "that would have",

"that’s": "that is",

"there’d": "there would",

"there’d’ve": "there would have",

"there’s": "there is",

"they’d": "they would",

"they’d’ve": "they would have",

"they’ll": "they will",

"they’ll’ve": "they will have",

"they’re": "they are",

"they’ve": "they have",

"to’ve": "to have",

"wasn’t": "was not",

"we’d": "we would",

"we’d’ve": "we would have",

"we’ll": "we will",

"we’ll’ve": "we will have",

"we’re": "we are",

"we’ve": "we have",

"weren’t": "were not",

"what’ll": "what will",

"what’ll’ve": "what will have",

"what’re": "what are",

"what’s": "what is",

"what’ve": "what have",

"when’s": "when is",

"when’ve": "when have",

"where’d": "where did",

"where’s": "where is",

"where’ve": "where have",

"who’ll": "who will",

"who’ll’ve": "who will have",

"who’s": "who is",

"who’ve": "who have",

"why’s": "why is",

"why’ve": "why have",

"will’ve": "will have",

"won’t": "will not",

"won’t’ve": "will not have",

"would’ve": "would have",

"wouldn’t": "would not",

"wouldn’t’ve": "would not have",

"y’all": "you all",

"y’all": "you all",

"y’all’d": "you all would",

"y’all’d’ve": "you all would have",

"y’all’re": "you all are",

"y’all’ve": "you all have",

"you’d": "you would",

"you’d’ve": "you would have",

"you’ll": "you will",

"you’ll’ve": "you will have",

"you’re": "you are",

"you’re": "you are",

"you’ve": "you have",

}

contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
# Function to clean the html from the article

def cleanhtml(raw_html):

    cleanr = re.compile('<.*?>')

    cleantext = re.sub(cleanr, '', raw_html)

    return cleantext



# Function expand the contractions if there's any

def expand_contractions(s, contractions_dict=contractions_dict):

    def replace(match):

        return contractions_dict[match.group(0)]

    return contractions_re.sub(replace, s)



# Function to preprocess the articles

def preprocessing(article):

    global article_sent

    

    # Converting to lowercase

    article = article.str.lower()

    

    # Removing the HTML

    article = article.apply(lambda x: cleanhtml(x))

    

    # Removing the email ids

    article = article.apply(lambda x: re.sub('\S+@\S+','', x))

    

    # Removing The URLS

    article = article.apply(lambda x: re.sub("((http\://|https\://|ftp\://)|(www.))+(([a-zA-Z0-9\.-]+\.[a-zA-Z]{2,4})|([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}))(/[a-zA-Z0-9%:/-_\?\.'~]*)?",'', x))

    

    # Removing the '\xa0'

    article = article.apply(lambda x: x.replace("\xa0", " "))

    

    # Removing the contractions

    article = article.apply(lambda x: expand_contractions(x))

    

    # Stripping the possessives

    article = article.apply(lambda x: x.replace("'s", ''))

    article = article.apply(lambda x: x.replace('’s', ''))

    article = article.apply(lambda x: x.replace("\'s", ''))

    article = article.apply(lambda x: x.replace("\’s", ''))

    

    # Removing the Trailing and leading whitespace and double spaces

    article = article.apply(lambda x: re.sub(' +', ' ',x))

    

    # Copying the article for the sentence tokenization

    article_sent = article.copy()

    

    # Removing punctuations from the article

    article = article.apply(lambda x: ''.join(word for word in x if word not in punctuation))

    

    # Removing the Trailing and leading whitespace and double spaces again as removing punctuation might

    # Lead to a white space

    article = article.apply(lambda x: re.sub(' +', ' ',x))

    

    # Removing the Stopwords

    article = article.apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))

    

    return article



# Function to normalize the word frequency which is used in the function word_frequency

def normalize(li_word):

    global normalized_freq

    normalized_freq = []

    for dictionary in li_word:

        max_frequency = max(dictionary.values())

        for word in dictionary.keys():

            dictionary[word] = dictionary[word]/max_frequency

        normalized_freq.append(dictionary)

    return normalized_freq



# Function to calculate the word frequency

def word_frequency(article_word):

    word_frequency = {}

    li_word = []

    for sentence in article_word:

        for word in word_tokenize(sentence):

            if word not in word_frequency.keys():

                word_frequency[word] = 1

            else:

                word_frequency[word] += 1

        li_word.append(word_frequency)

        word_frequency = {}

    normalize(li_word)

    return normalized_freq



# Function to Score the sentence which is called in the function sent_token

def sentence_score(li):

    global sentence_score_list

    sentence_score = {}

    sentence_score_list = []

    for list_, dictionary in zip(li, normalized_freq):

        for sent in list_:

            for word in word_tokenize(sent):

                if word in dictionary.keys():

                    if sent not in sentence_score.keys():

                        sentence_score[sent] = dictionary[word]

                    else:

                        sentence_score[sent] += dictionary[word]

        sentence_score_list.append(sentence_score)

        sentence_score = {}

    return sentence_score_list



# Function to tokenize the sentence

def sent_token(article_sent):

    sentence_list = []

    sent_token = []

    for sent in article_sent:

        token = sent_tokenize(sent)

        for sentence in token:

            token_2 = ''.join(word for word in sentence if word not in punctuation)

            token_2 = re.sub(' +', ' ',token_2)

            sent_token.append(token_2)

        sentence_list.append(sent_token)

        sent_token = []

    sentence_score(sentence_list)

    return sentence_score_list



# Function which generates the summary of the articles (This uses the 20% of the sentences with the highest score)

def summary(sentence_score_OwO):

    summary_list = []

    for summ in sentence_score_OwO:

        select_length = int(len(summ)*0.25)

        summary_ = nlargest(select_length, summ, key = summ.get)

        summary_list.append(".".join(summary_))

    return summary_list



# This Function can be used to generate the summary which uses the mean sentence score

#def summary(sentence_score_OwO):

#    summary_list = []

#    li_sen = []

#    for summ in sentence_score_OwO:

#        for sent, score in summ.items():

#            threshold_score = mean(list(summ.values()))

#            if score >= threshold_score:

#                li_sen.append(sent)

#            else:

#                continue

#        sugoi = ', '.join(li_sen)

#        summary_list.append(sugoi)

#        li_sen = []

#    return summary_list



# Functions to change the article string (if passed) to change it to generate a pandas series

def make_series(art):

    global dataframe

    data_dict = {'article' : [art]}

    dataframe = pd.DataFrame(data_dict)['article']

    return dataframe



# Function which is to be called to generate the summary which in further calls other functions alltogether

def article_summarize(artefact):

    

    if type(artefact) != pd.Series:

        artefact = make_series(artefact)

    

    df = preprocessing(artefact)

    

    word_normalization = word_frequency(df)

    

    sentence_score_OwO = sent_token(article_sent)

    

    summarized_article = summary(sentence_score_OwO)

    

    return summarized_article
# Generating the Word Cloud of the article using the preprocessing and make_series function mentioned below

def word_cloud(art):

    art_ = make_series(art)

    OwO = preprocessing(art_)

    wordcloud_ = WordCloud(height = 500, width = 1000, background_color = 'white').generate(art)

    plt.figure(figsize=(15, 10))

    plt.imshow(wordcloud_, interpolation='bilinear')

    plt.axis('off');
# Generating the summaries for the first 100 articles

summaries = article_summarize(df['article'][0:100])
print ("The Actual length of the article is : ", len(df['article'][0]))

df['article'][0]
print ("The length of the summarized article is : ", len(summaries[0]))

summaries[0]
word_cloud(df['article'][0])
print ("The Actual length of the article is : ", len(df['article'][50]))

df['article'][50]
print ("The length of the summarized article is : ", len(summaries[50]))

summaries[50]
word_cloud(df['article'][50])
print ("The Actual length of the article is : ", len(df['article'][99]))

df['article'][99]
print ("The length of the summarized article is : ", len(summaries[99]))

summaries[99]
word_cloud(df['article'][99])
print ("The Actual length of the article is : ", len(df['article'][75]))

df['article'][75]
print ("The length of the summarized article is : ", len(summaries[75]))

summaries[75]
word_cloud(df['article'][75])