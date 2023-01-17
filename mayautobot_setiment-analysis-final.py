import pandas as pd

import numpy as np

import re

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

import gensim

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score

import seaborn as sns

from wordcloud import WordCloud

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=FutureWarning)
#pd.set_option("display.max_colwidth", 300)

stp_words = stopwords.words('english')

stp_words.extend(['rt', 'vs', 'amp', 'quot', 'gt'])

print(stp_words)
#Read Given Tweets

tweet_data = pd.read_excel("../input/TWEET STACK.XLSX", dtype={'TweetFulltext' : str})

tweet_data.head(10)
def clean_tweets(tweet_txt):

    #Removing Handles

    tweet_txt_new = re.sub("@[\w]*", " ", tweet_txt)

    #Removing URLs

    tweet_txt_new = re.sub("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", " ", tweet_txt_new)

    #Removing punctation marks

    tweet_txt_new = re.sub("[^a-zA-Z#]", " ", tweet_txt_new)

    #Removing hashes

    tweet_txt_new = tweet_txt_new.replace('#', '')

    #Removing stop words

    words = [word.lower() for word in tweet_txt_new.split()]

    

    for stp_word in stp_words:

        if stp_word in words:

            words = list(filter(lambda a: a != stp_word, words))

    #Stemming words        

    stemmer = PorterStemmer()

    words_stemmed = [stemmer.stem(i) for i in words]

    return " ".join(words_stemmed)
# Take user handles

def extract_user_handles(tweet):

    user_handles = re.findall("@[\w]*", tweet)

    return "|".join(user_handles).replace("@", "")



def extract_hash_tags(tweet):

    tags = re.findall("#[\w]*", tweet)

    return "|".join(tags).replace("#", "")
tweet_data['TweetFulltext_cleaned'] = tweet_data.apply(lambda x: clean_tweets(x['TweetFulltext']), axis=1)

tweet_data['handles'] = tweet_data.apply(lambda x: extract_user_handles(x['TweetFulltext']), axis=1)

tweet_data['tags'] = tweet_data.apply(lambda x: extract_hash_tags(x['TweetFulltext']), axis=1)

tweet_data.head(10)
tweet_data = tweet_data.drop('TweetFulltext', axis=1)

tweet_data.head(10)
#Generating Top Handle Mentions

handle_dict = {}

tag_dict = {}



for i in range(tweet_data.shape[0]):

    user_handles = tweet_data.handles[i].split('|')

    tweet_tags = tweet_data.tags[i].split('|')

    

    # Counting occurence of user handles

    for user_handle in user_handles:

        if user_handle not in handle_dict:

            handle_dict[user_handle] = 0

        handle_dict[user_handle] += 1

    

    # Counting occurence of tags

    for tweet_tag in tweet_tags:

        if tweet_tag.lower() not in tag_dict:

            tag_dict[tweet_tag.lower()] = 0

        tag_dict[tweet_tag.lower()] += 1
if "" in handle_dict:

    del handle_dict[""]

handle_dict_sorted = sorted(handle_dict.items(), key=lambda item: item[1], reverse=True)

# handle, freq = zip(*handle_dict_sorted[:20])

df = pd.DataFrame(handle_dict_sorted[:20], columns=['User_Handle', 'Tweet_count'])

sns.set(style="darkgrid")

plt.figure(figsize=(20,6))

p = sns.barplot(y='User_Handle', x = "Tweet_count", data=df)
if "" in tag_dict:

    del tag_dict[""]

tag_dict_sorted = sorted(tag_dict.items(), key=lambda item: item[1], reverse=True)

df = pd.DataFrame(tag_dict_sorted[:20], columns=['Tags', 'Tweet_count'])

sns.set(style="darkgrid")

plt.figure(figsize=(20,6))

p = sns.barplot(y='Tags', x = "Tweet_count", data=df)
# Reading already tagged dataset

cols = ['sentiment','ids','date','flag','user','text']

training_data = pd.read_csv("../input/training.1600000.processed.noemoticon.csv",header=None, names=cols, encoding="ISO-8859-1")

training_data.head(10)
training_data = training_data.drop(['ids', 'date', 'flag', 'user'], axis=1)

training_data.head(10)
training_data.sentiment.value_counts()
# Changing the sentiment tagging as 0 -> 0 and 4 -> 1 in setiment column

training_data.sentiment = training_data.sentiment.map({0:0, 4:1})

training_data.sentiment.value_counts()
# Cleaning the tweets of the tagged dataset

training_data['tweet_txt_cleaned'] = training_data.apply(lambda x: clean_tweets(x['text']), axis=1)

training_data.head()
# Here we saw that few cleaned tweets have no text after cleaning. We have to remove those data from the dataset.

training_data[training_data.tweet_txt_cleaned == ""].sentiment.value_counts()
training_data = training_data[training_data.tweet_txt_cleaned != ""]

training_data.sentiment.value_counts()
training_data = training_data.drop(['text'], axis=1)
# Tokenize training data for creating word2vec vectors

training_data_tokenized = training_data.tweet_txt_cleaned.apply(lambda x: x.split())

word2vec_model = gensim.models.Word2Vec(training_data_tokenized, size=300, window=5, min_count=2, sg = 1, hs = 0, negative = 10, workers= 2, seed = 17)
def gen_tweet_vec(tokens, vocab_len):

    "Generates the vector for entire tweet as weighted average of the individual vectors of the words in the tweet"

    vec = np.zeros(vocab_len).reshape((1, vocab_len))

    token_cnt = 0

    for word in tokens:

        try:

            vec += word2vec_model[word].reshape((1, vocab_len))

            token_cnt += 1

        except KeyError:

            continue

    if token_cnt != 0:

        vec /= float(token_cnt)

    return vec
# Splitting the dataset in to test and train dataset.

train_data, test_data = train_test_split(training_data, test_size=0.2, random_state=51)

y_train_data = pd.DataFrame({'sentiment': train_data.values[:, 0]})

y_train_data = y_train_data.astype('int')

y_test_data = pd.DataFrame({'sentiment': test_data.values[:, 0]})

y_test_data = y_test_data.astype('int')

x_train_data = pd.DataFrame({'text': train_data.values[:, 1]})

x_test_data = pd.DataFrame({'text': test_data.values[:, 1]})
# Tokeninzing the cleaned tweets in test and train split.

x_train_data = x_train_data.text.apply(lambda x: x.split())

x_test_data = x_test_data.text.apply(lambda x: x.split())
# Generating word2vec vectors for train and test split.

# For calculating the sentence vector of each tweet, I have calculated the average of the the word2vec vectors of all the words that are present in our

# word2vecc model's dictionary



x_train_data_vectors = np.zeros((len(x_train_data), 300)) 

for i in range(len(x_train_data)):

    x_train_data_vectors[i,:] = gen_tweet_vec(x_train_data[i], 300)

x_train_data_vectors = pd.DataFrame(x_train_data_vectors)



x_test_data_vectors = np.zeros((len(x_test_data), 300)) 

for i in range(len(x_test_data)):

    x_test_data_vectors[i,:] = gen_tweet_vec(x_test_data[i], 300)

x_test_data_vectors = pd.DataFrame(x_test_data_vectors)
print(x_test_data_vectors.shape, y_test_data.shape, x_test_data.shape)
# Generating word2vec vector for given data (unlabelled dataset)

x_tweet_data_vectors = np.zeros((len(tweet_data.TweetFulltext_cleaned), 300)) 

for i in range(len(tweet_data.TweetFulltext_cleaned)):

    x_tweet_data_vectors[i,:] = gen_tweet_vec(tweet_data.TweetFulltext_cleaned[i], 300)

x_tweet_data_vectors = pd.DataFrame(x_tweet_data_vectors)
# Creating logistic regression model and gerating f1 score

lreg = LogisticRegression().fit(x_train_data_vectors, y_train_data)

prediction = lreg.predict(x_test_data_vectors)

f1_score(y_test_data, prediction)
#prediction of given data

prediction_given_data = lreg.predict(x_tweet_data_vectors)
tweet_data['sentiment'] = prediction_given_data

tweet_data['sentiment'].value_counts()
tweet_data[tweet_data.sentiment == 1].head(10)
tweet_data[tweet_data.sentiment == 0].head(10)
# Generating word-cloud to see the most prominent words in the negative sentiment tweets

# The size of the word is directly proportional to its occurence.

negative_setiment_df = tweet_data[tweet_data.sentiment == 0]

combined_corpus_text = ' '.join([text for text in negative_setiment_df['TweetFulltext_cleaned']])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(combined_corpus_text) 

plt.figure(figsize=(25, 25))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()
# Generating word-cloud to see the most prominent words in the positive sentiment tweets

# The size of the word is directly proportional to its occurence.

positive_setiment_df = tweet_data[tweet_data.sentiment == 1]

combined_corpus_text = ' '.join([text for text in positive_setiment_df['TweetFulltext_cleaned']])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(combined_corpus_text) 

plt.figure(figsize=(25, 25))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()