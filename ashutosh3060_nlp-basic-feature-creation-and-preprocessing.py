import numpy as np

import pandas as pd

import nltk

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

import textblob

from textblob import TextBlob, Word

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
def avg_word_len (sentence):

    words = sentence.split()

    avg_len = sum(len(word) for word in words)/len(words)

    return avg_len



def extract_ngrams(data, num):

    '''

    Function to generate n-grams from sentences

    '''

    n_grams = TextBlob(data).ngrams(num)

    return [ ' '.join(grams) for grams in n_grams]
# Dataset



train = pd.read_csv('../input/twitter-sentiment-analysis-hatred-speech/train.csv')
print(train.shape)

train.head()
# Creating a copy of train dataset for text analysis



df_train = train.copy()
df_train['char_count'] = df_train['tweet'].str.len()

df_train_sort_charcount = df_train.sort_values(by='char_count', ascending=False)

df_train_sort_charcount[['tweet', 'char_count']].head()
df_train['word_count'] = df_train['tweet'].apply(lambda x: len(str(x).split(" ")))

df_train_sort_wordcount = df_train.sort_values(by='word_count', ascending=False)

df_train_sort_wordcount[['tweet','word_count']].head()
# Number of hashtags in a tweet



df_train['hashtags'] = df_train['tweet'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))

df_train_sort_hashtags = df_train.sort_values(by='hashtags', ascending=False)

df_train_sort_hashtags[['tweet', 'hashtags']].head()
stop_words = stopwords.words('english')



df_train['stopwords'] = df_train['tweet'].apply(lambda x: len([i for i in x.split() if i in stop_words]))

df_train_sort_stopwords = df_train.sort_values(by='stopwords', ascending=False)

df_train_sort_stopwords[['tweet', 'stopwords']].head()
df_train['number_count'] = df_train['tweet'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))

df_train_sort_number_count = df_train.sort_values(by='number_count', ascending=False)

df_train_sort_number_count[['tweet', 'number_count']].head()
df_train['upper_word'] = df_train['tweet'].apply(lambda x: len([x for x in x.split() if x.isupper()]))

df_train_sort_uppercase = df_train.sort_values(by='upper_word', ascending=False)

df_train_sort_uppercase[['tweet', 'upper_word']].head()
df_train['avg_word_len'] = df_train['tweet'].apply(lambda x: round(avg_word_len(x),1))

df_train_sort_avg_word_len = df_train.sort_values(by='avg_word_len', ascending=True)

df_train_sort_avg_word_len[['tweet', 'avg_word_len']].head()
data = df_train['tweet'][0]

 

print("1-gram: ", extract_ngrams(data, 1))

print("2-gram: ", extract_ngrams(data, 2))

print("3-gram: ", extract_ngrams(data, 3))

print("4-gram: ", extract_ngrams(data, 4))
tf = df_train['tweet'][1:2].apply(lambda x: pd.value_counts(x.split())/len(x.split())).sum(axis=0).reset_index()

tf.columns = ['words', 'tf']

tf
for i,word in enumerate(tf['words']):

    tf.loc[i, 'idf'] = np.log(df_train.shape[0]/(len(df_train[df_train['tweet'].str.contains(word)])))    

tf
tfidf = TfidfVectorizer(max_features=10000, lowercase=True, analyzer='word', stop_words= 'english',ngram_range=(1,1))

df_train_tfidf = tfidf.fit_transform(df_train['tweet'])

df_train_tfidf
bag_of_words = CountVectorizer(max_features=10000, lowercase=True, ngram_range=(1,1),analyzer = "word")

df_train_bag_of_words = bag_of_words.fit_transform(df_train['tweet'])

df_train_bag_of_words
df_train['sentiment'] = df_train['tweet'][:20].apply(lambda x: TextBlob(x).sentiment[0])

df_train[['tweet','sentiment']].head(5)
df_train.head(3)
# Creating a copy of dataset to preprocess the data



df_train_dpp = df_train.copy()
df_train_dpp['tweet_lower'] = df_train_dpp['tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))

df_train_dpp[['tweet', 'tweet_lower']].head()

stop_words = stopwords.words('english')



df_train_dpp['tweet_stopwords'] = df_train_dpp['tweet_lower'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))

df_train_dpp[['tweet', 'tweet_stopwords']].head()
df_train_dpp['tweet_punc'] = df_train_dpp['tweet_stopwords'].str.replace('[^\w\s]', '')

df_train_dpp[['tweet', 'tweet_punc']].head()
# Frequency of common words in all the tweets



common_top20 = pd.Series(' '.join(df_train_dpp['tweet_punc']).split()).value_counts()[:20]

print(common_top20)





# Remove these top 20 freq words

common = list(common_top20.index)



df_train_dpp['tweet_comm_remv'] = df_train_dpp['tweet_punc'].apply(lambda x: " ".join(x for x in x.split() if x not in common))

df_train_dpp[['tweet','tweet_comm_remv']].head()
# Frequency of common words in all the tweets

rare_top20 = pd.Series(" ".join(df_train_dpp['tweet_comm_remv']).split()).value_counts()[-20:]

rare_top20



# Remove these top 20 common words

rare = list(rare_top20.index)



df_train_dpp['tweet_rare_remv'] = df_train_dpp['tweet_comm_remv'].apply(lambda x: " ".join(x for x in x.split() if x not in rare))

df_train_dpp[['tweet','tweet_rare_remv']].head()
# Using textblob



df_train_dpp['tweet_rare_remv'][:10].apply(lambda x: str(TextBlob(x).correct()))

df_train_dpp['tweet_rare_remv'][:10].apply(lambda x: TextBlob(x).words)
st = PorterStemmer()

df_train_dpp['tweet_rare_remv'][:10].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
df_train_dpp['tweet_rare_remv'][:10].apply(lambda x: " ".join(Word(word) for word in x.split()))