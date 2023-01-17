text = """Latanya Sweeney and Nick Diakopoulos pioneered the study of misbehavior in Google systems (Sweeney, 2013; Diakopoulos, 2013; Diakopoulos, 2016). Their work exposed instances of algorithmic defa- mation in Google searches and ads. Diakopoulos discussed a canonical example of such algorithmic defamation in which search engine auto- completion routines, fed a steady diet of historical user queries, learn to make incorrect defamatory or bigoted associations about people or groups of people.10 Sweeney showed that such learned negative associa- tions affect Google’s targeted ads. In her example, just searching for certain types of names led to advertising for criminal justice services, such as bail bonds or criminal record checking. Diakopoulos’s exam- ples included consistently defamatory associations for searches related to transgender issues.
Studies like Sweeney’s and Diakopoulos’s are archetypes in the growing field of data and algorithmic journalism. More news and research articles chronicle the many missteps of the algorithms that affect different parts of our lives, online and off. IBM’s Jeopardy- winning AI, Watson, famously had to have its excessive swearing habit corrected after its learning algorithms ingested some unsavory data (Madrigal, 2013). There have also been reports on the effects of Waze’s traffic routing algorithms on urban traffic patterns (Bradley, 2015). One revealing book describes the quirks of the data and algorithms underlying the popular OkCupid dating service (Rudder, 2014). More recently, former Facebook contractors revealed that Facebook’s news feed trend algorithm was actually the result of subjective input from a human panel (Tufekci, 2016)."""
# We can split the text-chunk into something like sentences.
split_text = text.split('.')
print(split_text)
# print out the first sentence

sentence_1 = split_text[0]
print(sentence_1)
# Let's create tokens from this sentence
tokens_sentence_1 = [word for word in sentence_1.split(' ')]
print(tokens_sentence_1)
# Let's lowercase all these tokens and clean up the \n (new line command)


tokens_sentence_1_lower = [word.lower().strip() for word in sentence_1.split(' ')]
print('### OUTPUT1 ###')
print(tokens_sentence_1_lower)
print('\n')
    
# Also we will replace "()" as well as make sure that only words lend in our list
tokens_sentence_1_lower = [word.replace('(','').replace(')','') for word in tokens_sentence_1_lower if word.isalpha()]

print('### OUTPUT2 ###')
print(tokens_sentence_1_lower)
# Removing stopwords

stopwords_en = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 
                'ourselves', 'you', "you're", "you've", "you'll", 
                "you'd", 'your', 'yours', 'yourself', 'yourselves', 
                'he', 'him', 'his', 'himself', 'she', "she's", 'her', 
                'hers', 'herself', 'it', "it's", 'its', 'itself', 
                'they', 'them', 'their', 'theirs', 'themselves', 'what', 
                'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 
                'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 
                'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
                'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 
                'between', 'into', 'through', 'during', 'before', 'after', 'above', 
                'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
                'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 
                'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 
                'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 
                'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 
                'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 
                'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', 
                "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', 
                "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 
                'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 
                'won', "won't", 'wouldn', "wouldn't"]
tokens_sentence_1_clean = [word for word in tokens_sentence_1_lower if word not in stopwords_en]
print(tokens_sentence_1_clean)
# Tokenizing sentences
from nltk.tokenize import sent_tokenize

# Tokenizing words
from nltk.tokenize import word_tokenize

# Tokenizing Tweets!
from nltk.tokenize import TweetTokenizer
# Let's get our stences.
# Note that the full-stops at the end of each sentence are still there
sentences = sent_tokenize(text)
print(sentences)
# Use word_tokenize to tokenize the first sentence: tokenized_sent
tokenized_sent = word_tokenize(sentences[0])

# Make a set of unique tokens in the entire text: unique_tokens
unique_tokens = set(word_tokenize(text))
print(unique_tokens)
# The preprocessing is donw for you

import pandas as pd

trump_tweets = pd.read_json('https://cdn.rawgit.com/SDS-AAU/M2-2018/2cbfe741/input/trump_twitter.json')
trump_tweets.set_index(pd.to_datetime(trump_tweets['created_at']), inplace=True)
trump_tweets
# time-indexing let's us perform neat things such as resampling
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
trump_tweets.resample('W').favorite_count.mean().plot()
trump_tweets.resample('W').retweet_count.mean().plot()
# Here we extract the tweets for year 2016. Can you also extract those for 2017 and 2018
trump_tweets_2016 = trump_tweets[trump_tweets.index.year == 2016]
# A quick lock at the created Dataframe
trump_tweets_2016.info()
# Can you find out the right column for the actual tweets?
tweets = trump_tweets_2016['text']
# We can use the tweet tokenizer to parse these tweets:

tknzr = TweetTokenizer()

# parse the tweets
tweets_tokenized = [tknzr.tokenize(tweet) for tweet in tweets]

# print out the 10 first tweets in parsed
print(tweets_tokenized[:10])
# Get out all hashtags using loops

# create an empty list: hashtags
hashtags = []

# Filter hashtags
for tweet in tweets_tokenized:
    hashtags.extend([word for word in tweet if word.startswith('#')])
    
# Print out the first 20 hashtags to check
print(hashtags[:20])
# Let't import the counter function
from collections import Counter
# Count all hashtags
hashtags_counter = Counter(hashtags)

# create an object and print out the most common 10 hashtags
most_common_10 = hashtags_counter.most_common(10)
print(most_common_10)
# Let's define a (a bit clunky but easy to read) function 
# that picks out the top10 hashtags for 1 year (performing the above)

def pick_top_10(year):
    tweet_df = trump_tweets[trump_tweets.index.year == year]
    tweets = tweet_df['text']
    tweets_tokenized = [tknzr.tokenize(tweet) for tweet in tweets]
    hashtags = []
    for tweet in tweets_tokenized:
        hashtags.extend([word for word in tweet if word.startswith('#')])
        hashtags_counter = Counter(hashtags)
    return dict(hashtags_counter.most_common(10))
years = [2015,2016,2017,2018]


top_10 = []

for year in years:
    top_10.append(pick_top_10(year))
top_10
# Importing stopwords
from nltk.corpus import stopwords
stopwords_en = stopwords.words('english')

# Let's import a lemmatizer from NLTK and try how it works
from nltk.stem import WordNetLemmatizer

# Instantiate the WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
# We already imported the data above and can use it right away

# Tokenize each tweet
trump_tweets['tokenized'] = trump_tweets['text'].map(lambda t: tknzr.tokenize(t))
# lowecase, strip and ensure we only include words
trump_tweets['tokenized'] = trump_tweets['tokenized'].map(
    lambda t: [word.lower().strip() for word in t if word.isalpha()])
# lemmarize and remove stopwords
trump_tweets['tokenized'] = trump_tweets['tokenized'].map(
    lambda t: [wordnet_lemmatizer.lemmatize(word) for word in t 
               if word not in stopwords_en])
# Quick check
trump_tweets['tokenized'][:10]
# We start by importing and initializing a Gensim Dictionary. 
# The dictionary will be used to map between words and IDs

from gensim.corpora.dictionary import Dictionary

# Create a Dictionary from the articles: dictionary
dictionary = Dictionary(trump_tweets['tokenized'])
# And this is how you can map back and forth
# Select the id for "hillary": hillary_id
hillary_id = dictionary.token2id.get('hillary')

# Use computer_id with the dictionary to print the word
print(dictionary.get(hillary_id))
# Create a Corpus: corpus
# We use a list comprehension to transform our abstracts into BoWs
corpus = [dictionary.doc2bow(tweet) for tweet in trump_tweets['tokenized']]
# Import the TfidfModel from Gensim
from gensim.models.tfidfmodel import TfidfModel

# Create and fit a new TfidfModel using the corpus: tfidf
tfidf = TfidfModel(corpus)

# Now we can transform the whole corpus
tfidf_corpus = tfidf[corpus]
# Just like before, we import the model
from gensim.models.lsimodel import LsiModel

# Fit a lsi model: lsi using the tfidf transformed corpus as input
lsi = LsiModel(tfidf_corpus, id2word=dictionary, num_topics=100)
# Inspect the first 10 topics
lsi.show_topics(num_topics=10)
# Create a transformed corpus using the lsi model from the tfidf corpus: lsi_corpus
lsi_corpus = lsi[tfidf_corpus]
# Load the MatrixSimilarity
from gensim.similarities import MatrixSimilarity

# Create the document-topic-matrix
document_topic_matrix = MatrixSimilarity(lsi_corpus)
document_topic_matrix = document_topic_matrix.index
pd.DataFrame(document_topic_matrix.dot(document_topic_matrix.T))
# Let's identify some clusters in our corpus

# We import KMeans form the Sklearn library
from sklearn.cluster import KMeans

# Instatiate a model with 4 clusters
kmeans = KMeans(n_clusters = 10)

# And fit it on our document-topic-matrix
kmeans.fit(document_topic_matrix)
# Let's annotate our abstracts with the assigned cluster number
trump_tweets['cluster'] = kmeans.labels_
# We can try to visualize our documents using TSNE - 
# an approach for visualizing high-dimensional data

# Import the module first
#from sklearn.manifold import TSNE

#faster as TSNE
import umap

visualization = umap.UMAP().fit_transform(document_topic_matrix)

# And instantiate
#tsne = TSNE()


# Let's try to boil down the 100 dimensions into 2
#visualization = tsne.fit_transform(document_topic_matrix)
# Import the plotting library

import matplotlib.pyplot as plt
import seaborn as sns
# Plot the trump_tweet map
plt.figure(figsize=(15,15))
sns.scatterplot(visualization[:,0],visualization[:,1], 
           data = trump_tweets, palette='RdBu', 
           hue=trump_tweets.cluster.values, legend='full')
# Collectiong

# Select a cluster e.g. 1
Cluster = 6

# Create an empty cluster token list: cluster_tweets
cluster_tweets = []

# Create a loop which iterates over all tokenized tweets in the 
# trump_tweets dataframe and extends the created list with them
for x in trump_tweets[trump_tweets['cluster'] == Cluster]['tokenized']:
    cluster_tweets.extend(x)
# Transfortm the selected tweets using the tfidf model

tweets_in_cluster_tfidf = tfidf[dictionary.doc2bow(cluster_tweets)]
# Sort the weights from highest to lowest: sorted_tfidf_weights
# this has been done for you
tweets_in_cluster_tfidf = sorted(tweets_in_cluster_tfidf, key=lambda w: w[1], reverse=True)

# Print the top 10 weighted words
for term_id, weight in tweets_in_cluster_tfidf[:10]:
    print(dictionary.get(term_id), weight)
# Tiny not too pretty graph
trump_tweets.groupby('cluster').resample('M')['cluster'].count().unstack().T.plot()
trump_tweets['is_retweet'] = trump_tweets['is_retweet'].astype(bool)
y = trump_tweets[trump_tweets['is_retweet'] == False].favorite_count.values
trump_tweets['indexing'] = range(len(trump_tweets))
x_selector = trump_tweets[trump_tweets['is_retweet'] == False]['indexing']
X = document_topic_matrix[x_selector,:]
# Splitting the dataset into the Training set and Test set (since we have a new output variable)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 21)
# Let's fit a simple linear regression model

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
# Let's fit a simple linear regression model

from sklearn.ensemble import GradientBoostingRegressor
regressor = GradientBoostingRegressor()

regressor.fit(X_train, y_train)
regressor.score(X_test, y_test)
y_pred = regressor.predict(X_test)
sns.regplot(y_test, y_pred)