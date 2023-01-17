text = """This book is about the economics of innovation and knowledge. One of the major conclusions drawn is that the perspectives standard economics imposes on society are biased, incomplete and inadequate. The focus on rational choice, allocation of scarce resources and equilibrium only captures some dimensions of the modern economy, notably short-term and static ones. Alternative perspectives, in which the focus is on learning as an interactive process and on processes of innovation, give visibility and direct attention to other, at least equally important and more dynamic, dimensions.
Social science is about human action and interaction, and it differs from natural science in several respects. It does not have access to laboratories where it is possible to organize controlled experiments. In spite of this, standard economics has gone far in adopting criteria and ideals from natural science, more precisely ideals that originate from Newtonian physics. This is reflected in standard economists’ conception of equilibrium as an ideal reference state and their tendency to focus exclusively on quantitative relations, also paired with in its excessive use of mathematics.
In this book, I insist that economics should remain a social science while also taking into account the complexity of the strivings and hopes of human beings. People cannot be reduced to algorithms or automatons. The basic assumption about rational behaviour in economic models (in which individu- als and firms act as if they know everything about the future) is absurd and leads to equally absurd conclusions and to dubious policy recommendations.
Taking a departure from more realistic assumptions about how and why people act as they do in society has implications for what constitutes a theory in social science. In social science, a theory should be regarded as a focusing device – no more and no less. This book presents two sets of theories or focus- ing devices – the innovation system and the learning economy that differ from those used in standard economics. These alternative focusing devices help us to see the core institutions in the economy (such as the market, the competition regime, the firm, the law, etc.) in a different light than that cast by mainstream economic theory.
What is currently presented as the only and necessary pathway for the economy and for economic policy aiming at competitiveness and growth at the national level actually undermines both. The only certain outcome of cur- rent national strategies with focus on fiscal balance and cost competitiveness is that the rich get richer and the poor stay poor. Using an alternative analytical perspective, where the focus is on processes of innovation and learning, points in other possible directions for institutional design and economic policy, where the focus is on collective entrepreneurship, knowledge sharing and international collaboration."""
# We can split the text-chunk into something like sentences.
split_text = text.split('.')
print(split_text)
# print out the first stentence
sentence_3 = split_text[2]
print(sentence_3)
# Let's create tokens
tokens_sentence_3 = [word for word in sentence_3.split(' ')]
print(tokens_sentence_3)
# Let's lowercase all these tokens and clean up the \n (new line command)
# Also we will replace "()" as well as make sure that only words lend in our list
tokens_sentence_3_lower = [word.lower().strip() for word in sentence_3.split(' ')]
print('### OUTPUT1 ###')
print(tokens_sentence_3_lower)
print('\n')
    
tokens_sentence_3_lower = [word.replace('(','').replace(')','') for word in tokens_sentence_3_lower if word.isalpha()]

print('### OUTPUT2 ###')
print(tokens_sentence_3_lower)

# Removing stopwords

stopwords_en = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
tokens_sentence_3_clean = [word for word in tokens_sentence_3_lower if word not in stopwords_en]
print(tokens_sentence_3_clean)
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
# Use word_tokenize to tokenize the third sentence: tokenized_sent
tokenized_sent = word_tokenize(sentences[2])

# Make a set of unique tokens in the entire scene: unique_tokens
unique_tokens = set(word_tokenize(text))
print(unique_tokens)
tweets = ["On behalf of @FLOTUS Melania & myself, THANK YOU for today's update & GREAT WORK! #SouthernBaptist @SendRelief,… https://t.co/4yZCeXCt6n",
"I will be going to Texas and Louisiana tomorrow with First Lady. Great progress being made! Spending weekend working at White House.",
"Stock Market up 5 months in a row!",
"'President Donald J. Trump Proclaims September 3, 2017, as a National Day of Prayer' #HurricaneHarvey #PrayForTexas… https://t.co/tOMfFWwEsN",
"Texas is healing fast thanks to all of the great men & women who have been working so hard. But still so much to do. Will be back tomorrow!"]
# We can use the tweet tokenizer to parse these tweets:

tknzr = TweetTokenizer()
tweets_tokenized = [tknzr.tokenize(tweet) for tweet in tweets]
print(tweets_tokenized)
# Get out all hashtags using loops

hashtags = []

for tweet in tweets_tokenized:
    hashtags.extend([word for word in tweet if word.startswith('#')])
    
print(hashtags)
# We import the Counter module from python's standard collections

from collections import Counter

word_tokenized = word_tokenize(text)
bow = Counter(word_tokenized)
print(bow.most_common())
# Let's add some preprocessing

from nltk.corpus import stopwords

english_stopwords = stopwords.words('english')

word_tokenized = word_tokenize(text)

# lowercasing
cleaned_word_tokenized = [word.lower().strip() for word in word_tokenized]
# replacing some unwanted things
cleaned_word_tokenized = [word.replace('(','').replace(')','') for word in cleaned_word_tokenized if word.isalpha()]
# removing stopwords
cleaned_word_tokenized = [word for word in cleaned_word_tokenized if word not in english_stopwords]

bow = Counter(cleaned_word_tokenized)
print(bow.most_common())
# Let's import a lemmatizer from NLTK and try how it works
from nltk.stem import WordNetLemmatizer

# Instantiate the WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

# Lemmatize all tokens into a new list: lemmatized
lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in cleaned_word_tokenized]

# Create the bag-of-words: bow
bow = Counter(lemmatized)

# Print the 10 most common tokens
print(bow.most_common(10))
# We start by importing the data, ~1900 Abstracts/Titles from Scopus
import pandas as pd

abstracts = pd.read_csv('../input/abstracts.csv')
# Let's inspect the data
abstracts.head()
# Tokenize each abstract
abstracts['tokenized'] = abstracts['Abstract'].map(lambda t: word_tokenize(t))
# lowecase, strip and ensure it's words
abstracts['tokenized'] = abstracts['tokenized'].map(lambda t: [word.lower().strip() for word in t if word.isalpha()])
# lemmarize and remove stopwords
abstracts['tokenized'] = abstracts['tokenized'].map(lambda t: [wordnet_lemmatizer.lemmatize(word) for word in t if word not in stopwords_en])
# We start by importing and initializing a Gensim Dictionary. 
# The dictionary will be used to map between words and IDs

from gensim.corpora.dictionary import Dictionary

# Create a Dictionary from the articles: dictionary
dictionary = Dictionary(abstracts['tokenized'])
# And this is how you can map back and forth
# Select the id for "firm": firm_id
firm_id = dictionary.token2id.get("firm")

# Use computer_id with the dictionary to print the word
print(dictionary.get(firm_id))
# Create a Corpus: corpus
# We use a list comprehension to transform our abstracts into BoWs
corpus = [dictionary.doc2bow(abstract) for abstract in abstracts['tokenized']]
# Print the first 10 word ids with their frequency counts from the fifth document
print(corpus[10][:10])

# This is the same what we did before when we were counting words with the Counter (just in big)
# Sort the doc for frequency: bow_doc
bow_doc = sorted(corpus[10], key=lambda w: w[1], reverse=True)

# Print the top 5 words of the document alongside the count
for word_id, word_count in bow_doc[:10]:
    print(dictionary.get(word_id), word_count)
# Import the TfidfModel from Gensim
from gensim.models.tfidfmodel import TfidfModel

# Create and fit a new TfidfModel using the corpus: tfidf
tfidf = TfidfModel(corpus)

# Calculate the tfidf weights of doc: tfidf_weights
tfidf_weights = tfidf[corpus[10]]

# Print the first five weights
print(tfidf_weights[:5])
# Sort the weights from highest to lowest: sorted_tfidf_weights
sorted_tfidf_weights = sorted(tfidf_weights, key=lambda w: w[1], reverse=True)

# Print the top 5 weighted words
for term_id, weight in sorted_tfidf_weights[:10]:
    print(dictionary.get(term_id), weight)
# Now we can transform the whole corpus
tfidf_corpus = tfidf[corpus]
# Just like before, we import the model
from gensim.models.lsimodel import LsiModel

# And we fir it on the tfidf_corpus pointing to the dictionary as reference and the number of topics.
# In more serious settings one would pick between 300-400
lsi = LsiModel(tfidf_corpus, id2word=dictionary, num_topics=100)
# Once the model is ready, we can inspect the topics
lsi.show_topics(num_topics=10)
# And just as before, we can use the trained model to transform the corpus
lsi_corpus = lsi[tfidf_corpus]
# Load the MatrixSimilarity
from gensim.similarities import MatrixSimilarity

# Create the document-topic-matrix
document_topic_matrix = MatrixSimilarity(lsi_corpus)
document_topic_matrix = document_topic_matrix.index
# Let's identify some clusters in our corpus

# We import KMeans form the Sklearn library
from sklearn.cluster import KMeans

# Instatiate a model with 4 clusters
kmeans = KMeans(n_clusters=10)

# And fit it on our matrix
kmeans.fit(document_topic_matrix)
# Let's annotate our abstracts with the assigned cluster number
abstracts['cluster'] = kmeans.labels_
# We can try to visualize our documents using TSNE - an approach for visualizing high-dimensional data

# Import the module first
from sklearn.manifold import TSNE

# And instantiate
tsne = TSNE()

# Let's try to boil down the 100 dimensions into 2
visualization = tsne.fit_transform(document_topic_matrix)
# Import plotting library

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(15,15))
sns.scatterplot(visualization[:,0],visualization[:,1], data = abstracts, palette='RdBu', hue=abstracts.cluster, legend='full')
# Preprocessing
abstracts['title_tok'] = abstracts['Title'].map(lambda t: word_tokenize(t))
abstracts['title_tok'] = abstracts['title_tok'].map(lambda t: [word.lower().strip() for word in t if word.isalpha()])
abstracts['title_tok'] = abstracts['title_tok'].map(lambda t: [wordnet_lemmatizer.lemmatize(word) for word in t if word not in stopwords_en])
# Collectiong

Cluster = 2

cluster_titles = []
for x in abstracts[abstracts['cluster'] == Cluster]['title_tok']:
    cluster_titles.extend(x)
# Transfortm into tf_idf format
titles_tfidf = tfidf[dictionary.doc2bow(cluster_titles)]
# Sort the weights from highest to lowest: sorted_tfidf_weights
titles_tfidf = sorted(titles_tfidf, key=lambda w: w[1], reverse=True)

# Print the top 5 weighted words
for term_id, weight in titles_tfidf[:20]:
    print(dictionary.get(term_id), weight)
