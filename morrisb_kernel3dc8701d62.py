# Set number of topics for LSI/LDA
nTopics = 8

# Number of maximal points to plot
points = 1000

# Set subject of the analysis
subject = 'Star Wars Episode IV'
# To store data
import pandas as pd

# To do linear algebra
import numpy as np

# To plot graphs
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import rgb2hex

# To create nicer graphs
import seaborn as sns

# To create interactive graphs
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

# To vectorize texts
from sklearn.feature_extraction.text import CountVectorizer
# To decompose texts
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
# To visualize high dimensional dataset
from sklearn.manifold import TSNE

# To tag words
from textblob import TextBlob

# To use new datatypes
from collections import Counter

# To stop words
from nltk.corpus import stopwords
stop = stopwords.words('english')
df = pd.read_table('../input/SW_EpisodeIV.txt', delim_whitespace=True, header=0, escapechar='\\').rename(columns={'dialogue':'text'}).dropna(subset=['text'], axis=0)
print('DataFrame Shape: {}'.format(df.shape))
df.head()
# Create vectorizer
countVectorizer = CountVectorizer(stop_words=stop)

# Vectorize text
vectorizedText = countVectorizer.fit_transform(df['text'].str.replace("'", '').values)
print('Shape Vectorized Text: {}'.format(vectorizedText.shape))
# Plot n most frequent words
n = 20


def nMostFrequentWords(n, countVectorizer, vectorizedText):    
    # Count word appearences in text
    vectorizedCount = np.sum(vectorizedText, axis=0)
    
    # Get word indices and counts
    wordIndices = np.flip(np.argsort(vectorizedCount), 1)
    wordCounts = np.flip(np.sort(vectorizedCount),1)

    # Create wordvectors to inverse-transform them
    wordVectors = np.zeros((n, vectorizedText.shape[1]))
    for i in range(n):
        wordVectors[i, wordIndices[0,i]] = 1

    # Inverse-transfrom the wordvectors
    words = [word[0].encode('ascii').decode('utf-8') for word in countVectorizer.inverse_transform(wordVectors)]

    # Return word and word-counts
    return (words, wordCounts[0, :n].tolist()[0])



# Get most frequent words with wordcounts
words, wordCounts = nMostFrequentWords(n=n, countVectorizer=countVectorizer, vectorizedText=vectorizedText)

# Create colormap
cmap = get_cmap('viridis')
colors = [rgb2hex(cmap(color)) for color in np.arange(0, 1.000001, 1/(n-1))]

# Create plot
data = go.Bar(x = words,
              y = wordCounts,
              marker = dict(color = colors))

layout = go.Layout(title = 'Most Frequent {} Words In {}'.format(n, subject),
                   xaxis = dict(title = 'Words'),
                   yaxis = dict(title = 'Count'))

fig = go.Figure(data=[data], layout=layout)
iplot(fig)
# Tags and descriptions
tag_dict = {"CC":"conjunction, coordinating; and, or, but",
                "CD":"cardinal number; five, three, 13%",
                "DT":"determiner; the, a, these",
                "EX":"existential there; there were six boys",
                "FW":"foreign word; mais",
                "IN":"conjunction, subordinating or preposition; of, on, before, unless",
                "JJ":"adjective; nice, easy",
                "JJR":"adjective, comparative; nicer, easier",
                "JJS":"adjective, superlative; nicest, easiest",
                "LS":"list item marker; ",
                "MD":"verb, modal auxillary; may, should",
                "NN":"noun, singular or mass; tiger, chair, laughter",
                "NNS":"noun, plural; tigers, chairs, insects",
                "NNP":"noun, proper singular; Germany, God, Alice",
                "NNPS":"noun, proper plural; we met two Christmases ago",
                "PDT":"predeterminer; both his children",
                "POS":"possessive ending; 's",
                "PRP":"pronoun, personal; me, you, it",
                "PRP$":"pronoun, possessive; my, your, our",
                "RB":"adverb; extremely, loudly, hard",
                "RBR":"adverb, comparative; better",
                "RBS":"adverb, superlative; best",
                "RP":"adverb, particle; about, off, up",
                "SYM":"symbol; %",
                "TO":"infinitival to; what to do?",
                "UH":"interjection; oh, oops, gosh",
                "VB":"verb, base form; think",
                "VBZ":"verb, 3rd person singular present; she thinks",
                "VBP":"verb, non-3rd person singular present; I think",
                "VBD":"verb, past tense; they thought",
                "VBN":"verb, past participle; a sunken ship",
                "VBG":"verb, gerund or present participle; thinking is fun",
                "WDT":"wh-determiner; which, whatever, whichever",
                "WP":"wh-pronoun, personal; what, who, whom",
                "WP$":"wh-pronoun, possessive; whose, whosever",
                "WRB":"wh-adverb; where, when"}
# Apply tag-function to DataFrame, stack tags and count them
tag_df = pd.DataFrame.from_records(df['text'].apply(lambda x: [tag for word, tag in TextBlob(x).pos_tags]).tolist()).stack().value_counts().reset_index().rename(columns={'index':'tag', 0:'count'})


# Create colormap
n = tag_df.shape[0]
cmap = get_cmap('viridis')
colors = [rgb2hex(cmap(color)) for color in np.arange(0, 1.000001, 1/(n-1))]

# Create plot
data = go.Bar(x = tag_df['tag'],
              y = tag_df['count'],
              text = tag_df['tag'].apply(lambda x: tag_dict[x] if x in tag_dict.keys() else x),
              marker = dict(color = colors))

layout = go.Layout(title = 'Most Frequent Tags In {}'.format(subject),
                   xaxis = dict(title = 'Type Of Word'),
                   yaxis = dict(title = 'Count'))

fig = go.Figure(data=[data], layout=layout)
iplot(fig)
# Create LSI and fit
lsiModel = TruncatedSVD(n_components=nTopics)
lsiTopicMatrix = lsiModel.fit_transform(vectorizedText)
print('Shape LSI Topic Matrix: {}'.format(lsiTopicMatrix.shape))

# Get most probable keys and all categories with counts
lsiKeys = lsiTopicMatrix.argmax(axis=1)
lsiCategories, lsiCounts = zip(*Counter(lsiKeys).items())
def getTopWords(n, lsiKeys, vectorizedText, countVectorizer):
    # Create empty array for mean
    wordMean = np.zeros((nTopics, vectorizedText.shape[1]))
    # Iterate over each topic
    for i in np.unique(lsiKeys):
        wordMean[i] += vectorizedText.toarray()[lsiKeys==i].mean(axis=0)
        
    # Sort and get the most frequent n words for each topic
    topWordsIndices = np.flip(np.argsort(wordMean, axis=1)[:, -n:], axis=1)
    topWordsPercentage = (np.divide(np.flip(np.sort(wordMean, axis=1)[:, -n:], axis=1), (np.sum(wordMean, axis=1)+0.0000001)[:, None])*100).astype(int)


    # Store all words for all topics
    topWords = []

    # Iterate over the topics with its indices
    for i, (topic, percentage) in enumerate(zip(topWordsIndices, topWordsPercentage)):
        # Store all words for one topic
        topicWords = []

        if i in np.unique(lsiKeys):
            # Iterate over the indices for the topic
            for index, percent in zip(topic, percentage):
                # Create a wordvector for the index
                wordVector = np.zeros((vectorizedText.shape[1]))
                wordVector[index] = 1
                # Inverse-transfor the wordvector
                word = countVectorizer.inverse_transform(wordVector)[0][0]
                # Store the word
                topicWords.append('{}% '.format(percent) + word.encode('ascii').decode('utf-8'))
        # Store all words for the topic
        topWords.append(', '.join(topicWords))

    return topWords
# Get top n words
topWords = getTopWords(5, lsiKeys, vectorizedText, countVectorizer)

# Print the topics and its words
for i, words in enumerate(topWords):
    print('Topic {}: {}'.format(i, words))
# Sort data
lsiCategoriesSorted, lsiCountsSorted = zip(*sorted(zip(lsiCategories, lsiCounts)))

# Create labels
topWords = getTopWords(5, lsiKeys, vectorizedText, countVectorizer)
labels = ['Topic {}'.format(i) for i in lsiCategoriesSorted]

# Create colormap
n = nTopics
cmap = get_cmap('viridis')
colors = [rgb2hex(cmap(color)) for color in np.arange(0, 1.000001, 1/(n-1))]

# Create plot
data = go.Bar(x = labels,
              y = lsiCountsSorted,
              text = [word for word in topWords if word],
              marker = dict(color = colors))

layout = go.Layout(title = 'Most Frequent LSI Topics In {}'.format(subject),
                   xaxis = dict(title = 'Topic'),
                   yaxis = dict(title = 'Count'))

fig = go.Figure(data=[data], layout=layout)
iplot(fig)
# Transform high dimensional dataset to visualize in 2D
tsneModel = TSNE(n_components=2, perplexity=50, learning_rate=100, n_iter=2000, verbose=1, random_state=0, angle=0.75)
tsneModelVectors = tsneModel.fit_transform(lsiTopicMatrix)
# Create colormap
n = nTopics
cmap = get_cmap('tab10')
colors = [rgb2hex(cmap(color)) for color in np.arange(0, 1.000001, 1/(n-1))]

# Get n top words
topWords = getTopWords(3, lsiKeys, vectorizedText, countVectorizer)


# Create plot
data = []
# Iterate over each topic
for topic in range(nTopics):
    # Mask for a single topic
    mask = lsiKeys==topic
    # Mask for sampling
    sample_mask = np.zeros(mask.sum()).astype(bool)
    sample_mask[:int(points/nTopics)] = True
    np.random.shuffle(sample_mask)
    
    scatter = go.Scatter(x = tsneModelVectors[mask,0][sample_mask],
                         y = tsneModelVectors[mask,1][sample_mask],
                         name = 'Topic {}: {}'.format(topic, topWords[topic]),
                         mode = 'markers',
                         text = df[mask]['text'][sample_mask],
                         marker = dict(color = colors[topic]))
    data.append(scatter)

layout = go.Layout(title = 't-SNE Clustering of {} LSI Topics'.format(nTopics),
                   showlegend=True,
                   hovermode = 'closest')

fig = go.Figure(data=data, layout=layout)
iplot(fig)
# Create LDA and fit
ldaModel = LatentDirichletAllocation(n_components=nTopics, learning_method='online', random_state=0, verbose=0)
ldaTopicMatrix = ldaModel.fit_transform(vectorizedText)
print('Shape LSI Topic Matrix: {}'.format(ldaTopicMatrix.shape))

# Get most probable keys and all categories with counts
ldaKeys = ldaTopicMatrix.argmax(axis=1)
ldaCategories, ldaCounts = zip(*Counter(ldaKeys).items())
# Get top n words
topWords = getTopWords(5, ldaKeys, vectorizedText, countVectorizer)

# Print the topics and its words
for i, words in enumerate(topWords):
    print('Topic {}: {}'.format(i, words))
# Sort data
ldaCategoriesSorted, ldaCountsSorted = zip(*sorted(zip(ldaCategories, ldaCounts)))

# Create labels
topWords = getTopWords(5, ldaKeys, vectorizedText, countVectorizer)
labels = ['Topic {}'.format(i) for i in ldaCategoriesSorted]

# Create colormap
n = nTopics
cmap = get_cmap('viridis')
colors = [rgb2hex(cmap(color)) for color in np.arange(0, 1.000001, 1/(n-1))]

# Create plot
data = go.Bar(x = labels,
              y = ldaCountsSorted,
              text = [word for word in topWords if word],
              marker = dict(color = colors))

layout = go.Layout(title = 'Most Frequent LDA Topics In {}'.format(subject),
                   xaxis = dict(title = 'Topic'),
                   yaxis = dict(title = 'Count'))

fig = go.Figure(data=[data], layout=layout)
iplot(fig)
# Transform high dimensional dataset to visualize in 2D
tsneModel = TSNE(n_components=2, perplexity=50, learning_rate=100, n_iter=2000, verbose=1, random_state=0, angle=0.75)
tsneModelVectors = tsneModel.fit_transform(ldaTopicMatrix)
# Create colormap
n = nTopics
cmap = get_cmap('tab10')
colors = [rgb2hex(cmap(color)) for color in np.arange(0, 1.000001, 1/(n-1))]

# Get n top words
topWords = getTopWords(3, ldaKeys, vectorizedText, countVectorizer)


# Create plot
data = []
# Iterate over each topic
for topic in range(nTopics):
    # Mask for a single topic
    mask = ldaKeys==topic
    # Mask for sampling
    sample_mask = np.zeros(mask.sum()).astype(bool)
    sample_mask[:int(points/nTopics)] = True
    np.random.shuffle(sample_mask)
    
    scatter = go.Scatter(x = tsneModelVectors[mask,0][sample_mask],
                         y = tsneModelVectors[mask,1][sample_mask],
                         name = 'Topic {}: {}'.format(topic, topWords[topic]),
                         mode = 'markers',
                         text = df[mask]['text'][sample_mask],
                         marker = dict(color = colors[topic]))
    data.append(scatter)

layout = go.Layout(title = 't-SNE Clustering of {} LDA Topics'.format(nTopics),
                   showlegend=True,
                   hovermode = 'closest')

fig = go.Figure(data=data, layout=layout)
iplot(fig)
# Create LDA and fit
pcaModel = PCA(n_components=nTopics, random_state=0)
pcaTopicMatrix = pcaModel.fit_transform(vectorizedText.toarray())
print('Shape PCA Topic Matrix: {}'.format(pcaTopicMatrix.shape))

# Get most probable keys and all categories with counts
pcaKeys = pcaTopicMatrix.argmax(axis=1)
pcaCategories, pcaCounts = zip(*Counter(pcaKeys).items())
# Get top n words
topWords = getTopWords(5, pcaKeys, vectorizedText, countVectorizer)

# Print the topics and its words
for i, words in enumerate(topWords):
    print('Topic {}: {}'.format(i, words))
# Sort data
pcaCategoriesSorted, pcaCountsSorted = zip(*sorted(zip(pcaCategories, pcaCounts)))

# Create labels
topWords = getTopWords(5, pcaKeys, vectorizedText, countVectorizer)
labels = ['Topic {}'.format(i) for i in pcaCategoriesSorted]

# Create colormap
n = nTopics
cmap = get_cmap('viridis')
colors = [rgb2hex(cmap(color)) for color in np.arange(0, 1.000001, 1/(n-1))]

# Create plot
data = go.Bar(x = labels,
              y = pcaCountsSorted,
              text = [word for word in topWords if word],
              marker = dict(color = colors))

layout = go.Layout(title = 'Most Frequent PCA Topics In {}'.format(subject),
                   xaxis = dict(title = 'Topic'),
                   yaxis = dict(title = 'Count'))

fig = go.Figure(data=[data], layout=layout)
iplot(fig)
# Transform high dimensional dataset to visualize in 2D
tsneModel = TSNE(n_components=2, perplexity=50, learning_rate=100, n_iter=2000, verbose=1, random_state=0, angle=0.75)
tsneModelVectors = tsneModel.fit_transform(pcaTopicMatrix)
# Create colormap
n = nTopics
cmap = get_cmap('tab10')
colors = [rgb2hex(cmap(color)) for color in np.arange(0, 1.000001, 1/(n-1))]

# Get n top words
topWords = getTopWords(3, pcaKeys, vectorizedText, countVectorizer)


# Create plot
data = []
# Iterate over each topic
for topic in range(nTopics):
    # Mask for a single topic
    mask = pcaKeys==topic
    # Mask for sampling
    sample_mask = np.zeros(mask.sum()).astype(bool)
    sample_mask[:int(points/nTopics)] = True
    np.random.shuffle(sample_mask)
    
    scatter = go.Scatter(x = tsneModelVectors[mask,0][sample_mask],
                         y = tsneModelVectors[mask,1][sample_mask],
                         name = 'Topic {}: {}'.format(topic, topWords[topic]),
                         mode = 'markers',
                         text = df[mask]['text'][sample_mask],
                         marker = dict(color = colors[topic]))
    data.append(scatter)

layout = go.Layout(title = 't-SNE Clustering of {} PCA Topics'.format(nTopics),
                   showlegend=True,
                   hovermode = 'closest')

fig = go.Figure(data=data, layout=layout)
iplot(fig)
# Create LDA and fit
spcaModel = SparsePCA(n_components=nTopics, random_state=0)
spcaTopicMatrix = spcaModel.fit_transform(vectorizedText.toarray())
print('Shape SPCA Topic Matrix: {}'.format(spcaTopicMatrix.shape))

# Get most probable keys and all categories with counts
spcaKeys = spcaTopicMatrix.argmax(axis=1)
spcaCategories, spcaCounts = zip(*Counter(spcaKeys).items())
# Get top n words
topWords = getTopWords(5, spcaKeys, vectorizedText, countVectorizer)

# Print the topics and its words
for i, words in enumerate(topWords):
    print('Topic {}: {}'.format(i, words))
# Sort data
spcaCategoriesSorted, spcaCountsSorted = zip(*sorted(zip(spcaCategories, spcaCounts)))

# Create labels
topWords = getTopWords(5, spcaKeys, vectorizedText, countVectorizer)
labels = ['Topic {}'.format(i) for i in spcaCategoriesSorted]

# Create colormap
n = nTopics
cmap = get_cmap('viridis')
colors = [rgb2hex(cmap(color)) for color in np.arange(0, 1.000001, 1/(n-1))]

# Create plot
data = go.Bar(x = labels,
              y = spcaCountsSorted,
              text = [word for word in topWords if word],
              marker = dict(color = colors))

layout = go.Layout(title = 'Most Frequent SPCA Topics In {}'.format(subject),
                   xaxis = dict(title = 'Topic'),
                   yaxis = dict(title = 'Count'))

fig = go.Figure(data=[data], layout=layout)
iplot(fig)
# Transform high dimensional dataset to visualize in 2D
tsneModel = TSNE(n_components=2, perplexity=50, learning_rate=100, n_iter=2000, verbose=1, random_state=0, angle=0.75)
tsneModelVectors = tsneModel.fit_transform(spcaTopicMatrix)
# Create colormap
n = nTopics
cmap = get_cmap('tab10')
colors = [rgb2hex(cmap(color)) for color in np.arange(0, 1.000001, 1/(n-1))]

# Get n top words
topWords = getTopWords(3, spcaKeys, vectorizedText, countVectorizer)


# Create plot
data = []
# Iterate over each topic
for topic in range(nTopics):
    # Mask for a single topic
    mask = spcaKeys==topic
    # Mask for sampling
    sample_mask = np.zeros(mask.sum()).astype(bool)
    sample_mask[:int(points/nTopics)] = True
    np.random.shuffle(sample_mask)
    
    scatter = go.Scatter(x = tsneModelVectors[mask,0][sample_mask],
                         y = tsneModelVectors[mask,1][sample_mask],
                         name = 'Topic {}: {}'.format(topic, topWords[topic]),
                         mode = 'markers',
                         text = df[mask]['text'][sample_mask],
                         marker = dict(color = colors[topic]))
    data.append(scatter)

layout = go.Layout(title = 't-SNE Clustering of {} SPCA Topics'.format(nTopics),
                   showlegend=True,
                   hovermode = 'closest')

fig = go.Figure(data=data, layout=layout)
iplot(fig)
text = "Hey, Han Solo what's up?"

textVector = countVectorizer.transform([text])
newTransformedVector = spcaModel.transform(textVector.toarray())
topic = np.argmax(newTransformedVector)
print('Topic {}: {} '.format(topic, text))
