# Import modules
import numpy as np
import pandas as pd
import nltk

# Set seed for reproducibility
np.random.seed(5)

# Read in IMDb and Wikipedia movie data (both in same file)
movies_df = pd.read_csv ('../input/movies/movies.csv')

print("Number of movies loaded: %s " % (len(movies_df)))

# Display the data
movies_df.head ()
# Combine wiki_plot and imdb_plot into a single column
movies_df['plot'] = movies_df['wiki_plot'].astype(str) + "\n" + \
                 movies_df['imdb_plot'].astype(str)

# Inspect the new DataFrame
movies_df.head()
movies_df.shape
movies_df.info ()
import re
# Tokenize a paragraph into sentences and store in sent_tokenized
sent_tokenized = [sent for sent in nltk.sent_tokenize("""
                        Today (May 19, 2016) is his only daughter's wedding. 
                        Vito Corleone is the Godfather.
                        """)]

# Word Tokenize first sentence from sent_tokenized, save as words_tokenized
words_tokenized = [word for word in nltk.word_tokenize (sent_tokenized[0])]

# Remove tokens that do not contain any letters from words_tokenized

filtered = [word for word in words_tokenized if re.search('[a-zA-Z]', word)]

# Display filtered words to observe words after tokenization
filtered
# Import the SnowballStemmer to perform stemming
from nltk.stem.snowball import SnowballStemmer

# Create an English language SnowballStemmer object
stemmer = SnowballStemmer("english")

# Print filtered to observe words without stemming
print("Without stemming: ", filtered)

# Stem the words from filtered and store in stemmed_words
stemmed_words = [stemmer.stem (word) for word in filtered]

# Print the stemmed_words to observe words after stemming
print("After stemming:   ", stemmed_words)
# Define a function to perform both stemming and tokenization
def tokenize_and_stem (text):
    
    # Tokenize by sentence, then by word
    tokens = [word for sent in nltk.sent_tokenize (text) for word in nltk.word_tokenize (sent)]
    
    # Filter out raw tokens to remove noise
    filtered_tokens = [token for token in tokens if re.search('[a-zA-Z]', token)]
    
    # Stem the filtered_tokens
    stems = [stemmer.stem (word) for word in filtered_tokens]
    
    return stems

words_stemmed = tokenize_and_stem("Today (May 19, 2016) is his only daughter's wedding.")
print(words_stemmed)
words_stemmed = {}
for i in range (movies_df.shape [0]):
    words_stemmed.update ({i : tokenize_and_stem (movies_df.iloc [i,5])})
words_stemmed [0][0:5]
from sklearn.feature_extraction.text import CountVectorizer
vector = CountVectorizer (stop_words='english')
vector_count = vector.fit_transform (words_stemmed [0])
vector.get_feature_names()[0:5]
len (vector.get_feature_names())
df = pd.DataFrame (vector_count.toarray (), columns = vector.get_feature_names())
df = df.transpose ()
df
df ['count'] = df.sum (axis = 1)
df ['count'].nlargest (10)
# Import TfidfVectorizer to create TF-IDF vectors
from sklearn.feature_extraction.text import TfidfVectorizer

# Instantiate TfidfVectorizer object with stopwords and tokenizer
# parameters for efficient processing of text
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem,
                                 ngram_range=(1,3))
# Fit and transform the tfidf_vectorizer with the "plot" of each movie
# to create a vector representation of the plot summaries
tfidf_matrix = tfidf_vectorizer.fit_transform ([x for x in movies_df["plot"]])

print(tfidf_matrix.shape)
tfidf_matrix.toarray ()
tfidf_vectorizer.get_feature_names ()
# Import k-means to perform clusters
from sklearn.cluster import KMeans

# Create a KMeans object with 5 clusters and save as km
km = KMeans (n_clusters=5)

# Fit the k-means object with tfidf_matrix
km.fit (tfidf_matrix)

clusters = km.labels_.tolist()

# Create a column cluster to denote the generated cluster for each movie
movies_df["cluster"] = clusters

# Display number of films per cluster (clusters from 0 to 4)
movies_df['cluster'].value_counts() 
# Import cosine_similarity to calculate similarity of movie plots
from sklearn.metrics.pairwise import cosine_similarity

# Calculate the similarity distance
similarity_distance = 1 - cosine_similarity (tfidf_matrix)
similarity_distance
# Import matplotlib.pyplot for plotting graphs
import matplotlib.pyplot as plt

# Configure matplotlib to display the output inline
%matplotlib inline

# Import modules necessary to plot dendrogram
from scipy.cluster.hierarchy import linkage, dendrogram
# Create mergings matrix 
mergings = linkage (similarity_distance, method='complete')

# Plot the dendrogram, using title as label column
dendrogram_ = dendrogram (mergings,
               labels=[x for x in movies_df["title"]],
               leaf_rotation=90,
               leaf_font_size=16,
)

# Adjust the plot
fig = plt.gcf()
_ = [lbl.set_color('r') for lbl in plt.gca().get_xmajorticklabels()]
fig.set_size_inches(108, 21)

# Show the plotted dendrogram
plt.show()
# Answer the question 
ans = "Gladiator"
print(ans)
