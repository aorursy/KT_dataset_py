# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.probability import FreqDist
from scipy.cluster.hierarchy import ward, dendrogram, leaves_list, to_tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
# Open the csv file which contains the text of the book
bg = pd.read_csv('../input/bhagavad-gita.csv')
titles = bg['title'].tolist()
texts = bg['verse_text_no_samdhis'].tolist()
# Create tokentype counts and frequency distribution plots for examination (optional)
full_text = ' '.join(texts)
tokens = nltk.word_tokenize(full_text)
sortedset = sorted(set(tokens))

sortedset_counts = {}
for tokentype in sortedset:
    sortedset_counts[tokentype] = tokens.count(tokentype)
sortedset_counts = sorted(((v,k) for k,v in sortedset_counts.items()), reverse=True)

fdist = FreqDist(tokens)
fdist.plot(50, cumulative=True)
# Create a function for creating a vocabulary frame and for the parameter definition of the TfidfVectorizer

def tokenizer(text):
    tokens = [word for word in nltk.word_tokenize(text)]
    return tokens

totalvocab = []
for i in tqdm(texts):
    allwords_tokenized = tokenizer(i)
    totalvocab.extend(allwords_tokenized)

vocab_frame = pd.DataFrame({'words': totalvocab})
# Define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=1.0, max_features=200000,
                                 min_df=1, use_idf=True,
                                 tokenizer=tokenizer, ngram_range=(1,3))
# Fit the vectorizer to texts
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
print(tfidf_matrix.shape)
tfidf_matrix_array = tfidf_matrix.toarray()
# Get the distance matrix
dist = 1 - cosine_similarity(tfidf_matrix)
# Hierarchical clustering
# Define the linkage_matrix using ward clustering pre-computed distances
linkage_matrix = ward(dist)
# Plot the dendrogram

# Set size
fig, ax = plt.subplots(figsize=(25, 100))
ax = dendrogram(linkage_matrix, orientation="right", labels=titles, distance_sort=True, leaf_font_size=10);

# Show plot with tight layout
plt.tight_layout()
# Create a function to get the titles of verses and their distance from the rows of the linkage matrix
linkage_matrix_tree = to_tree(linkage_matrix, rd=True)

def get_verses(linkage_matrix_row_number):
    first_node_id = int(linkage_matrix[linkage_matrix_row_number][0])
    second_node_id = int(linkage_matrix[linkage_matrix_row_number][1])
    if first_node_id in leaves_list(linkage_matrix):
        first = titles[first_node_id]
    else:
        node_id_list = linkage_matrix_tree[1][first_node_id].pre_order()
        first = []
        for n_id in node_id_list:
            first.append(titles[n_id])
    if second_node_id in leaves_list(linkage_matrix):
        second = titles[second_node_id]
    else:
        node_id_list = linkage_matrix_tree[1][second_node_id].pre_order()
        second = []
        for n_id in node_id_list:
            second.append(titles[n_id])
    verses = [first, second, linkage_matrix[linkage_matrix_row_number][2]]
    return verses
# Get the 40 most similar verse-pairs or pairs of verse-pairs
for i in range(40):
    print(get_verses(i))