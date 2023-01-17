# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
import numpy as np 
import pandas as pd
import json
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from nltk.util import ngrams
import collections

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))
        os.path.join(dirname, filename)
# Reading in the metadata on the dataset
root = "kaggle/input/CORD-19-research-challenge/"
meta_path = root + "metadata.csv"
meta_df = pd.read_csv(meta_path)
meta_df.head()
# Gets all of the file paths in a list such that they can be read in at one time
json_paths = glob.glob(root + "**/*.json", recursive=True)
json_paths = json_paths[0:len(json_paths)//8]
print("Number of papers: ", len(json_paths))
def all_json_to_df(jsons):
    """
    Creates 2 dataframes containing each paper's paper_id, abstract, and body text in lists.
    One is split on words and used in creating the trigrams, the other is a simple string relating
    to the content of the paper.
    
    :param jsons: glob of json paths to read in
    :return: 2 Dataframes, one character based and one split on words
    """
    d = {"paper_id": [], "paper_title": [], "text": []}
    d_split = {"paper_id": [], "text": []}
    
    # Loop through all of the papers
    for file_path in jsons:
        with open(file_path) as file:
            content = json.load(file)

            # Add the paper title and paper IDs to the DFs
            d['paper_title'] += [content['metadata']['title']]
            d['paper_id'] += [content['paper_id']]
            d_split['paper_id'] += [content['paper_id']]
            
            # Loop through the abstract sentences and append them into a single string
            abstract = content.get('abstract', [])
            text = ""
            for ab in abstract:
                text += ab['text']
            body = content['body_text']
            for bo in body:
                text += bo['text']
                
            # Add text to both DFs, splitting by words on one DF
            d['text'] += [text]
            d_split['text'] += [text.split(" ")]
    return pd.DataFrame(d), pd.DataFrame(d_split)

                        
covid_df, split_df = all_json_to_df(json_paths)
covid_df.head()
# For every paper, grab their list of 3-grams
grams = [ngrams(paper, 3) for paper in split_df['text']]

# Count how many times every trigram is found
grams = [collections.Counter(gram) for gram in grams]

# Generate the trigram corpus, consisting of a concatenated string of the x-most
# common phrases in a given paper
dataset = []
for gram in grams:
    # Get the n-most common grams
    phrases = gram.most_common(5)
    
    # Concatenate these phrases into a string
    vector = ' '
    for phrase, _ in phrases:
        vector = vector + ' ' + ' '.join(phrase)
        
    # Append to the dataset as this paper's vector
    dataset.append(vector)
    
# Fit the vectorizer based on the phrase corpus
vectorizer = TfidfVectorizer()
vectorizer.fit(dataset)

# Transform the input dataset of papers into their respective feature vectors
covid_matrix = vectorizer.transform(covid_df['text'].to_numpy()).toarray() * vectorizer.idf_

# Perform KMeans on the transformed papers to generate related clusters
clusters = 9 
kmeans = KMeans(n_clusters = clusters).fit(covid_matrix)

# Printing their bincounts to get a size of each cluster
print(np.bincount(kmeans.labels_))
# Continuously query the model to get related clusters from a paper abstract/sentence
while True:
    # Grab the query from the user
    query = [input("Enter keywords to search for ('!q' to quit): ")]
    
    # Check for sentinel value
    if query[0] == "!q":
        break
        
    # Vectorize the query through the TFID Vectorizer
    vec_query = vectorizer.transform(query).toarray() 
    vec_query = vec_query * vectorizer.idf_

    # Get more related cluster
    cluster = kmeans.predict(vec_query)

    # Get the indices of the labels associated with this cluster from the Kmeans model
    indices = np.where(kmeans.labels_ == cluster)
    
    # Grab a random index within the bounds of the cluster indices
    index = np.random.randint(0, len(indices[0]), 5)
    
    # Correlate that back to the paper index and grab from dataframe
    paper_index = indices[0][index]
    
    print("List of related papers:")
    for idx in paper_index:
        print("")
        print(covid_df['paper_title'][idx])
    