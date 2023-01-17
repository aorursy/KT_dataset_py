# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import glob
from sklearn.feature_extraction.text import TfidfVectorizer

# Import modules to get n-grams
import collections
from nltk.util import ngrams

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))
        os.path.join(dirname, filename)
        
# Any results you write to the current directory are saved as output.
# Checking out the metadata 
# I don't really know what we'd do with it tbh but like trying pandas so
root = "/kaggle/input/CORD-19-research-challenge/"
meta_path = root + "metadata.csv"
meta_df = pd.read_csv(meta_path)
meta_df.head()
## Trying to assemble the actual data in some meaningful way
# Found this glob thing in somone else's submission, actually seems dope in this situation
json_paths = glob.glob(root + "**/*.json", recursive=True)
json_paths = json_paths[0:len(json_paths)//6]
len(json_paths)
#f = open(json_paths[0])
#a = json.load(f)
def all_json_to_df(jsons):
    # Currently takes like 3 minutes so idk maybe make this faster somehow
    # Dictionary containing each paper's paper_id, abstract, and body text in lists.
    # Hopefully, the paper_id, abstract, and body_text at index n all line up to the same paper
    # Ok nevermind, since there are some texts with no abstract, we're just going to lump the abstract and the bodytext together. That way we can at least match up id and text
    # Will need to test that at some point 
    d = {"paper_id": [], "text": []}
    for file_path in jsons:
        with open(file_path) as file:
            content = json.load(file)
            d['paper_id'] += [content['paper_id']]
            abstract = content.get('abstract', [])
            text = ""
            for ab in abstract:
                text += ab['text']
            body = content['body_text']
            for bo in body:
                text += bo['text']
            d['text'] += [text.split(" ")]
    return pd.DataFrame(d)
                        
covid_df = all_json_to_df(json_paths)
covid_df.head()
stopwords_path = "/kaggle/input/stopwords/stopwords/english"
stopwords = np.loadtxt(stopwords_path, dtype=str)
stopwords # I know some of them look weird but it's cuz they're the stemmed versions of common words
# So I was trying for a while to make an efficient function that removed stopwords, but then found this neat solution on Stack Overflow instead
covid_df['text'].apply(lambda x: [word for word in x if word not in stopwords]) # This is fucking english omg Python why 
# Also so I ran this for like 20 minutes and it did not complete.... :_(
# For every paper, grab their list of 4-grams, in this case
grams = [ngrams(paper, 3) for paper in covid_df['text']]
# Count how many times every n-gram is found
grams = [collections.Counter(gram) for gram in grams]
# Check out the most common n-grams in the first paper
grams[0].most_common(10)
# P ez, just look at the documentation for TfidfVectorizer
# If you ever forget, a Tf-idf value is just a numeric representation of how often that word appears relative to its corpus
def vectorize_sum_shiz(text):
    vectorized_df = []
    
    # Loop through each paper, getting the tfid vector of it
    for paper in text:
        vectorizer = TfidfVectorizer()
        vectorized_df.append(vectorizer.fit_transform(paper))
        
    # Return the dataset
    return vectorized_df
for x in grams:
    print(x)

# Vectorize and perform dimensionality reduction
covid_matrx = vectorize_sum_shiz(grams) # Turns our text into a matrix of TF-IDF features
pca = PCA(n_components = .95) # Cuz .95 seems good I guess
covid_reduced = pca.fit_transform(covid_matrx) # Fit model on TF-IDF features, perform dimensionality reduction 
# Doing k-means things (elbow method) nevermind, we're doing 9 clusters just cuz
# Fuzzy k-means
clusters = 9 # Just cuz
kmeans = KMeans(n_clusters = clusters, random_state = 69).fit(covid_reduced)
n_grams = 3

# Query the model, make sure to n-gram the query 
while True:
    # Grab the query from the user
    query = input("Enter keywords to search for ('!q' to quit): ").split()
    
    # Check for sentinel value
    if query == "!q":
        break
    
    # Perform a 3-gram transform on the query and then vectorize
    grams = [query[i:i+n_grams] for i in range(len(query)-n_grams+1)]
    
    # Vectorize the grams through the TFID Vectorizer
    vec_query = vectorize_sum_shiz(grams)
    print(vec_query)
    
    # Perform PCA dim reduction
    vec_reduced = pca.fit_transforms(vec_query)
    
    # Get cluster likelihoods
    clusters = kmeans.predict(vec_reduced)
    print(clusters)
    
    # Get papers based on threshold coming back
    
    # Do formatting here or something, need to see how it comes back first
