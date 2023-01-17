import os
import pandas as pd
import json
from tqdm import tqdm
import numpy as np
root_path = '/kaggle/input/CORD-19-research-challenge/'

dirs = [
    root_path+'biorxiv_medrxiv/biorxiv_medrxiv/pdf_json',
    root_path + 'comm_use_subset/comm_use_subset/pdf_json',
    root_path+'custom_license/custom_license/pdf_json',
    root_path+'noncomm_use_subset/noncomm_use_subset/pdf_json',
]
documents = []
for d in dirs:
    for file in tqdm(os.listdir(d)):
        j = json.load(open(d+f"/{file}","rb"))
        title = j['metadata']['title']
        
        abstract = ""
        if len(j['abstract']) > 0:
            abstract = j['abstract'][0]["text"]
            
        text = ''
        for t in j["body_text"]:
            text += t['text'] + "\n\n"
            
        documents += [[title, abstract, text]] 

df = pd.DataFrame(documents, columns=["title", 'abstract','text'])
df.size
            
    
df.head()
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

processed_abstracts = []
for abstract in tqdm(df['abstract'].tolist()):
    # Normalization
    normalized = re.sub(r"[^a-zA-Z0-9]", " ", abstract.lower())

    # Tokenization
    words = word_tokenize(normalized)

    # Removing stopwords
    no_stopwords = [w for w in words if w not in stopwords.words("english")]

    # Stemming 
    p = PorterStemmer()
    stemmed = [p.stem(w) for w in no_stopwords]

    processed_abstracts += [stemmed]
df['abstract'][df['abstract'].str.contains("smoker")].iloc[1]
df['abstract'][df['abstract'].str.contains("risk") & df['abstract'].str.contains("factors")]
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = df['text']

vectorizer = TfidfVectorizer()
X = vectorizer.fit(corpus)
# print(vectorizer.get_feature_names())

# print(X.shape)
riskFactorsTFIDF = vectorizer.transform(df['text'][df['abstract'].str.contains("smoking")])
feature_names = np.array(vectorizer.get_feature_names())


def get_top_tf_idf_words(response, top_n=2):
    sorted_nzs = np.argsort(response.data)[:-(top_n+1):-1]
    print(sorted_nzs)
    return feature_names[response.indices[sorted_nzs]]

get_top_tf_idf_words(riskFactorsTFIDF, 5)
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
 
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

def display_scores(vectorizer, tfidf_result):
    # http://stackoverflow.com/questions/16078015/
    scores = zip(vectorizer.get_feature_names(),
                 np.asarray(tfidf_result.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    for item in sorted_scores:
        print("{0:50} Score: {1}".format(item[0], item[1]))
 
display_scores(vectorizer, vectorizer.transform(df['text'][df['abstract'].str.contains("smoking")].iloc[1]))
# you only needs to do this once, this is a mapping of index to 
feature_names=vectorizer.get_feature_names()
 
# get the document that we want to extract keywords from
doc= df['text'][df['abstract'].str.contains("smoking")].iloc[3]
 
#generate tf-idf for the given document
tf_idf_vector=vectorizer.transform([doc])
 
#sort the tf-idf vectors by descending order of scores
sorted_items=sort_coo(tf_idf_vector.tocoo())
 
#extract only the top n; n here is 10
keywords=extract_topn_from_vector(feature_names,sorted_items,10)
 
# now print the results
print("\n===Keywords===")
for k in keywords:
    print(k,keywords[k])

path = "../input/googles-trained-word2vec-model-in-python/GoogleNews-vectors-negative300.bin"

import gensim
model = gensim.models.KeyedVectors.load_word2vec_format(path,binary=True)
model.most_similar(positive=['weed'], topn = 20)

