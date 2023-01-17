import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
# Config
root_path = '../input/homework4_v2/'  # Relative path of homework data

# TF-IDF
max_df = 0.95        # Ignore words with high df. (Similar effect to stopword filtering)
min_df = 5           # Ignore words with low df.
smooth_idf = True    # Smooth idf weights by adding 1 to df.
sublinear_tf = True  # Replace tf with 1 + log(tf).

# Rocchio (Below is a param set called Ide Dec-Hi)
alpha = 1
beta = 0.75
gamma = 0.15
rel_count = 5   # Use top-5 relevant documents to update query vector.
nrel_count = 1  # Use only the most non-relevant document to update query vector.
iters = 5
with open(root_path + 'doc_list.txt') as file:
    doc_list = [line.rstrip() for line in file]
    
with open(root_path + 'query_list.txt') as file:
    query_list = [line.rstrip() for line in file]
documents, queries = [], []

for doc_name in doc_list:
    with open(root_path + 'Document/' + doc_name) as file:
        doc = ' '.join([word for line in file.readlines()[3:] for word in line.split()[:-1]])
        documents.append(doc)

for query_name in query_list:
    with open(root_path + 'Query/' + query_name) as file:
        query = ' '.join([word for line in file.readlines() for word in line.split()[:-1]])
        queries.append(query)
# Build TF-IDF vectors of docs and queries
vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df,
                             smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)
doc_tfidfs = vectorizer.fit_transform(documents).toarray()
query_vecs = vectorizer.transform(queries).toarray()

# Rank documents based on cosine similarity
cos_sim = cosine_similarity(query_vecs, doc_tfidfs)
rankings = np.flip(cos_sim.argsort(), axis=1)
for _ in range(iters):
    
    # Update query vectors with Rocchio algorithm
    rel_vecs = doc_tfidfs[rankings[:, :rel_count]].mean(axis=1)
    nrel_vecs = doc_tfidfs[rankings[:, -nrel_count:]].mean(axis=1)
    query_vecs = alpha * query_vecs + beta * rel_vecs - gamma * nrel_vecs
    
    # Rerank documents based on cosine similarity
    cos_sim = cosine_similarity(query_vecs, doc_tfidfs)
    rankings = np.flip(cos_sim.argsort(axis=1), axis=1)
with open('submission.csv', mode='w') as file:
    file.write('Query,RetrievedDocuments\n')
    for query_name, ranking in zip(query_list, rankings):
        ranked_docs = ' '.join([doc_list[idx] for idx in ranking])
        file.write('%s,%s\n' % (query_name, ranked_docs))