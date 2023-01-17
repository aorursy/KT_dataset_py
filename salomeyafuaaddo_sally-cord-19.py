# importing relevant libraries

import numpy as np # linear algebra

import pandas as pd

import glob

import json



import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer

import re

from nltk.corpus import stopwords

from allennlp.commands.elmo import ElmoEmbedder

from sklearn.cluster import DBSCAN

from sklearn import metrics

from sklearn.preprocessing import StandardScaler







import time

import random





plt.style.use('ggplot')
# loading our data

root_path = '/kaggle/input/CORD-19-research-challenge/'

metadata_path = f'{root_path}/metadata.csv'

meta_df = pd.read_csv(metadata_path, dtype={

    'pubmed_id': str,

    'Microsoft Academic Paper ID': str, 

    'doi': str

})

meta_df.head()
# having a quick peek at our data

meta_df.info()
# extracting only the abstracts from the metadata

abs_text = meta_df["abstract"].to_list()

#abs_text[0:10]
# preprocess text by taking out regular stop words

def preprocess(text, remove_stopwords=True):

   

    text = re.sub("[^a-zA-Z]"," ", str(text))

   

    words = text.lower().split()

    

    if remove_stopwords:

        stops = set(stopwords.words("english"))

        words = [w for w in words if not w in stops]



    cleaned_abstract=[]

    

    for word in words:

        cleaned_abstract.append(word)



    

    return(cleaned_abstract)



# a list of my preprocessed text

cleaned_abstract_body = []

for text in abs_text:

    cleaned_abstract_body.append( " ".join(preprocess(text)))
# I am working with 57k abstract

#cleaned_abstract_body[0:5]

abstracts = [i.split() for i in cleaned_abstract_body]

#print(abstracts) # r is a list of lists. Each list contained in the bigger list is the abstract(each) with its tokenised words

    
word_embeddings = []

elmo = ElmoEmbedder()

# vectorising the entire abstract

# vectorising 50 abstracts becos it takes infinity to run like 1000

text_vector = [elmo.embed_sentence(reviews)[0] for reviews in abstracts[0:2]]

# text_vector is a list containing the word vectors for the words in each abstract

text_vector
X=(np.concatenate(text_vector))



# Now we cluster our vector X using sklearn DBSCAN

eps = 0.3

min_samples = 4

   

X = StandardScaler().fit_transform(np.asarray(X))

plt.rcParams.update({'figure.max_open_warning': 0})



   

db = DBSCAN(eps, min_samples,algorithm="kd_tree").fit(X)



core_samples_mask = np.zeros_like(db.labels_, dtype=bool)

core_samples_mask[db.core_sample_indices_] = True

labels = db.labels_



n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print(n_clusters_)

n_noise_ = list(labels).count(-1)





# Black removed and is used for noise instead.

unique_labels = set(labels)

fig = plt.figure()

colors = [plt.cm.Spectral(each)

          for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):

    if k == -1:

        # Black used for noise.

        col = [0, 0, 0, 1]



    class_member_mask = (labels == k)



    xy = X[class_member_mask & core_samples_mask]

    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),

             markeredgecolor='k', markersize=14)



    xy = X[class_member_mask & ~core_samples_mask]

    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),

             markeredgecolor='k', markersize=6)





plt.title('Estimated number of clusters: %d' % n_clusters_)



fig.savefig('cluster_image.png', bbox_inches='tight')



# extracting the abstratcs from the clusters



index2 = []

counter = 0

for i in labels:

    if i != -1: # remove the noise labels

        index2.append(counter)

    counter+=1



label2 = []

for j in index2:

    k = labels[j]

    label2.append(k) # these are the labels for points in each cluster





d = {"Index":index2, "Labels":label2}



df = pd.DataFrame(d)



s = pd.Series(["Labels"])
# extracting the indices of the labels in the cluster

index_kk = []



for i in range(len(label2)):

    p= df[df['Labels']==i]

    

    index_kk.append(p.Index.tolist())

#print("index for each label:",index_kk )

index_labels = [x for x in index_kk if x != []]

#print ("These are the indices for each label",index_labels)
# extracting the reviews in each cluster

empty_lst = []

for i in index_labels:

    empty_lst.append([cleaned_abstract_body[j] for j in i])



#print ("these are the abstracts found in each cluster",empty_lst)	



#for i in range(n_clusters_):

    #print("Cluster %d:" % i,' %s' % empty_lst[i])

    #print("Cluster %d:" % i,' %s' % len(empty_lst[i]))



# all abstracts in a cluster is now a single token

clust = []

for i in range(len(empty_lst)):

    t= ' '.join(empty_lst[i])

    clust.append(t)

#print("These are the abstracts in a cluster",clust)



# building word-cluster count matrix



def fn_tdm_df(docs, xColNames = None):

    ''' create a term document matrix as pandas DataFrame

    with **kwargs you can pass arguments of CountVectorizer

    if xColNames is given the dataframe gets columns Names'''



    #initialize the  vectorizer

    vectorizer = CountVectorizer()

    x1 = vectorizer.fit_transform(clust)

    #create dataFrame

    df = pd.DataFrame(x1.toarray().transpose(), index = vectorizer.get_feature_names()) 

    

    if xColNames is not None:

        df.columns =  xColNames



    return df



for i in range(len(clust[:5])):

        

    count_matrix=fn_tdm_df(docs=clust, xColNames =None)

#print("This is the count matrix",count_matrix.head())
# building word-cluster PMI matrix



def pmi(df):

    '''

    Calculate the positive pointwise mutal information score for each entry

    https://en.wikipedia.org/wiki/Pointwise_mutual_information

    We use the log( p(y|x)/p(y) ), y being the column, x being the row

    '''

    # Get numpy array from pandas df

    arr = df.as_matrix()



    # p(y|x) probability of each t1 overlap within the row

    row_totals = arr.sum(axis=1).astype(float)

    prob_cols_given_row = (arr.T / row_totals).T



    # p(y) probability of each t1 in the total set

    col_totals = arr.sum(axis=0).astype(float)

    prob_of_cols = col_totals / sum(col_totals)



    # PMI: log( p(y|x) / p(y) )

    # This is the same data, normalized

    ratio = prob_cols_given_row / prob_of_cols

    ratio[ratio==0] = 0.00001

    _pmi = np.log(ratio)

    _pmi[_pmi < 0] = 0



    return _pmi



# PMI matrix into a data frame

c=pmi(count_matrix)

vectorizer = CountVectorizer()

x1 = vectorizer.fit_transform(clust)

PMI_df = pd.DataFrame(c, index = vectorizer.get_feature_names()) 
# extracting the labels of each cluster with a positive pmi score into a list

List = []

for i in range(

len(empty_lst)):

    col=PMI_df[i]

    List.append(col[col>0].sort_values(ascending=False).to_dict())

#print('These are the label vectors for each cluster')



for i in range(n_clusters_):

    print("Cluster_labels %d:\n" % i,' %s' % List[i])
# These are the keywords with their corresponding PMI scores

for i in range(n_clusters_):

    print("Cluster_labels %d:\n" % i,' %s' % List[0])
len(List[0])
# these are the words that DBSCAN clustered

empty_lst[:5]
type(c)
# APPLY LDA TO EXTRACT KEY WORDS
