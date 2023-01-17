# Import all of our libs

import re

import string

import json



from ast import literal_eval

from collections import defaultdict



import pandas as pd

import numpy as np



import matplotlib.pyplot as plt



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.preprocessing import Normalizer

from sklearn.pipeline import make_pipeline



from tqdm.notebook import tqdm
# Load Preprocessed Data

preprocessed_df = pd.read_csv('/kaggle/input/coronawhy/titles_abstracts_processed_03282020.csv')

valid_df = preprocessed_df[(preprocessed_df.sentence != 'None') & (preprocessed_df.language == 'en')]
valid_titles_df = valid_df[valid_df.section == "title"]

valid_abstracts_df = valid_df[valid_df.section == "abstract"].groupby("_id")
def clean(token):

    token = token.lower()

    token = token.translate(str.maketrans('', '', string.punctuation))

    return token



def is_valid(token):

    if not token:

        return False

    if token.isnumeric():

        return False

    if token == "pron":

        return False

    return True
# Build data set of lemmas, and combined vectors fo all absracts

data_set_by_id = defaultdict(dict)

for _id, abstracts_df in tqdm(valid_abstracts_df):

    lemmas = []

    for lemma in abstracts_df.lemma:

        lemma = literal_eval(lemma)

        lemma = [clean(i) for i in lemma]

        lemma = [i for i in lemma if is_valid(i)]

        lemmas += lemma

    

    vector_sum = np.zeros([200])

    for vector in abstracts_df.w2vVector[:2]:

        vector_np = np.fromstring(vector[1:-1], dtype=float, sep=' ')

        if vector_np.shape == vector_sum.shape:

            vector_sum += vector_np

    vector_sum = np.nan_to_num(vector_sum)

    

    data_set_by_id[_id]["lemmas"] = lemmas

    data_set_by_id[_id]["vector_sum"] = vector_sum

    data_set_by_id[_id]["title_sentence"] = valid_titles_df[valid_titles_df._id==_id].sentence
abstracts_vect_data_set =  []

reference_in_order = []

for key, values in tqdm(data_set_by_id.items()):

    abstracts_vect_data_set.append(values["vector_sum"])

    reference_in_order.append(key)

    

abstracts_vect_data_set_np = np.asarray(abstracts_vect_data_set)
abstracts_vect_data_set_np.shape
def get_sphere_kmeans(k):

    kmeans = KMeans(n_clusters=k,init='random', random_state=0)

    normalizer = Normalizer(copy=False)

    return make_pipeline(normalizer, kmeans)



class UnsupervisedClustering:

    

    def __init__(self, k, tfidf_ngrams=(1,2), tfidf_features=100, color_map=None):

        self.k = k

        self.kmeans = get_sphere_kmeans(self.k)

        self.pca = PCA(n_components=2)

        self.tfidf = {

            i: TfidfVectorizer(

                max_features=tfidf_features,

                stop_words='english',

                ngram_range=tfidf_ngrams,

                tokenizer= lambda x: x,

                preprocessor=lambda x: x

            )

            for i in range(self.k)

        }

        self.color_map = color_map or {

            0: '#4287f5', 1: '#8c70e0', 2: '#e9f238', 3: '#f23333', 4: '#2cbfba',

            5: '#ccc0ba', 6: '#4700f9', 7: '#f6f900', 8: '#00f91d', 9: '#da8c49'

        }

        

    def fit(self, data, id_order, ref_data):

        kmeans_labels = self.kmeans.fit_predict(data)

        

        clustered_data = defaultdict(list)

        for idx, label in tqdm(enumerate(kmeans_labels)):

            clustered_data[label].append(

                ref_data[id_order[idx]]["lemmas"]

            )

            

        top_features_by_cluster = {}

        for cluster, lemma_corpus in clustered_data.items():

            self.tfidf[cluster].fit_transform(lemma_corpus)

            indices = np.argsort(self.tfidf[cluster].idf_)[::-1]

            features = self.tfidf[cluster].get_feature_names()

            top_features_by_cluster[cluster] = [features[i] for i in indices[:10]]

            

        return kmeans_labels, top_features_by_cluster



    def plot(self, data, labels, num_points=100):

        reduced_data = self.pca.fit_transform(data)

        fig, ax = plt.subplots()

        for index, instance in enumerate(reduced_data[:num_points]):

            pca_comp_1, pca_comp_2 = reduced_data[index]

            color = self.color_map[labels[index]]

            ax.scatter(pca_comp_1, pca_comp_2, c=color)

        plt.show()
data = abstracts_vect_data_set_np

bigram_cluster = UnsupervisedClustering(5, tfidf_ngrams=(2,3))

labels, top = bigram_cluster.fit(data, reference_in_order, data_set_by_id)
bigram_cluster.plot(data, labels)
top