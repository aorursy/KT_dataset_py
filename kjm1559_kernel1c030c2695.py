# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json # to read csv file

from tqdm import tqdm 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

data_list = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        data_list.append(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



# read meta data

print(data_list[0])

all_meta_data = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')

data_list = pd.DataFrame(data_list, columns=['file_name'])



# remove column what doesn't have full text.

exist_text_data = all_meta_data[all_meta_data.abstract.isnull() != True]

exist_text_data = exist_text_data[exist_text_data.abstract != 'Unknown']

# exist_text_data = all_meta_data.fillna(value=-1)

# exist_text_data = exist_text_data[exist_text_data.abstract != -1]



# all_meta_data.abstract.isnull()[all_meta_data.abstract.isnull() != True].index.tolist()

all_meta_data[all_meta_data.abstract.isnull() != True][52:53]

# collect full texts

# train_text = []

# for idata in tqdm(exist_text_data.itertuples(), total=len(exist_text_data), position=0):

#     texts = ''

#     try:

#         with open(data_list[data_list.file_name.str.contains(idata.sha)].values[0][0], "r") as f:

#             dict_json = json.load(f)

#     except:

#         print(idata.sha)

#     for dd in dict_json['body_text']:

#         texts += ' ' + dd['text']

#     train_text.append(texts.split(' '))
corpus = [data for data in all_meta_data.abstract if type(data) == type('')]

train_text = [dd.split(' ') for dd in corpus]

word_counter = {}

for i in range(len(train_text)):

    for j in range(len(train_text[i])):

        if train_text[i][j] in word_counter:

            word_counter[train_text[i][j]] += 1

        else:

            word_counter[train_text[i][j]] = 1

word_counter_df = pd.DataFrame(word_counter.values(), index=list(word_counter.keys()))

word_counter_df = word_counter_df.sort_values(by=0, ascending=False)

word_counter_df[: 30].index.tolist()
# tf-idf

from sklearn.feature_extraction.text import TfidfVectorizer





tfidfv = TfidfVectorizer(max_features=2**10, stop_words=word_counter_df[: 30].index.tolist(), ngram_range=(2,2)).fit(corpus)

print(tfidfv.vocabulary_)



# # full text version

# from sklearn.feature_extraction.text import TfidfVectorizer



# corpus = [' '.join(data) for data in train_text]

# tfidfv = TfidfVectorizer().fit(corpus)

# print(tfidfv.vocabulary_)



# vocabulary diction, to find using number

find_dic = {}

for key in tfidfv.vocabulary_.keys():

    find_dic[tfidfv.vocabulary_[key]] = key
import gensim

# from sklearn.decomposition import PCA

# import matplotlib.pylab as plt



model = gensim.models.Word2Vec(train_text, size=20, window=5, min_count=1, workers=4)



# #to visualize vector values

# vector_data = []

# for key in model.wv.vocab.keys():

#     vector_data.append(model.wv.get_vector(key))

# pca = PCA(n_components=2)

# pca_vector_data = pca.fit_transform(vector_data)

# # to nkow what word2vec is good

# plt.scatter(pca_vector_data[:,0], pca_vector_data[:,1])

# pca_select_vector_data = pca.transform([model.wv.get_vector('Pregnancy').tolist()])

# plt.scatter(pca_select_vector_data[:,0], pca_select_vector_data[:,1])

# pca_select_vector_data = pca.transform([model.wv.get_vector('Obesity').tolist()])

# plt.scatter(pca_select_vector_data[:,0], pca_select_vector_data[:,1])

# pca_select_vector_data = pca.transform([model.wv.get_vector('Smoking').tolist()])

# plt.scatter(pca_select_vector_data[:,0], pca_select_vector_data[:,1])

# plt.show()



# # for test that word2vec is good

# model.wv.most_similar(positive = ['Smoking', 'Pregnancy', 'Obesity', 'Malnutrition'], negative = '', topn=20)

# model.wv.most_similar(positive = ['pneumonia', 'respiratory', 'transmissible', 'co-detection', 'cough'], negative = '', topn=20)



print(model.wv.most_similar(positive=['congenital', 'heart', 'rubella', 'intrauterine', 'disease', 'suggestive', 'etiologic']))

print(corpus[0])

# model.wv.v

print(model.wv.most_similar('antibody'))
model.wv.most_similar(positive=['corona', 'risk', 'factor'])
# test_string = []

# kk = tfidfv.transform(corpus[:10]).toarray()

# index = np.argsort(kk[1])[::-1]

# for i in index[:20]:

#     print(find_dic[i], ':', kk[1][i])

#     test_string.append(find_dic[i])
feature_data = []

except_words = []

# except_words = ['the', 'of', 'and', 'in', 'to', 'were', 'virus', 'cells', 'patients',

#        'protein', 'was', 'viral', 'infection', 'viruses', 'influenza',

#        'respiratory', 'cell', 'for', 'with', 'health']

# except_words = ['of', 'the', 'end', 'in', 'and', 'to', 'for', 'were', 'was', 'viruses', 'virus', 'with', 'viral', 'patients', 'is', \

#                 'that', 'disease', 'replication', 'infections', 'are', 'cells', 'immune', 'respiratory', 'clinical', 'by', 'protein', \

#                 'exposure', 'antiviral', 'against', 'activity', 'mice', 'infected', 'on', 'diseases', 'from', 'infection', 'influenza', \

#                 'or', 'as', 'expression', 'proteins', 'host', 'cell', 'zoonotic', 'expressed', 'coronavirus', 'gene', 'human', 'samples', \

#                 'response', 'we', 'vaccine', 'cases', 'health', 'their', 'can', 'experiments', 'binding', 'at', 'exposed', 'experimental', \

#                 'infectious', 'existing', 'exhibited', 'evolution', 'evidence', 'children', 'detection', 'specific', 'treatment', 'associated', \

#                 'during', 'be', 'two', 'its', 'induced', 'estimated', 'et', 'examined', 'evaluation', 'even', 'an', 'events', 'environmental', \

#                 'risk', 'epithelial', 'been', 'epitopes', 'type', 'epitope', 'epidemiology', 'evaluated', 'evaluate', 'data', 'established', \

#                 'essential', 'control', 'have', 'epithelial', 'been', 'epitopes', 'type', 'epitope', 'epidemiology', 'evaluated', 'evaluate', \

#                 'data', 'established', 'essential', 'control', 'have', 'it', 'factor', 'epidemiological', 'especially', 'expressing', 'between', \

#                 'full', 'epidemics', 'epidemic', 'function', 'based', 'may', 'frequently', 'this', 'frequency', 'free', 'model', 'formation', \

#                 'food', 'four', 'found', 'new', 'factors', 'fold', 'first', 'five', 'form', 'following', 'followed', 'focus', 'analysis', \

#                 'transmission', 'species', 'sequence', 'pathogens', 'high', 'strains', 'responses', 'entry', 'using', 'system', 'positive', \

#                 'after', 'detected', 'group', 'findings', 'has', 'antibodies', 'field', 'fever', 'feline', 'genome', 'other', 'few', \

#                 'enzyme', 'both', 'used', 'finally', 'than', '19', 'had', 'fecal', 'our', 'but', 'more', 'these', 'features', 'family', \

#                 'different', 'envelope', 'environment', 'failure', 'diversity', 'all', 'diverse', 'studies', 'discovery', 'most', \

#                 'distribution', 'discussed', 'discuss']#[dd[0] for dd in model.wv.most_similar('The')] + [dd[0] for dd in model.wv.most_similar('the')] + [dd[0] for dd in model.wv.most_similar('of')] + [dd[0] for dd in model.wv.most_similar('and')]

word_counter = {}

for idata in tqdm(corpus, position=0):

    tmp_idata = tfidfv.transform([idata]).toarray()[0]

    index = np.argsort(tmp_idata)[::-1]

    tmp_feature = []

    for i in index:

#         print(find_dic[i].split(' ')[0], find_dic[i].split(' ')[1])

        if find_dic[i] in except_words:

            continue

        elif (find_dic[i].split(' ')[0] in model.wv.vocab) & (find_dic[i].split(' ')[1] in model.wv.vocab):

#             if find_dic[i] in word_counter:

#                 word_counter[find_dic[i]] += 1

#             else:

#                 word_counter[find_dic[i]] = 1

            tmp_feature += model.wv.get_vector(find_dic[i].split(' ')[0]).tolist()

            tmp_feature += model.wv.get_vector(find_dic[i].split(' ')[1]).tolist()

        if len(tmp_feature) >= 1000: # 20 words

            break

    feature_data.append(tmp_feature)

        
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=8, random_state=0).fit(feature_data)
# import matplotlib.pylab as plt

# plt.plot(word_counter.values())

# plt.show()

# except_list = []

# for key in word_counter.keys():

#     if word_counter[key] > 1000:

#         except_list.append(key)

# pd.DataFrame(word_counter.values(), index=word_counter.keys()).sort_values(by=0, ascending=False)[:20].index

# word_counter.values
# # from sklearn.manifold import TSNE

# # tsne_vector_data = TSNE(n_components=2).fit_transform(feature_data)



# from sklearn.manifold import TSNE



# tsne = TSNE(verbose=1, perplexity=5)

# X_embedded = tsne.fit_transform(feature_data)
corpus[52]
np.where((kmeans.labels_ ==5) == True)[0][:20]

# np.array(corpus)[kmeans.labels_ ==1][0]
corpus[141]
corpus[9]
# feature_data
# from sklearn.decomposition import PCA

# import matplotlib.pylab as plt

# # kk = tfidfv.transform(feature_data).toarray()

# pca = PCA(n_components=2)

# pca_vector_data = pca.fit_transform(feature_data)

# # to know what word2vec is good

# for i in range(8):

#     plt.scatter(pca_vector_data[kmeans.labels_ == i,0], pca_vector_data[kmeans.labels_ == i,1])

# # plt.scatter(pca_vector_data[:,0], pca_vector_data[:,1])

# plt.show()
# # tmp_idata = tfidfv.transform([idata]).toarray()[0]



# from sklearn.decomposition import PCA

# import matplotlib.pylab as plt

# # kk = tfidfv.transform(feature_data).toarray()

# pca = PCA(n_components=2)

# pca_vector_data = pca.fit_transform(tfidfv.transform(corpus).toarray())

# # to know what word2vec is good

# for i in range(8):

#     plt.scatter(pca_vector_data[kmeans.labels_ == i,0], pca_vector_data[kmeans.labels_ == i,1])

# plt.show()
%matplotlib inline

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



pca = PCA(n_components=3)

pca_result = pca.fit_transform(feature_data)



ax = plt.figure(figsize=(16,10)).gca(projection='3d')

ax.scatter(

    xs=pca_result[:,0], 

    ys=pca_result[:,1], 

    zs=pca_result[:,2], 

    c=kmeans.labels_, 

    cmap='tab10'

)

ax.set_xlabel('pca-one')

ax.set_ylabel('pca-two')

ax.set_zlabel('pca-three')

plt.title("PCA Covid-19 Abstract (3D) - Clustered (K-Means) - Tf-idf with word2vec")

# plt.savefig("plots/pca_covid19_label_TFID_3d.png")

plt.show()
# s_index = 8



# data = []

# # print(kk[s_index][np.where(kk[s_index] > 0)])

# # print(np.where(kk[s_index] > 0))

# for i in np.where(kk[s_index] > 0)[0]:

#     data.append([find_dic[i], kk[s_index][i]])

# df = pd.DataFrame(data, columns=['word', 'tfidf'])

# print(corpus[s_index])

# df.sort_values(by='tfidf', ascending=False)[:20]