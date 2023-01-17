# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# import the used lib

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import sklearn

from matplotlib import pyplot as plt

import json

import glob

from wordcloud import WordCloud

import re


root_path = '/kaggle/input/CORD-19-research-challenge/'

metadata_path = f'{root_path}/metadata.csv'

meta_df = pd.read_csv(metadata_path, dtype={

    'title': str,

    'pubmed_id': str,

    'Microsoft Academic Paper ID': str, 

    'doi': str

})

meta_df.head(2)
meta_df.info()
all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)

len(all_json)
with open(all_json[10]) as file:

    print(all_json[10])

    first_entry = json.load(file)

#     print(json.dumps(first_entry, indent=4)) 

    

keys = first_entry.keys()

print(keys)

for ky in keys:

    print(len(ky))

    

print(type(first_entry))



for key in first_entry.keys():

    print(key)

    value = first_entry[key]

    if type(value).__name__ == 'list':

        print(value[0].keys())

    elif type(value).__name__ == 'dict':

        print(value.keys())

    else:

        print(value)
class FileReader:

    def __init__(self, file_path):

        with open(file_path) as file:

            content = json.load(file)

            self.paper_id = content['paper_id']

            self.title = content['metadata']['title']

            self.abstract = []

            self.body_text = []

            # Abstract

            for entry in content['abstract']:

                self.abstract.append(entry['text'])

            # Body text

            for entry in content['body_text']:

                self.body_text.append(entry['text'])

            self.abstract = '\n'.join(self.abstract)

            self.body_text = '\n'.join(self.body_text)

            

    def __repr__(self):

        return f'paper_id:{self.paper_id}:\ntitle:{self.title}\nabstract:{self.abstract[:200]}...\nbody:{self.body_text[:200]}...'

first_entry = FileReader(all_json[0])

print(first_entry)
def get_breaks(content, length):

    data = ""

    words = content.split(' ')

    total_chars = 0



    # add break every length characters

    for i in range(len(words)):

        total_chars += len(words[i])

        if total_chars > length:

            data = data + "<br>" + words[i]

            total_chars = 0

        else:

            data = data + " " + words[i]

    return data
dict_ = {'paper_id': [], 'abstract': [], 'body_text': [], 'authors': [], 'title': [], 'journal': [], 'publish_time': [], 'abstract_summary': []}

for idx, entry in enumerate(all_json):

    if idx % (len(all_json) // 10) == 0:

        print(f'Processing index: {idx} of {len(all_json)}')

    content = FileReader(entry)

    

    # get metadata information

    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]

    # no metadata, skip this paper

    if len(meta_data) == 0:

        continue

    

    dict_['paper_id'].append(content.paper_id)

    dict_['abstract'].append(content.abstract)

    dict_['body_text'].append(content.body_text)

    

    # also create a column for the summary of abstract to be used in a plot

    if len(content.abstract) == 0: 

        # no abstract provided

        dict_['abstract_summary'].append("Not provided.")

    elif len(content.abstract.split(' ')) > 100:

        # abstract provided is too long for plot, take first 300 words append with ...

        info = content.abstract.split(' ')[:100]

        summary = get_breaks(' '.join(info), 40)

        dict_['abstract_summary'].append(summary + "...")

    else:# abstract is short enough

        summary = get_breaks(content.abstract, 40)

        dict_['abstract_summary'].append(summary)

        

    # get metadata information

    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]

    

    try:

        # if more than one author

        authors = meta_data['authors'].values[0].split(';')

        if len(authors) > 2:

            # more than 2 authors, may be problem when plotting, so take first 2 append with ...

            dict_['authors'].append(". ".join(authors[:2]) + "...")

        else:

            # authors will fit in plot

            dict_['authors'].append(". ".join(authors))

    except Exception as e:

        # if only one author - or Null valie

        dict_['authors'].append(meta_data['authors'].values[0])

    

    # add the title information, add breaks when needed

    try:

        title = get_breaks(meta_data['title'].values[0], 40)

        dict_['title'].append(title)

    # if title was not provided

    except Exception as e:

        dict_['title'].append(meta_data['title'].values[0])

    

    # add the journal information and publish time

    dict_['journal'].append(meta_data['journal'].values[0])

    dict_['publish_time'].append(meta_data['publish_time'].values[0])

    

df_covid = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text', 'authors', 'title', 'journal', 'publish_time', 'abstract_summary'])

df_covid.head(2)
df_covid.info()
dict_ = None
df_covid['abstract_word_count'] = df_covid['abstract'].apply(lambda x: len(x.strip().split()))

df_covid['body_word_count'] = df_covid['body_text'].apply(lambda x: len(x.strip().split()))

df_covid.head(2)
df_covid.describe(include='all')
output_path = './clean_covid_data.csv'

df_covid.drop_duplicates(['abstract','body_text'], inplace=True)

df_covid.describe(include='all')

df_covid.to_csv(output_path)
# clean_data_path = './clean_covid_data.csv'

# df_covid = pd.read_csv(clean_data_path)
# df_title = df_covid.loc[:, ["title"]].dropna()

# df_title.info()

# df_abstract = df_covid.loc[:, ["abstract"]].dropna()

# df_abstract.info()
# def lower_case(x):

#     return x.lower()



# df_title["title"] = df_title['title'].apply(lambda x: lower_case(x))

# df_title["title"] = df_title['title'].apply(lambda x: x.strip())

# df_title["title"] = df_title['title'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))

# df_title["title"] = df_title['title'].apply(lambda x: re.sub(' +',' ',x))

# titles = ' '.join(df_title["title"])





# df_abstract["abstract"] = df_abstract['abstract'].apply(lambda x: lower_case(x))

# df_abstract["abstract"] = df_abstract['abstract'].apply(lambda x: x.strip())

# df_abstract["abstract"] = df_abstract['abstract'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))

# df_abstract["abstract"] = df_abstract['abstract'].apply(lambda x: re.sub(' +',' ',x))

# abstracts = ' '.join(df_abstract["abstract"])



# print(titles[:100])

# print(abstracts[:200])
# from nltk.corpus import stopwords

# import scipy.misc

# from matplotlib.pyplot import imread

# stopword = stopwords.words('english')  # remove the stop words



# wordcloud_title = WordCloud(max_font_size=None, background_color='white', 

#                       collocations=False, stopwords=stopword,

#                       width=1000, height=1000).generate(titles)



# wordcloud_abstract = WordCloud(max_font_size=None, background_color='white', 

#                       collocations=False, stopwords=stopword,

#                       width=1000, height=1000).generate(abstracts)

# plt.figure(figsize=(15,15))

# plt.subplot(1,2,1)

# plt.axis("off")

# plt.imshow(wordcloud_title)

# plt.title('Common Words in Title')

# plt.subplot(1,2,2)

# plt.axis("off")

# plt.imshow(wordcloud_abstract)

# plt.title('Common Words in Abstract')

# plt.show()
# import nltk

# import time



# wnl = nltk.stem.WordNetLemmatizer()



# word_count = {}



# def Pos_tag(text, publish_time):

#     token = nltk.word_tokenize(text)

#     pos = nltk.pos_tag(token)

#     try:

#         timeStruct = time.strptime(publish_time, "%Y-%m-%d")

#     except:

#         timeStruct = time.strptime(publish_time, "%Y")

#     return token, pos, timeStruct.tm_year



# # 'NN*', 'VB*'

# def add2vocab(pos_tag):

#     for w, p in pos_tag:

#         if (re.match('NN',p) or re.match('VB',p)) and w not in stopword and w.isalnum() and len(w) > 1:

#             w = w.lower()

#             if re.match('NN',p):

#                 w = wnl.lemmatize(w, pos='n')  

#             if re.match('VB',p):

#                 w = wnl.lemmatize(w, pos='v')

#             if w in word_count:

#                 word_count[w] += 1

#             else:

#                 word_count[w] = 1  
# df_subset = df_covid.loc[:, ['abstract','publish_time']].dropna()

# all_pos = []

# all_year = []

# for idx, (abstract, publish_time) in df_subset.iterrows():

#     token, pos, year = Pos_tag(abstract, publish_time)

#     add2vocab(pos)

#     all_pos.append(pos)

#     all_year.append(year)
# # vocab = [k for k,v in word_count if v > 5]

# word_count_sort = sorted(word_count.items(), key=lambda d: d[1], reverse=True)

# vocab = [k for k,v in word_count_sort[:100]]

# count = [v for k,v in word_count_sort[:100]]

# print(vocab)
# plt.figure(figsize=(10,10))

# plt.barh(range(len(vocab[:50])), count[:50], height=0.3, color='steelblue', alpha=0.8)      # 从下往上画

# plt.yticks(range(len(vocab[:50])), vocab)

# # plt.xlim(30,47)

# plt.xlabel("frequency")

# plt.title("Most Frequent words")

# # for x, y in enumerate(count):

# #     plt.text(y + 0.2, x - 0.1, '%s' % y)

# plt.show()
# max_year = max(all_year)

# min_year = min(all_year)

# print('articles are from %d year to %d year.' % (min_year, max_year))

# publish_count = np.zeros(max_year-min_year+1)

# for y in all_year:

#     publish_count[y-min_year] += 1

# year_list = list(range(min_year, max_year+1))
# # choose_vocab = ['virus', 'infection', 'cell', 'protein', 'disease', 'patient', 'gene', 'respiratory',

# #                 'rna', 'vaccine', 'sample', 'strain', 'expression', 'level', 'antibody', 'pathogen', 'assay', 

# #                 'detect', 'factor', 'mouse', 'associate', 'treatment', 'coronavirus', 'influenza', 'target', 

# #                 'replication', 'development', 'demonstrate', 'risk', 'outbreak', 'mechanism', 'detection', 

# #                 'review', 'indicate', 'child', 'function', 'population', 'structure', 'transmission', 'region', 'research', 'sars', 'conclusion', 'change', 'induce', 'syndrome', 'genome', 'infect', 'process', 'determine', 'interaction', 'age', 'approach', 'receptor', 'animal', 'specie', 'evaluate', 'acid', 'drug', 'observe', 'dna', 'reveal', 'investigate']

# word2ix = {word:ix for ix, word in enumerate(vocab)}

# matrix = np.zeros((max_year-min_year+1) * len(vocab)).reshape(max_year-min_year+1, len(vocab))

# for pos, year in zip(all_pos, all_year):

#     for w,p in pos:

#         if re.match('NN',p):

#             w = wnl.lemmatize(w, pos='n')

#             if w in vocab:

#                 matrix[year-min_year][word2ix[w]] += 1

#         elif re.match('VB',p):

#             w = wnl.lemmatize(w, pos='v')

#             if w in vocab:

#                 matrix[year-min_year][word2ix[w]] += 1
# # sub_axix = filter(lambda x:x%200 == 0, x_axix)

# small_matrix = matrix[:-1,:20].copy()  # 1957-2019, top20 words

# plt.figure(figsize=(15,10))

# plt.title('Words Trend')

# size1, size2 = small_matrix.shape

# year_num = year_list[:-1]

# colors = ['g', 'r', 'b', 'k', 'y', 'c', 'm']

# for idx in range(size2):

#     plt.plot(year_num, list(small_matrix[:, idx]), color=colors[idx%7], label=vocab[idx])

# plt.plot(year_num, publish_count[:-1], 'r*', label='publications')

# plt.legend() # 显示图例



# plt.xlabel('year')

# plt.ylabel('word frequency')

# plt.show()
# small_matrix = None
# small_matrix = matrix[-20:-1,:20].copy()  # 2000-2019, top20 words

# small_count = publish_count[-20:-1]

# for idx in range(small_matrix.shape[0]):

#     small_matrix[idx,:] = small_matrix[idx,:]/small_count[idx]

# plt.figure(figsize=(15,10))

# plt.title('Words Trend')

# size1, size2 = small_matrix.shape

# year_num = year_list[-20:-1]

# print(year_num)

# colors = ['g', 'r', 'b', 'k', 'y', 'c', 'm']

# for idx in range(size2):

#     plt.plot(year_num, list(small_matrix[:, idx]), color=colors[idx%7], label=vocab[idx])

# plt.legend() 



# plt.xlabel('year')

# plt.ylabel('word frequency')

# plt.show()
# matrix = None

# small_matrix = None
# cluster_covid = df_covid.loc[:, ["paper_id", "title", "abstract"]].dropna()

# cluster_covid = cluster_covid.drop_duplicates(['abstract','paper_id', 'title'])
# from sklearn.feature_extraction.text import CountVectorizer 

# from sklearn.feature_extraction.text import TfidfTransformer 



# vectorizer = CountVectorizer()

# transformer = TfidfTransformer()  # 这里需要限制维度，占用内存太大了



# X = vectorizer.fit_transform(list(cluster_covid['abstract']))

# tfidf = transformer.fit_transform(X).toarray()

# word = vectorizer.get_feature_names()

# print(tfidf.shape)
# from sklearn.decomposition import PCA

# from sklearn.cluster import KMeans

# pca_sk = PCA(n_components=3)



# pca_result = pca_sk.fit_transform(tfidf)



# y_preds = KMeans(n_clusters=6,random_state=0).fit(pca_result)



# labels = y_preds.labels_

# centers = y_preds.cluster_centers_
# import seaborn as sns

# plt.figure(figsize=(10,10))



# # sns settings

# sns.set(rc={'figure.figsize':(15,15)})



# # colors

# palette = sns.color_palette("bright", len(set(y)))



# # plot

# sns.scatterplot(pca_result[:,0], pca_result[:,1], hue=y, legend='full', palette=palette)

# plt.title("PCA Covid-19 Articles - Clustered (K-Means) - Tf-idf with Plain Text")

# # plt.savefig("plots/pca_covid19_label_TFID.png")

# plt.show()