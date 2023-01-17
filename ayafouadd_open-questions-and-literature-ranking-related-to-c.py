import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

import json

from IPython.display import Image

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('ggplot')
root_path = '/kaggle/input/CORD-19-research-challenge/'

metadata_path = f'{root_path}/metadata.csv'

meta_df = pd.read_csv(metadata_path, dtype={

    'pubmed_id': str,

    'Microsoft Academic Paper ID': str, 

    'doi': str

})

meta_df.head()
meta_df.info()
all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)

len(all_json)
class FileReader:

    def __init__(self, file_path):

        with open(file_path) as file:

            content = json.load(file)

            self.paper_id = content['paper_id']

            self.abstract = []

            self.body_text = []

            # Abstract

            if not 'abstract' in content.keys():

                self.abstract = ''

                self.body_text= ''

                return

            for entry in content['abstract']:

                self.abstract.append(entry['text'])

            # Body text

            for entry in content['body_text']:

                self.body_text.append(entry['text'])

            self.abstract = '\n'.join(self.abstract)

            self.body_text = '\n'.join(self.body_text)

    def __repr__(self):

        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'

first_row = FileReader(all_json[0])

print(first_row)
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
dict_ = {'paper_id': [], 'doi':[], 'abstract': [], 'body_text': [], 'authors': [], 'title': [], 'journal': [], 'abstract_summary': []}

for idx, entry in enumerate(all_json):

    if idx % (len(all_json) // 10) == 0:

        print(f'Processing index: {idx} of {len(all_json)}')

    

    try:

        content = FileReader(entry)

    except Exception as e:

        continue  # invalid paper format, skip

    

    # get metadata information

    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]

    # no metadata, skip this paper

    if len(meta_data) == 0:

        continue

    

    dict_['abstract'].append(content.abstract)

    dict_['paper_id'].append(content.paper_id)

    dict_['body_text'].append(content.body_text)

    

    # also create a column for the summary of abstract to be used in a plot

    if len(content.abstract) == 0: 

        # no abstract provided

        dict_['abstract_summary'].append("Not provided.")

    elif len(content.abstract.split(' ')) > 100:

        # abstract provided is too long for plot, take first 100 words append with ...

        info = content.abstract.split(' ')[:100]

        summary = get_breaks(' '.join(info), 40)

        dict_['abstract_summary'].append(summary + "...")

    else:

        # abstract is short enough

        summary = get_breaks(content.abstract, 40)

        dict_['abstract_summary'].append(summary)

        

    # get metadata information

    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]

    

    try:

        # if more than one author

        authors = meta_data['authors'].values[0].split(';')

        if len(authors) > 2:

            # if more than 2 authors, take them all with html tag breaks in between

            dict_['authors'].append(get_breaks('. '.join(authors), 40))

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

    

    # add the journal information

    dict_['journal'].append(meta_data['journal'].values[0])

    

    # add doi

    dict_['doi'].append(meta_data['doi'].values[0])

    

df_covid = pd.DataFrame(dict_, columns=['paper_id', 'doi', 'abstract', 'body_text', 'authors', 'title', 'journal', 'abstract_summary'])

df_covid.head()
df_covid
df_covid['abstract_word_count'] = df_covid['abstract'].apply(lambda x: len(x.strip().split()))  # word count in abstract

df_covid['body_word_count'] = df_covid['body_text'].apply(lambda x: len(x.strip().split()))  # word count in body

df_covid['body_unique_words']=df_covid['body_text'].apply(lambda x:len(set(str(x).split())))  # number of unique words in body

df_covid.head()
df_covid.info()
df_covid['abstract'].describe(include='all')


df_covid.drop_duplicates(['abstract', 'body_text'], inplace=True)

df_covid['abstract'].describe(include='all')
df_covid['body_text'].describe(include='all')
df_covid.dropna(inplace=True)

df_covid.columns
df_covid.head()
df_covid.describe()
df = df_covid
from tqdm import tqdm

from langdetect import detect

from langdetect import DetectorFactory



# set seed

DetectorFactory.seed = 0



# hold label - language

languages = []



# go through each text

for ii in tqdm(range(0,len(df))):

    # split by space into list, take the first x intex, join with space

    text = df.iloc[ii]['body_text'].split(" ")

    

    lang = "en"

    try:

        if len(text) > 50:

            lang = detect(" ".join(text[:50]))

        elif len(text) > 0:

            lang = detect(" ".join(text[:len(text)]))

    # ught... beginning of the document was not in a good format

    except Exception as e:

        all_words = set(text)

        try:

            lang = detect(" ".join(all_words))

        # what!! :( let's see if we can find any text in abstract...

        except Exception as e:

            

            try:

                # let's try to label it through the abstract then

                lang = detect(df.iloc[ii]['abstract_summary'])

            except Exception as e:

                lang = "unknown"

                pass

    

    # get the language    

    languages.append(lang)
from pprint import pprint



languages_dict = {}

for lang in set(languages):

    languages_dict[lang] = languages.count(lang)

    

print("Total: {}\n".format(len(languages)))

pprint(languages_dict)
df['language'] = languages

plt.bar(range(len(languages_dict)), list(languages_dict.values()), align='center')

plt.xticks(range(len(languages_dict)), list(languages_dict.keys()))

plt.title("Distribution of Languages in Dataset")

plt.show()
df = df[df['language'] == 'en'] 

df.info()
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz

import en_core_sci_lg
#NLP 

from IPython.utils import io

import spacy

from spacy.lang.en.stop_words import STOP_WORDS

import string



punctuations = string.punctuation

stopwords = list(STOP_WORDS)

stopwords[:10]
custom_stop_words = [

    'doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure', 

    'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.', 

    'al.', 'Elsevier', 'PMC', 'CZI', 'www'

]



for w in custom_stop_words:

    if w not in stopwords:

        stopwords.append(w)
import re

# Parser

parser = en_core_sci_lg.load(disable=["tagger", "ner"])

parser.max_length = 7000000



def spacy_tokenizer(sentence):

    mytokens = parser(sentence)

    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]

    mytokens = [ re.sub('[0-9%]','',word) for word in mytokens ]

    mytokens = " ".join([i for i in mytokens])

    return mytokens
from tqdm import tqdm

import numpy as np

import math
tqdm.pandas()

df["processed_text"] = df["abstract"].progress_apply(spacy_tokenizer)
df['abstract'].replace('', np.nan, inplace=True)

df.dropna(inplace=True)

df.info()
df['processed_word_count'] = df["processed_text"].apply(lambda x: len(x.strip().split()))

sns.distplot(df['processed_word_count'])

df['abstract_word_count'].describe()
from sklearn.feature_extraction.text import TfidfVectorizer

text = df['processed_text'].values

vectorizer = TfidfVectorizer(max_features=4096)

X = vectorizer.fit_transform(text) 

terms = vectorizer.get_feature_names()
terms[0:20]
from sklearn.decomposition import PCA



pca = PCA(n_components=0.95, random_state=42)

X_reduced= pca.fit_transform(X.toarray())

X_reduced.shape
from sklearn.cluster import KMeans
Image(filename='/kaggle/input/kaggle-resources/kmeans.PNG', width=800, height=800)
# from sklearn import metrics

# from scipy.spatial.distance import cdist

# %matplotlib inline

# from matplotlib import pyplot as plt



# # run kmeans with many different k

# distortions = []

# K = range(10, 35)

# for k in tqdm(K):

#     k_means = KMeans(n_clusters=k, random_state=42).fit(X_reduced)

#     k_means.fit(X_reduced)

#     distortions.append(sum(np.min(cdist(X_reduced, k_means.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

#     #print('Found distortion for {} clusters'.format(k))
# X_line = [K[0], K[-1]]

# Y_line = [distortions[0], distortions[-1]]



# # Plot the elbow

# plt.plot(K, distortions, 'b-')

# plt.plot(X_line, Y_line, 'r')

# plt.xlabel('k')

# plt.ylabel('Distortion')

# plt.title('The Elbow Method showing the optimal k')

# plt.show()
k = 20

kmeans = KMeans(n_clusters=k, random_state=42)

y_pred = kmeans.fit_predict(X_reduced)

df['y'] = y_pred
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
for i in range(20):

    print('Cluster %d:' % i),

    for ind in order_centroids[i, :20]:

        print('%s' % terms[ind])
# from sklearn.manifold import TSNE



# tsne = TSNE(verbose=1, perplexity=100, random_state=42)

# X_embedded = tsne.fit_transform(X.toarray())
# # sns settings

# sns.set(rc={'figure.figsize':(15,15)})



# # colors

# palette = sns.color_palette("bright", 1)



# # plot

# sns.scatterplot(X_embedded[:,0], X_embedded[:,1], palette=palette)

# plt.title('t-SNE with no Labels')

# plt.savefig("t-sne_covid19.png")

# plt.show()
# %matplotlib inline

# from matplotlib import pyplot as plt

# import seaborn as sns



# # sns settings

# sns.set(rc={'figure.figsize':(15,15)})



# # colors

# palette = sns.hls_palette(20, l=.4, s=.9)



# # plot

# sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y_pred, legend='full', palette=palette)

# plt.title('t-SNE with Kmeans Labels')

# plt.savefig("improved_cluster_tsne.png")

# plt.show()
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=10)

knn_model.fit(X_reduced,df['y'].values )
def predict_nearest_neighbour(model,sentence):

    print("Prediction")

    sentence = spacy_tokenizer(sentence)

    print (sentence)

    X = vectorizer.transform([sentence])

    X = pca.transform(X.toarray())

    predicted = model.kneighbors(X, 100)

    return predicted
sentence = "What has been published about ethical and social science considerations?"

sentence_1 = 'What do we know about virus genetics, origin and evolution ?'

sentence_2 = 'What is known about transmission, incubation, and environmental stability?'

sentence_3 = 'Create summary tables that address risk factors related to COVID-19'

sentence_4 = 'What do we know about COVID-19 risk factors?'

sentence_5 = 'What has been published about medical care?'

sentence_6 = 'What do we know about diagnostics and surveillance?'

sentence_7 = 'What do we know about vaccines and therapeutics?'
res = predict_nearest_neighbour(knn_model, sentence_1)

print (res)
inds =[]

for i,dist in enumerate(res[0][0]):

    if dist > 1:

        inds.append(res[1][0][i])

print (inds)

        

    
for i in range(10):

    index = inds[i]

    print (df.iloc[index]['abstract'])

    print ("virus appeared", df.iloc[index]['abstract'].count('virus'), 'times')

    print ("genetic appeared", df.iloc[index]['abstract'].count('genetic'), 'times')

    print ("origin appeared", df.iloc[index]['abstract'].count('origin'), 'times')

    print ("evolution appeared", df.iloc[index]['abstract'].count('evolution'), 'times')
for i in range(10):

    index = inds[i]

    print (df.iloc[index])
res = predict_nearest_neighbour(knn_model, sentence_4)

print (res)
inds =[]

for i,dist in enumerate(res[0][0]):

    if dist > 1:

        inds.append(res[1][0][i])

print (inds)
for i in range(10):

    index = inds[i]

    print (df.iloc[index]['abstract'])

    print ("risk appeared", df.iloc[index]['abstract'].count('risk'), 'times')

    print ("factor appeared", df.iloc[index]['abstract'].count('factor'), 'times')

    print ("COVID-19", df.iloc[index]['abstract'].count('covid-19'), 'times')
for i in range(10):

    index = inds[i]

    print (df.iloc[index])
import pickle

pickle.dump(kmeans, open("k-means_abstract.pkl", "wb"))

pickle.dump(knn_model, open("knn_model_abstract.pkl", "wb"))



#kmeans = pickle.load(open("k-means_model.pkl", "rb"))