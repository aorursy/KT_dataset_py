# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        os.path.join(dirname, filename)



# Any results you write to the current directory are saved as output.
!ls /kaggle/input//covid19-filtered-dataset/
import numpy as np 

import pandas as pd 

import glob

import json



import matplotlib.pyplot as plt

plt.style.use('ggplot')
meta_df = pd.read_csv('/kaggle/input//covid19-filtered-dataset/covid_19_full_text_files.csv', dtype={

    'pubmed_id': str,

    'Microsoft Academic Paper ID': str, 

    'doi': str

})

meta_df.head()
meta_df.info()
import re,string



meta_df['text'] = meta_df['text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

meta_df.head()
def lower_case(input_str):

    input_str = input_str.lower()

    return input_str



meta_df['text'] = meta_df['text'].apply(lambda x: lower_case(x))

meta_df.head()
text = meta_df.drop(["source_x","pmcid","pubmed_id","license","publish_time","Microsoft Academic Paper ID","WHO #Covidence","has_pdf_parse","has_pmc_xml_parse","full_text_file","tag_disease_covid19"], axis=1)

text.head()
text.drop_duplicates(['text'], inplace=True)

len(text)
text['text'].describe(include='all')
para_list=pd.DataFrame(columns=['cord_uid','sha','paper_id','doi','journal','title','authors','abstract','text','url'])

i=0

for index,bodyText in text.iterrows(): 

    big_data_list=[]

    para=bodyText['text'].split('\n')

    for par in para:

      data_list=[]

      data_list.append(bodyText['cord_uid'])

      data_list.append(bodyText['sha'])

      data_list.append(bodyText['paper_id'])

      data_list.append(bodyText['doi'])

      data_list.append(bodyText['journal'])

      data_list.append(bodyText['title'])

      data_list.append(bodyText['authors'])

      data_list.append(bodyText['abstract'])

      data_list.append(par)

      data_list.append(bodyText['url'])

      big_data_list.append(data_list)

    para_df=pd.DataFrame(columns=['cord_uid','sha','paper_id','doi','journal','title','authors','abstract','text','url'], data=big_data_list)

    para_list=para_list.append(para_df)

    i+=1

#     print(i)

print("length: "+str(len(para_list)))
para_list.head()
para_list=para_list[para_list['text'].map(len) > 65]

len(para_list)
para_list.head()
from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer = TfidfVectorizer(max_features=2**12)

X = vectorizer.fit_transform(para_list['text'].values)

X.shape
from sklearn.cluster import MiniBatchKMeans



k = 20

kmeans = MiniBatchKMeans(n_clusters=k)

y_pred = kmeans.fit_predict(X)

y = y_pred

y
from sklearn.decomposition import PCA



pca = PCA(n_components=3)

pca_result = pca.fit_transform(X.toarray())
%matplotlib inline

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



ax = plt.figure(figsize=(16,10)).gca(projection='3d')

ax.scatter(

    xs=pca_result[:,0], 

    ys=pca_result[:,1], 

    zs=pca_result[:,2], 

    c=y, 

    cmap='tab10'

)

ax.set_xlabel('pca-one')

ax.set_ylabel('pca-two')

ax.set_zlabel('pca-three')

plt.title("PCA Covid-19 Articles (paragraph) (3D) - Clustered (K-Means,k=20) - Tf-idf with Plain Text")

# plt.savefig("plots/pca_covid19_label_TFID_3d.png")

plt.show()
para_list['Cluster']=y

para_list.head()
import nltk

import os

import string

import numpy as np

import copy

import pandas as pd

import pickle

import re

import math



nltk.download("popular")

!pip install num2words

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem import PorterStemmer

from collections import Counter

from num2words import num2words







def remove_stop_words(data):

    stop_words = stopwords.words('english')

    words = word_tokenize(str(data))

    new_text = ""

    for w in words:

        if w not in stop_words and len(w) > 1:

            new_text = new_text + " " + w

    return new_text



def stemming(data):

    stemmer= PorterStemmer()

    

    tokens = word_tokenize(str(data))

    new_text = ""

    for w in tokens:

        new_text = new_text + " " + stemmer.stem(w)

    return new_text

def convert_numbers(data):

    tokens = word_tokenize(str(data))

    new_text = ""

    for w in tokens:

        try:

            w = num2words(int(w))

        except:

            a = 0

        new_text = new_text + " " + w

    new_text = np.char.replace(new_text, "-", " ")

    return new_text



def convert_lower_case(data):

    return np.char.lower(data)



def remove_punctuation(data):

    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"

    for i in range(len(symbols)):

        data = np.char.replace(data, symbols[i], ' ')

        data = np.char.replace(data, "  ", " ")

    data = np.char.replace(data, ',', '')

    return data



def remove_apostrophe(data):

    return np.char.replace(data, "'", "")





def preprocess(data):

    data = convert_lower_case(data)

    data = remove_punctuation(data) #remove comma seperately

    data = remove_apostrophe(data)

    data = remove_stop_words(data)

    data = convert_numbers(data)

    data = stemming(data)

    data = remove_punctuation(data)

    data = convert_numbers(data)

    data = stemming(data) #needed again as we need to stem the words

    data = remove_punctuation(data) #needed again as num2word is giving few hypens and commas fourty-one

    data = remove_stop_words(data) #needed again as num2word is giving stop words 101 - one hundred and one

    return data
processed_text=[]

i=0

for t in para_list['text']:

    processed_text.append(word_tokenize(str(preprocess(t))))

    if i%1000==0:

        print("text: "+str(i))

    i+=1



print(len(processed_text))
N=len(processed_text)

DF = {}



for i in range(N):

    tokens = processed_text[i]

    for w in tokens:

        try:

            DF[w].add(i)

        except:

            DF[w] = {i}

for i in DF:

    DF[i] = len(DF[i])
total_vocab_size = len(DF)

total_vocab_size
total_vocab = [x for x in DF]

total_vocab[:20]
def doc_freq(word):

    c = 0

    try:

        c = DF[word]

    except:

        pass

    return c
from collections import Counter

doc = 0



tf_idf = {}



for i in range(N):

    

    tokens = processed_text[i]

    

    counter = Counter(tokens)

    words_count = len(tokens)

    

    for token in np.unique(tokens):

        

        tf = counter[token]/words_count

        df = doc_freq(token)

        idf = np.log((N+1)/(df+1))

        

        tf_idf[doc, token] = tf*idf



    doc += 1

    

len(tf_idf)
def matching_score(k, query):

    tokens = word_tokenize(str(query))



    print("Matching Score")

    print("\nQuery:", query)

    print("")

    print(tokens)

    

    query_weights = {}



    for key in tf_idf:

        

        if key[1] in tokens:

            try:

                query_weights[key[0]] += tf_idf[key]

            except:

                query_weights[key[0]] = tf_idf[key]

    

    query_weights = sorted(query_weights.items(), key=lambda x: x[1], reverse=True)



    print("")

    

    l = []

    

    for i in query_weights[:10]:

        l.append(i[0])

    

    print(l)

    print(para_list.iloc[l[0]])

    



matching_score(10, "pregnant woman")
def cosine_sim(a, b):

    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

    return cos_sim
len(tf_idf)
D = np.zeros((N, total_vocab_size))

cnt=0

for i in tf_idf:

    if cnt%100000==0:

        print(cnt)

    try:

        ind = total_vocab.index(i[1])

        D[i[0]][ind] = tf_idf[i]

    except:

        pass

    cnt+=1
def gen_vector(tokens):



    Q = np.zeros((len(total_vocab)))

    

    counter = Counter(tokens)

    words_count = len(tokens)



    query_weights = {}

    

    for token in np.unique(tokens):

        

        tf = counter[token]/words_count

        df = doc_freq(token)

        idf = math.log((N+1)/(df+1))



        try:

            ind = total_vocab.index(token)

            Q[ind] = tf*idf

        except:

            pass

    return Q
def cosine_similarity(k, query):

    # print("Cosine Similarity")

    preprocessed_query = preprocess(query)

    tokens = word_tokenize(str(preprocessed_query))

    

    # print("\nQuery:", query)

    # print("")

    # print(tokens)

    

    d_cosines = []

    

    query_vector = gen_vector(tokens)

    

    for d in D:

        d_cosines.append(cosine_sim(query_vector, d))

        

    out = np.array(d_cosines).argsort()[-k:][::-1]

    

    result=pd.DataFrame(columns=['cord_uid','sha','paper_id','doi','journal','title','authors','abstract','text','url'])

    for file in out:

      found=result[result['paper_id'].str.contains(str(para_list.iloc[file]['paper_id']),na=False)]

      if len(found)!=0:

        indx=result.index[result['paper_id'] == para_list.iloc[file]['paper_id']]

        result.loc[indx]['text']=result.loc[indx]['text'] +"***************************************************"+ para_list.iloc[file]['text']

      else:

        result=result.append(para_list.iloc[file])

    return result
pd.options.display.max_colwidth=70

questions=['Smoking, pre-existing pulmonary disease',

          'Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities',

          'Neonates and pregnant women',

          'Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences',

          'Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors',

          'Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups',

          'Susceptibility of populations',

          'Public health mitigation measures that could be effective for control']

Q = cosine_similarity(10, questions[0])

Q2 = cosine_similarity(10, questions[1])

Q3 = cosine_similarity(10, questions[2])

Q4 = cosine_similarity(10, questions[3])

Q5 = cosine_similarity(10, questions[4])

Q6 = cosine_similarity(10, questions[5])

Q7 = cosine_similarity(10, questions[6])

Q8 = cosine_similarity(10, questions[7])

Q.to_csv('./smokingPreExistingPulmonaryDisease.csv', index=False)

Q2.to_csv('./coinfectionsComorbities.csv', index=False)

Q3.to_csv('./neonatesPregnantWomen.csv', index=False)

Q4.to_csv('./socioeconomicBehavioralFactors.csv', index=False)

Q5.to_csv('./transmissionDynamicsVirus.csv', index=False)

Q6.to_csv('./severityDiseaseIncludingRisk.csv', index=False)

Q7.to_csv('./susceptibilityOfPopulations.csv', index=False)

Q8.to_csv('./publicHealthMitigationMeasures.csv', index=False)

Q8['title']
Q.head()
(para_list.loc[para_list['Cluster'] == 18]).head()