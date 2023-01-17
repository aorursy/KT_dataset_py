import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import nltk as nk

%matplotlib inline
data = pd.read_csv('../input/jobs_data.csv' )
data.tail()
data['ID'] = data.iloc[:,0]
data = data[['ID','title','jobFunction','industry']]
data.head()
data.shape
data['title']
data.info()
data.describe().transpose()
data['tit_ind'] = data['title'] + data['industry']  
data['tit_ind'].head()
import re
from nltk.corpus import stopwords

from nltk.corpus import words

from nltk.stem.porter import PorterStemmer

#nltk.download('stopwords')

#nltk.download('words')
corpus = []

for i in range(0, len(data['tit_ind'])):

    jobtitle = re.sub('[^a-zA-Z]', ' ', data['tit_ind'][i])

    jobtitle = jobtitle.lower()

    jobtitle = jobtitle.split()

    ps = PorterStemmer()

    jobtitle = [ps.stem(word) for word in jobtitle if not word in set(stopwords.words('english'))] # this will also remove arabic word

    jobtitle = ' '.join(jobtitle)

    corpus.append(jobtitle)
corpus 
data['function'] = data['jobFunction'] + data['industry'] 
corpus_2 = []

for i in range(0, len(data['function'])):

    jobfunction = re.sub('[^a-zA-Z]', ' ', data['function'][i])

    jobfunction = jobfunction.lower()

    jobfunction = jobfunction.split()

    ps = PorterStemmer()

    jobfunction = [ps.stem(word) for word in jobfunction if not word in set(stopwords.words('english'))] # this will also remove arabic word

    jobfunction = ' '.join(jobfunction)

    corpus_2.append(jobfunction)
corpus_2 
d = {'ID': data['ID'] ,'job': corpus}
df_job = pd.DataFrame(data = d)
df_job.head()
df_func = pd.DataFrame({'function' :corpus_2})
df_func.head()
from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer()

tfidf_job = tf.fit_transform(df_job.iloc[:,1])

tfidf_func = tf.transform(df_func.iloc[:,0])
tfidf_func.shape


from sklearn.metrics.pairwise import cosine_similarity



cos_similarity_tfidf = map(lambda x: cosine_similarity(tfidf_job,x), tfidf_func )
cosine_similarity(tfidf_job,tfidf_func)
output2 = list(cos_similarity_tfidf)
output2 
def get_recommendation(jobtitle):

    #recommendation = pd.DataFrame(columns = ['ID','title' , 'Recommended function', 'score'])

    id = list(data['title']).index(jobtitle)

    recommended = data['jobFunction'][id]

    score = output2[id][0][0]

    result = {'ID': id , 'title':jobtitle ,'Recommended function' : recommended , 'score':score  }

    result = pd.DataFrame(result , index = range(1))

    return result
get_recommendation('Odoo Developer')