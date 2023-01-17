# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import sqlite3
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from gensim.models import Word2Vec

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from tqdm import tqdm

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
conn = sqlite3.connect('../input/database.sqlite')
filtered_data = pd.read_sql_query(""" SELECT * FROM Reviews WHERE Score != 3 """, conn)

def partition(x):
    if x < 3:
        return 0
    return 1

actualScore = filtered_data['Score']
positiveNegative = actualScore.map(partition) 
filtered_data['Score'] = positiveNegative
print("Number of data points in our data", filtered_data.shape)
filtered_data.head(5)
display= pd.read_sql_query("""
SELECT *
FROM Reviews
WHERE Score != 3 AND UserId="AR5J8UI46CURR"
ORDER BY ProductID
""", conn)
display.head()
sorted_data=filtered_data.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
final=sorted_data.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, keep='first', inplace=False)
final.shape
(final['Id'].size*1.0)/(filtered_data['Id'].size*1.0)*100
display= pd.read_sql_query("""
SELECT *
FROM Reviews
WHERE Score != 3 AND Id=44737 OR Id=64422
ORDER BY ProductID
""", conn)

display.head()
final=final[final.HelpfulnessNumerator<=final.HelpfulnessDenominator]
print(final.shape)
final['Score'].value_counts()
stop = set(stopwords.words('english'))
sno = nltk.stem.SnowballStemmer('english')

def cleanhtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', sentence)
    return cleantext
def cleanpunc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    return  cleaned
i=0
str1=' '
final_string=[]
all_positive_words=[] 
all_negative_words=[] 
s=''
for sent in tqdm(final['Text'].values):
    filtered_sentence=[]
    #print(sent);
    sent=cleanhtml(sent) 
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):    
                if(cleaned_words.lower() not in stop):
                    s=(sno.stem(cleaned_words.lower())).encode('utf8')
                    filtered_sentence.append(s)
                    if (final['Score'].values)[i] == 1: 
                        all_positive_words.append(s) 
                    if(final['Score'].values)[i] == 0:
                        all_negative_words.append(s) 
                else:
                    continue
            else:
                continue 
    str1 = b" ".join(filtered_sentence) #final string of cleaned words

    final_string.append(str1)
    i+=1

final['CleanedText']= final_string
final['CleanedText']= final['CleanedText'].str.decode("utf-8")
data_pos = final[final['Score'] == 1].sample(n = 1000)
print('Shape of positive reviews', data_pos.shape)
print()

data_neg = final[final['Score'] == 0].sample(n = 1000)
print('Shape of negative reviews', data_neg.shape)
print()

final_reviews = pd.concat([data_pos, data_neg])
print('Shape of final reviews', final_reviews.shape)

score_2000 = final_reviews['Score']
sample_2000 = final_reviews['CleanedText']
count_vect = CountVectorizer(ngram_range=(1,1))

std_scaler = StandardScaler(with_mean=False)

sample_2000 = count_vect.fit_transform(sample_2000)
sample_2000 = std_scaler.fit_transform(sample_2000)
sample_2000 = sample_2000.todense()

print(sample_2000.shape)
model = TSNE(n_components=2, random_state=0)
tsne = model.fit_transform(sample_2000)

tsne_data = np.vstack((tsne.T, score_2000)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "score"))

sns.FacetGrid(tsne_df, hue='score', size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.show()
tf_idf_vect = TfidfVectorizer(ngram_range=(1,2))
final_tf_idf = tf_idf_vect.fit_transform(final_reviews['CleanedText'])
final_tf_idf = final_tf_idf.todense()

print(final_tf_idf.shape)
model = TSNE(n_components=2, random_state=0)
tsne = model.fit_transform(final_tf_idf)

tsne_data = np.vstack((tsne.T, score_2000)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "score"))

sns.FacetGrid(tsne_df, hue='score', size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.show()
list_of_sent=[]
for sent in final_reviews['CleanedText'].values:
    list_of_sent.append(sent.split())
w2v_model = Word2Vec(list_of_sent, min_count=5, size=50, workers=4)

w2v_words = w2v_model[w2v_model.wv.vocab]
len(w2v_words)
sent_vectors = []
for sent in tqdm(list_of_sent):
    sent_vec = np.zeros(50)
    cnt_words =0
    for word in sent:
        if word in w2v_words:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
    if cnt_words != 0:
        sent_vec /= cnt_words
    sent_vectors.append(sent_vec)
    
print(len(sent_vectors))
print(len(sent_vectors[0]))
model = TSNE(n_components=2, random_state=0)
tsne = model.fit_transform(sent_vectors)

tsne_data = np.vstack((tsne.T, score_2000)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "score"))

sns.FacetGrid(tsne_df, hue='score', size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.show()
model = TfidfVectorizer()
tf_idf_matrix = model.fit_transform(final_reviews['CleanedText'].values)
dictionary = dict(zip(model.get_feature_names(), list(model.idf_)))
tfidf_feat = model.get_feature_names() # tfidf words/col-names

tfidf_sent_vectors = []
row=0;
for sent in tqdm(list_of_sent):
    sent_vec = np.zeros(50) 
    weight_sum =0; 
    for word in sent: 
        if word in w2v_words:
            vec = w2v_model.wv[word]
            tf_idf = dictionary[word]*(sent.count(word)/len(sent))
            sent_vec += (vec * tf_idf)
            weight_sum += tf_idf
    if weight_sum != 0:
        sent_vec /= weight_sum
    tfidf_sent_vectors.append(sent_vec)
    row += 1
model = TSNE(n_components=2, random_state=15)
tfidf_w2v_points = model.fit_transform(tfidf_sent_vectors)

tsne_data = np.vstack((tfidf_w2v_points.T, score_2000)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=('Dim_1', 'Dim_2', 'score'))

sns.FacetGrid(tsne_df, hue='score', size=5).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.show()