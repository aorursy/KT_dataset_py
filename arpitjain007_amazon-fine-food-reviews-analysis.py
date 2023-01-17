import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import nltk

import string

import matplotlib.pyplot as plt

import seaborn as sns

import os



os.chdir("/kaggle/input/amazon-reviews/")

from dataset import *

from utils import *

os.chdir("/kaggle/working/")



# Modules for handling text data

from sklearn.feature_extraction.text import TfidfTransformer , TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer 

from nltk.stem.porter import PorterStemmer

from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer 

from nltk.stem.wordnet import WordNetLemmatizer

#Evaluation Metrics

from sklearn.metrics import auc , roc_curve , confusion_matrix

sqldatapath = "../input/amazon-fine-food-reviews/database.sqlite"



filter_data= SQLdata(sqldatapath)
filter_data.head()
actualScore = filter_data['Score']

positiveNegative = actualScore.map(partition) 

filter_data['Score'] = positiveNegative
filter_data[filter_data['UserId']=='AZY10LLTJ71NX']
# Sorting the data by Product ID 

sorted_data = filter_data.sort_values('ProductId' , ascending=True , axis=0 , inplace=False)



# Dropping the duplicates

final = sorted_data.drop_duplicates(subset={'UserId' , 'ProfileName', 'Time' , 'Text'} , inplace=False , keep='first')
filter_data[filter_data['HelpfulnessNumerator']> filter_data['HelpfulnessDenominator']]
final = final[final['HelpfulnessNumerator']< final['HelpfulnessDenominator']]
print('The final shape of the data is',final.shape)

print('The count of positive and negative reviews \n', final['Score'].value_counts())
# Get all the stopwords from English Language

stop = set(stopwords.words('english'))



# Initialize the Snowball stemming

snow = SnowballStemmer('english')
print(stop)
print(snow.stem('tasty'))
i =0

str1 = ' '

final_string = []

all_pos_words = []

all_neg_words = []

s = " "

import time

from tqdm import tqdm

start = time.time()



for sent in tqdm(final['Text'].values):

    filtered_sent =[]

    sent = cleanhtml(sent)  # remove HTML tags

    for w in sent.split():  

        for clean_words in cleanpunc(w).split():

            if ((clean_words.isalpha()) & (len(clean_words)>2)):

                if (clean_words.lower() not in stop):

                    s = (snow.stem(clean_words.lower())).encode('utf8')

                    filtered_sent.append(s)   # storing all filterd words 

                    if (final['Score'].values[i] == 'Positive'):

                        all_pos_words.append(s)  # storing all pos words

                    if (final['Score'].values[i] == 'Negative'):

                        all_neg_words.append(s)  # storing all neg words

                else:

                    continue

            else:

                continue

    str1 = b" ".join(filtered_sent)

    final_string.append(str1)

    

    i+=1



end = time.time()



print('The total time to run the cell: ', (end-start))
final['CleanedText'] = final_string
final.head()
# conn = sqlite3.connect('/kaggle/working/final.sqlite')

# c= conn.cursor()

# conn.text_factory = str

# final.to_sql('Amazon Reviews' , conn , if_exists='replace')
count_vect = CountVectorizer()

final_count = count_vect.fit_transform(final['Text'].values)
print("The type of vectors", type(final_count))

print('THE shape of the vector', final_count.get_shape())
#computing frequently occuring words in positive rewiew and negative rewiew

freq_pos = nltk.FreqDist(all_pos_words)

freq_neg = nltk.FreqDist(all_neg_words)



print("Most Common positive words: " , freq_pos.most_common(20))

print("\nMost Common positive words: " , freq_neg.most_common(20))
# Bi-gram 



count_vect = CountVectorizer(ngram_range=(1,2))

final_bi_gram_count = count_vect.fit_transform(final['Text'].values)
print("The shape of new vector matrix after Bi_gram " , final_bi_gram_count.get_shape())
tf_idf_vect = TfidfVectorizer(ngram_range=(1,2))

fina__tfidf = tf_idf_vect.fit_transform(final['Text'].values)
print("the shape of the vector after TF-IDF: ", fina__tfidf.get_shape())
features = tf_idf_vect.get_feature_names()

len(features)
# checking few features 

features[109000:109010]
# convert a row of sparsematrix into array

print(fina__tfidf[3,:].toarray()[0])
# getting top n features of TF-IDF



def top_n_feat(row , features , top=25):

    

    topn_ids= np.argsort(row)[::-1][:top]  #sorting the vector and reversing and pick top 25

    topn_feat = [(features[i] , row[i]) for i in topn_ids]

    df = pd.DataFrame(topn_feat)

    df.columns = [ 'Features' , 'tfidf']

    return df
top_tfidf = top_n_feat(fina__tfidf[2,:].toarray()[0] , features)
top_tfidf
from gensim.models import word2vec , KeyedVectors
model = KeyedVectors.load_word2vec_format('../input/gnewsvector/GoogleNews-vectors-negative300.bin' , binary=True)
model.wv['computer']
model.wv.similarity('woman' , 'man')
model.wv.most_similar('woman')
import gensim

from tqdm import tqdm

i=0

list_of_sent =[]



for sent in tqdm(final['Text'].values):

    filtered_sent=[]

    sent= cleanhtml(sent)

    for w in sent.split():

        for cleanW in cleanpunc(w).split():

            if (cleanW.isalpha()):

                filtered_sent.append(cleanW.lower())

            else:

                continue

    list_of_sent.append(filtered_sent)

    
print(final['Text'].values[0])
print(list_of_sent[0])
word2vecmodel = gensim.models.Word2Vec(list_of_sent , min_count=5, size=50 , workers=4)

words = list(word2vecmodel.wv.vocab)

print(len(words))
word2vecmodel.wv.most_similar('tasty')
word2vecmodel.wv.most_similar('like')
count_vect_feat = count_vect.get_feature_names()

count_vect_feat.index('like')

print(count_vect_feat[574544])
# Compute Word to vector



sent_vectors = []



for sent in tqdm(list_of_sent):    # looping over all the rewiew

    sent_vec = np.zeros(50)         # len of the word vector

    count_word =0                  #count the valid word in the sentence

    for word in sent:              #looping over each word in the sentence

        try:

            vec =word2vecmodel.wv[word] #converting each word to vector

            sent_vec+=vec                #adding vector to the defined vector

            count_word+=1               #counting the number of words converted. used later for average

        except:

            continue

    sent_vec/= count_word

    sent_vectors.append(sent_vec)
print(len(sent_vectors))

print(len(sent_vectors[0]))
tfidf_sent_vectors =[]

row=0

for sent in tqdm(list_of_sent[:10]):

    sent_vec= np.zeros(50)

    weight_sum=0

    for word in sent:

        try:

            vec = word2vecmodel.wv[word]

            tfidf = fina__tfidf[row , features.index(word)]

            sent_vec+= (vec*tfidf)

            weight_sum +=tfidf

        except:

            pass

    sent_vec/=weight_sum

    tfidf_sent_vectors.append(sent_vec)

    row+=1