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

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%matplotlib inline



import sqlite3

import pandas as pd

import numpy as np

import nltk

import string

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.metrics import roc_curve, auc

from nltk.stem.porter import PorterStemmer # nltk is Natural Language Processing Toolkit

import pickle

def saveindisk(obj,filename):

    pickle.dump(obj,open(filename+".p","wb"), protocol=4)

def openfromdisk(filename):

    temp = pickle.load(open(filename+".p","rb"))

    return temp
con = sqlite3.connect("/kaggle/input/amazon-fine-food-reviews/database.sqlite")



# For positive reviews score should be 4,5 and for negatie reviews score should b 1,2

# So we are eliminating all those where score is 3 as it is being considered as a neutral point

filtered_data = pd.read_sql_query("""

SELECT *

FROM Reviews

WHERE Score != 3

""",con)



# In this function from column Score we are removing the rating and putting 'negative' or 'positive' comment in score column

def partition(x):

    if x<3:

        return 'negative'

    return 'positive'



actualScore = filtered_data['Score']

positiveNegative = actualScore.map(partition)

filtered_data['Score'] = positiveNegative
filtered_data.shape

filtered_data.head()
display = pd.read_sql_query("""

SELECT *

FROM Reviews 

WHERE Score != 3 AND UserId='AR5J8UI46CURR'

ORDER BY ProductId

""",con)

display
# Sorting data acc to ProductId

sorted_data = filtered_data.sort_values('ProductId', axis=0, ascending=True)
final = sorted_data.drop_duplicates(subset={"UserId", "ProfileName", "Time", "Text"}, keep='first', inplace=False)

# keep says to keep the first data and discard the rest duplicate data

final.shape
(final['Id'].size*1.0)/(filtered_data['Id'].size*1.0)*100
display = pd.read_sql_query("""

SELECT *

FROM Reviews 

WHERE Score != 3 AND Id=44737 OR Id=64422

ORDER BY ProductId

""",con)

display
final=final[final.HelpfulnessNumerator<=final.HelpfulnessDenominator]
print(final.shape)

final['Score'].value_counts()
count_vect = CountVectorizer() # In scikit-learn

final_counts = count_vect.fit_transform(final['Text'].values)
type(final_counts)
final_counts.get_shape()
# Find sentence containing html tags

import re # re is regular expression

i=0

for sent in final['Text'].values:

    if (len(re.findall('<.*?>', sent))):

        print(i)

        print(sent)

        break

    i+=1
import string

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer



stop=set(stopwords.words('english')) # Set of stopwords 

sno = nltk.stem.SnowballStemmer('english')



def cleanhtml(sentence):

    cleanr = re.compile('<.*?>')

    cleantext = re.sub(cleanr, ' ', sentence) # sub is substitute. html tags will be replaced by space

    return cleantext



def cleanpunc(sentence):

    cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)

    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)

    return cleaned



print(stop)

print(sno.stem('tasty'))
i=0

strl = ' '

final_string =[]

all_positive_words=[] # Store words from +ve reviews

all_negative_words=[] # Store words from -ve reviews

s=''

for sent in final['Text'].values:

    filtered_sentence=[]

    sent=cleanhtml(sent) # Remove html tags

    for w in sent.split():

        for cleaned_words in cleanpunc(w).split():

            if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):

                if(cleaned_words.lower() not in stop):

                    s = (sno.stem(cleaned_words.lower())).encode('utf8')

                    filtered_sentence.append(s)

                    if (final['Score'].values)[i] == 'positive':

                        all_positive_words.append(s)

                    if (final['Score'].values)[i] == 'negative':

                        all_negative_words.append(s)

                else:

                    continue

            else:

                continue

                

    strl = b" ".join(filtered_sentence)

    final_string.append(strl)

    i+=1
final['CleanedText'] = final_string # Adding a cloumn of CleanedText
final.head(3)

# Store final table into SQL table

conn = sqlite3.connect('final.sqlite')

c=conn.cursor()

conn.text_factory = str

final.to_sql('Reviews', conn, schema=None, if_exists='replace')
freq_dist_positive = nltk.FreqDist(all_positive_words)

freq_dist_negative = nltk.FreqDist(all_negative_words)

print("Most common +ve words : ", freq_dist_positive.most_common(20))

print("Most common -ve words : ", freq_dist_negative.most_common(20))
# Bi-gram

count_vect = CountVectorizer(ngram_range=(1,2))

final_bigram_counts = count_vect.fit_transform(final['Text'].values)
final_bigram_counts.get_shape()
tf_idf_vect = TfidfVectorizer(ngram_range=(1,2))

final_tf_idf = tf_idf_vect.fit_transform(final['Text'].values) # final_tf_idf is a sparse matrix being created
final_tf_idf.get_shape()
features = tf_idf_vect.get_feature_names()

len(features)
features[100000:100010]
# To get vector for any review let its for review 3

print(final_tf_idf[3,:].toarray()[0]) # Returns a numpy array
def top_tfidf_feats(row, features, top_n = 25):

    topn_ids = np.argsort(row)[::-1][:top_n]

    top_feats = [(features[i], row[i]) for i in topn_ids]

    df = pd.DataFrame(top_feats)

    df.columns = ['feature', 'tfidf']

    return df



top_tfidf = top_tfidf_feats(final_tf_idf[1,:].toarray()[0], features, 25)
top_tfidf # Top 25 words(unigram or bigram) of review 1 from vector v1
# Using goggle news word2vector

#from gensim.models import Word2Vwec

#from gensim.models import KeyedVectors

#import pickle



#model = KeyedVectors.load_word2vec_format('GoggleNews-vectors-negative300.bin.gz') # 300 dim representation of word 2 vector
#model.wv['computer'] # wv is worrdvector
#model.wv.similarity('woman', 'man') # Similarity between man and woman. Max similarity can be 1 and min is 0
#model.wv.most_similar('woman') # It will return the most similar word in decreasing order
#model.wv.most_similar('tasty')
# Train your own word2vec model using your own text corpus

import gensim

i=0

list_of_sent=[]

for sent in final['Text'].values:

    filtered_sentence=[]

    sent = cleanhtml(sent)

    for w in sent.split():

        for cleaned_words in cleanpunc(w).split():

            if(cleaned_words.isalpha()):

                filtered_sentence.append(cleaned_words.lower())

            else:

                continue

    list_of_sent.append(filtered_sentence)
print(final['Text'].values[0])

print(list_of_sent[0]) # converting words to list
# To train

w2v_model=gensim.models.Word2Vec(list_of_sent, min_count=5, size=50, workers=4)

# min_count says that if a word doesn't occur at least 5 times then do not construct its w2v

# size is dim of vector v
words=list(w2v_model.wv.vocab)

print(len(words))
w2v_model.wv.most_similar('tasty')
w2v_model.wv.most_similar('like')
count_vect_feat = count_vect.get_feature_names()

count_vect_feat.index('like')

print(count_vect_feat[64055])
sent_vectors = []

for sent in list_of_sent:

    sent_vec = np.zeros(50)

    cnt_words = 0

    for word in sent:

        try:

            vec = w2v_model.wv[word]

            sent_vec += vec

            cnt_words += 1

        except:

            pass

    sent_vec /= cnt_words

    sent_vectors.append(sent_vec)

print(len(sent_vectors))

print(len(sent_vectors[0]))
# tfidf_feat = tf_idf_vect.get_feature_names()

# tfidf_sent_vectors =[]

# row=0

# for sent in list_of_sent:

    # sent_vec = np.zeros(50)

    # weight_sum = 0

    # for word in sent:

        # try:

            # vec = w2v_model.wv[word]

            # tfidf = final_tf_idf[row, tfidf_feat.index(word)]

            # sent_vec += vec*tf_idf

            # weight_sum += tf_idf

        # except:

            # pass

    # sent_vec /= weight_sum

    # tfidf_sent_vectors.append(sent_vec)

    # row += 1