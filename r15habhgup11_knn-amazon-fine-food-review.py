%matplotlib inline

import warnings

warnings.filterwarnings("ignore")





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

from nltk.stem.porter import PorterStemmer



import re

# Tutorial about Python regular expressions: https://pymotw.com/2/re/

import string

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer



from gensim.models import Word2Vec

from gensim.models import KeyedVectors

import pickle



from tqdm import tqdm

import os
# using SQLite Table to read data.

con = sqlite3.connect('../input/database.sqlite') 



# filtering only positive and negative reviews i.e. 

# not taking into consideration those reviews with Score=3

# SELECT * FROM Reviews WHERE Score != 3 LIMIT 500000, will give top 500000 data points

# you can change the number to any other number based on your computing power



# filtered_data = pd.read_sql_query(""" SELECT * FROM Reviews WHERE Score != 3 LIMIT 500000""", con) 

# for tsne assignment you can take 5k data points



filtered_data = pd.read_sql_query(""" SELECT * FROM Reviews WHERE Score != 3 LIMIT 50000""", con) 



# Give reviews with Score>3 a positive rating(1), and reviews with a score<3 a negative rating(0).

def partition(x):

    if x < 3:

        return 0

    return 1



#changing reviews with score less than 3 to be positive and vice-versa

actualScore = filtered_data['Score']

positiveNegative = actualScore.map(partition) 

filtered_data['Score'] = positiveNegative

print("Number of data points in our data", filtered_data.shape)

filtered_data.head(3)
display = pd.read_sql_query("""

SELECT UserId, ProductId, ProfileName, Time, Score, Text, COUNT(*)

FROM Reviews

GROUP BY UserId

HAVING COUNT(*)>1

""", con)
print(display.shape)

display.head()
display[display['UserId']=='AZY10LLTJ71NX']
display['COUNT(*)'].sum()
display= pd.read_sql_query("""

SELECT *

FROM Reviews

WHERE Score != 3 AND UserId="AR5J8UI46CURR"

ORDER BY ProductID

""", con)

display.head()
#Sorting data according to ProductId in ascending order

sorted_data=filtered_data.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
#Deduplication of entries

final=sorted_data.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, keep='first', inplace=False)

final.shape
#Checking to see how much % of data still remains

(final['Id'].size*1.0)/(filtered_data['Id'].size*1.0)*100
display= pd.read_sql_query("""

SELECT *

FROM Reviews

WHERE Score != 3 AND Id=44737 OR Id=64422

ORDER BY ProductID

""", con)



display.head()
final=final[final.HelpfulnessNumerator<=final.HelpfulnessDenominator]
#Before starting the next phase of preprocessing lets see the number of entries left

print(final.shape)



#How many positive and negative reviews are present in our dataset?

final['Score'].value_counts()
# printing some random reviews

sent_0 = final['Text'].values[0]

print(sent_0)

print("="*50)



sent_1000 = final['Text'].values[1000]

print(sent_1000)

print("="*50)



sent_1500 = final['Text'].values[1500]

print(sent_1500)

print("="*50)



sent_4900 = final['Text'].values[4900]

print(sent_4900)

print("="*50)
# remove urls from text python: https://stackoverflow.com/a/40823105/4084039

sent_0 = re.sub(r"http\S+", "", sent_0)

sent_1000 = re.sub(r"http\S+", "", sent_1000)

sent_150 = re.sub(r"http\S+", "", sent_1500)

sent_4900 = re.sub(r"http\S+", "", sent_4900)



print(sent_0)
# https://stackoverflow.com/questions/16206380/python-beautifulsoup-how-to-remove-all-tags-from-an-element

from bs4 import BeautifulSoup



soup = BeautifulSoup(sent_0, 'lxml')

text = soup.get_text()

print(text)

print("="*50)



soup = BeautifulSoup(sent_1000, 'lxml')

text = soup.get_text()

print(text)

print("="*50)



soup = BeautifulSoup(sent_1500, 'lxml')

text = soup.get_text()

print(text)

print("="*50)



soup = BeautifulSoup(sent_4900, 'lxml')

text = soup.get_text()

print(text)
# https://stackoverflow.com/a/47091490/4084039

import re



def decontracted(phrase):

    # specific

    phrase = re.sub(r"won't", "will not", phrase)

    phrase = re.sub(r"can\'t", "can not", phrase)



    # general

    phrase = re.sub(r"n\'t", " not", phrase)

    phrase = re.sub(r"\'re", " are", phrase)

    phrase = re.sub(r"\'s", " is", phrase)

    phrase = re.sub(r"\'d", " would", phrase)

    phrase = re.sub(r"\'ll", " will", phrase)

    phrase = re.sub(r"\'t", " not", phrase)

    phrase = re.sub(r"\'ve", " have", phrase)

    phrase = re.sub(r"\'m", " am", phrase)

    return phrase
sent_1500 = decontracted(sent_1500)

print(sent_1500)

print("="*50)
#remove words with numbers python: https://stackoverflow.com/a/18082370/4084039

sent_0 = re.sub("\S*\d\S*", "", sent_0).strip()

print(sent_0)
#remove spacial character: https://stackoverflow.com/a/5843547/4084039

sent_1500 = re.sub('[^A-Za-z0-9]+', ' ', sent_1500)

print(sent_1500)
# https://gist.github.com/sebleier/554280

# we are removing the words from the stop words list: 'no', 'nor', 'not'

# <br /><br /> ==> after the above steps, we are getting "br br"

# we are including them into stop words list

# instead of <br /> if we have <br/> these tags would have revmoved in the 1st step



stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\

            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \

            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\

            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \

            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \

            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \

            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\

            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\

            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\

            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \

            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \

            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\

            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\

            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \

            'won', "won't", 'wouldn', "wouldn't"])
# Combining all the above stundents 

from tqdm import tqdm

preprocessed_reviews = []

# tqdm is for printing the status bar

for sentance in tqdm(final['Text'].values):

    sentance = re.sub(r"http\S+", "", sentance)

    sentance = BeautifulSoup(sentance, 'lxml').get_text()

    sentance = decontracted(sentance)

    sentance = re.sub("\S*\d\S*", "", sentance).strip()

    sentance = re.sub('[^A-Za-z]+', ' ', sentance)

    # https://gist.github.com/sebleier/554280

    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)

    preprocessed_reviews.append(sentance.strip())
preprocessed_reviews[1500]
## Similartly you can do preprocessing for review summary also.
#BoW

count_vect = CountVectorizer() #in scikit-learn

count_vect.fit(preprocessed_reviews)

print("some feature names ", count_vect.get_feature_names()[:10])

print('='*50)



final_counts = count_vect.transform(preprocessed_reviews)

# print(final_counts)

# print("the type of count vectorizer ",type(final_counts))

# print("the shape of out text BOW vectorizer ",final_counts.get_shape())

# print("the number of unique words ", final_counts.get_shape()[1])

# print(final['Score'])
#bi-gram, tri-gram and n-gram



#removing stop words like "not" should be avoided before building n-grams

# count_vect = CountVectorizer(ngram_range=(1,2))

# please do read the CountVectorizer documentation http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html



# you can choose these numebrs min_df=10, max_features=5000, of your choice

count_vect = CountVectorizer(ngram_range=(1,2), min_df=10, max_features=5000)

final_bigram_counts = count_vect.fit_transform(preprocessed_reviews)

print("the type of count vectorizer ",type(final_bigram_counts))

print("the shape of out text BOW vectorizer ",final_bigram_counts.get_shape())

print("the number of unique words including both unigrams and bigrams ", final_bigram_counts.get_shape()[1])
tf_idf_vect = TfidfVectorizer(ngram_range=(1,2), min_df=10)

tf_idf_vect.fit(preprocessed_reviews)

print("some sample features(unique words in the corpus)",tf_idf_vect.get_feature_names()[0:10])

print('='*50)



final_tf_idf = tf_idf_vect.transform(preprocessed_reviews)

print("the type of count vectorizer ",type(final_tf_idf))

print("the shape of out text TFIDF vectorizer ",final_tf_idf.get_shape())

print("the number of unique words including both unigrams and bigrams ", final_tf_idf.get_shape()[1])
# Train your own Word2Vec model using your own text corpus

i=0

list_of_sentance=[]

for sentance in preprocessed_reviews:

    list_of_sentance.append(sentance.split())
# Using Google News Word2Vectors



# in this project we are using a pretrained model by google

# its 3.3G file, once you load this into your memory 

# it occupies ~9Gb, so please do this step only if you have >12G of ram

# we will provide a pickle file wich contains a dict , 

# and it contains all our courpus words as keys and  model[word] as values

# To use this code-snippet, download "GoogleNews-vectors-negative300.bin" 

# from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit

# it's 1.9GB in size.





# http://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/#.W17SRFAzZPY

# you can comment this whole cell

# or change these varible according to your need



is_your_ram_gt_16g=False

want_to_use_google_w2v = False

want_to_train_w2v = True



if want_to_train_w2v:

    # min_count = 5 considers only words that occured atleast 5 times

    w2v_model=Word2Vec(list_of_sentance,min_count=5,size=50, workers=4)

    print(w2v_model.wv.most_similar('great'))

    print('='*50)

    print(w2v_model.wv.most_similar('worst'))

    

elif want_to_use_google_w2v and is_your_ram_gt_16g:

    if os.path.isfile('GoogleNews-vectors-negative300.bin'):

        w2v_model=KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

        print(w2v_model.wv.most_similar('great'))

        print(w2v_model.wv.most_similar('worst'))

    else:

        print("you don't have gogole's word2vec file, keep want_to_train_w2v = True, to train your own w2v ")
w2v_words = list(w2v_model.wv.vocab)

print("number of words that occured minimum 5 times ",len(w2v_words))

print("sample words ", w2v_words[0:50])
# average Word2Vec

# compute average word2vec for each review.

sent_vectors = []; # the avg-w2v for each sentence/review is stored in this list

for sent in tqdm(list_of_sentance): # for each review/sentence

    sent_vec = np.zeros(50) # as word vectors are of zero length 50, you might need to change this to 300 if you use google's w2v

    cnt_words =0; # num of words with a valid vector in the sentence/review

    for word in sent: # for each word in a review/sentence

        if word in w2v_words:

            vec = w2v_model.wv[word]

            sent_vec += vec

            cnt_words += 1

    if cnt_words != 0:

        sent_vec /= cnt_words

    sent_vectors.append(sent_vec)

print(len(sent_vectors))

print(len(sent_vectors[0]))
# S = ["abc def pqr", "def def def abc", "pqr pqr def"]

model = TfidfVectorizer()

tf_idf_matrix = model.fit_transform(preprocessed_reviews)

# we are converting a dictionary with word as a key, and the idf as a value

dictionary = dict(zip(model.get_feature_names(), list(model.idf_)))
# TF-IDF weighted Word2Vec

tfidf_feat = model.get_feature_names() # tfidf words/col-names

# final_tf_idf is the sparse matrix with row= sentence, col=word and cell_val = tfidf



tfidf_sent_vectors = []; # the tfidf-w2v for each sentence/review is stored in this list

row=0;

for sent in tqdm(list_of_sentance): # for each review/sentence 

    sent_vec = np.zeros(50) # as word vectors are of zero length

    weight_sum =0; # num of words with a valid vector in the sentence/review

    for word in sent: # for each word in a review/sentence

        if word in w2v_words and word in tfidf_feat:

            vec = w2v_model.wv[word]

#             tf_idf = tf_idf_matrix[row, tfidf_feat.index(word)]

            # to reduce the computation we are 

            # dictionary[word] = idf value of word in whole courpus

            # sent.count(word) = tf valeus of word in this review

            tf_idf = dictionary[word]*(sent.count(word)/len(sent))

            sent_vec += (vec * tf_idf)

            weight_sum += tf_idf

    if weight_sum != 0:

        sent_vec /= weight_sum

    tfidf_sent_vectors.append(sent_vec)

    row += 1
# Importing Library

import numpy as np

import pandas as pd

import sklearn

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

from collections import Counter

from sklearn.metrics import roc_auc_score

from sklearn.metrics import confusion_matrix

import seaborn as sns
#Spliting Traing Test and CrossValidation

X=preprocessed_reviews

X=np.array(X)

y = np.array(final['Score'])

X_1, X_test, y_1, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

X_tr, X_cv, y_tr, y_cv = train_test_split(X_1, y_1, test_size=0.3,random_state=1) 
#converting Reviews to Bag of words

count_vect = CountVectorizer()

final_X_tr=count_vect.fit_transform(X_tr)

final_X_test=count_vect.transform(X_test)

final_X_cv=count_vect.transform(X_cv)
#Calculating for finding Best K

roc_tr=[]

roc_cv=[]

k_value=[]

max_auc_score=0

K_best=0

for i in range(1,100,4):

    # instantiate learning model (k = 100)

    knn = KNeighborsClassifier(algorithm='brute',metric='minkowski',n_neighbors=i)



    # fitting the model on train data

    knn.fit(final_X_tr, y_tr)



    # predict the response on the crossvalidation 

    pred_cv = knn.predict_proba(final_X_cv)

    pred_cv=(pred_cv)[:,1]

    roc_cv.append(roc_auc_score(y_cv,pred_cv))

    

     # predict the response on the traininig

    pred_tr = knn.predict_proba(final_X_tr)

    pred_tr=(pred_tr)[:,1]

    roc_tr.append(roc_auc_score(y_tr,pred_tr))

    k_value.append(i)

    

    #finding best k using max value of auc score

    if roc_auc_score(y_cv,pred_cv)>max_auc_score:

        k_best=i

        max_auc_score=roc_auc_score(y_cv,pred_cv)

print(k_best)        

print(max_auc_score)

k1=k_best

auc1=max_auc_score
# plotting curve between K vs Train and Cross validation Data

plt.plot(k_value,roc_cv ,label="AUC cv")

plt.plot(k_value,roc_tr,label="AUC train")

plt.legend()

plt.title('AUC Score vs K')

plt.xlabel('K')

plt.ylabel('AUC')

plt.show()
# Training the model using best K    

knn = KNeighborsClassifier(algorithm='brute',metric='minkowski',n_neighbors=k_best)

knn.fit(final_X_tr, y_tr)

#predicting probability on Test data

pred_test = knn.predict_proba(final_X_test)

pred_test=(pred_test)[:,1]

#predicting probablity of Training data

pred_tr = knn.predict_proba(final_X_tr)

pred_tr=(pred_tr)[:,1]



#Plotting Roc Curve



#fiding fpr and tpr on Traing and Test Data

fpr, tpr, threshold = metrics.roc_curve(y_test, pred_test)

fpr1, tpr1, threshold1 = metrics.roc_curve(y_tr, pred_tr)



#plotting

plt.plot(fpr,tpr ,label="characterstics on Test data")

plt.plot(fpr1,tpr1 ,label="characterstics on Train data")

plt.legend()

plt.title('ROC on best K')

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.show()
#finding Confusion_matrix

predic=knn.predict(final_X_test)

conf_mat = confusion_matrix(y_test, predic)

print(conf_mat)
#plotting Confusion Matrix

class_label = ["negative", "positive"]

df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)

sns.heatmap(df, annot = True,fmt="d")

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.show()
# Please write all the code with proper documentation

#Spliting Traing Test and CrossValidation

X=preprocessed_reviews

X=np.array(X)

y = np.array(final['Score'])

X_1, X_test, y_1, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

X_tr, X_cv, y_tr, y_cv = train_test_split(X_1, y_1, test_size=0.3,random_state=1) 
#converting Reviews to tf_idf_vec

tf_idf_vect = TfidfVectorizer(ngram_range=(1,2),min_df=10 )

final_X_tr=tf_idf_vect.fit_transform(X_tr)

final_X_test=tf_idf_vect.transform(X_test)

final_X_cv=tf_idf_vect.transform(X_cv)
#Calculating for finding Best K

roc_tr=[]

roc_cv=[]

k_value=[]

max_auc_score=0

K_best=0

for i in range(1,100,4):

    # instantiate learning model (k = 100)

    knn = KNeighborsClassifier(algorithm='brute',metric='cosine',n_neighbors=i)



    # fitting the model on train data

    knn.fit(final_X_tr, y_tr)



    # predict the response on the crossvalidation 

    pred_cv = knn.predict_proba(final_X_cv)

    pred_cv=(pred_cv)[:,1]

    roc_cv.append(roc_auc_score(y_cv,pred_cv))

    

     # predict the response on the traininig

    pred_tr = knn.predict_proba(final_X_tr)

    pred_tr=(pred_tr)[:,1]

    roc_tr.append(roc_auc_score(y_tr,pred_tr))

    k_value.append(i)

    

    #finding best k using max value of auc score

    if roc_auc_score(y_cv,pred_cv)>max_auc_score:

        k_best=i

        max_auc_score=roc_auc_score(y_cv,pred_cv)

print(k_best)        

k2=k_best

auc2=max_auc_score        

        
print(max_auc_score)
# plotting curve between K vs Train and Cross validation Data

plt.plot(k_value,roc_cv ,label="AUC cv")

plt.plot(k_value,roc_tr,label="AUC train")

plt.legend()

plt.title('AUC Score vs K')

plt.xlabel('K')

plt.ylabel('AUC')

plt.show()
# Training the model using best K    

knn = KNeighborsClassifier(algorithm='brute',metric='minkowski',n_neighbors=k_best)

knn.fit(final_X_tr, y_tr)

#predicting probability on Test data

pred_test = knn.predict_proba(final_X_test)

pred_test=(pred_test)[:,1]

#predicting probablity of Training data

pred_tr = knn.predict_proba(final_X_tr)

pred_tr=(pred_tr)[:,1]
#Plotting Roc Curve



#fiding fpr and tpr on Traing and Test Data

fpr, tpr, threshold = metrics.roc_curve(y_test, pred_test)

fpr1, tpr1, threshold1 = metrics.roc_curve(y_tr, pred_tr)



#plotting

plt.plot(fpr,tpr ,label="characterstics on Test data")

plt.plot(fpr1,tpr1 ,label="characterstics on Train data")

plt.legend()

plt.title('ROC on best K')

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.show()
#finding Confusion_matrix

predic=knn.predict(final_X_test)

conf_mat = confusion_matrix(y_test, predic)

print(conf_mat)
#plotting Confusion Matrix

class_label = ["negative", "positive"]

df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)

sns.heatmap(df, annot = True,fmt="d")

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.show()
X=preprocessed_reviews

y = np.array(final['Score'])

X_1, X_test, y_1, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

X_tr, X_cv, y_tr, y_cv = train_test_split(X_1, y_1, test_size=0.3,random_state=1)

# Please write all the code with proper documentation

# average Word2Vec

# compute average word2vec for each review.

list_of_sentance_tr=[]

for sentance in X_tr:

    list_of_sentance_tr.append(sentance.split())

final_X_tr = []; # the avg-w2v for each sentence/review is stored in this list

for sent in tqdm(list_of_sentance_tr): # for each review/sentence

    sent_vec = np.zeros(50) # as word vectors are of zero length 50, you might need to change this to 300 if you use google's w2v

    cnt_words =0; # num of words with a valid vector in the sentence/review

    for word in sent: # for each word in a review/sentence

        if word in w2v_words:

            vec = w2v_model.wv[word]

            sent_vec += vec

            cnt_words += 1

    if cnt_words != 0:

        sent_vec /= cnt_words

    final_X_tr.append(sent_vec)

    

    

list_of_sentance_cv=[]

for sentance in X_cv:

    list_of_sentance_cv.append(sentance.split())    

final_X_cv = []; # the avg-w2v for each sentence/review is stored in this list

for sent in tqdm(list_of_sentance_cv): # for each review/sentence

    sent_vec = np.zeros(50) # as word vectors are of zero length 50, you might need to change this to 300 if you use google's w2v

    cnt_words =0; # num of words with a valid vector in the sentence/review

    for word in sent: # for each word in a review/sentence

        if word in w2v_words:

            vec = w2v_model.wv[word]

            sent_vec += vec

            cnt_words += 1

    if cnt_words != 0:

        sent_vec /= cnt_words

    final_X_cv.append(sent_vec)    

    

    

list_of_sentance_test=[]

for sentance in X_test:

    list_of_sentance_test.append(sentance.split())    

final_X_test = []; # the avg-w2v for each sentence/review is stored in this list

for sent in tqdm(list_of_sentance_test): # for each review/sentence

    sent_vec = np.zeros(50) # as word vectors are of zero length 50, you might need to change this to 300 if you use google's w2v

    cnt_words =0; # num of words with a valid vector in the sentence/review

    for word in sent: # for each word in a review/sentence

        if word in w2v_words:

            vec = w2v_model.wv[word]

            sent_vec += vec

            cnt_words += 1

    if cnt_words != 0:

        sent_vec /= cnt_words

    final_X_test.append(sent_vec)    

#Calculating for finding Best K

roc_tr=[]

roc_cv=[]

k_value=[]

max_auc_score=0

K_best=0

for i in tqdm(range(1,100,4)):

    # instantiate learning model (k = 100)

    knn = KNeighborsClassifier(algorithm='brute',metric='minkowski',n_neighbors=i)



    # fitting the model on train data

    knn.fit(final_X_tr, y_tr)



    # predict the response on the crossvalidation 

    pred_cv = knn.predict_proba(final_X_cv)

    pred_cv=(pred_cv)[:,1]

    roc_cv.append(roc_auc_score(y_cv,pred_cv))

    

     # predict the response on the traininig

    pred_tr = knn.predict_proba(final_X_tr)

    pred_tr=(pred_tr)[:,1]

    roc_tr.append(roc_auc_score(y_tr,pred_tr))

    k_value.append(i)

    

    #finding best k using max value of auc score

    if roc_auc_score(y_cv,pred_cv)>max_auc_score:

        k_best=i

        max_auc_score=roc_auc_score(y_cv,pred_cv)

print("best K is",k_best)

print("max AUC Score is",max_auc_score)

k3=k_best

auc3=max_auc_score
# plotting curve between K vs Train and Cross validation Data

plt.plot(k_value,roc_cv ,label="AUC cv")

plt.plot(k_value,roc_tr,label="AUC train")

plt.legend()

plt.title('AUC Score vs K')

plt.xlabel('K')

plt.ylabel('AUC')

plt.show()
# Training the model using best K    

knn = KNeighborsClassifier(algorithm='brute',metric='minkowski',n_neighbors=k_best)

knn.fit(final_X_tr, y_tr)

#predicting probability on Test data

pred_test = knn.predict_proba(final_X_test)

pred_test=(pred_test)[:,1]

#predicting probablity of Training data

pred_tr = knn.predict_proba(final_X_tr)

pred_tr=(pred_tr)[:,1]
#Plotting Roc Curve



#fiding fpr and tpr on Train and Test Data

fpr, tpr, threshold = metrics.roc_curve(y_test, pred_test)

fpr1, tpr1, threshold1 = metrics.roc_curve(y_tr, pred_tr)



#plotting

plt.plot(fpr,tpr ,label="characterstics on Test data")

plt.plot(fpr1,tpr1 ,label="characterstics on Train data")

plt.legend()

plt.title('ROC on best K')

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.show()
#finding Confusion_matrix

predic=knn.predict(final_X_test)

conf_mat = confusion_matrix(y_test, predic)

print(conf_mat)
#plotting Confusion Matrix

class_label = ["negative", "positive"]

df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)

sns.heatmap(df, annot = True,fmt="d")

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.show()
# Please write all the code with proper documentation

# TF-IDF weighted Word2Vec

tfidf_feat = model.get_feature_names() # tfidf words/col-names

# final_tf_idf is the sparse matrix with row= sentence, col=word and cell_val = tfidf



list_of_sentance_tr=[]

for sentance in X_tr:

    list_of_sentance_tr.append(sentance.split())

final_X_tr = []; # the tfidf-w2v for each sentence/review is stored in this list

row=0;

for sent in tqdm(list_of_sentance_tr): # for each review/sentence 

    sent_vec = np.zeros(50) # as word vectors are of zero length

    weight_sum =0; # num of words with a valid vector in the sentence/review

    for word in sent: # for each word in a review/sentence

        if word in w2v_words and word in tfidf_feat:

            vec = w2v_model.wv[word]

#             tf_idf = tf_idf_matrix[row, tfidf_feat.index(word)]

            # to reduce the computation we are 

            # dictionary[word] = idf value of word in whole courpus

            # sent.count(word) = tf valeus of word in this review

            tf_idf = dictionary[word]*(sent.count(word)/len(sent))

            sent_vec += (vec * tf_idf)

            weight_sum += tf_idf

    if weight_sum != 0:

        sent_vec /= weight_sum

    final_X_tr.append(sent_vec)

    row += 1



    

list_of_sentance_cv=[]

for sentance in X_cv:

    list_of_sentance_cv.append(sentance.split())

final_X_cv = []; # the tfidf-w2v for each sentence/review is stored in this list

row=0;

for sent in tqdm(list_of_sentance_cv): # for each review/sentence 

    sent_vec = np.zeros(50) # as word vectors are of zero length

    weight_sum =0; # num of words with a valid vector in the sentence/review

    for word in sent: # for each word in a review/sentence

        if word in w2v_words and word in tfidf_feat:

            vec = w2v_model.wv[word]

#             tf_idf = tf_idf_matrix[row, tfidf_feat.index(word)]

            # to reduce the computation we are 

            # dictionary[word] = idf value of word in whole courpus

            # sent.count(word) = tf valeus of word in this review

            tf_idf = dictionary[word]*(sent.count(word)/len(sent))

            sent_vec += (vec * tf_idf)

            weight_sum += tf_idf

    if weight_sum != 0:

        sent_vec /= weight_sum

    final_X_cv.append(sent_vec)

    row += 1 

    

    

list_of_sentance_test=[]

for sentance in X_test:

    list_of_sentance_test.append(sentance.split())

final_X_test = []; # the tfidf-w2v for each sentence/review is stored in this list

row=0;

for sent in tqdm(list_of_sentance_test): # for each review/sentence 

    sent_vec = np.zeros(50) # as word vectors are of zero length

    weight_sum =0; # num of words with a valid vector in the sentence/review

    for word in sent: # for each word in a review/sentence

        if word in w2v_words and word in tfidf_feat:

            vec = w2v_model.wv[word]

#             tf_idf = tf_idf_matrix[row, tfidf_feat.index(word)]

            # to reduce the computation we are 

            # dictionary[word] = idf value of word in whole courpus

            # sent.count(word) = tf valeus of word in this review

            tf_idf = dictionary[word]*(sent.count(word)/len(sent))

            sent_vec += (vec * tf_idf)

            weight_sum += tf_idf

    if weight_sum != 0:

        sent_vec /= weight_sum

    final_X_test.append(sent_vec)

    row += 1    
#Calculating for finding Best K

roc_tr=[]

roc_cv=[]

k_value=[]

max_auc_score=0

K_best=0

for i in tqdm(range(1,100,4)):

    # instantiate learning model (k = 100)

    knn = KNeighborsClassifier(algorithm='brute',metric='minkowski',n_neighbors=i)



    # fitting the model on train data

    knn.fit(final_X_tr, y_tr)



    # predict the response on the crossvalidation 

    pred_cv = knn.predict_proba(final_X_cv)

    pred_cv=(pred_cv)[:,1]

    roc_cv.append(roc_auc_score(y_cv,pred_cv))

    

     # predict the response on the traininig

    pred_tr = knn.predict_proba(final_X_tr)

    pred_tr=(pred_tr)[:,1]

    roc_tr.append(roc_auc_score(y_tr,pred_tr))

    k_value.append(i)

    

    #finding best k using max value of auc score

    if roc_auc_score(y_cv,pred_cv)>max_auc_score:

        k_best=i

        max_auc_score=roc_auc_score(y_cv,pred_cv)

print("best K is",k_best)

print("max AUC Score is",max_auc_score)

k4=k_best

auc4=max_auc_score
# plotting curve between K vs Train and Cross validation Data

plt.plot(k_value,roc_cv ,label="AUC cv")

plt.plot(k_value,roc_tr,label="AUC train")

plt.legend()

plt.title('AUC Score vs K')

plt.xlabel('K')

plt.ylabel('AUC')

plt.show()
# Training the model using best K    

knn = KNeighborsClassifier(algorithm='brute',metric='minkowski',n_neighbors=k_best)

knn.fit(final_X_tr, y_tr)

#predicting probability on Test data

pred_test = knn.predict_proba(final_X_test)

pred_test=(pred_test)[:,1]

#predicting probablity of Training data

pred_tr = knn.predict_proba(final_X_tr)

pred_tr=(pred_tr)[:,1]
#Plotting Roc Curve



#fiding fpr and tpr on Train and Test Data

fpr, tpr, threshold = metrics.roc_curve(y_test, pred_test)

fpr1, tpr1, threshold1 = metrics.roc_curve(y_tr, pred_tr)



#plotting

plt.plot(fpr,tpr ,label="characterstics on Test data")

plt.plot(fpr1,tpr1 ,label="characterstics on Train data")

plt.legend()

plt.title('ROC on best K')

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.show()
#finding Confusion_matrix

predic=knn.predict(final_X_test)

conf_mat = confusion_matrix(y_test, predic)

print(conf_mat)
#plotting Confusion Matrix

class_label = ["negative", "positive"]

df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)

sns.heatmap(df, annot = True,fmt="d")

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.show()
# Please write all the code with proper documentation

#Spliting Traing Test and CrossValidation

X=preprocessed_reviews

X=np.array(X)

y = np.array(final['Score'])



X, X_waste, y, y_waste = train_test_split(X, y, test_size=0.70, random_state=1)

X_1, X_test, y_1, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

X_tr, X_cv, y_tr, y_cv = train_test_split(X_1, y_1, test_size=0.3,random_state=1) 

#converting Reviews to Bag of words

count_vect = CountVectorizer()

final_X_tr=count_vect.fit_transform(X_tr)

final_X_test=count_vect.transform(X_test)

final_X_cv=count_vect.transform(X_cv)
#converting them todense bcz kd tree doest work on sparse matrics

final_X_tr=final_X_tr.todense()

final_X_test=final_X_test.todense()

final_X_cv=final_X_cv.todense()
#Calculating for finding Best K

roc_tr=[]

roc_cv=[]

k_value=[]

max_auc_score=0

K_best=0

for i in tqdm(range(1,100,20)):

    # instantiate learning model (k = 100)

    knn = KNeighborsClassifier(algorithm='kd_tree',metric='minkowski',n_neighbors=i)



    # fitting the model on train data

    knn.fit(final_X_tr, y_tr)



    # predict the response on the crossvalidation 

    pred_cv = knn.predict_proba(final_X_cv)

    pred_cv=(pred_cv)[:,1]

    roc_cv.append(roc_auc_score(y_cv,pred_cv))

    

     # predict the response on the traininig

    pred_tr = knn.predict_proba(final_X_tr)

    pred_tr=(pred_tr)[:,1]

    roc_tr.append(roc_auc_score(y_tr,pred_tr))

    k_value.append(i)

    

    #finding best k using max value of auc score

    if roc_auc_score(y_cv,pred_cv)>max_auc_score:

        k_best=i

        max_auc_score=roc_auc_score(y_cv,pred_cv)

print("best k is",k_best)

print("Max AUC is",max_auc_score)

k5=k_best

auc5=max_auc_score
# plotting curve between K vs Train and Cross validation Data

plt.plot(k_value,roc_cv ,label="AUC cv")

plt.plot(k_value,roc_tr,label="AUC train")

plt.legend()

plt.title('AUC Score vs K')

plt.xlabel('K')

plt.ylabel('AUC')

plt.show()
# Training the model using best K    

knn = KNeighborsClassifier(algorithm='kd_tree',metric='minkowski',n_neighbors=k_best)

knn.fit(final_X_tr, y_tr)

#predicting probability on Test data

pred_test = knn.predict_proba(final_X_test)

pred_test=(pred_test)[:,1]

#predicting probablity of Training data

pred_tr = knn.predict_proba(final_X_tr)

pred_tr=(pred_tr)[:,1]

#Plotting Roc Curve



#fiding fpr and tpr on Traing and Test Data

fpr, tpr, threshold = metrics.roc_curve(y_test, pred_test)

fpr1, tpr1, threshold1 = metrics.roc_curve(y_tr, pred_tr)



#plotting

plt.plot(fpr,tpr ,label="characterstics on Test data")

plt.plot(fpr1,tpr1 ,label="characterstics on Train data")

plt.legend()

plt.title('ROC on best K')

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.show()
#finding Confusion_matrix

predic=knn.predict(final_X_test)

conf_mat = confusion_matrix(y_test, predic)

print(conf_mat)
#plotting Confusion Matrix

class_label = ["negative", "positive"]

df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)

sns.heatmap(df, annot = True,fmt="d")

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.show()
# Please write all the code with proper documentation

X_1, X_test, y_1, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

X_tr, X_cv, y_tr, y_cv = train_test_split(X_1, y_1, test_size=0.3,random_state=1) 
#converting Reviews to tf_idf_vec

tf_idf_vect = TfidfVectorizer(ngram_range=(1,2),min_df=10)

final_X_tr=tf_idf_vect.fit_transform(X_tr)

final_X_test=tf_idf_vect.transform(X_test)

final_X_cv=tf_idf_vect.transform(X_cv)



#converting them todense bcz kd tree doest work on sparse matrics

final_X_tr=final_X_tr.todense()

final_X_test=final_X_test.todense()

final_X_cv=final_X_cv.todense()
#Calculating for finding Best K

roc_tr=[]

roc_cv=[]

k_value=[]

max_auc_score=0

K_best=0

for i in tqdm(range(1,100,20)):

    # instantiate learning model (k = 100)

    knn = KNeighborsClassifier(algorithm='kd_tree',metric='minkowski',n_neighbors=i)



    # fitting the model on train data

    knn.fit(final_X_tr, y_tr)



    # predict the response on the crossvalidation 

    pred_cv = knn.predict_proba(final_X_cv)

    pred_cv=(pred_cv)[:,1]

    roc_cv.append(roc_auc_score(y_cv,pred_cv))

    

     # predict the response on the traininig

    pred_tr = knn.predict_proba(final_X_tr)

    pred_tr=(pred_tr)[:,1]

    roc_tr.append(roc_auc_score(y_tr,pred_tr))

    k_value.append(i)

    

    #finding best k using max value of auc score

    if roc_auc_score(y_cv,pred_cv)>max_auc_score:

        k_best=i

        max_auc_score=roc_auc_score(y_cv,pred_cv)

print(k_best) 

print(max_auc_score)

k6=k_best

auc6=max_auc_score
# plotting curve between K vs Train and Cross validation Data

plt.plot(k_value,roc_cv ,label="AUC cv")

plt.plot(k_value,roc_tr,label="AUC train")

plt.legend()

plt.title('AUC Score vs K')

plt.xlabel('K')

plt.ylabel('AUC')

plt.show()
# Training the model using best K    

knn = KNeighborsClassifier(algorithm='kd_tree',metric='minkowski',n_neighbors=k_best)

knn.fit(final_X_tr, y_tr)

#predicting probability on Test data

pred_test = knn.predict_proba(final_X_test)

pred_test=(pred_test)[:,1]

#predicting probablity of Training data

pred_tr = knn.predict_proba(final_X_tr)

pred_tr=(pred_tr)[:,1]
#Plotting Roc Curve



#fiding fpr and tpr on Traing and Test Data

fpr, tpr, threshold = metrics.roc_curve(y_test, pred_test)

fpr1, tpr1, threshold1 = metrics.roc_curve(y_tr, pred_tr)



#plotting

plt.plot(fpr,tpr ,label="characterstics on Test data")

plt.plot(fpr1,tpr1 ,label="characterstics on Train data")

plt.legend()

plt.title('ROC on best K')

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.show()
#finding Confusion_matrix

predic=knn.predict(final_X_test)

conf_mat = confusion_matrix(y_test, predic)

print(conf_mat)
#plotting Confusion Matrix

class_label = ["negative", "positive"]

df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)

sns.heatmap(df, annot = True,fmt="d")

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.show()
# Please write all the code with proper documentation

X_1, X_test, y_1, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

X_tr, X_cv, y_tr, y_cv = train_test_split(X_1, y_1, test_size=0.3,random_state=1)

# Please write all the code with proper documentation

# average Word2Vec

# compute average word2vec for each review.

list_of_sentance_tr=[]

for sentance in X_tr:

    list_of_sentance_tr.append(sentance.split())

final_X_tr = []; # the avg-w2v for each sentence/review is stored in this list

for sent in tqdm(list_of_sentance_tr): # for each review/sentence

    sent_vec = np.zeros(50) # as word vectors are of zero length 50, you might need to change this to 300 if you use google's w2v

    cnt_words =0; # num of words with a valid vector in the sentence/review

    for word in sent: # for each word in a review/sentence

        if word in w2v_words:

            vec = w2v_model.wv[word]

            sent_vec += vec

            cnt_words += 1

    if cnt_words != 0:

        sent_vec /= cnt_words

    final_X_tr.append(sent_vec)

    

    

list_of_sentance_cv=[]

for sentance in X_cv:

    list_of_sentance_cv.append(sentance.split())    

final_X_cv = []; # the avg-w2v for each sentence/review is stored in this list

for sent in tqdm(list_of_sentance_cv): # for each review/sentence

    sent_vec = np.zeros(50) # as word vectors are of zero length 50, you might need to change this to 300 if you use google's w2v

    cnt_words =0; # num of words with a valid vector in the sentence/review

    for word in sent: # for each word in a review/sentence

        if word in w2v_words:

            vec = w2v_model.wv[word]

            sent_vec += vec

            cnt_words += 1

    if cnt_words != 0:

        sent_vec /= cnt_words

    final_X_cv.append(sent_vec)    

    

    

list_of_sentance_test=[]

for sentance in X_test:

    list_of_sentance_test.append(sentance.split())    

final_X_test = []; # the avg-w2v for each sentence/review is stored in this list

for sent in tqdm(list_of_sentance_test): # for each review/sentence

    sent_vec = np.zeros(50) # as word vectors are of zero length 50, you might need to change this to 300 if you use google's w2v

    cnt_words =0; # num of words with a valid vector in the sentence/review

    for word in sent: # for each word in a review/sentence

        if word in w2v_words:

            vec = w2v_model.wv[word]

            sent_vec += vec

            cnt_words += 1

    if cnt_words != 0:

        sent_vec /= cnt_words

    final_X_test.append(sent_vec)    

#Calculating for finding Best K

roc_tr=[]

roc_cv=[]

k_value=[]

max_auc_score=0

K_best=0

for i in tqdm(range(1,100,20)):

    # instantiate learning model (k = 100)

    knn = KNeighborsClassifier(algorithm='kd_tree',metric='minkowski',n_neighbors=i)



    # fitting the model on train data

    knn.fit(final_X_tr, y_tr)



    # predict the response on the crossvalidation 

    pred_cv = knn.predict_proba(final_X_cv)

    pred_cv=(pred_cv)[:,1]

    roc_cv.append(roc_auc_score(y_cv,pred_cv))

    

     # predict the response on the traininig

    pred_tr = knn.predict_proba(final_X_tr)

    pred_tr=(pred_tr)[:,1]

    roc_tr.append(roc_auc_score(y_tr,pred_tr))

    k_value.append(i)

    

    #finding best k using max value of auc score

    if roc_auc_score(y_cv,pred_cv)>max_auc_score:

        k_best=i

        max_auc_score=roc_auc_score(y_cv,pred_cv)

print("best K is",k_best)

print("max AUC Score is",max_auc_score)

k7=k_best

auc7=max_auc_score
# plotting curve between K vs Train and Cross validation Data

plt.plot(k_value,roc_cv ,label="AUC cv")

plt.plot(k_value,roc_tr,label="AUC train")

plt.legend()

plt.title('AUC Score vs K')

plt.xlabel('K')

plt.ylabel('AUC')

plt.show()
# Training the model using best K    

knn = KNeighborsClassifier(algorithm='kd_tree',metric='minkowski',n_neighbors=k_best)

knn.fit(final_X_tr, y_tr)

#predicting probability on Test data

pred_test = knn.predict_proba(final_X_test)

pred_test=(pred_test)[:,1]

#predicting probablity of Training data

pred_tr = knn.predict_proba(final_X_tr)

pred_tr=(pred_tr)[:,1]
#Plotting Roc Curve



#fiding fpr and tpr on Train and Test Data

fpr, tpr, threshold = metrics.roc_curve(y_test, pred_test)

fpr1, tpr1, threshold1 = metrics.roc_curve(y_tr, pred_tr)



#plotting

plt.plot(fpr,tpr ,label="characterstics on Test data")

plt.plot(fpr1,tpr1 ,label="characterstics on Train data")

plt.legend()

plt.title('ROC on best K')

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.show()
#finding Confusion_matrix

predic=knn.predict(final_X_test)

conf_mat = confusion_matrix(y_test, predic)

print(conf_mat)
#plotting Confusion Matrix

class_label = ["negative", "positive"]

df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)

sns.heatmap(df, annot = True,fmt="d")

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.show()
# Please write all the code with proper documentation

# Please write all the code with proper documentation

# TF-IDF weighted Word2Vec

tfidf_feat = model.get_feature_names() # tfidf words/col-names

# final_tf_idf is the sparse matrix with row= sentence, col=word and cell_val = tfidf



list_of_sentance_tr=[]

for sentance in X_tr:

    list_of_sentance_tr.append(sentance.split())

final_X_tr = []; # the tfidf-w2v for each sentence/review is stored in this list

row=0;

for sent in tqdm(list_of_sentance_tr): # for each review/sentence 

    sent_vec = np.zeros(50) # as word vectors are of zero length

    weight_sum =0; # num of words with a valid vector in the sentence/review

    for word in sent: # for each word in a review/sentence

        if word in w2v_words and word in tfidf_feat:

            vec = w2v_model.wv[word]

#             tf_idf = tf_idf_matrix[row, tfidf_feat.index(word)]

            # to reduce the computation we are 

            # dictionary[word] = idf value of word in whole courpus

            # sent.count(word) = tf valeus of word in this review

            tf_idf = dictionary[word]*(sent.count(word)/len(sent))

            sent_vec += (vec * tf_idf)

            weight_sum += tf_idf

    if weight_sum != 0:

        sent_vec /= weight_sum

    final_X_tr.append(sent_vec)

    row += 1



    

list_of_sentance_cv=[]

for sentance in X_cv:

    list_of_sentance_cv.append(sentance.split())

final_X_cv = []; # the tfidf-w2v for each sentence/review is stored in this list

row=0;

for sent in tqdm(list_of_sentance_cv): # for each review/sentence 

    sent_vec = np.zeros(50) # as word vectors are of zero length

    weight_sum =0; # num of words with a valid vector in the sentence/review

    for word in sent: # for each word in a review/sentence

        if word in w2v_words and word in tfidf_feat:

            vec = w2v_model.wv[word]

#             tf_idf = tf_idf_matrix[row, tfidf_feat.index(word)]

            # to reduce the computation we are 

            # dictionary[word] = idf value of word in whole courpus

            # sent.count(word) = tf valeus of word in this review

            tf_idf = dictionary[word]*(sent.count(word)/len(sent))

            sent_vec += (vec * tf_idf)

            weight_sum += tf_idf

    if weight_sum != 0:

        sent_vec /= weight_sum

    final_X_cv.append(sent_vec)

    row += 1 

    

    

list_of_sentance_test=[]

for sentance in X_test:

    list_of_sentance_test.append(sentance.split())

final_X_test = []; # the tfidf-w2v for each sentence/review is stored in this list

row=0;

for sent in tqdm(list_of_sentance_test): # for each review/sentence 

    sent_vec = np.zeros(50) # as word vectors are of zero length

    weight_sum =0; # num of words with a valid vector in the sentence/review

    for word in sent: # for each word in a review/sentence

        if word in w2v_words and word in tfidf_feat:

            vec = w2v_model.wv[word]

#             tf_idf = tf_idf_matrix[row, tfidf_feat.index(word)]

            # to reduce the computation we are 

            # dictionary[word] = idf value of word in whole courpus

            # sent.count(word) = tf valeus of word in this review

            tf_idf = dictionary[word]*(sent.count(word)/len(sent))

            sent_vec += (vec * tf_idf)

            weight_sum += tf_idf

    if weight_sum != 0:

        sent_vec /= weight_sum

    final_X_test.append(sent_vec)

    row += 1    
#Calculating for finding Best K

roc_tr=[]

roc_cv=[]

k_value=[]

max_auc_score=0

K_best=0

for i in tqdm(range(1,100,20)):

    # instantiate learning model (k = 100)

    knn = KNeighborsClassifier(algorithm='kd_tree',metric='minkowski',n_neighbors=i)



    # fitting the model on train data

    knn.fit(final_X_tr, y_tr)



    # predict the response on the crossvalidation 

    pred_cv = knn.predict_proba(final_X_cv)

    pred_cv=(pred_cv)[:,1]

    roc_cv.append(roc_auc_score(y_cv,pred_cv))

    

     # predict the response on the traininig

    pred_tr = knn.predict_proba(final_X_tr)

    pred_tr=(pred_tr)[:,1]

    roc_tr.append(roc_auc_score(y_tr,pred_tr))

    k_value.append(i)

    

    #finding best k using max value of auc score

    if roc_auc_score(y_cv,pred_cv)>max_auc_score:

        k_best=i

        max_auc_score=roc_auc_score(y_cv,pred_cv)

print("best K is",k_best)

print("max AUC Score is",max_auc_score)

k8=k_best

auc8=max_auc_score
# plotting curve between K vs Train and Cross validation Data

plt.plot(k_value,roc_cv ,label="AUC cv")

plt.plot(k_value,roc_tr,label="AUC train")

plt.legend()

plt.title('AUC Score vs K')

plt.xlabel('K')

plt.ylabel('AUC')

plt.show()
# Training the model using best K    

knn = KNeighborsClassifier(algorithm='kd_tree',metric='minkowski',n_neighbors=k_best)

knn.fit(final_X_tr, y_tr)

#predicting probability on Test data

pred_test = knn.predict_proba(final_X_test)

pred_test=(pred_test)[:,1]

#predicting probablity of Training data

pred_tr = knn.predict_proba(final_X_tr)

pred_tr=(pred_tr)[:,1]
#Plotting Roc Curve



#fiding fpr and tpr on Train and Test Data

fpr, tpr, threshold = metrics.roc_curve(y_test, pred_test)

fpr1, tpr1, threshold1 = metrics.roc_curve(y_tr, pred_tr)



#plotting

plt.plot(fpr,tpr ,label="characterstics on Test data")

plt.plot(fpr1,tpr1 ,label="characterstics on Train data")

plt.legend()

plt.title('ROC on best K')

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.show()
#finding Confusion_matrix

predic=knn.predict(final_X_test)

conf_mat = confusion_matrix(y_test, predic)

print(conf_mat)
#plotting Confusion Matrix

class_label = ["negative", "positive"]

df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)

sns.heatmap(df, annot = True,fmt="d")

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.show()
# Please compare all your models using Prettytable library

from prettytable import PrettyTable    

x = PrettyTable()

x.field_names = ["Vectorizer", "Model", "Hyperameter", "AUC"]

x.add_row(["BOW","Brute",k1,auc1])

x.add_row(["TFIDF","Brute",k2,auc2])

x.add_row(["AwgW2V","Brute",k3,auc3])

x.add_row(["TFIDF W2V","Brute",k4,auc4])

x.add_row(["BOW","k_d tree",k5,auc5])

x.add_row(["TFIDF","k_d tree",k6,auc6])

x.add_row(["AwgW2V","k_d tree",k7,auc7])

x.add_row(["TFIDF W2V","k_d tree",k8,auc8])

print(x)




