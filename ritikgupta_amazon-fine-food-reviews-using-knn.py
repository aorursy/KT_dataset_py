import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import sqlite3
import numpy as np
import pandas as pd

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

import re # Tutorial about Python regular expressions: https://pymotw.com/2/re/
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

from tqdm import tqdm
# using SQLite Table to read data
con = sqlite3.connect('../input/amazon-fine-food-reviews/database.sqlite')

# filtering only positive and negative reviews i.e. 
# not taking into consideration those reviews with Score=3
# SELECT * FROM Reviews WHERE Score != 3 LIMIT 500000, will give top 500000 data points
# you can change the number to any other number based on your computing power

# filtered_data = pd.read_sql_query(""" SELECT * FROM Reviews WHERE Score != 3 LIMIT 500000""", con) 


filtered_data = pd.read_sql_query(""" SELECT * FROM Reviews WHERE Score != 3 LIMIT 100000""", con) 

# Give reviews with Score>3 a positive rating(1), and reviews with a score<3 a negative rating(0).
def partition(x):
    if x < 3:
        return 0
    return 1

# changing reviews with score less than 3 to be positive and vice-versa
actualScore = filtered_data['Score']
positiveNegative = actualScore.map(partition)
filtered_data['Score'] = positiveNegative
print("Number of data points in our data", filtered_data.shape)
filtered_data.head(3)


# Just look in the data and analysis
display = pd.read_sql_query("""
SELECT UserId, ProductId, ProfileName, Time, Score, Text, COUNT(*)
FROM Reviews GROUP BY UserId 
HAVING COUNT(*)>1""",con)
print(display.shape)
display.head()
display['COUNT(*)'].sum()
display = pd.read_sql_query("""
SELECT * FROM Reviews
WHERE Score != 3 AND UserId = "AR5J8UI46CURR"
ORDER BY ProductID
""",con)
display.head()
#Sorting data according to ProductId in ascending order
sorted_data=filtered_data.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
# Deduplication of entries
final = sorted_data.drop_duplicates(subset = {"UserId","ProfileName","Time","Text"},keep = 'first',inplace = False)
final.shape
# Checking to see how much % of data still remains
(final['Id'].size*1.0)/(filtered_data['Id'].size*1.0)*100
display = pd.read_sql_query("""
SELECT * FROM Reviews
WHERE Score !=3 AND Id = 44737 OR Id = 64422
ORDER BY ProductID
""", con)

display.head()
final = final[final.HelpfulnessNumerator<= final.HelpfulnessDenominator]
#Before starting the next phase of preprocessing lets see the number of entries left
print(final.shape)

#How many positive and negative reviews are present in our dataset?
final['Score'].value_counts()
# printing Some random reviews
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

# Remove urls from text python: https://stackoverflow.com/a/40823105/4084039
sent_0 = re.sub(r"http\S+","",sent_0)
sent_1000 = re.sub(r"http\S+","",sent_1000)
sent_1500 = re.sub(r"http\S+","",sent_1500)
sent_4900 = re.sub(r"http\S+","",sent_4900)
print(sent_0)
# https://stackoverflow.com/questions/16206380/python-beautifulsoup-how-to-remove-all-tags-from-an-element
from bs4 import BeautifulSoup

soup = BeautifulSoup(sent_0,'lxml')
text = soup.get_text()
print(text)
print("="*50)

soup = BeautifulSoup(sent_1000,'lxml')
text = soup.get_text()
print(text)
print("="*50)

soup = BeautifulSoup(sent_1500,'lxml')
text = soup.get_text()
print(text)
print("="*50)

soup = BeautifulSoup(sent_4900,'lxml')
text = soup.get_text()
print(text)
print("="*50)
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
#remove words with numbers python: https://stackoverflow.com/a/18082370/4084039
sent_0 = re.sub("\S*\d\S*","",sent_0).strip()
print(sent_0)
#remove spacial character: https://stackoverflow.com/a/5843547/4084039
sent_1500 = re.sub('[^A-Za-z0-9]+',' ',sent_1500)
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
from tqdm import tqdm # tqdm is for printing the status bar
preprocessed_reviews = []

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
# Combining all the above stundents
from tqdm import tqdm # tqdm is for printing the status bar
preprocessed_summary = []

for sentance in tqdm(final['Summary'].values):
    sentance = re.sub(r"http\S+", "", sentance)
    sentance = BeautifulSoup(sentance, 'lxml').get_text()
    sentance = decontracted(sentance)
    sentance = re.sub("\S*\d\S*", "", sentance).strip()
    sentance = re.sub('[^A-Za-z]+', ' ', sentance)
    # https://gist.github.com/sebleier/554280
    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)
    preprocessed_summary.append(sentance.strip())
preprocessed_summary[1500]
#BoW 
count_vect = CountVectorizer()
count_vect.fit(preprocessed_reviews)
print("some features name ", count_vect.get_feature_names()[:10])
print("="*50)

final_counts = count_vect.transform(preprocessed_reviews)
print("The type of count vectorizer ",type(final_counts))
print("the shape of out text BOW vectorizer ",final_counts.get_shape())
print("the number of unique words ", final_counts.get_shape()[1])
#bi-gram, tri-gram and n-gram

#removing stop words like "not" should be avoided before building n-grams
# count_vect = CountVectorizer(ngram_range=(1,2))
# please do read the CountVectorizer documentation http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

# you can choose these numebrs min_df=10, max_features=5000, of your choice

count_vect = CountVectorizer(ngram_range = (1,2),min_df = 10, max_features = 5000)
final_bigram_counts = count_vect.fit_transform(preprocessed_reviews)
print("the type of count vectorizer ",type(final_bigram_counts))
print("the shape of out text BOW vectorizer ",final_bigram_counts.get_shape())
print("the number of unique words including both unigrams and bigrams ", final_bigram_counts.get_shape()[1])
tf_idf_vect = TfidfVectorizer(ngram_range = (1,2),min_df = 10)
tf_idf_vect.fit(preprocessed_reviews)
print("Some sample features(unique words in the corpus)",tf_idf_vect.get_feature_names()[:10])
print("="*50)

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
    print(w2v_model.wv.most_similar('fantastic'))
    print('='*50)
    print(w2v_model.wv.most_similar('worst'))
    
elif want_to_use_google_w2v and is_your_ram_gt_16g:
    if os.path.isfile('GoogleNews-vectors-negative300.bin'):
        w2v_model=KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        print(w2v_model.wv.most_similar('great'))
        print(w2v_model.wv.most_similar('worst'))
    else:
        print("you don't have gogole's word2vec file, keep want_to_train_w2v = True, to train your own w2v ")
w2v_model.wv["dog"]
w2v_words = list(w2v_model.wv.vocab)
print("Number of words that occured minimum 5 times ", len(w2v_words)) # Because we took min count = 5 in word2vec 
print("Sample words ", w2v_words[0:50])
# average Word2Vec
# Compute average word2Vec for each review.
sent_vectors = [] # the avg-w2v for each sentence/review is stored in this list
for sent in tqdm(list_of_sentance): # for each review/sentence
    sent_vec = np.zeros(50) # as word vectors are of zero length 50, you might need to change this to 300 if we use google's w2v
    cnt_words = 0 # num of word in a review/sentence
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
dictinory = dict(zip(model.get_feature_names(),list(model.idf_)))
dictinory
print(tf_idf_matrix)
# Tf-IDF weighted word2vec
tfidf_feat = model.get_feature_names() # tfidf words/col-names
# final_tf_idf is the sparse metrix with row = sentence, col = words and cell_val = tfidf
tfidf_sent_vectors = [] # the tfidf-w2v for each sentence/review is stored in this list
row = 0
for sent in tqdm(list_of_sentance): # for each review/sentence
    sent_vec = np.zeros(50) # as word vectors are of zero length
    weight_sum = 0 # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/ sentence
        if word in w2v_words and word in tfidf_feat:
            vec = w2v_model.wv[word]
            # tf_idf = tf_idf_matrix[row, tfidf_feat.index(word)]
            # to reduce the computation we are 
            # dictionary[word] = idf value of word in whole courpus
            # sent.count(word) = tf valeus of word in this review
            tf_idf = dictinory[word]*(sent.count(word)/len(sent))
            sent_vec += (vec * tf_idf)
            weight_sum += tf_idf
    if weight_sum != 0:
        sent_vec /= weight_sum
    tfidf_sent_vectors.append(sent_vec)
    row += 1
# Import libraries
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
# Importing the data
X = preprocessed_reviews
y = np.array(final['Score'])

# Spliting the data into train,CV, test 
X_1, X_test, y_1, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_tr, X_cv, y_tr, y_cv = train_test_split(X_1, y_1, test_size=0.3)
# CountVectorizer for train, CV, test data
count_vect = CountVectorizer()
final_Xtr = count_vect.fit_transform(X_tr) # We use fit_transform() for train and transform() for test and cv
final_Xcv = count_vect.transform(X_cv)
final_Xtest = count_vect.transform(X_test)

# Implementation 
auc_train = []
auc_cv = []
K = list(range(1,50,4))

for i in tqdm(K):
    knn = KNeighborsClassifier(n_neighbors = i, weights = 'uniform',algorithm = 'brute',leaf_size = 30, p = 2, metric = 'cosine')
    knn.fit(final_Xtr,y_tr)
    
    pred_tr = knn.predict_proba(final_Xtr)[:,1]
    auc_train.append(roc_auc_score(y_tr,pred_tr))
    
    pred_cv = knn.predict_proba(final_Xcv)[:,1]
    auc_cv.append(roc_auc_score(y_cv,pred_cv))

# Graph between AUC V/s K
fig = plt.figure()
ax = plt.subplot(111)

ax.plot(K,auc_train, label = 'AUC Train')
ax.plot(K,auc_cv, label = 'AUC CV')

plt.title('AUC V/s K')
plt.xlabel('K')
plt.ylabel('AUC')

plt.legend()
plt.show()

    


auc_cv
# ROC Curve for K = 30 seens best hyperparameter
knn = KNeighborsClassifier(n_neighbors = 30, weights = 'uniform', algorithm = 'brute', leaf_size = 30, p = 2, metric = 'cosine')
knn.fit(final_Xtr,y_tr)

pred_tr = knn.predict_proba(final_Xtr)[:,1]
fpr_tr,tpr_tr,thresholds_tr = metrics.roc_curve(y_tr,pred_tr)

pred_test = knn.predict_proba(final_Xtest)[:,1]
fpr_test,tpr_test,thresholds_test = metrics.roc_curve(y_test,pred_test)

# Plot the Graph between TPR AND FPR
fig = plt.figure()
ax = plt.subplot()
ax.plot(fpr_tr,tpr_tr,label = 'Train ROC , auc = '+str(roc_auc_score(y_tr,pred_tr)))
ax.plot(fpr_test,tpr_test,label = 'Test ROC , auc = '+str(roc_auc_score(y_test,pred_test)))

plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')

ax.legend()
plt.show()
# Confusion Metrix
knn = KNeighborsClassifier(n_neighbors = 30, weights = 'uniform',algorithm = 'brute', leaf_size = 30,p = 2, metric = 'cosine')
knn.fit(final_Xtr,y_tr)

predict_test = knn.predict(final_Xtest)

conf_mat = confusion_matrix(y_test, predict_test)
class_label = ["Negative", "Positive"]

df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)

sns.heatmap(df,annot = True, fmt = "d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Preparing the data
X = preprocessed_reviews
y = np.array(final['Score'])

# Spliting the data into train, cv and test
X_1, X_test, y_1, y_test = train_test_split(X,y, test_size = .3,random_state = 0)
X_tr, X_cv, y_tr, y_cv = train_test_split(X_1,y_1, test_size = 0.3)

# Tfidf
tf_idf_vect = TfidfVectorizer(ngram_range = (1,2), min_df = 10)

final_Xtr = tf_idf_vect.fit_transform(X_tr)
final_Xcv = tf_idf_vect.transform(X_cv)
final_Xtest = tf_idf_vect.transform(X_test)

# KNN
auc_cv = []
auc_train = []
K = list(range(1,50,4))

for i in tqdm(K):
    knn = KNeighborsClassifier(n_neighbors = i, weights = 'uniform', algorithm = 'brute',leaf_size = 30,p = 2, metric = 'cosine')
    knn.fit(final_Xtr, y_tr)
    
    pred_tr = knn.predict_proba(final_Xtr)[:,1]
    auc_train.append(roc_auc_score(y_tr, pred_tr))
    
    pred_cv = knn.predict_proba(final_Xcv)[:,1]
    auc_cv.append(roc_auc_score(y_cv, pred_cv))

# Graph between AUC vs K
fig = plt.figure()
ax = plt.subplot(111)

ax.plot(K, auc_train, label= 'AUC train')
ax.plot(K, auc_cv, label = 'AUC CV')

plt.title('AUC v/s K')
plt.xlabel('K')
plt.ylabel('AUC')

ax.legend()
plt.show()
auc_cv
# ROC curve for K = 49 seen best hyperparameter
knn = KNeighborsClassifier(n_neighbors = 49, weights = 'uniform', algorithm = 'brute', leaf_size = 30, p = 2, metric = 'cosine')
knn.fit(final_Xtr, y_tr)

pred_tr = knn.predict_proba(final_Xtr)[:,1]
fpr_tr, tpr_tr, thresholds_tr = metrics.roc_curve(y_tr, pred_tr)

pred_test = knn.predict_proba(final_Xtest)[:,1]
fpr_test, tpr_test, thresholds_test = metrics.roc_curve(y_test, pred_test)

# Graph between TPR AND FPR
fig = plt.figure()
ax = plt.subplot(111)

ax.plot(fpr_tr, tpr_tr,label = 'Train ROC, auc = '+str(roc_auc_score(y_tr, pred_tr)))
ax.plot(fpr_test, tpr_test,label = 'Test ROC, auc = '+str(roc_auc_score(y_test, pred_test)))

plt.title("ROC")
plt.xlabel("FPR")
plt.ylabel("TPR")

ax.legend()
plt.show()
# Confusion matrix
knn = KNeighborsClassifier(n_neighbors = 49, weights = 'uniform', algorithm = 'brute', leaf_size = 30, p = 2, metric = 'cosine')
knn.fit(final_Xtr, y_tr)

predict_test = knn.predict(final_Xtest)

conf_mat = confusion_matrix(y_test, predict_test)
class_label = ["Negative", "Positive"]
df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)

sns.heatmap(df, annot = True, fmt = "d")

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True label")
plt.show()
# Preparing the data
X = preprocessed_reviews
y = np.array(final['Score'])

# Spliting the data into Train, CV and Test
X_1, X_test, y_1, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
X_tr, X_cv, y_tr, y_cv = train_test_split(X_1, y_1, test_size = 0.3)

# Word2Vec

list_of_sentance_tr = []

for sentance in X_tr:
    list_of_sentance_tr.append(sentance.split())
w2v_model = Word2Vec(list_of_sentance_tr, min_count = 5, size = 50, workers = 4)
w2v_words = list(w2v_model.wv.vocab)


def word2Vec(X):
    list_of_sentance = []
    
    for sentance in X:
        list_of_sentance.append(sentance.split())
    
    sent_vectors = []
    
    for sent in tqdm(list_of_sentance):
        sent_vec = np.zeros(50)
        cnt_words = 0
        
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
    
    return sent_vectors

final_Xtr = word2Vec(X_tr)    
final_Xcv = word2Vec(X_cv)    
final_Xtest = word2Vec(X_test)    
# KNN
auc_train = []
auc_cv = []
K = list(range(1,50,4))

for i in tqdm(K):
    knn = KNeighborsClassifier(n_neighbors = i, weights = 'uniform', algorithm = 'brute', leaf_size = 30, p = 2,metric = 'cosine')
    knn.fit(final_Xtr,y_tr)
    
    pred_tr = knn.predict_proba(final_Xtr)[:,1]
    auc_train.append(roc_auc_score(y_tr,pred_tr))
    
    pred_cv = knn.predict_proba(final_Xcv)[:,1]
    auc_cv.append(roc_auc_score(y_cv,pred_cv))
    

# Graph between AUC v/s K
fig = plt.figure()
ax = plt.subplot(111)

ax.plot(K,auc_train, label = "AUC train")
ax.plot(K,auc_cv, label = "AUC CV")

plt.title('AUC vs K')
plt.xlabel('K')
plt.ylabel('AUC')

ax.legend()
plt.show()
    
# ROC curve for K = 30 seen best hyperparameter
knn = KNeighborsClassifier(n_neighbors = 30, weights = 'uniform', algorithm = 'brute', leaf_size = 30, p = 2, metric = 'cosine')
knn.fit(final_Xtr, y_tr)

pred_tr = knn.predict_proba(final_Xtr)[:,1]
fpr_tr, tpr_tr, thresholds_tr = metrics.roc_curve(y_tr, pred_tr)

pred_test = knn.predict_proba(final_Xtest)[:,1]
fpr_test, tpr_test, thresholds_test = metrics.roc_curve(y_test, pred_test)

# Graph between TPR AND FPR
fig = plt.figure()
ax = plt.subplot(111)

ax.plot(fpr_tr, tpr_tr,label = 'Train ROC, auc = '+str(roc_auc_score(y_tr, pred_tr)))
ax.plot(fpr_test, tpr_test,label = 'Test ROC, auc = '+str(roc_auc_score(y_test, pred_test)))

plt.title("ROC")
plt.xlabel("FPR")
plt.ylabel("TPR")

ax.legend()
plt.show()
# Confusion Matrix
knn = KNeighborsClassifier(n_neighbors = 30, weights = 'uniform', algorithm = 'brute', leaf_size = 30, p = 2, metric = 'cosine')
knn.fit(final_Xtr, y_tr)

predict_test = knn.predict(final_Xtest)

conf_mat = confusion_matrix(y_test, predict_test)
class_label = ["Negative", "Positive"]
df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)

sns.heatmap(df, annot = True, fmt = "d")

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True label")
plt.show()
# Preparing the data
X = preprocessed_reviews
y = np.array(final['Score'])

# Spliting the data
X_1, X_test, y_1, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
X_tr, X_cv, y_tr,y_cv = train_test_split(X_1,y_1, test_size = 0.3)


list_of_sentance_train = []
for sentance in X_tr:
    list_of_sentance_train.append(sentance.split())

w2v_model  = Word2Vec(list_of_sentance_train, min_count = 5, size = 50, workers = 4)
w2v_words = list(w2v_model.wv.vocab)

tf_idf_vect = TfidfVectorizer(ngram_range = (1,2), min_df = 10, max_features = 500)
tf_idf_matrix = tf_idf_vect.fit_transform(X_tr)

tfidf_feat = tf_idf_vect.get_feature_names()
dictionary = dict(zip(tf_idf_vect.get_feature_names(), list(tf_idf_vect.idf_)))

# TFIDF W2V Function
def tfidf_W2V(X):
    list_of_sentance = []
    
    for sentance in X:
        list_of_sentance.append(sentance.split())
    
    tfidf_sent_vectors = []
    row = 0
    
    for sent in tqdm(list_of_sentance):
        sent_vec = np.zeros(50)
        weight_sum = 0
        
        for word in sent:
            if word in w2v_words and word in tfidf_feat:
                vec = w2v_model.wv[word]
                tf_idf = dictionary[word]*(sent.count(word)/len(sent))
                sent_vec += (vec * tf_idf)
                weight_sum  += tf_idf
        
        if weight_sum != 0:
            sent_vec /= weight_sum
        
        tfidf_sent_vectors.append(sent_vec)
        row +=1
    print(len(tfidf_sent_vectors))
    print(len(tfidf_sent_vectors[0]))
    
    return tfidf_sent_vectors

final_Xtr = tfidf_W2V(X_tr)
final_Xcv = tfidf_W2V(X_cv)
final_Xtest = tfidf_W2V(X_test)
# KNN
auc_train = []
auc_cv = []
K = list(range(1,50,4))

for i in tqdm(K):
    knn = KNeighborsClassifier(n_neighbors = i, weights = 'uniform', algorithm = 'brute', leaf_size = 30, p = 2,metric = 'cosine')
    knn.fit(final_Xtr,y_tr)
    
    pred_tr = knn.predict_proba(final_Xtr)[:,1]
    auc_train.append(roc_auc_score(y_tr,pred_tr))
    
    pred_cv = knn.predict_proba(final_Xcv)[:,1]
    auc_cv.append(roc_auc_score(y_cv,pred_cv))
    

# Graph between AUC v/s K
fig = plt.figure()
ax = plt.subplot(111)

ax.plot(K,auc_train, label = "AUC train")
ax.plot(K,auc_cv, label = "AUC CV")

plt.title('AUC vs K')
plt.xlabel('K')
plt.ylabel('AUC')

ax.legend()
plt.show()
    
# ROC curve for K = 30 seen best hyperparameter
knn = KNeighborsClassifier(n_neighbors = 30, weights = 'uniform', algorithm = 'brute', leaf_size = 30, p = 2, metric = 'cosine')
knn.fit(final_Xtr, y_tr)

pred_tr = knn.predict_proba(final_Xtr)[:,1]
fpr_tr, tpr_tr, thresholds_tr = metrics.roc_curve(y_tr, pred_tr)

pred_test = knn.predict_proba(final_Xtest)[:,1]
fpr_test, tpr_test, thresholds_test = metrics.roc_curve(y_test, pred_test)

# Graph between TPR AND FPR
fig = plt.figure()
ax = plt.subplot(111)

ax.plot(fpr_tr, tpr_tr,label = 'Train ROC, auc = '+str(roc_auc_score(y_tr, pred_tr)))
ax.plot(fpr_test, tpr_test,label = 'Test ROC, auc = '+str(roc_auc_score(y_test, pred_test)))

plt.title("ROC")
plt.xlabel("FPR")
plt.ylabel("TPR")

ax.legend()
plt.show()
# Confusion Matrix
knn = KNeighborsClassifier(n_neighbors = 30, weights = 'uniform', algorithm = 'brute', leaf_size = 30, p = 2, metric = 'cosine')
knn.fit(final_Xtr, y_tr)

predict_test = knn.predict(final_Xtest)

conf_mat = confusion_matrix(y_test, predict_test)
class_label = ["Negative", "Positive"]
df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)

sns.heatmap(df, annot = True, fmt = "d")

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True label")
plt.show()
# Preparing the data
X = preprocessed_reviews[:30000]
y = np.array(final['Score'])[:30000]

# Spliting the data
X_1, X_test, y_1, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
X_tr, X_cv, y_tr,y_cv = train_test_split(X_1,y_1, test_size = 0.3)

# Count Vectorizer for train, cv and test
count_vect  = CountVectorizer(min_df = 10, max_features = 500) # fit_transform is used for train
final_Xtr = count_vect.fit_transform(X_tr).toarray()
final_Xcv = count_vect.transform(X_cv).toarray()
final_Xtest = count_vect.transform(X_test).toarray()

# KNN
auc_cv = []
auc_train = []
K = list(range(1,50,4))
for i in tqdm(K):
    knn = KNeighborsClassifier(n_neighbors = i,weights = "uniform", algorithm = "kd_tree")
    knn.fit(final_Xtr,y_tr)
    
    pred_Xtr = knn.predict_proba(final_Xtr)[:,1]
    auc_train.append(roc_auc_score(y_tr,pred_Xtr))
    
    pred_Xcv = knn.predict_proba(final_Xcv)[:,1]
    auc_cv.append(roc_auc_score(y_cv,pred_Xcv))
    
# Graph between AUC and K

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(K,auc_train, label = "AUC train")
ax.plot(K,auc_cv, label = "AUC CV")

plt.title('AUC vs K')
plt.xlabel('K')
plt.ylabel('AUC')

ax.legend()
plt.show()
    

# ROC curve for K = 30 seen best hyperparameter
knn = KNeighborsClassifier(n_neighbors = 30, weights = 'uniform', algorithm = 'kd_tree')
knn.fit(final_Xtr, y_tr)

pred_tr = knn.predict_proba(final_Xtr)[:,1]
fpr_tr, tpr_tr, thresholds_tr = metrics.roc_curve(y_tr, pred_tr)

pred_test = knn.predict_proba(final_Xtest)[:,1]
fpr_test, tpr_test, thresholds_test = metrics.roc_curve(y_test, pred_test)

# Graph between TPR AND FPR
fig = plt.figure()
ax = plt.subplot(111)

ax.plot(fpr_tr, tpr_tr,label = 'Train ROC, auc = '+str(roc_auc_score(y_tr, pred_tr)))
ax.plot(fpr_test, tpr_test,label = 'Test ROC, auc = '+str(roc_auc_score(y_test, pred_test)))

plt.title("ROC")
plt.xlabel("FPR")
plt.ylabel("TPR")

ax.legend()
plt.show()
# Confusion Matrix
knn = KNeighborsClassifier(n_neighbors = 30, weights = 'uniform', algorithm = 'kd_tree')
knn.fit(final_Xtr, y_tr)

predict_test = knn.predict(final_Xtest)

conf_mat = confusion_matrix(y_test, predict_test)
class_label = ["Negative", "Positive"]
df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)

sns.heatmap(df, annot = True, fmt = "d")

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True label")
plt.show()
# Preparing the dat
X = preprocessed_reviews[:10000]
y = np.array(final['Score'])[:10000]

# Spliting data into train, CV and test 
X_1, X_test, y_1, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
X_tr, X_cv, y_tr,y_cv = train_test_split(X_1,y_1, test_size = 0.3)

# TFIDF
tf_idf_vect = TfidfVectorizer(ngram_range = (1,2), min_df  = 10, max_features = 500)
final_Xtr = tf_idf_vect.fit_transform(X_tr).toarray()
final_Xcv = tf_idf_vect.transform(X_cv).toarray()
final_Xtest = tf_idf_vect.transform(X_test).toarray()

# KNN
auc_train = []
auc_cv = []
K = list(range(1,50,4))

for i in tqdm(K):
    knn = KNeighborsClassifier(n_neighbors = i, weights = 'uniform', algorithm = 'kd_tree')
    knn.fit(final_Xtr,y_tr)
    
    pred_tr = knn.predict_proba(final_Xtr)[:,1]
    auc_train.append(roc_auc_score(y_tr,pred_tr))
    
    pred_cv = knn.predict_proba(final_Xcv)[:,1]
    auc_cv.append(roc_auc_score(y_cv,pred_cv))
    

# Graph between AUC v/s K
fig = plt.figure()
ax = plt.subplot(111)

ax.plot(K,auc_train, label = "AUC train")
ax.plot(K,auc_cv, label = "AUC CV")

plt.title('AUC vs K')
plt.xlabel('K')
plt.ylabel('AUC')

ax.legend()
plt.show()

# ROC curve for K = 37 seen best hyperparameter
knn = KNeighborsClassifier(n_neighbors = 37, weights = 'uniform', algorithm = 'kd_tree')
knn.fit(final_Xtr, y_tr)

pred_tr = knn.predict_proba(final_Xtr)[:,1]
fpr_tr, tpr_tr, thresholds_tr = metrics.roc_curve(y_tr, pred_tr)

pred_test = knn.predict_proba(final_Xtest)[:,1]
fpr_test, tpr_test, thresholds_test = metrics.roc_curve(y_test, pred_test)

# Graph between TPR AND FPR
fig = plt.figure()
ax = plt.subplot(111)

ax.plot(fpr_tr, tpr_tr,label = 'Train ROC, auc = '+str(roc_auc_score(y_tr, pred_tr)))
ax.plot(fpr_test, tpr_test,label = 'Test ROC, auc = '+str(roc_auc_score(y_test, pred_test)))

plt.title("ROC")
plt.xlabel("FPR")
plt.ylabel("TPR")

ax.legend()
plt.show()
# Confusion Matrix
knn = KNeighborsClassifier(n_neighbors = 37, weights = 'uniform', algorithm = 'kd_tree')
knn.fit(final_Xtr, y_tr)

predict_test = knn.predict(final_Xtest)

conf_mat = confusion_matrix(y_test, predict_test)
class_label = ["Negative", "Positive"]
df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)

sns.heatmap(df, annot = True, fmt = "d")

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True label")
plt.show()
# Preparing the data
X = preprocessed_reviews[:30000]
y = np.array(final['Score'])[:30000]

# Spliting the data into Train, CV and Test
X_1, X_test, y_1, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
X_tr, X_cv, y_tr, y_cv = train_test_split(X_1, y_1, test_size = 0.3)

# Word2Vec

list_of_sentance_tr = []

for sentance in X_tr:
    list_of_sentance_tr.append(sentance.split())
w2v_model = Word2Vec(list_of_sentance_tr, min_count = 5, size = 50, workers = 4)
w2v_words = list(w2v_model.wv.vocab)


def word2Vec(X):
    list_of_sentance = []
    
    for sentance in X:
        list_of_sentance.append(sentance.split())
    
    sent_vectors = []
    
    for sent in tqdm(list_of_sentance):
        sent_vec = np.zeros(50)
        cnt_words = 0
        
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
    
    return sent_vectors

final_Xtr = word2Vec(X_tr)    
final_Xcv = word2Vec(X_cv)    
final_Xtest = word2Vec(X_test)    
# KNN
auc_train = []
auc_cv = []
K = list(range(1,50,4))

for i in tqdm(K):
    knn = KNeighborsClassifier(n_neighbors = i, weights = 'uniform', algorithm = 'kd_tree')
    knn.fit(final_Xtr,y_tr)
    
    pred_tr = knn.predict_proba(final_Xtr)[:,1]
    auc_train.append(roc_auc_score(y_tr,pred_tr))
    
    pred_cv = knn.predict_proba(final_Xcv)[:,1]
    auc_cv.append(roc_auc_score(y_cv,pred_cv))
    

# Graph between AUC v/s K
fig = plt.figure()
ax = plt.subplot(111)

ax.plot(K,auc_train, label = "AUC train")
ax.plot(K,auc_cv, label = "AUC CV")

plt.title('AUC vs K')
plt.xlabel('K')
plt.ylabel('AUC')

ax.legend()
plt.show()
    
# ROC curve for K = 33 seen best hyperparameter
knn = KNeighborsClassifier(n_neighbors = 33, weights = 'uniform', algorithm = 'kd_tree')
knn.fit(final_Xtr, y_tr)

pred_tr = knn.predict_proba(final_Xtr)[:,1]
fpr_tr, tpr_tr, thresholds_tr = metrics.roc_curve(y_tr, pred_tr)

pred_test = knn.predict_proba(final_Xtest)[:,1]
fpr_test, tpr_test, thresholds_test = metrics.roc_curve(y_test, pred_test)

# Graph between TPR AND FPR
fig = plt.figure()
ax = plt.subplot(111)

ax.plot(fpr_tr, tpr_tr,label = 'Train ROC, auc = '+str(roc_auc_score(y_tr, pred_tr)))
ax.plot(fpr_test, tpr_test,label = 'Test ROC, auc = '+str(roc_auc_score(y_test, pred_test)))

plt.title("ROC")
plt.xlabel("FPR")
plt.ylabel("TPR")

ax.legend()
plt.show()
# Confusion Matrix
knn = KNeighborsClassifier(n_neighbors = 33, weights = 'uniform', algorithm = 'kd_tree')
knn.fit(final_Xtr, y_tr)

predict_test = knn.predict(final_Xtest)

conf_mat = confusion_matrix(y_test, predict_test)
class_label = ["Negative", "Positive"]
df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)

sns.heatmap(df, annot = True, fmt = "d")

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True label")
plt.show()
# Preparing the data
X = preprocessed_reviews[:30000]
y = np.array(final['Score'])[:30000]

# Spliting the data
X_1, X_test, y_1, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
X_tr, X_cv, y_tr,y_cv = train_test_split(X_1,y_1, test_size = 0.3)


list_of_sentance_train = []
for sentance in X_tr:
    list_of_sentance_train.append(sentance.split())

w2v_model  = Word2Vec(list_of_sentance_train, min_count = 5, size = 50, workers = 4)
w2v_words = list(w2v_model.wv.vocab)

tf_idf_vect = TfidfVectorizer(ngram_range = (1,2), min_df = 10, max_features = 500)
tf_idf_matrix = tf_idf_vect.fit_transform(X_tr)

tfidf_feat = tf_idf_vect.get_feature_names()
dictionary = dict(zip(tf_idf_vect.get_feature_names(), list(tf_idf_vect.idf_)))

# TFIDF W2V Function
def tfidf_W2V(X):
    list_of_sentance = []
    
    for sentance in X:
        list_of_sentance.append(sentance.split())
    
    tfidf_sent_vectors = []
    row = 0
    
    for sent in tqdm(list_of_sentance):
        sent_vec = np.zeros(50)
        weight_sum = 0
        
        for word in sent:
            if word in w2v_words and word in tfidf_feat:
                vec = w2v_model.wv[word]
                tf_idf = dictionary[word]*(sent.count(word)/len(sent))
                sent_vec += (vec * tf_idf)
                weight_sum  += tf_idf
        
        if weight_sum != 0:
            sent_vec /= weight_sum
        
        tfidf_sent_vectors.append(sent_vec)
        row +=1
    print(len(tfidf_sent_vectors))
    print(len(tfidf_sent_vectors[0]))
    
    return tfidf_sent_vectors

final_Xtr = tfidf_W2V(X_tr)
final_Xcv = tfidf_W2V(X_cv)
final_Xtest = tfidf_W2V(X_test)
# KNN
auc_train = []
auc_cv = []
K = list(range(1,50,4))

for i in tqdm(K):
    knn = KNeighborsClassifier(n_neighbors = i, weights = 'uniform', algorithm = 'kd_tree')
    knn.fit(final_Xtr,y_tr)
    
    pred_tr = knn.predict_proba(final_Xtr)[:,1]
    auc_train.append(roc_auc_score(y_tr,pred_tr))
    
    pred_cv = knn.predict_proba(final_Xcv)[:,1]
    auc_cv.append(roc_auc_score(y_cv,pred_cv))
    

# Graph between AUC v/s K
fig = plt.figure()
ax = plt.subplot(111)

ax.plot(K,auc_train, label = "AUC train")
ax.plot(K,auc_cv, label = "AUC CV")

plt.title('AUC vs K')
plt.xlabel('K')
plt.ylabel('AUC')

ax.legend()
plt.show()
    
# ROC curve for K = 40 seen best hyperparameter
knn = KNeighborsClassifier(n_neighbors = 40, weights = 'uniform', algorithm = 'kd_tree')
knn.fit(final_Xtr, y_tr)

pred_tr = knn.predict_proba(final_Xtr)[:,1]
fpr_tr, tpr_tr, thresholds_tr = metrics.roc_curve(y_tr, pred_tr)

pred_test = knn.predict_proba(final_Xtest)[:,1]
fpr_test, tpr_test, thresholds_test = metrics.roc_curve(y_test, pred_test)

# Graph between TPR AND FPR
fig = plt.figure()
ax = plt.subplot(111)

ax.plot(fpr_tr, tpr_tr,label = 'Train ROC, auc = '+str(roc_auc_score(y_tr, pred_tr)))
ax.plot(fpr_test, tpr_test,label = 'Test ROC, auc = '+str(roc_auc_score(y_test, pred_test)))

plt.title("ROC")
plt.xlabel("FPR")
plt.ylabel("TPR")

ax.legend()
plt.show()
# Confusion Matrix
knn = KNeighborsClassifier(n_neighbors = 40, weights = 'uniform', algorithm = 'kd_tree')
knn.fit(final_Xtr, y_tr)

predict_test = knn.predict(final_Xtest)

conf_mat = confusion_matrix(y_test, predict_test)
class_label = ["Negative", "Positive"]
df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)

sns.heatmap(df, annot = True, fmt = "d")

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True label")
plt.show()
from prettytable import PrettyTable    
x = PrettyTable()
x.field_names = ["Vectirizer","Model", "Hyper parameter(K)","AUC" ]
x.field_names = ["Vectorizer", "Model", "Hyperameter", "AUC"]
x.add_row(["BOW","Brute",30,0.832])
x.add_row(["TFIDF","Brute",49,0.869])
x.add_row(["AwgW2V","Brute",30,0.885])
x.add_row(["TFIDF W2V","Brute",30,0.817])
x.add_row(["BOW","k_d tree",30,0.738])
x.add_row(["TFIDF","k_d tree",37,0.790])
x.add_row(["AwgW2V","k_d tree",33,0.832])
x.add_row(["TFIDF W2V","k_d tree",40,0.778])
print(x)
