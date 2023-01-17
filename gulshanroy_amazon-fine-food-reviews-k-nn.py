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



filtered_data = pd.read_sql_query(""" SELECT * FROM Reviews WHERE Score != 3""", con) 



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
final['cleaned_text']=preprocessed_reviews
data_pos = final[final["Score"] == 1].sample(n = 10000)

data_neg = final[final["Score"] == 0].sample(n = 10000)

final1 = pd.concat([data_pos, data_neg])

final1.shape
Y = final1['Score'].values

X = final1['cleaned_text'].values

print(Y.shape)

print(type(Y))

print(X.shape)

print(type(X))
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt



X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=12)

X_train,X_cv,Y_train,Y_cv=train_test_split(X_train,Y_train,test_size=0.2,random_state=12)

print('='*100)

print("After splitting")

print(X_train.shape,Y_train.shape)

print(X_cv.shape,Y_cv.shape)

print(X_test.shape,Y_test.shape)
vectorizer=CountVectorizer()

vectorizer=vectorizer.fit(X_train)

X_train_bow=vectorizer.transform(X_train)

X_cv_bow=vectorizer.transform(X_cv)

X_test_bow=vectorizer.transform(X_test)

print('='*100)

print("After transform")

print(X_train_bow.shape,Y_train.shape)

print(X_cv_bow.shape,Y_cv.shape)

print(X_test_bow.shape,Y_cv.shape)
print(Y_train.shape)

print(Y_cv.shape)
train_auc = []

cv_auc = []

K = list(range(1, 150, 10))

for i in tqdm(K):

    neigh = KNeighborsClassifier(n_neighbors=i, algorithm='brute')

    neigh.fit(X_train_bow, Y_train)

    # roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class

    # not the predicted outputs

    Y_train_pred =  neigh.predict_proba(X_train_bow)[:,1]

    Y_cv_pred =  neigh.predict_proba(X_cv_bow)[:,1]

    

    train_auc.append(roc_auc_score(Y_train,Y_train_pred))

    cv_auc.append(roc_auc_score(Y_cv, Y_cv_pred))



plt.plot(K, train_auc, label='Train AUC')

plt.scatter(K, train_auc, label='Train AUC')

plt.plot(K, cv_auc, label='CV AUC')

plt.scatter(K, cv_auc, label='CV AUC')

plt.legend()

plt.xlabel("K: hyperparameter")

plt.ylabel("AUC")

plt.title("ERROR PLOTS")

plt.show()
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

neighbors = list(range(80, 100, 2))

cv_score = []

for k in tqdm(neighbors):

    knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute')

    scores = cross_val_score(knn, X_train_bow, Y_train, cv=10, scoring='accuracy')

    cv_score.append(scores.mean())
MSE = [1 - x for x in cv_score]

optimal_k1 = neighbors[MSE.index(min(MSE))]

print("_" * 101)

print("Optimal number of neighbors: ", optimal_k1)

print("_" * 101)

print("Missclassification error for each k values: ", np.round(MSE, 3))

print("_" * 101)



plt.plot(neighbors, MSE)

plt.title("Number of neighbors and error")

plt.xlabel("Number of neighbors")

plt.ylabel("Missclassification error")

plt.show()
optimal_model = KNeighborsClassifier(n_neighbors=optimal_k1,algorithm='brute')

optimal_model.fit(X_train_bow, Y_train)

prediction = optimal_model.predict(X_test_bow)
train_fpr, train_tpr, thresholds = roc_curve(Y_train, optimal_model.predict_proba(X_train_bow)[:,1])

test_fpr, test_tpr, thresholds = roc_curve(Y_test, optimal_model.predict_proba(X_test_bow)[:,1])

AUC1=str(auc(test_fpr, test_tpr))

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))

plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))

plt.legend()

plt.xlabel("K: hyperparameter")

plt.ylabel("AUC")

plt.title("ERROR PLOTS")

plt.show()
training_accuracy = optimal_model.score(X_train_bow, Y_train)

training_error = 1 - training_accuracy

test_accuracy = accuracy_score(Y_test, prediction)

test_error = 1 - test_accuracy





print("_" * 101)

print("Training Accuracy: ", training_accuracy)

print("Train Error: ", training_error)

print("Test Accuracy: ", test_accuracy)

print("Test Error: ", test_error)

print("_" * 101)
import seaborn as sb

conf_matrix = confusion_matrix(Y_test, optimal_model.predict(X_test_bow))

class_label = [0, 1]

df_conf_matrix = pd.DataFrame(

    conf_matrix, index=class_label, columns=class_label)

sb.heatmap(df_conf_matrix, annot=True, fmt='d')

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

print("_" * 101)
from sklearn.metrics import classification_report

print(classification_report(Y_test, prediction))
# Please write all the code with proper documentation

tf_idf_vect = TfidfVectorizer(ngram_range=(1,2), min_df=5)

tf_idf_vect.fit(X_train)

X_train_tfidf= tf_idf_vect.transform(X_train)

X_cv_tfidf=tf_idf_vect.transform(X_cv)

X_test_tfidf=tf_idf_vect.transform(X_test)
train_auc = []

cv_auc = []

K = list(range(1, 400, 10))

for i in tqdm(K):

    neigh = KNeighborsClassifier(n_neighbors=i, algorithm='brute')

    neigh.fit(X_train_tfidf, Y_train)

    # roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class

    # not the predicted outputs

    Y_train_pred =  neigh.predict_proba(X_train_tfidf)[:,1]

    Y_cv_pred =  neigh.predict_proba(X_cv_tfidf)[:,1]

    

    train_auc.append(roc_auc_score(Y_train,Y_train_pred))

    cv_auc.append(roc_auc_score(Y_cv, Y_cv_pred))



plt.plot(K, train_auc, label='Train AUC')

plt.scatter(K, train_auc, label='Train AUC')

plt.plot(K, cv_auc, label='CV AUC')

plt.scatter(K, cv_auc, label='CV AUC')

plt.legend()

plt.xlabel("K: hyperparameter")

plt.ylabel("AUC")

plt.title("ERROR PLOTS")

plt.show()
neighbors = list(range(151, 400,10))

cv_score = []

for k in tqdm(neighbors):

    knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute')

    scores = cross_val_score(knn, X_train_tfidf, Y_train, cv=10, scoring='accuracy')

    cv_score.append(scores.mean())
MSE = [1 - x for x in cv_score]

optimal_k2 = neighbors[MSE.index(min(MSE))]

print("_" * 101)

print("Optimal number of neighbors: ", optimal_k2)

print("_" * 101)

print("Missclassification error for each k values: ", np.round(MSE, 3))

print("_" * 101)



plt.plot(neighbors, MSE)

plt.title("Number of neighbors and error")

plt.xlabel("Number of neighbors")

plt.ylabel("Missclassification error")

plt.show()
optimal_model = KNeighborsClassifier(n_neighbors=optimal_k2, algorithm='brute')

optimal_model.fit(X_train_tfidf, Y_train)

prediction = optimal_model.predict(X_test_tfidf)
train_fpr, train_tpr, thresholds = roc_curve(Y_train, optimal_model.predict_proba(X_train_tfidf)[:,1])

test_fpr, test_tpr, thresholds = roc_curve(Y_test, optimal_model.predict_proba(X_test_tfidf)[:,1])

AUC2=str(auc(test_fpr, test_tpr))

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))

plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))

plt.legend()

plt.xlabel("K: hyperparameter")

plt.ylabel("AUC")

plt.title("ERROR PLOTS")

plt.show()
training_accuracy = optimal_model.score(X_train_tfidf, Y_train)

training_error = 1 - training_accuracy

test_accuracy = accuracy_score(Y_test, prediction)

test_error = 1 - test_accuracy





print("_" * 101)

print("Training Accuracy: ", training_accuracy)

print("Train Error: ", training_error)

print("Test Accuracy: ", test_accuracy)

print("Test Error: ", test_error)

print("_" * 101)
import seaborn as sb

conf_matrix = confusion_matrix(Y_test, optimal_model.predict(X_test_tfidf))

class_label = [0, 1]

df_conf_matrix = pd.DataFrame(

    conf_matrix, index=class_label, columns=class_label)

sb.heatmap(df_conf_matrix, annot=True, fmt='d')

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

print("_" * 101)
from sklearn.metrics import classification_report

print(classification_report(Y_test, prediction))
# Please write all the code with proper documentation

i=0

list_of_sentance_train=[]

for sentance in X_train:

    list_of_sentance_train.append(sentance.split())



# this line of code trains your w2v model on the give list of sentances

w2v_model=Word2Vec(list_of_sentance_train,min_count=5,size=50, workers=4)



w2v_words = list(w2v_model.wv.vocab)

print("number of words that occured minimum 5 times ",len(w2v_words))

print("sample words ", w2v_words[0:50])
# average Word2Vec

# compute average word2vec for each review.

sent_vectors_train = []; # the avg-w2v for each sentence/review is stored in this list

for sent in tqdm(list_of_sentance_train): # for each review/sentence

    sent_vec = np.zeros(50) # as word vectors are of zero length 50, you might need to change this to 300 if you use google's w2v

    cnt_words =0; # num of words with a valid vector in the sentence/review

    for word in sent: # for each word in a review/sentence

        if word in w2v_words:

            vec = w2v_model.wv[word]

            sent_vec += vec

            cnt_words += 1

    if cnt_words != 0:

        sent_vec /= cnt_words

    sent_vectors_train.append(sent_vec)

sent_vectors_train = np.array(sent_vectors_train)

print(sent_vectors_train.shape)

print(sent_vectors_train[0])
i=0

list_of_sentance_cv=[]

for sentance in X_cv:

    list_of_sentance_cv.append(sentance.split())

# average Word2Vec

# compute average word2vec for each review.

sent_vectors_cv = []; # the avg-w2v for each sentence/review is stored in this list

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

    sent_vectors_cv.append(sent_vec)

sent_vectors_cv = np.array(sent_vectors_cv)

print(sent_vectors_cv.shape)

print(sent_vectors_cv[0])
i=0

list_of_sentance_test=[]

for sentance in X_test:

    list_of_sentance_test.append(sentance.split())

# average Word2Vec

# compute average word2vec for each review.

sent_vectors_test = []; # the avg-w2v for each sentence/review is stored in this list

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

    sent_vectors_test.append(sent_vec)

sent_vectors_test = np.array(sent_vectors_test)

print(sent_vectors_test.shape)

print(sent_vectors_test[0])
train_auc = []

cv_auc = []

K = list(range(1,150,10))

for i in K:

    neigh = KNeighborsClassifier(n_neighbors=i, algorithm='brute')

    neigh.fit(sent_vectors_train, Y_train)

    # roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class

    # not the predicted outputs

    Y_train_pred =  neigh.predict_proba(sent_vectors_train)[:,1]

    Y_cv_pred =  neigh.predict_proba(sent_vectors_cv)[:,1]

    

    train_auc.append(roc_auc_score(Y_train,Y_train_pred))

    cv_auc.append(roc_auc_score(Y_cv, Y_cv_pred))



plt.plot(K, train_auc, label='Train AUC')

plt.scatter(K, train_auc, label='Train AUC')

plt.plot(K, cv_auc, label='CV AUC')

plt.scatter(K, cv_auc, label='CV AUC')

plt.legend()

plt.xlabel("K: hyperparameter")

plt.ylabel("AUC")

plt.title("ERROR PLOTS")

plt.show()
neighbors = list(range(31,45, 2))

cv_score = []

for k in tqdm(neighbors):

    knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute')

    scores = cross_val_score(knn, sent_vectors_train, Y_train, cv=10, scoring='f1')

    cv_score.append(scores.mean())
MSE = [1 - x for x in cv_score]

optimal_k3 = neighbors[MSE.index(min(MSE))]

print("_" * 101)

print("Optimal number of neighbors: ", optimal_k3)

print("_" * 101)

print("Missclassification error for each k values: ", np.round(MSE, 3))

print("_" * 101)



plt.plot(neighbors, MSE)

plt.title("Number of neighbors and error")

plt.xlabel("Number of neighbors")

plt.ylabel("Missclassification error")

plt.show()
optimal_model = KNeighborsClassifier(n_neighbors=optimal_k3, algorithm='brute')

optimal_model.fit(sent_vectors_train, Y_train)

prediction = optimal_model.predict(sent_vectors_test)
train_fpr, train_tpr, thresholds = roc_curve(Y_train, optimal_model.predict_proba(sent_vectors_train)[:,1])

test_fpr, test_tpr, thresholds = roc_curve(Y_test, optimal_model.predict_proba(sent_vectors_test)[:,1])

AUC3=str(auc(test_fpr, test_tpr))

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))

plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))

plt.legend()

plt.xlabel("K: hyperparameter")

plt.ylabel("AUC")

plt.title("ERROR PLOTS")

plt.show()
training_accuracy = optimal_model.score(sent_vectors_train, Y_train)

training_error = 1 - training_accuracy

test_accuracy = accuracy_score(Y_test, prediction)

test_error = 1 - test_accuracy





print("_" * 101)

print("Training Accuracy: ", training_accuracy)

print("Train Error: ", training_error)

print("Test Accuracy: ", test_accuracy)

print("Test Error: ", test_error)

print("_" * 101)
import seaborn as sb

conf_matrix = confusion_matrix(Y_test, optimal_model.predict(sent_vectors_test))

class_label = [0, 1]

df_conf_matrix = pd.DataFrame(

    conf_matrix, index=class_label, columns=class_label)

sb.heatmap(df_conf_matrix, annot=True, fmt='d')

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

print("_" * 101)
from sklearn.metrics import classification_report

print(classification_report(Y_test, prediction))
# S = ["abc def pqr", "def def def abc", "pqr pqr def"]

model = TfidfVectorizer()

tf_idf_matrix = model.fit(X_train)

# we are converting a dictionary with word as a key, and the idf as a value

dictionary = dict(zip(model.get_feature_names(), list(model.idf_)))
# TF-IDF weighted Word2Vec

i=0

list_of_sentance_train=[]

for sentance in X_train:

    list_of_sentance_train.append(sentance.split())

tfidf_feat = tf_idf_vect.get_feature_names() # tfidf words/col-names

# final_tf_idf is the sparse matrix with row= sentence, col=word and cell_val = tfidf



tfidf_sent_vectors_train = []; # the tfidf-w2v for each sentence/review is stored in this list

row=0;

for sent in tqdm(list_of_sentance_train): # for each review/sentence 

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

    tfidf_sent_vectors_train.append(sent_vec)

    row += 1
# TF-IDF weighted Word2Vec

i=0

list_of_sentance_cv=[]

for sentance in X_cv:

    list_of_sentance_cv.append(sentance.split())

tfidf_feat = tf_idf_vect.get_feature_names() # tfidf words/col-names

# final_tf_idf is the sparse matrix with row= sentence, col=word and cell_val = tfidf



tfidf_sent_vectors_cv = []; # the tfidf-w2v for each sentence/review is stored in this list

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

    tfidf_sent_vectors_cv.append(sent_vec)

    row += 1
# TF-IDF weighted Word2Vec

i=0

list_of_sentance_test=[]

for sentance in X_test:

    list_of_sentance_test.append(sentance.split())

tfidf_feat = tf_idf_vect.get_feature_names() # tfidf words/col-names

# final_tf_idf is the sparse matrix with row= sentence, col=word and cell_val = tfidf



tfidf_sent_vectors_test = []; # the tfidf-w2v for each sentence/review is stored in this list

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

    tfidf_sent_vectors_test.append(sent_vec)

    row += 1
train_auc = []

cv_auc = []

K = list(range(1,150,10))

for i in K:

    neigh = KNeighborsClassifier(n_neighbors=i, algorithm='brute')

    neigh.fit(tfidf_sent_vectors_train, Y_train)

    # roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class

    # not the predicted outputs

    Y_train_pred =  neigh.predict_proba(tfidf_sent_vectors_train)[:,1]

    Y_cv_pred =  neigh.predict_proba(tfidf_sent_vectors_cv)[:,1]

    

    train_auc.append(roc_auc_score(Y_train,Y_train_pred))

    cv_auc.append(roc_auc_score(Y_cv, Y_cv_pred))



plt.plot(K, train_auc, label='Train AUC')

plt.scatter(K, train_auc, label='Train AUC')

plt.plot(K, cv_auc, label='CV AUC')

plt.scatter(K, cv_auc, label='CV AUC')

plt.legend()

plt.xlabel("K: hyperparameter")

plt.ylabel("AUC")

plt.title("ERROR PLOTS")

plt.show()
neighbors = list(range(21, 50, 2))

cv_score = []

for k in tqdm(neighbors):

    knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute')

    scores = cross_val_score(knn, tfidf_sent_vectors_train, Y_train, cv=10, scoring='f1')

    cv_score.append(scores.mean())  
MSE = [1 - x for x in cv_score]

optimal_k4 = neighbors[MSE.index(min(MSE))]

print("_" * 101)

print("Optimal number of neighbors: ", optimal_k4)

print("_" * 101)

print("Missclassification error for each k values: ", np.round(MSE, 3))

print("_" * 101)



plt.plot(neighbors, MSE)

plt.title("Number of neighbors and error")

plt.xlabel("Number of neighbors")

plt.ylabel("Missclassification error")

plt.show()
optimal_model = KNeighborsClassifier(n_neighbors=optimal_k4, algorithm='brute')

optimal_model.fit(tfidf_sent_vectors_train, Y_train)

prediction = optimal_model.predict(tfidf_sent_vectors_test)
train_fpr, train_tpr, thresholds = roc_curve(Y_train, optimal_model.predict_proba(tfidf_sent_vectors_train)[:,1])

test_fpr, test_tpr, thresholds = roc_curve(Y_test, optimal_model.predict_proba(tfidf_sent_vectors_test)[:,1])

AUC4=str(auc(test_fpr, test_tpr))

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))

plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))

plt.legend()

plt.xlabel("K: hyperparameter")

plt.ylabel("AUC")

plt.title("ERROR PLOTS")

plt.show()
training_accuracy = optimal_model.score(tfidf_sent_vectors_train, Y_train)

training_error = 1 - training_accuracy

test_accuracy = accuracy_score(Y_test, prediction)

test_error = 1 - test_accuracy





print("_" * 101)

print("Training Accuracy: ", training_accuracy)

print("Train Error: ", training_error)

print("Test Accuracy: ", test_accuracy)

print("Test Error: ", test_error)

print("_" * 101)



import seaborn as sb
conf_matrix = confusion_matrix(Y_test, optimal_model.predict(tfidf_sent_vectors_test))

class_label = [0, 1]

df_conf_matrix = pd.DataFrame(

    conf_matrix, index=class_label, columns=class_label)

sb.heatmap(df_conf_matrix, annot=True, fmt='d')

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

print("_" * 101)
from sklearn.metrics import classification_report

print(classification_report(Y_test, prediction))
data_pos1 = final[final["Score"] == 1].sample(n = 1000)

data_neg1 = final[final["Score"] == 0].sample(n = 1000)

final2 = pd.concat([data_pos1, data_neg1])

final2.shape



A = final2['cleaned_text'].values

B = final2['Score'].values

print(A.shape)

print(type(B))

print(B.shape)

print(type(B))
A_train,A_test,B_train,B_test=train_test_split(A,B,test_size=0.2,random_state=12)

A_train,A_cv,B_train,B_cv=train_test_split(A_train,B_train,test_size=0.2,random_state=12)

print('='*100)

print("After splitting")

print(A_train.shape,B_train.shape)

print(A_cv.shape,B_cv.shape)

print(A_test.shape,B_test.shape)
vectorizer=CountVectorizer(min_df=10, max_features=500)

vectorizer=vectorizer.fit(A_train)

A_train_bow_kdtree=vectorizer.transform(A_train)

A_cv_bow_kdtree=vectorizer.transform(A_cv)

A_test_bow_kdtree=vectorizer.transform(A_test)

print('='*100)

print("After transform")

print(A_train_bow_kdtree.shape,B_train.shape)

print(A_cv_bow_kdtree.shape,B_cv.shape)

print(A_test_bow_kdtree.shape,B_cv.shape)
train_auc = []

cv_auc = []

K = list(range(1,200,10))

for i in K:

    neigh = KNeighborsClassifier(n_neighbors=i, algorithm='kd_tree')

    neigh.fit(A_train_bow_kdtree.todense(), B_train)

    # roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class

    # not the predicted outputs

    B_train_pred =  neigh.predict_proba(A_train_bow_kdtree.todense())[:,1]

    B_cv_pred =  neigh.predict_proba(A_cv_bow_kdtree.todense())[:,1]

    

    train_auc.append(roc_auc_score(B_train,B_train_pred))

    cv_auc.append(roc_auc_score(B_cv, B_cv_pred))



plt.plot(K, train_auc, label='Train AUC')

plt.scatter(K, train_auc, label='Train AUC')

plt.plot(K, cv_auc, label='CV AUC')

plt.scatter(K, cv_auc, label='CV AUC')

plt.legend()

plt.xlabel("K: hyperparameter")

plt.ylabel("AUC")

plt.title("ERROR PLOTS")

plt.show()
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

neighbors = list(range(31, 71, 2))

cv_score = []

for k in tqdm(neighbors):

    knn = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree')

    scores = cross_val_score(knn, A_train_bow_kdtree.todense(), B_train, cv=10, scoring='accuracy')

    cv_score.append(scores.mean())
MSE = [1 - x for x in cv_score]

optimal_k5 = neighbors[MSE.index(min(MSE))]

print("_" * 101)

print("Optimal number of neighbors: ", optimal_k5)

print("_" * 101)

print("Missclassification error for each k values: ", np.round(MSE, 3))

print("_" * 101)



plt.plot(neighbors, MSE)

plt.title("Number of neighbors and error")

plt.xlabel("Number of neighbors")

plt.ylabel("Missclassification error")

plt.show()
optimal_model = KNeighborsClassifier(n_neighbors=optimal_k5,algorithm='kd_tree')

optimal_model.fit(A_train_bow_kdtree.todense(), B_train)

prediction = optimal_model.predict(A_test_bow_kdtree.todense())
train_fpr, train_tpr, thresholds = roc_curve(B_train, optimal_model.predict_proba(A_train_bow_kdtree.todense())[:,1])

test_fpr, test_tpr, thresholds = roc_curve(B_test, optimal_model.predict_proba(A_test_bow_kdtree.todense())[:,1])

AUC5=str(auc(test_fpr, test_tpr))

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))

plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))

plt.legend()

plt.xlabel("K: hyperparameter")

plt.ylabel("AUC")

plt.title("ERROR PLOTS")

plt.show()
training_accuracy = optimal_model.score(A_train_bow_kdtree.todense(), B_train)

training_error = 1 - training_accuracy

test_accuracy = accuracy_score(B_test, prediction)

test_error = 1 - test_accuracy





print("_" * 101)

print("Training Accuracy: ", training_accuracy)

print("Train Error: ", training_error)

print("Test Accuracy: ", test_accuracy)

print("Test Error: ", test_error)

print("_" * 101)
import seaborn as sb

conf_matrix = confusion_matrix(B_test, optimal_model.predict(A_test_bow_kdtree.todense()))

class_label = [0, 1]

df_conf_matrix = pd.DataFrame(

    conf_matrix, index=class_label, columns=class_label)

sb.heatmap(df_conf_matrix, annot=True, fmt='d')

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

print("_" * 101)
from sklearn.metrics import classification_report

print(classification_report(B_test, prediction))
tf_idf_vect = TfidfVectorizer(ngram_range=(1,2), min_df=5, max_df=100)

tf_idf_vect.fit(A_train)

A_train_tfidf =tf_idf_vect.transform(A_train)

A_cv_tfidf=tf_idf_vect.transform(A_cv)

A_test_tfidf=tf_idf_vect.transform(A_test)

print(A_train_tfidf.shape)
train_auc = []

cv_auc = []

K = list(range(1,200,10))

for i in K:

    neigh = KNeighborsClassifier(n_neighbors=i, algorithm='kd_tree')

    neigh.fit(A_train_tfidf.todense(), B_train)

    # roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class

    # not the predicted outputs

    B_train_pred =  neigh.predict_proba(A_train_tfidf.todense())[:,1]

    B_cv_pred =  neigh.predict_proba(A_cv_tfidf.todense())[:,1]

    

    train_auc.append(roc_auc_score(B_train,B_train_pred))

    cv_auc.append(roc_auc_score(B_cv, B_cv_pred))



plt.plot(K, train_auc, label='Train AUC')

plt.scatter(K, train_auc, label='Train AUC')

plt.plot(K, cv_auc, label='CV AUC')

plt.scatter(K, cv_auc, label='CV AUC')

plt.legend()

plt.xlabel("K: hyperparameter")

plt.ylabel("AUC")

plt.title("ERROR PLOTS")

plt.show()
from sklearn.metrics import accuracy_score

neighbors = list(range(109,125,2))

cv_score = []

for k in tqdm(neighbors):

    knn = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree')

    scores = cross_val_score(knn, A_train_tfidf.todense(), B_train, cv=3, scoring='accuracy')

    cv_score.append(scores.mean())
MSE = [1 - x for x in cv_score]

optimal_k6 = neighbors[MSE.index(min(MSE))]

print("_" * 101)

print("Optimal number of neighbors: ", optimal_k6)

print("_" * 101)

print("Missclassification error for each k values: ", np.round(MSE, 3))

print("_" * 101)



plt.plot(neighbors, MSE)

plt.title("Number of neighbors and error")

plt.xlabel("Number of neighbors")

plt.ylabel("Missclassification error")

plt.show()
optimal_model = KNeighborsClassifier(n_neighbors=optimal_k6, algorithm='kd_tree')

optimal_model.fit(A_train_tfidf.todense(), B_train)

prediction = optimal_model.predict(A_test_tfidf.todense())
train_fpr, train_tpr, thresholds = roc_curve(B_train, optimal_model.predict_proba(A_train_tfidf.todense())[:,1])

test_fpr, test_tpr, thresholds = roc_curve(B_test, optimal_model.predict_proba(A_test_tfidf.todense())[:,1])

AUC6=str(auc(test_fpr, test_tpr))

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))

plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))

plt.legend()

plt.xlabel("K: hyperparameter")

plt.ylabel("AUC")

plt.title("ERROR PLOTS")

plt.show()
training_accuracy = optimal_model.score(A_train_tfidf.todense(), B_train)

training_error = 1 - training_accuracy

test_accuracy = accuracy_score(B_test, prediction)

test_error = 1 - test_accuracy





print("_" * 101)

print("Training Accuracy: ", training_accuracy)

print("Train Error: ", training_error)

print("Test Accuracy: ", test_accuracy)

print("Test Error: ", test_error)

print("_" * 101)
import seaborn as sb

conf_matrix = confusion_matrix(B_test, optimal_model.predict(A_test_tfidf.todense()))

class_label = [0, 1]

df_conf_matrix = pd.DataFrame(

    conf_matrix, index=class_label, columns=class_label)

sb.heatmap(df_conf_matrix, annot=True, fmt='d')

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

print("_" * 101)
from sklearn.metrics import classification_report

print(classification_report(B_test, prediction))
i=0

list_of_sentance_train=[]

for sentance in X_train:

    list_of_sentance_train.append(sentance.split())



# this line of code trains your w2v model on the give list of sentances

w2v_model=Word2Vec(list_of_sentance_train,min_count=5,size=50, workers=4)



w2v_words = list(w2v_model.wv.vocab)

print("number of words that occured minimum 5 times ",len(w2v_words))

print("sample words ", w2v_words[0:50])
# average Word2Vec

# compute average word2vec for each review.

sent_vectors_train = []; # the avg-w2v for each sentence/review is stored in this list

for sent in tqdm(list_of_sentance_train): # for each review/sentence

    sent_vec = np.zeros(50) # as word vectors are of zero length 50, you might need to change this to 300 if you use google's w2v

    cnt_words =0; # num of words with a valid vector in the sentence/review

    for word in sent: # for each word in a review/sentence

        if word in w2v_words:

            vec = w2v_model.wv[word]

            sent_vec += vec

            cnt_words += 1

    if cnt_words != 0:

        sent_vec /= cnt_words

    sent_vectors_train.append(sent_vec)

sent_vectors_train = np.array(sent_vectors_train)

print(sent_vectors_train.shape)

print(sent_vectors_train[0])
i=0

list_of_sentance_cv=[]

for sentance in X_cv:

    list_of_sentance_cv.append(sentance.split())

# average Word2Vec

# compute average word2vec for each review.

sent_vectors_cv = []; # the avg-w2v for each sentence/review is stored in this list

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

    sent_vectors_cv.append(sent_vec)

sent_vectors_cv = np.array(sent_vectors_cv)

print(sent_vectors_cv.shape)

print(sent_vectors_cv[0])
i=0

list_of_sentance_test=[]

for sentance in X_test:

    list_of_sentance_test.append(sentance.split())

# average Word2Vec

# compute average word2vec for each review.

sent_vectors_test = []; # the avg-w2v for each sentence/review is stored in this list

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

    sent_vectors_test.append(sent_vec)

sent_vectors_test = np.array(sent_vectors_test)

print(sent_vectors_test.shape)

print(sent_vectors_test[0])
train_auc = []

cv_auc = []

K = list(range(1,200,10))

for i in tqdm(K):

    neigh = KNeighborsClassifier(n_neighbors=i, algorithm='kd_tree')

    neigh.fit(sent_vectors_train, Y_train)

    # roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class

    # not the predicted outputs

    Y_train_pred =  neigh.predict_proba(sent_vectors_train)[:,1]

    Y_cv_pred =  neigh.predict_proba(sent_vectors_cv)[:,1]

    

    train_auc.append(roc_auc_score(Y_train,Y_train_pred))

    cv_auc.append(roc_auc_score(Y_cv, Y_cv_pred))



plt.plot(K, train_auc, label='Train AUC')

plt.scatter(K, train_auc, label='Train AUC')

plt.plot(K, cv_auc, label='CV AUC')

plt.scatter(K, cv_auc, label='CV AUC')

plt.legend()

plt.xlabel("K: hyperparameter")

plt.ylabel("AUC")

plt.title("ERROR PLOTS")

plt.show()
neighbors = list(range(9,35,2))

cv_score = []

for k in tqdm(neighbors):

    knn = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree')

    scores = cross_val_score(knn, sent_vectors_train, Y_train, cv=10, scoring='f1')

    cv_score.append(scores.mean())
MSE = [1 - x for x in cv_score]

optimal_k7 = neighbors[MSE.index(min(MSE))]

print("_" * 101)

print("Optimal number of neighbors: ", optimal_k7)

print("_" * 101)

print("Missclassification error for each k values: ", np.round(MSE, 3))

print("_" * 101)



plt.plot(neighbors, MSE)

plt.title("Number of neighbors and error")

plt.xlabel("Number of neighbors")

plt.ylabel("Missclassification error")

plt.show()
optimal_model = KNeighborsClassifier(n_neighbors=optimal_k7, algorithm='kd_tree')

optimal_model.fit(sent_vectors_train, Y_train)

prediction = optimal_model.predict(sent_vectors_test)
train_fpr, train_tpr, thresholds = roc_curve(Y_train, neigh.predict_proba(sent_vectors_train)[:,1])

test_fpr, test_tpr, thresholds = roc_curve(Y_test, neigh.predict_proba(sent_vectors_test)[:,1])

AUC7=str(auc(test_fpr, test_tpr))

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))

plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))

plt.legend()

plt.xlabel("K: hyperparameter")

plt.ylabel("AUC")

plt.title("ERROR PLOTS")

plt.show()
training_accuracy = optimal_model.score(sent_vectors_train, Y_train)

training_error = 1 - training_accuracy

test_accuracy = accuracy_score(Y_test, prediction)

test_error = 1 - test_accuracy





print("_" * 101)

print("Training Accuracy: ", training_accuracy)

print("Train Error: ", training_error)

print("Test Accuracy: ", test_accuracy)

print("Test Error: ", test_error)

print("_" * 101)
import seaborn as sb

conf_matrix = confusion_matrix(Y_test, optimal_model.predict(sent_vectors_test))

class_label = [0, 1]

df_conf_matrix = pd.DataFrame(

    conf_matrix, index=class_label, columns=class_label)

sb.heatmap(df_conf_matrix, annot=True, fmt='d')

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

print("_" * 101)
from sklearn.metrics import classification_report

print(classification_report(Y_test, prediction))
# S = ["abc def pqr", "def def def abc", "pqr pqr def"]

model = TfidfVectorizer()

tf_idf_matrix = model.fit(X_train)

# we are converting a dictionary with word as a key, and the idf as a value

dictionary = dict(zip(model.get_feature_names(), list(model.idf_)))
# TF-IDF weighted Word2Vec

i=0

list_of_sentance_train=[]

for sentance in X_train:

    list_of_sentance_train.append(sentance.split())

tfidf_feat = tf_idf_vect.get_feature_names() # tfidf words/col-names

# final_tf_idf is the sparse matrix with row= sentence, col=word and cell_val = tfidf



tfidf_sent_vectors_train = []; # the tfidf-w2v for each sentence/review is stored in this list

row=0;

for sent in tqdm(list_of_sentance_train): # for each review/sentence 

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

    tfidf_sent_vectors_train.append(sent_vec)

    row += 1
#TF-IDF weighted Word2Vec

i=0

list_of_sentance_cv=[]

for sentance in X_cv:

    list_of_sentance_cv.append(sentance.split())

tfidf_feat = tf_idf_vect.get_feature_names() # tfidf words/col-names

# final_tf_idf is the sparse matrix with row= sentence, col=word and cell_val = tfidf



tfidf_sent_vectors_cv = []; # the tfidf-w2v for each sentence/review is stored in this list

row=0;

for sent in tqdm(list_of_sentance_cv): # for each review/sentence 

    sent_vec = np.zeros(50) # as word vectors are of zero length

    weight_sum =0; # num of words with a valid vector in the sentence/review

    for word in sent: # for each word in a review/sentence

        if word in w2v_words and word in tfidf_feat:

            vec = w2v_model.wv[word]

            # tf_idf = tf_idf_matrix[row, tfidf_feat.index(word)]

            # to reduce the computation we are 

            # dictionary[word] = idf value of word in whole courpus

            # sent.count(word) = tf valeus of word in this review

            tf_idf = dictionary[word]*(sent.count(word)/len(sent))

            sent_vec += (vec * tf_idf)

            weight_sum += tf_idf

    if weight_sum != 0:

        sent_vec /= weight_sum

    tfidf_sent_vectors_cv.append(sent_vec)

    row += 1
# TF-IDF weighted Word2Vec

i=0

list_of_sentance_test=[]

for sentance in X_test:

    list_of_sentance_test.append(sentance.split())

tfidf_feat = tf_idf_vect.get_feature_names() # tfidf words/col-names

# final_tf_idf is the sparse matrix with row= sentence, col=word and cell_val = tfidf



tfidf_sent_vectors_test = []; # the tfidf-w2v for each sentence/review is stored in this list

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

    tfidf_sent_vectors_test.append(sent_vec)

    row += 1
train_auc = []

cv_auc = []

K = list(range(1,200,10))

for i in tqdm(K):

    neigh = KNeighborsClassifier(n_neighbors=i, algorithm='kd_tree')

    neigh.fit(sent_vectors_train, Y_train)

    # roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class

    # not the predicted outputs

    Y_train_pred =  neigh.predict_proba(sent_vectors_train)[:,1]

    Y_cv_pred =  neigh.predict_proba(sent_vectors_cv)[:,1]

    

    train_auc.append(roc_auc_score(Y_train,Y_train_pred))

    cv_auc.append(roc_auc_score(Y_cv, Y_cv_pred))



plt.plot(K, train_auc, label='Train AUC')

plt.scatter(K, train_auc, label='Train AUC')

plt.plot(K, cv_auc, label='CV AUC')

plt.scatter(K, cv_auc, label='CV AUC')

plt.legend()

plt.xlabel("K: hyperparameter")

plt.ylabel("AUC")

plt.title("ERROR PLOTS")

plt.show()
from sklearn.model_selection import cross_val_score

neighbors = list(range(13, 50, 2))

cv_score = []

for k in tqdm(neighbors):

    knn = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree')

    scores = cross_val_score(knn, tfidf_sent_vectors_train, Y_train, cv=10, scoring='accuracy')

    cv_score.append(scores.mean())
MSE = [1 - x for x in cv_score]

optimal_k8 = neighbors[MSE.index(min(MSE))]

print("_" * 101)

print("Optimal number of neighbors: ", optimal_k8)

print("_" * 101)

print("Missclassification error for each k values: ", np.round(MSE, 3))

print("_" * 101)



plt.plot(neighbors, MSE)

plt.title("Number of neighbors and error")

plt.xlabel("Number of neighbors")

plt.ylabel("Missclassification error")

plt.show()
optimal_model = KNeighborsClassifier(n_neighbors=optimal_k8, algorithm='kd_tree')

optimal_model.fit(tfidf_sent_vectors_train, Y_train)

prediction = optimal_model.predict(tfidf_sent_vectors_test)
train_fpr, train_tpr, thresholds = roc_curve(Y_train, neigh.predict_proba(tfidf_sent_vectors_train)[:,1])

test_fpr, test_tpr, thresholds = roc_curve(Y_test, neigh.predict_proba(tfidf_sent_vectors_test)[:,1])

AUC8=str(auc(test_fpr, test_tpr))

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))

plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))

plt.legend()

plt.xlabel("K: hyperparameter")

plt.ylabel("AUC")

plt.title("ERROR PLOTS")

plt.show()
training_accuracy = optimal_model.score(tfidf_sent_vectors_train, Y_train)

training_error = 1 - training_accuracy

test_accuracy = accuracy_score(Y_test, prediction)

test_error = 1 - test_accuracy





print("_" * 101)

print("Training Accuracy: ", training_accuracy)

print("Train Error: ", training_error)

print("Test Accuracy: ", test_accuracy)

print("Test Error: ", test_error)

print("_" * 101)
import seaborn as sb

conf_matrix = confusion_matrix(Y_test, optimal_model.predict(tfidf_sent_vectors_test))

class_label = [0, 1]

df_conf_matrix = pd.DataFrame(

    conf_matrix, index=class_label, columns=class_label)

sb.heatmap(df_conf_matrix, annot=True, fmt='d')

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

print("_" * 101)
from sklearn.metrics import classification_report

print(classification_report(Y_test, prediction))
from prettytable import PrettyTable

comparison = PrettyTable()

comparison.field_names = ["Vectorizer", "Model", "Hyperparameter", "AUC"]

comparison.add_row(["BOW", 'brute', optimal_k1, np.round(float(AUC1),3)])

comparison.add_row(["TFIDF", 'brute', optimal_k2, np.round(float(AUC2),3)])

comparison.add_row(["AVG W2V", 'brute', optimal_k3, np.round(float(AUC3),3)])

comparison.add_row(["Weighted W2V", 'brute', optimal_k4,np.round(float(AUC4),3)])

comparison.add_row(["BOW", 'kd_tree', optimal_k5, np.round(float(AUC5),3)])

comparison.add_row(["TFIDF", 'kd_tree', optimal_k6, np.round(float(AUC6),3)])

comparison.add_row(["AVG W2V", 'kd_tree', optimal_k7, np.round(float(AUC7),3)])

comparison.add_row(["Weighted W2V", 'kd_tree', optimal_k8, np.round(float(AUC8),3)])

print(comparison)