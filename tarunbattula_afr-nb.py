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



filtered_data = pd.read_sql_query(""" SELECT * FROM Reviews WHERE Score != 3 """, con) 



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
final['preprocessed_reviews']=preprocessed_reviews #adding a column of CleanedText which displays the data after pre-processing of the review 

final.head(3)
## Similartly you can do preprocessing for review summary also.

from tqdm import tqdm

preprocessed_reviews_Summary = []

# tqdm is for printing the status bar

for sentance in tqdm(final['Summary'].values):

    sentance = re.sub(r"http\S+", "", sentance)

    #sentance=BeautifulSoup(sentance,'lxml').get_text()

    sentance = decontracted(sentance)

    sentance = re.sub("\S*\d\S*", "", sentance).strip()

    sentance = re.sub('[^A-Za-z]+', ' ', sentance)

    # https://gist.github.com/sebleier/554280

    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)

    preprocessed_reviews_Summary.append(sentance.strip())
final[' preprocessed_reviews_Summary']= preprocessed_reviews_Summary #adding a column of CleanedText which displays the data after pre-processing of the review 

final.head(3)
# store final table into an SQlLite table for future.

conn = sqlite3.connect('final.sqlite')

c=conn.cursor()

conn.text_factory = str

final.to_sql('Reviews', conn, schema=None, if_exists='replace', index=True, index_label=None, chunksize=None, dtype=None)
import sqlite3

con = sqlite3.connect("final.sqlite")

cleaned_data = pd.read_sql_query("select * from Reviews", con)

cleaned_data.head(3)

#cleaned_data.shape
# Sampling positive and negative reviews

positive_points = cleaned_data[cleaned_data['Score'] == 1].sample(

    n=50000, random_state=0)

negative_points = cleaned_data[cleaned_data['Score'] == 0].sample(

    n=50000, random_state=0)

total_points = pd.concat([positive_points, negative_points])



# Sorting based on time

total_points['Time'] = pd.to_datetime(

    total_points['Time'], origin='unix', unit='s')

total_points = total_points.sort_values('Time')
X = total_points['preprocessed_reviews']

Y = total_points['Score']
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

from sklearn.model_selection import train_test_split



# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, shuffle=Flase)# this is for time series split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30) # this is random splitting

X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.30) # this is random splitting





print(X_train.shape, y_train.shape)

print(X_cv.shape, y_cv.shape)

print(X_test.shape, y_test.shape)



print("="*100)



from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

vectorizer.fit(X_train) # fit has to happen only on train data



# we use the fitted CountVectorizer to convert the text to vector

X_train_bow = vectorizer.transform(X_train)

X_cv_bow = vectorizer.transform(X_cv)

X_test_bow = vectorizer.transform(X_test)



print("After vectorizations")

print(X_train_bow.shape, y_train.shape)

print(X_cv_bow.shape, y_cv.shape)

print(X_test_bow.shape, y_test.shape)

print("="*100)
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt



train_auc = []

cv_auc = []

K = np.arange(0.00001, 1, .001)

for i in K:

    neigh = MultinomialNB(alpha= i)

    neigh.fit(X_train_bow, y_train)

    # roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class

    # not the predicted outputs

    y_train_pred =  neigh.predict_proba(X_train_bow)[:,1]

    y_cv_pred =  neigh.predict_proba(X_cv_bow)[:,1]

    

    train_auc.append(roc_auc_score(y_train,y_train_pred))

    cv_auc.append(roc_auc_score(y_cv, y_cv_pred))



plt.plot(K, train_auc, label='Train AUC')

plt.plot(K, cv_auc, label='CV AUC')

plt.legend()

plt.xlabel("K: hyperparameter")

plt.ylabel("AUC")

plt.title("ERROR PLOTS")

plt.show()
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve

from sklearn.metrics import roc_curve, auc





neigh = MultinomialNB(alpha= .1)

neigh.fit(X_train_bow, y_train)

# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class

# not the predicted outputs



train_fpr, train_tpr, thresholds = roc_curve(y_train, neigh.predict_proba(X_train_bow)[:,1])

test_fpr, test_tpr, thresholds = roc_curve(y_test, neigh.predict_proba(X_test_bow)[:,1])



plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))

plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))

plt.legend()

plt.xlabel("K: hyperparameter")

plt.ylabel("AUC")

plt.title("ERROR PLOTS")

plt.show()
# To get all the features name 



bow_features = vectorizer.get_feature_names()



# To count feature for each class while fitting the model

# Number of samples encountered for each (class, feature) during fitting



feat_count = neigh.feature_count_

feat_count.shape
# Number of samples encountered for each class during fitting



neigh.class_count_



feature_prob = pd.DataFrame(neigh.feature_log_prob_, columns = bow_features)

feature_prob_tr = feature_prob.T

feature_prob_tr.shape

# To show top 10 feature from Positive class

# Feature Importance



print("\n\n Top 10 Positive Features:-\n",feature_prob_tr[1].sort_values(ascending = False)[0:10])
# To show top 10 feature from negative class

# Feature Importance

print("Top 10 Negative Features:-\n",feature_prob_tr[0].sort_values(ascending = False)[0:10])
from sklearn.metrics import confusion_matrix

print("Train confusion matrix")

Train_CF= confusion_matrix(y_train, neigh.predict(X_train_bow))

print(Train_CF)

print("Test confusion matrix")

Test_CF=confusion_matrix(y_test, neigh.predict(X_test_bow))

print(Test_CF)
# plot confusion matrix to describe the performance of classifier.

import seaborn as sns

class_label = ["negative", "positive"]

df_cm = pd.DataFrame(Test_CF, index = class_label, columns = class_label)

sns.heatmap(df_cm, annot = True, fmt = "d")

plt.title(" Confusiion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer



tf_idf_vect = TfidfVectorizer(ngram_range=(1,2),min_df=10)

X_train_tfidf = tf_idf_vect.fit_transform(X_train)



X_CV_tfidf = tf_idf_vect.transform(X_cv)

# Convert test text data to its vectorizor

X_test_tfidf = tf_idf_vect.transform(X_test)



print("After vectorizations")

print(X_train_tfidf.shape, y_train.shape)

print(X_CV_tfidf.shape, y_cv.shape)

print(X_test_tfidf.shape, y_test.shape)

print("="*100)
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt



train_auc = []

cv_auc = []

K = np.arange(0.00001, 1, .001)

for i in K:

    neigh_tfidf = MultinomialNB(alpha= i)

    neigh_tfidf.fit(X_train_tfidf, y_train)

    # roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class

    # not the predicted outputs

    y_train_pred =  neigh_tfidf.predict_proba(X_train_tfidf)[:,1]

    y_cv_pred =  neigh_tfidf.predict_proba(X_CV_tfidf)[:,1]

    

    train_auc.append(roc_auc_score(y_train,y_train_pred))

    cv_auc.append(roc_auc_score(y_cv, y_cv_pred))



plt.plot(K, train_auc, label='Train AUC')

plt.plot(K, cv_auc, label='CV AUC')

plt.legend()

plt.xlabel("K: hyperparameter")

plt.ylabel("AUC")

plt.title("ERROR PLOTS")

plt.show()
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve

from sklearn.metrics import roc_curve, auc





neigh_tfidf = MultinomialNB(alpha= .1)

neigh_tfidf.fit(X_train_tfidf, y_train)

# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class

# not the predicted outputs



train_fpr, train_tpr, thresholds = roc_curve(y_train, neigh_tfidf.predict_proba(X_train_tfidf)[:,1])

test_fpr, test_tpr, thresholds = roc_curve(y_test, neigh_tfidf.predict_proba(X_test_tfidf)[:,1])



plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))

plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))

plt.legend()

plt.xlabel("K: hyperparameter")

plt.ylabel("AUC")

plt.title("ERROR PLOTS")

plt.show()



print("="*100)
# To get all the features name 



tfidf_features = tf_idf_vect.get_feature_names()



# To count feature for each class while fitting the model

# Number of samples encountered for each (class, feature) during fitting



feat_count = neigh_tfidf.feature_count_

feat_count.shape
neigh_tfidf.class_count_
feature_prob_tfidf = pd.DataFrame(neigh_tfidf.feature_log_prob_,columns = tfidf_features)

feature_prob_tr_tfidf = feature_prob.T

feature_prob_tr_tfidf.shape
# To show top 10 feature from Positive class

# Feature Importance



print("\n\n Top 10 Positive Features:-\n",feature_prob_tr_tfidf[1].sort_values(ascending = False)[0:10])
# To show top 10 feature from negative class

# Feature Importance

print("Top 10 Negative Features:-\n",feature_prob_tr_tfidf[0].sort_values(ascending = False)[0:10])
from sklearn.metrics import confusion_matrix

print("Train confusion matrix")

Train_CF= confusion_matrix(y_train, neigh_tfidf.predict(X_train_tfidf))

print(Train_CF)

print("Test confusion matrix")

Test_CF=confusion_matrix(y_test, neigh_tfidf.predict(X_test_tfidf))

print(Test_CF)
# plot confusion matrix to describe the performance of classifier.

import seaborn as sns

class_label = ["negative", "positive"]

df_cm = pd.DataFrame(Test_CF, index = class_label, columns = class_label)

sns.heatmap(df_cm, annot = True, fmt = "d")

plt.title(" Confusiion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.show()
from prettytable import PrettyTable

    

xp = PrettyTable()



xp.field_names = ["vectorizer", "model", "hyper parameter", "Auc"]



xp.add_row(["BOW","Naive Bayes", 0.1,91.7])

xp.add_row(["TFIDF","Naive Bayes", 0.1,95.3])

print(xp)