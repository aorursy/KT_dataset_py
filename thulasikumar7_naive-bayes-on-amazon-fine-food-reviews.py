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

from sklearn.preprocessing import StandardScaler



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

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

from collections import Counter

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_validate

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import roc_auc_score 

import matplotlib.pyplot as plt
# using the SQLite Table to read data.

con = sqlite3.connect('../input/database.sqlite')

#con = sqlite3.connect('database.sqlite') 



#filtering only positive and negative reviews i.e. 

# not taking into consideration those reviews with Score=3

filtered_data = pd.read_sql_query("""SELECT * FROM Reviews WHERE Score != 3 LIMIT 100000""", con) 



# Give reviews with Score>3 a positive rating, and reviews with a score<3 a negative rating.

def partition(x):

    if x < 3:

        return 0

    return 1



#changing reviews with score less than 3 to be positive and vice-versa

actualScore = filtered_data['Score']

positiveNegative = actualScore.map(partition) 

filtered_data['Score'] = positiveNegative

print("Number of data points in our data", filtered_data.shape)

filtered_data.head(5)

display = pd.read_sql_query(""" SELECT UserId, ProductId, ProfileName, Time, Score, Text, COUNT(*) FROM Reviews GROUP BY UserId HAVING COUNT(*)>1 """, con)

print(display.shape)

display.head()

display[display['UserId']=='AZY10LLTJ71NX']
display['COUNT(*)'].sum()
display= pd.read_sql_query(""" SELECT * FROM Reviews WHERE Score != 3 AND UserId="AR5J8UI46CURR" ORDER BY ProductID """, con)

display.head()

#Sorting data according to ProductId in ascending order

sorted_data = filtered_data.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
#Deduplication of entries

final = sorted_data.drop_duplicates(subset = {"UserId","ProfileName","Time","Text"}, keep ='first', inplace=False)

final.shape
#Checking to see how much % of data still remains

(final['Id'].size*1.0)/(filtered_data['Id'].size*1.0)*100
display= pd.read_sql_query(""" SELECT * FROM Reviews WHERE Score != 3 AND Id=44737 OR Id=64422 ORDER BY ProductID """, con)

display.head()

final=final[final.HelpfulnessNumerator<=final.HelpfulnessDenominator]
#Before starting the next phase of preprocessing lets see the number of entries left

print(final.shape)



#How many positive and negative reviews are present in our dataset?

print(final['Score'].value_counts())

final['Score'].value_counts().plot(kind='bar')
final['Time']=pd.to_datetime(final['Time'],unit='s')

final=final.sort_values(by='Time')

final.head(5)
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
sent_0 = re.sub(r"http\S+", "", sent_0)

sent_1000 = re.sub(r"http\S+", "", sent_1000)

sent_150 = re.sub(r"http\S+", "", sent_1500)

sent_4900 = re.sub(r"http\S+", "", sent_4900) 

print(sent_0)
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

# instead of <br /> if we have <br/> these tags would have revmoved in the 1st ste

stop = set(stopwords.words('english')) #set of stopwords

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

    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stop)

    preprocessed_reviews.append(sentance.strip())

preprocessed_reviews[1500] 



 
print(len(preprocessed_reviews))

final.shape
final ['preprocessed_reviews']= preprocessed_reviews

final.head(5)
# store final table into an SQlLite table for future.

conn = sqlite3.connect('final.sqlite')

c=conn.cursor()

conn.text_factory = str

final.to_sql('Reviews', conn,  schema=None, if_exists='replace',index=True, index_label=None, chunksize=None, dtype=None)

conn.close()

#Loading data

conn = sqlite3.connect('final.sqlite')

data=pd.read_sql_query("""select * from Reviews""",conn)

#splitting data into Train, C.V and Test

X_train, X_test, y_train, y_test = train_test_split(final ['preprocessed_reviews'], final['Score'], test_size=0.33) 

X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.33)

print("Train:",X_train.shape,y_train.shape)

print("CV:",X_cv.shape,y_cv.shape)

print("Test:",X_test.shape,y_test.shape)



def optimal_alpha(X_train,y_train,X_cv,y_cv):

    train_auc = [] 

    cv_auc = []

    alpha_values = [10**i for i in range(-5,5)]

    for i in alpha_values:

        #neigh = KNeighborsClassifier(n_neighbors=i)

        clf=MultinomialNB(alpha=i)

        clf.fit(X_train, y_train) 

        # roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of t 

        # not the predicted outputs

        y_train_pred =  clf.predict_proba(X_train)[:,1] 

        y_cv_pred =  clf.predict_proba(X_cv)[:,1]

        train_auc.append(roc_auc_score(y_train,y_train_pred)) 

        cv_auc.append(roc_auc_score(y_cv, y_cv_pred)) 



    plt.plot(np.log(alpha_values), train_auc, label='Train AUC') 

    plt.plot(np.log(alpha_values), cv_auc, label='CV AUC')

    plt.legend()

    plt.xlabel("alpha_values: hyperparameter")

    plt.ylabel("AUC")

    plt.title("ERROR PLOTS")

    plt.show()

    print(cv_auc)

    print("Maximum Auv value: ",max(cv_auc))

    print("Index : ",cv_auc.index(max(cv_auc)))



    
def top_features(vectorizer,clf,n):

    features=vectorizer.get_feature_names()

    log_prob = clf.feature_log_prob_

    feature_prob = pd.DataFrame(log_prob, columns = features)

    feature_prob_tr = feature_prob.T

    feature_prob_tr.shape

    print("Top 10 Negative Features:-\n",feature_prob_tr[0].sort_values(ascending = False)[0:n])

    print("\n\n Top 10 Positive Features:-\n",feature_prob_tr[1].sort_values(ascending = False)[0:n])
vectorizer = CountVectorizer(ngram_range=(1,2))

vectorizer.fit(X_train)

#vectorizer.fit(X_train) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector

X_train_bow = vectorizer.transform(X_train)

X_cv_bow = vectorizer.transform(X_cv)

X_test_bow = vectorizer.transform(X_test) 

print("After vectorizations")

print(X_train_bow.shape, y_train.shape) 

print(X_cv_bow.shape, y_cv.shape)

print(X_test_bow.shape, y_test.shape) 
optimal_alpha(X_train_bow,y_train,X_cv_bow,y_cv)
from sklearn.metrics import roc_curve, auc



clf=MultinomialNB(alpha=0.01)

clf.fit(X_train_bow, y_train) 



# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the p 

# not the predicted outputs 

train_fpr,train_tpr,thresholds = roc_curve(y_train, clf.predict_proba(X_train_bow)[:,1] )

test_fpr,test_tpr,thresholds = roc_curve(y_test, clf.predict_proba(X_test_bow)[:,1]) 

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr))) 

plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))                                             

plt.legend()

plt.xlabel("alpha_values: hyperparameter") 

plt.ylabel("AUC") 

plt.title("ERROR PLOTS") 

plt.show()                                         
from sklearn.metrics import confusion_matrix 

print("Train confusion matrix") 

print(confusion_matrix(y_train, clf.predict(X_train_bow))) 

print("Test confusion matrix")

print(confusion_matrix(y_test, clf.predict(X_test_bow)))

cm_test=confusion_matrix(y_test, clf.predict(X_test_bow))

import seaborn as sns

class_label = ["negative", "positive"]

df_cm = pd.DataFrame(cm_test, index = class_label, columns = class_label)

sns.heatmap(df_cm, annot = True, fmt = "d")

plt.title("Confusiion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.show()

 
top_features(vectorizer,clf,10)
vect = TfidfVectorizer(ngram_range=(1,2))

tf_idf_vect = vect.fit(X_train)

# we use the fitted CountVectorizer to convert the text to vector

X_train_tfidf = tf_idf_vect.transform(X_train)

X_cv_tfidf = tf_idf_vect.transform(X_cv)

X_test_tfidf = tf_idf_vect.transform(X_test) 

print("After vectorizations")

print(X_train_tfidf.shape, y_train.shape) 

print(X_cv_tfidf.shape, y_cv.shape)

print(X_test_tfidf.shape, y_test.shape) 

optimal_alpha(X_train_tfidf,y_train,X_cv_tfidf,y_cv)
from sklearn.metrics import roc_curve, auc

model=MultinomialNB(alpha=0.01)

model.fit(X_train_tfidf, y_train) 



# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the p 

# not the predicted outputs 

train_fpr,train_tpr,thresholds = roc_curve(y_train, clf.predict_proba(X_train_tfidf)[:,1] )

test_fpr,test_tpr,thresholds = roc_curve(y_test, clf.predict_proba(X_test_tfidf)[:,1]) 

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr))) 

plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))                                             

plt.legend()

plt.xlabel("alpha_values: hyperparameter") 

plt.ylabel("AUC") 

plt.title("ERROR PLOTS") 

plt.show()

    
from sklearn.metrics import confusion_matrix 

print("Train confusion matrix") 

print(confusion_matrix(y_train, clf.predict(X_train_tfidf))) 

print("Test confusion matrix")

print(confusion_matrix(y_test, clf.predict(X_test_tfidf)))

cm_test=confusion_matrix(y_test, clf.predict(X_test_tfidf))

import seaborn as sns

class_label = ["negative", "positive"]

df_cm = pd.DataFrame(cm_test, index = class_label, columns = class_label)

sns.heatmap(df_cm, annot = True, fmt = "d")

plt.title("Confusiion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.show()

 
top_features(vect,model,10)
data=[["Hyper parameter(α)",0.01,0.01],["CV Auc",0.89,0.922],["Test Auc",0.844,0.921],["TNR","1979","1595"],["FPR","2630","3014"],["FNR","413","184"],["TPR","23944","24173"]]

result=pd.DataFrame(data,columns=["Result",'Bag of Words','TFIDF'])



result