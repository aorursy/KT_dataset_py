import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm

import sqlite3

import re

from bs4 import BeautifulSoup

import string

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from gensim.models import Word2Vec

from gensim.models import KeyedVectors

from sklearn.model_selection import GridSearchCV

import pickle

import gc

from sklearn.metrics import roc_auc_score, auc, roc_curve, confusion_matrix

from sklearn.linear_model import LogisticRegression

# Converting to CSR_Matrix..

from scipy.sparse import csr_matrix
db = '/kaggle/input/amazon-fine-food-reviews/database.sqlite'

connection = sqlite3.connect(db)





df_filtered = pd.read_sql_query(""" SELECT * FROM Reviews WHERE Score != 3 """,connection)





print("Number of data points in our data", df_filtered.shape)

# df_filtered = df_filtered.head(3000)
df_filtered['Score'] = df_filtered['Score'].apply(lambda x: 1 if x>3 else 0)

df_filtered['Score'].head(3)
#Sorting data according to ProductId in ascending order

df_sorted=df_filtered.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')

#Deduplication of entries

df = df_sorted.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, keep='first', inplace=False)

print(df.shape)

df.head(3)
df = df[df['HelpfulnessNumerator'] <= df['HelpfulnessDenominator']]

df.shape
#checking how much data still remains



print(f'{round((df.shape[0]/df_filtered.shape[0])*100,2)}%')
print(df['Score'].value_counts())

values = df['Score'].value_counts().values

sns.barplot(x=['Positive','Negative'],y=values)

plt.show()
# replacing some phrases like won't with will not



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



preprocessed_reviews = []

# tqdm is for printing the status bar

for sentance in tqdm(df['Text'].values):

    sentance = re.sub(r"http\S+", "", sentance)

    # removing html tags

    sentance = BeautifulSoup(sentance, 'lxml').get_text()

    sentance = decontracted(sentance)

    # removing extra spaces and numbers

    sentance = re.sub("\S*\d\S*", "", sentance).strip()

    # removing non alphabels

    sentance = re.sub('[^A-Za-z]+', ' ', sentance)

    # https://gist.github.com/sebleier/554280

    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)

    preprocessed_reviews.append(sentance.strip())
#combining required columns

df['clean_text'] = preprocessed_reviews

df = df[['Time','clean_text','Score']]

#reseting index

df = df.reset_index(drop=True)
#sampling 100k points 

df_10k = df.sample(50000)

#sorting 100kpoints based on time

df_10k['Time'] = pd.to_datetime(df_10k['Time'],unit='s')

df_10k = df_10k.sort_values('Time')

#reseting index

df_10k = df_10k.reset_index(drop=True)
df_10k['Score'].value_counts()
#splitting data to train.cv and test

from sklearn.model_selection import train_test_split

x = df_10k['clean_text']

y = df_10k['Score']

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,stratify=y)

X_tr,X_cv,y_tr,y_cv = train_test_split(X_train,y_train,test_size=0.3,stratify=y_train)
bow = CountVectorizer()

bow.fit(X_tr)

X_train_bow = bow.transform(X_tr)

X_cv_bow = bow.transform(X_cv)

X_test_bow = bow.transform(X_test)



print('shape of X_train_bow is {}'.format(X_train_bow.get_shape()))

print('shape of X_cv_bow is {}'.format(X_cv_bow.get_shape()))

print('shape of X_test_bow is {}'.format(X_test_bow.get_shape()))


C = [0.001,0.01,0.1,1,10,100]

train_auc = []

cv_auc = []



for c in C:

    model = LogisticRegression(penalty='l2',C=c,solver='liblinear')

    model.fit(X_train_bow,y_tr)

    y_tr_pred = model.predict(X_train_bow)

    y_cv_pred = model.predict(X_cv_bow)

    train_auc.append(roc_auc_score(y_tr,y_tr_pred))

    cv_auc.append(roc_auc_score(y_cv,y_cv_pred))

plt.grid(True)

plt.plot(np.log(C),train_auc,label='Train AUC')

plt.plot(np.log(C),cv_auc,label='CV AUC')

plt.scatter(np.log(C),train_auc)

plt.scatter(np.log(C),cv_auc)

plt.legend()

plt.xlabel("C: hyperparameter")

plt.ylabel("AUC")

plt.title("ERROR PLOTS")

plt.show()
model = LogisticRegression(penalty='l2',C=0.1,solver='liblinear')

model.fit(X_train_bow,y_tr)

y_tr_pred = model.predict(X_train_bow)

y_cv_pred = model.predict(X_cv_bow)

train_fpr, train_tpr, thresholds = roc_curve(y_tr, model.predict_proba(X_train_bow)[:,1])

test_fpr, test_tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test_bow)[:,1])



plt.grid(True)

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))

plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))

plt.legend()

plt.xlabel("fpr")

plt.ylabel("tpr")

plt.title("ROC CURVE FOR OPTIMAL K")

plt.show()



#Area under ROC curve

print('Area under train roc {}'.format(auc(train_fpr, train_tpr)))

print('Area under test roc {}'.format(auc(test_fpr, test_tpr)))

feats = bow.get_feature_names()

coefs = model.coef_.reshape(-1,1)

dff = pd.DataFrame(coefs,columns=['coef'],index=feats)

top_neg = dff.sort_values(ascending=True,by='coef').head(10)

top_pos = dff.sort_values(ascending=False,by='coef').head(10)

print('Top 10 Positive features')

print(top_pos)

print('-'*50)

print('Top 10 Negative features')

print(top_neg)
W = model.coef_
#noise

epsilon = 0.00005

# adding noise X_ = X + epsilon

X_ = X_train_bow.data + epsilon 

print(X_.shape)

X_train_bow_dash = csr_matrix((X_, X_train_bow.indices, X_train_bow.indptr), shape=X_train_bow.shape)

print(X_train_bow_dash.shape)
model = LogisticRegression(penalty='l2',C=0.1,solver='liblinear')

model.fit(X_train_bow_dash,y_tr)

W_ = model.coef_


epsilon2 = 0.000006

W = W + epsilon2

W_ = W_ + epsilon2
change = abs((W - W_)/(W))

percentage_change = change*100

percentage_change = percentage_change[0]





# Printing Percentiles :

for i in range(10, 101, 10):

    print("{}th Percentile value : {}".format(i, np.percentile(percentage_change, i)))

    

print('--'*50)



for i in range(90, 101):

    print("{}th Percentile value : {}".format(i, np.percentile(percentage_change, i)))



print('--'*50)



for i in range(1, 11):

    print("{}th Percentile value : {}".format((i*1.0/10 + 99), np.percentile(percentage_change, i*1.0/10 + 99)))
feats = bow.get_feature_names()

change_ = percentage_change.reshape(-1,1)

pertub_df = pd.DataFrame(change_,columns=['change'],index=feats)

pertub_df.reset_index(inplace=True)

pertub_df = pertub_df.rename(columns={'index':'features'})

print(pertub_df.shape)

# pertub_df_sorted = pertub_df.sort_values(ascending=False,by=['change'])

pertub_df.tail(3)
#removing features with high change (> 99.9th percentile value)

pertub_df = pertub_df[pertub_df['change'] < 0.5838385159697708]

print(pertub_df.shape)

pertub_df.tail(3)
import gc

idx = pertub_df.index.to_list()

# our features get reduced from 43064 to 43020. We will pick only those columns



X_train_d = X_train_bow.todense()[:,idx]

gc.collect()

X_test_d = X_test_bow.todense()[:,idx]

X_cv_d = X_cv_bow.todense()[:,idx]

print(X_train_d.shape)

print(X_cv_d.shape)

print(X_test_d.shape)
from scipy import sparse

X_train_d = sparse.csr_matrix(X_train_d)

X_cv_d = sparse.csr_matrix(X_cv_d)

X_test_d = sparse.csr_matrix(X_test_d)
#hyperparameter tuning



C = [0.001,0.01,0.1,1,10,100]

train_auc = []

cv_auc = []



for c in C:

    model = LogisticRegression(penalty='l2',C=c,solver='liblinear')

    model.fit(X_train_d,y_tr)

    y_tr_pred = model.predict(X_train_d)

    y_cv_pred = model.predict(X_cv_d)

    train_auc.append(roc_auc_score(y_tr,y_tr_pred))

    cv_auc.append(roc_auc_score(y_cv,y_cv_pred))

plt.grid(True)

plt.plot(np.log(C),train_auc,label='Train AUC')

plt.plot(np.log(C),cv_auc,label='CV AUC')

plt.scatter(np.log(C),train_auc)

plt.scatter(np.log(C),cv_auc)

plt.legend()

plt.xlabel("C: hyperparameter")

plt.ylabel("AUC")

plt.title("ERROR PLOTS")

plt.show()
model = LogisticRegression(penalty='l2',C=1,solver='liblinear')

model.fit(X_train_d,y_tr)

y_tr_pred = model.predict(X_train_d)

y_cv_pred = model.predict(X_cv_d)

train_fpr, train_tpr, thresholds = roc_curve(y_tr, model.predict_proba(X_train_d)[:,1])

test_fpr, test_tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test_d)[:,1])



plt.grid(True)

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))

plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))

plt.legend()

plt.xlabel("fpr")

plt.ylabel("tpr")

plt.title("ROC CURVE FOR OPTIMAL K")

plt.show()



#Area under ROC curve

print('Area under train roc {}'.format(auc(train_fpr, train_tpr)))

print('Area under test roc {}'.format(auc(test_fpr, test_tpr)))

feats = pertub_df.features.to_list()

coefs = model.coef_.reshape(-1,1)

dff = pd.DataFrame(coefs,columns=['coef'],index=feats)

top_neg = dff.sort_values(ascending=True,by='coef').head(10)

top_pos = dff.sort_values(ascending=False,by='coef').head(10)

print('Top 10 Positive features')

print(top_pos)

print('-'*50)

print('Top 10 Negative features')

print(top_neg)