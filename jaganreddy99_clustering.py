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

print(os.listdir("../input"))
# using SQLite Table to read data.

con = sqlite3.connect(r'../input/amazon-fine-food-reviews/database.sqlite') 



# filtering only positive and negative reviews i.e. 

# not taking into consideration those reviews with Score=3

# SELECT * FROM Reviews WHERE Score != 3 LIMIT 500000, will give top 500000 data points

# you can change the number to any other number based on your computing power



# filtered_data = pd.read_sql_query(""" SELECT * FROM Reviews WHERE Score != 3 LIMIT 500000""", con) 

# for tsne assignment you can take 5k data points



filtered_data = pd.read_sql_query(""" SELECT * FROM Reviews WHERE Score != 3 LIMIT 200000""", con) 



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
from sklearn.model_selection import train_test_split

x,x_test,y,y_test = train_test_split(preprocessed_reviews,final['Score'],train_size=0.8)

x_train,x_cv,y_train,y_cv = train_test_split(x,y,train_size=0.8)
bag_words = CountVectorizer()

x_train_bag= bag_words.fit_transform(x_train)

x_test_bag= bag_words.transform(x_test)

x_cv_bag= bag_words.transform(x_cv)



print('After vectorizing shape of x Train',x_train_bag.shape)

print('After vectorizing shape of x Test',x_test_bag.shape)

print('After vectorizing shape of x CV',x_cv_bag.shape)
#bi-gram, tri-gram and n-gram



#removing stop words like "not" should be avoided before building n-grams

# count_vect = CountVectorizer(ngram_range=(1,2))

# please do read the CountVectorizer documentation http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html



# you can choose these numebrs min_df=10, max_features=5000, of your choice

count_vect = CountVectorizer(ngram_range=(1,2), min_df=10, max_features=5000)



x_train_bigr= count_vect.fit_transform(x_train)

x_test_bigr= count_vect.transform(x_test)

x_cv_bigr= count_vect.transform(x_cv)



print('After vectorizing shape of x Train',x_train_bigr.shape)

print('After vectorizing shape of x Test',x_test_bigr.shape)

print('After vectorizing shape of x CV',x_cv_bigr.shape)
tfidf_words = TfidfVectorizer(ngram_range=(1,2), min_df=10)



x_train_tfidf= tfidf_words.fit_transform(x_train)

x_test_tfidf= tfidf_words.transform(x_test)

x_cv_tfidf= tfidf_words.transform(x_cv)

print('After vectorizing shape of x Train',x_train_tfidf.shape)

print('After vectorizing shape of x Test',x_test_tfidf.shape)

print('After vectorizing shape of x CV',x_cv_tfidf.shape)
# Train your own Word2Vec model using your own text corpus

i=0

list_of_sentance=[]

for sentance in x_train:

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

 # for each review/sentence

x_train_avgw2v= []# the avg-w2v for each sentence/review is stored in this list

for sent in tqdm(list_of_sentance): # for each review/sentence

    sent_vec = np.zeros(50) # as word vectors are of zero length

    cnt_words =0; # num of words with a valid vector in the sentence/review

    for word in sent: # for each word in a review/sentence

        try:

            vec = w2v_model.wv[word]

            sent_vec += vec

            cnt_words += 1

        except:

            pass

    sent_vec /= cnt_words

    x_train_avgw2v.append(sent_vec)

print(len(x_train_avgw2v))

print(len(x_train_avgw2v[0]))

list_sent_test=[]



for sentance in x_test:

    list_sent_test.append(sentance.split())

x_test_avgw2v= []; # the avg-w2v for each sentence/review is stored in this list

for sent in list_sent_test: # for each review/sentence

    sent_vec = np.zeros(50) # as word vectors are of zero length

    cnt_words =0; # num of words with a valid vector in the sentence/review

    for word in sent: # for each word in a review/sentence

        try:

            vec = w2v_model.wv[word]

            sent_vec += vec

            cnt_words += 1

        except:

            pass

    sent_vec /= cnt_words

    x_test_avgw2v.append(sent_vec)

print(len(x_test_avgw2v))

print(len(x_test_avgw2v[0]))
list_sent_cv=[]

for sentance in x_cv:

    list_sent_cv.append(sentance.split())

x_cv_avgw2v= []; # the avg-w2v for each sentence/review is stored in this list

for sent in list_sent_cv: # for each review/sentence

    sent_vec = np.zeros(50) # as word vectors are of zero length

    cnt_words =0; # num of words with a valid vector in the sentence/review

    for word in sent: # for each word in a review/sentence

        try:

            vec = w2v_model.wv[word]

            sent_vec += vec

            cnt_words += 1

        except:

            pass

    sent_vec /= cnt_words

    x_cv_avgw2v.append(sent_vec)

print(len(x_cv_avgw2v))

print(len(x_cv_avgw2v[0]))
x_train_avgw2v=np.nan_to_num(x_train_avgw2v)

x_test_avgw2v=np.nan_to_num(x_test_avgw2v)

x_cv_avgw2v=np.nan_to_num(x_cv_avgw2v)
# S = ["abc def pqr", "def def def abc", "pqr pqr def"]

model = TfidfVectorizer()

tf_idf_matrix = model.fit_transform(x_train)

# we are converting a dictionary with word as a key, and the idf as a value

dictionary = dict(zip(model.get_feature_names(), list(model.idf_)))
# TF-IDF weighted Word2Vec

tfidf_feat = model.get_feature_names() # tfidf words/col-names

# final_tf_idf is the sparse matrix with row= sentence, col=word and cell_val = tfidf



x_train_tfidfwv = []; # the tfidf-w2v for each sentence/review is stored in this list

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

    x_train_tfidfwv.append(sent_vec)

    row += 1
x_test_tfidfwv = []; # the tfidf-w2v for each sentence/review is stored in this list

row=0;

for sent in tqdm(list_sent_test): # for each review/sentence 

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

    x_test_tfidfwv.append(sent_vec)

    row += 1
x_cv_tfidfwv = []; # the tfidf-w2v for each sentence/review is stored in this list

row=0;

for sent in tqdm(list_sent_cv): # for each review/sentence 

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

    x_cv_tfidfwv.append(sent_vec)

    row += 1
x_train_tfidfwv=np.nan_to_num(x_train_tfidfwv)

x_test_tfidfwv=np.nan_to_num(x_test_tfidfwv)

x_cv_tfidfwv=np.nan_to_num(x_cv_tfidfwv)
#method to train the model

#method to train the model

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score,f1_score,confusion_matrix,classification_report,accuracy_score

def train_rf(x_train,y_train,x_cv,y_cv):

    #n_estimators = [int(x) for x in np.linspace(start =50, stop = 300, num = 6)]

    n_estimators = [i for i in range(1,200,50)]

    max_depth =  [1,11,51,101]

    max_features  = [int(x) for x in np.linspace(start =30, stop = 50, num = 10)]

    #min_samples_leaf= [10,50,100,200,500]

    auc_scores = []

    for n in n_estimators:

        for d in max_depth:

            clf_rf = RandomForestClassifier(n_estimators=n,max_depth=d)

            #clf_rf = RandomForestClassifier(n_estimators=n)

            clf_rf.fit(x_train,y_train)

            pred_y_train = clf_rf.predict(x_train)

            probs = clf_rf.predict_proba(x_cv)

            preds = probs[:,1]

            fpr, tpr, threshold = metrics.roc_curve(y_cv, preds)

            roc_auc_cv = metrics.auc(fpr, tpr)

            print('-'*60)

            print('For n_estimators',n,'and max_depth',d,'CV AUC score is',roc_auc_cv)

            probs_t = clf_rf.predict_proba(x_train)

            preds_t = probs_t[:,1]

            fpr_t, tpr_t, threshold = metrics.roc_curve(y_train, preds_t)

            roc_auc_cv_t = metrics.auc(fpr_t, tpr_t)

            

            print('For n_estimators',n,'and max_depth',d,'train AUC score is',roc_auc_cv_t)

            auc_scores.append(roc_auc_cv)

            print('-'*60)

    scores = np.array(auc_scores).reshape(len(n_estimators),len(max_depth))

    plt.figure(figsize=(20,7))

    sns.heatmap(scores,annot=True,fmt='g',cmap="YlGnBu")

    plt.xlabel('max_depth')

    plt.ylabel('n_estimators')

    plt.title('Auc score for different max_depth and estimator values')

    plt.xticks(np.arange(len(max_depth))+0.5,max_depth,rotation=45)

    plt.yticks(np.arange(len(n_estimators))+0.5,n_estimators,rotation=45)

    plt.show()





def best_rf(x_train,y_train,x_test,y_test,n_estimators,max_depth):

        #clf_rf = RandomForestClassifier()

        clf_rf = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)

        clf_rf.fit(x_train,y_train)

        probs = clf_rf.predict_proba(x_test)

        preds = probs[:,1]

        fpr, tpr, threshold = metrics.roc_curve(y_test, preds)

        roc_auc = metrics.auc(fpr, tpr)

        plt.title('Receiver Operating Characteristic')

        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

        plt.legend(loc = 'lower right')

        plt.plot([0, 1], [0, 1],'r--')

        plt.ylabel('True Positive Rate')

        plt.xlabel('False Positive Rate')

        plt.show()

        predict_y_train = clf_rf.predict(x_train)

        print('For number of estomators = ', n_estimators,'and max_depth value',max_depth, "The train f1  score is:",np.round(f1_score(y_train,predict_y_train),4))

        predict_y_test = clf_rf.predict(x_test)

        print('For number of estomators = ', n_estimators,'and max_depth value',max_depth, "The test f1  score is:",np.round(f1_score(y_test,predict_y_test),4))

        acc_t = accuracy_score(y_train,predict_y_train)

        print('Accuracy on train data is ',acc_t)

        acc = accuracy_score(y_test,predict_y_test)

        print('Accuracy on test data is ',acc)

        c_1 = confusion_matrix(y_train, predict_y_train)

        C = confusion_matrix(y_test, predict_y_test)

        print("-"*20, "Confusion matrix on train data", "-"*20)

        plt.figure(figsize=(20,7))

    

        sns.heatmap(c_1, annot=True, cmap="YlGnBu", fmt="d")

        plt.xlabel('Predicted Class')

        plt.ylabel('Original Class')

        plt.show()

        print("-"*20, "Confusion matrix on test data", "-"*20)

        plt.figure(figsize=(20,7))

    

        sns.heatmap(C, annot=True, cmap="YlGnBu", fmt="d")

        plt.xlabel('Predicted Class')

        plt.ylabel('Original Class')

        plt.show()

        print(classification_report(y_test, predict_y_test))

        return clf_rf



    
#method to print inportant features

from wordcloud import WordCloud

def imp_words(clf,vectorizer):

    names = vectorizer.get_feature_names()

    names = np.array(names)

    importances = clf.feature_importances_

    # Sort feature importances in descending order

    indices = np.argsort(importances)[::-1][:20]

    words = names[indices]

    print('Top 20 words')

    wordcloud = WordCloud(background_color='white',

                          width=1200,

                          height=1000

                         ).generate(str(words))



    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()
train_rf(x_train_bag,y_train,x_cv_bag,y_cv)
clf = best_rf(x_train_bag,y_train,x_test_bag,y_test,151,101)
# Please write all the code with proper documentation

imp_words(clf,bag_words)

# Please write all the code with proper 

train_rf(x_train_tfidf,y_train,x_cv_tfidf,y_cv)
clf = best_rf(x_train_tfidf,y_train,x_test_tfidf,y_test,151,101)
# Please write all the code with proper documentation

imp_words(clf,tfidf_words)
# Please write all the code with proper documentation

train_rf(x_train_avgw2v,y_train,x_cv_avgw2v,y_cv)
best_rf(x_train_avgw2v,y_train,x_test_avgw2v,y_test,151,11)
# Please write all the code with proper documentation

train_rf(x_train_tfidfwv,y_train,x_cv_tfidfwv,y_cv)
best_rf(x_train_tfidfwv,y_train,x_test_tfidfwv,y_test,151,11)
from sklearn.ensemble import GradientBoostingClassifier

import lightgbm as lgbm

import re

from xgboost import XGBClassifier

from sklearn.metrics import f1_score,f1_score,confusion_matrix,classification_report,accuracy_score

def train_gdbt(x_train,y_train,x_cv,y_cv):

    #n_estimators = [int(x) for x in np.linspace(start =50, stop = 300, num = 6)]

    n_estimators = [i for i in range(1,200,50)]

    max_depth =  [1,11,51,101]

    max_features  = [int(x) for x in np.linspace(start =30, stop = 50, num = 10)]

    #min_samples_leaf= [10,50,100,200,500]

    auc_scores = []

    for n in n_estimators:

        for d in max_depth:

            clf_rf = XGBClassifier(n_estimators=n,max_depth=d,booster='gbtree')

            #clf_rf = RandomForestClassifier(n_estimators=n)

            clf_rf.fit(x_train,y_train)

            pred_y_train = clf_rf.predict(x_train)

            probs = clf_rf.predict_proba(x_cv)

            preds = probs[:,1]

            fpr, tpr, threshold = metrics.roc_curve(y_cv, preds)

            roc_auc_cv = metrics.auc(fpr, tpr)

            print('-'*60)

            print('For n_estimators',n,'and max_depth',d,'CV AUC score is',roc_auc_cv)

            probs_t = clf_rf.predict_proba(x_train)

            preds_t = probs_t[:,1]

            fpr_t, tpr_t, threshold = metrics.roc_curve(y_train, preds_t)

            roc_auc_cv_t = metrics.auc(fpr_t, tpr_t)

            

            print('For n_estimators',n,'and max_depth',d,'train AUC score is',roc_auc_cv_t)

            auc_scores.append(roc_auc_cv)

            print('-'*60)

    scores = np.array(auc_scores).reshape(len(n_estimators),len(max_depth))

    plt.figure(figsize=(20,7))

    sns.heatmap(scores,annot=True,fmt='g',cmap="YlGnBu")

    plt.xlabel('max_depth')

    plt.ylabel('n_estimators')

    plt.title('Auc score for different max_depth and estimator values')

    plt.xticks(np.arange(len(max_depth))+0.5,max_depth,rotation=45)

    plt.yticks(np.arange(len(n_estimators))+0.5,n_estimators,rotation=45)

    plt.show()





# Please write all the code with proper documentation

train_gdbt(x_train_bag,y_train,x_cv_bag,y_cv)
# Please write all the code with proper documentation
# Please write all the code with proper documentation
# Please write all the code with proper documentation
# Please compare all your models using Prettytable library