# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
import warnings
warnings.filterwarnings("ignore")
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Reviews.csv',encoding='Latin-1')
data.head()
data.info()
mk = data.isnull().sum()
mk_2 = mk[mk>0]
print(mk_2)
data = data.fillna("")
mk = data.isnull().sum()
mk_2 = mk[mk>0]
print(mk_2)
data.info()
data.Text
df_caseFolding = data.copy()
df_caseFolding['Summary'] = data['Summary'].str.lower()
df_caseFolding['Text'] = data['Text'].str.lower()
df_caseFolding.columns = df_caseFolding.columns.str.lower()
df_caseFolding.head()
df_caseFolding.info()
import re

def pre_processing(review):
    clean_str = review.lower() #lowercase
    clean_str = re.sub(r'<[^>]+>', ' ', clean_str) #buang_tag html
    clean_str = re.sub(r"(?:\@|https?\://)\S+", " ", clean_str) #buang username dan url
    clean_str = re.sub(r'[^\w\s]',' ',clean_str) #buang punctuation
    rpt_regex = re.compile(r"(.)\1{1,}", re.IGNORECASE) #regex kata yang berulang kaya haiiii
    clean_str = re.sub('\s+', ' ', clean_str) # remove extra space
    clean_str = clean_str.strip() #trim depan belakang
    
    return(clean_str)
df_regex = df_caseFolding.copy()
df_regex['text'] = df_regex['text'].apply(pre_processing)
df_regex['summary'] = df_regex['summary'].apply(pre_processing)
df_regex.text
import pandas as pd
import nltk

df_token = df_regex.copy()
df_token["text"] = df_token["text"].apply(nltk.word_tokenize)
df_token["summary"] = df_token["summary"].apply(nltk.word_tokenize)
print(df_token.text.head())
print(df_token.summary.head())
from nltk.corpus import stopwords
stop = stopwords.words('english')

df_stop = df_token.copy()
df_stop['summary'] = df_stop['summary'].apply(lambda x: [item for item in x if item not in stop])
df_stop['text'] = df_stop['text'].apply(lambda x: [item for item in x if item not in stop])
df_stop.head()
from textblob import Word

df_lemma = df_stop.copy()
df_lemma['text'] = df_lemma['text'].apply(lambda x: [Word(word).lemmatize() for word in x])
df_lemma['summary'] = df_lemma['summary'].apply(lambda x: [Word(word).lemmatize() for word in x])
print(df_lemma.text.head())
print(df_lemma.summary.head())
df_stop.text
df_lemma.text
df_lemma.score.unique()
from gensim.models import Word2Vec, KeyedVectors

import matplotlib.pyplot as plt
import seaborn as sns

numeric = df_lemma.dtypes[df_lemma.dtypes != 'object'].index
corr = df_lemma[['helpfulnessnumerator','helpfulnessdenominator','score']].corr()
plt.figure(figsize = (15,15))
sns.heatmap(corr, annot = True, fmt = '.2f')
df_lemma['score'].value_counts().plot(kind='bar')

#hasilnya imbalance dataset
filtered_data = df_lemma.drop(df_lemma[df_lemma.score == 3].index)
filtered_data['score'].value_counts().plot(kind='bar')
filtered_data.shape
# Here are replacing review score 1,2 as negative (0) and 4,5 as a positive(1). we are skipping review score 3 considering it as a neutral.
def partition(x):
    if x<3:
        return 0
    return 1

actualScore = filtered_data['score']
positiveNegative = actualScore.map(partition)
filtered_data['score'] = positiveNegative
filtered_data['score'].value_counts().plot(kind='bar')
filtered_data.time
import datetime

#filtered_data["time"] = filtered_data["time"].map(lambda t: datetime.datetime.fromtimestamp(int(t)).strftime('%Y-%m-%d %H:%M:%S'))

final1 = filtered_data.sort_values('productid',axis=0,kind="quicksort", ascending=True)
#final = sortedData.drop_duplicates(subset={"userid","profilename","time","text"},keep="first",inplace=False)

final1 = final1[final1.helpfulnessnumerator <= final1.helpfulnessdenominator]

#As data is huge, due to computation limitation we will randomly select data. we will try to pick data in a way so that it doesn't make data imbalance problem
finalp1 = final1[final1.score == 1]
finalp1 = finalp1.sample(frac=0.035,random_state=1) #0.055

finaln1 = final1[final1.score == 0]
finaln1 = finaln1.sample(frac=0.15,random_state=1) #0.25

final1 = pd.concat([finalp1,finaln1],axis=0)

#sording data by timestamp so that it can be devided in train and test dataset for time based slicing.
final1 = final1.sort_values('time',axis=0,kind="quicksort", ascending=True).reset_index(drop=True)


print(final1.shape)
print(final1['score'].value_counts().plot(kind='bar'))
import datetime

#filtered_data["time"] = filtered_data["time"].map(lambda t: datetime.datetime.fromtimestamp(int(t)).strftime('%Y-%m-%d %H:%M:%S'))

final2 = filtered_data.sort_values('productid',axis=0,kind="quicksort", ascending=True)
#final = sortedData.drop_duplicates(subset={"userid","profilename","time","text"},keep="first",inplace=False)

final2 = final2[final2.helpfulnessnumerator <= final2.helpfulnessdenominator]
#As data is huge, due to computation limitation we will randomly select data. we will try to pick data in a way so that it doesn't make data imbalance problem
finalp2 = final2[final2.score == 1]
finalp2 = finalp2.sample(frac=0.055,random_state=1) #0.055

finaln2 = final2[final2.score == 0]
finaln2 = finaln2.sample(frac=0.25,random_state=1) #0.25

final2 = pd.concat([finalp2,finaln2],axis=0)

#sording data by timestamp so that it can be devided in train and test dataset for time based slicing.
final2 = final2.sort_values('time',axis=0,kind="quicksort", ascending=True).reset_index(drop=True)


print(final2.shape)
print(final2['score'].value_counts().plot(kind='bar'))
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

def find_optimal_k(X_train,y_train, myList):
   
    #creating odd list of K for KNN
    #myList = list(range(0,40))
    neighbors = list(filter(lambda x: x % 2 != 0, myList))

    # empty list that will hold cv scores
    cv_scores = []
    kfold = KFold(n_splits=10, random_state=7)
    # perform 10-fold cross validation
    for k in neighbors:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=kfold, scoring='accuracy')
        cv_scores.append(scores.mean())

    # changing to misclassification error
    MSE = [1 - x for x in cv_scores]

    # determining best k
    optimal_k = neighbors[MSE.index(min(MSE))]
    print('\nThe optimal number of neighbors is %d.' % optimal_k)


    plt.figure(figsize=(10,6))
    plt.plot(list(filter(lambda x: x % 2 != 0, myList)),MSE,color='blue', linestyle='dashed', marker='o',
             markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')

    print("the misclassification error for each k value is : ", np.round(MSE,3))
    
    return optimal_k
# import gensim
# i=0
# str1=''
# list_of_sent=[]
# final_string_for_tfidf = []
# for sent in final2['text'].values:
#     filtered_sentence=[]
#     #sent=cleanhtml(sent)
#     str1 = ''
#     for w in sent:
#         for cleaned_words in w:
#             if((cleaned_words.isalpha()) & (cleaned_words.lower() not in stop)):    
#                 filtered_sentence.append(cleaned_words.lower())
#                 str1 += " "+cleaned_words.lower() 
#             else:
#                 continue
#     #str1 = b" ".join(filtered_sentence) #final string of cleaned words
            
#     #final_string_for_tfidf.append(str1)
#     list_of_sent.append(filtered_sentence)
#     final_string_for_tfidf.append((str1).strip())
w2v_model=gensim.models.Word2Vec(final2['text'],min_count=5,size=50, workers=4)  
sent_vectors = []; 
for sent in final2['text']: 
    sent_vec = np.zeros(50)
    cnt_words =0; 
    for word in sent: 
        try:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
        except:
            pass
    sent_vec /= cnt_words
    sent_vectors.append(sent_vec)
    
import math
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

X = sent_vectors #final_w2v_count
y = final2['score']

X_train =  sent_vectors[:math.ceil(len(final2)*.7)]  #final_w2v_count
X_test = sent_vectors[math.ceil(len(final2)*.7):] #final_w2v_count
y_train = y[:math.ceil(len(final2)*.7)]
y_test =  y[math.ceil(len(final2)*.7):]
myList = list(range(0,10))

optimal_k = find_optimal_k(X_train ,y_train,myList)
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
import scikitplot.metrics as skplt

skplt.plot_confusion_matrix(y_test ,pred)
print(classification_report(y_test ,pred))
print("Accuracy for KNN model with Word2Vec is ",round(accuracy_score(y_test ,pred),3))
