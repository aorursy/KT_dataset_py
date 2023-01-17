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
food_reviews = pd.read_csv('../input/ds5230usml-project/Reviews.csv')

food_reviews.head()
food_reviews['ProductId'].nunique()
food_reviews['UserId'].nunique()
food_reviews.isnull().sum()
food_reviews_score = food_reviews[['Score']].dropna()



ax=food_reviews_score.Score.value_counts().plot(kind='bar')

fig = ax.get_figure()

fig.savefig("score.png");
food_reviews_filtered = food_reviews[food_reviews['Score']!=3]       #Neutral reviews removed
# Give reviews with Score>3 a positive rating, and reviews with a score<3 a negative rating.

def partition(x):

    if x < 3:

        return 'negative'

    return 'positive'



# Applying polarity function on Score column of filtered_data

actualScore = food_reviews_filtered['Score']

positiveNegative = actualScore.map(partition) 

food_reviews_filtered['Score'] = positiveNegative
print(food_reviews_filtered.shape)

# Removing duplicate entries

food_reviews_filtered_sorted=food_reviews_filtered.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')

food_reviews_final=food_reviews_filtered_sorted.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, keep='first', inplace=False)
print(food_reviews_final.shape)
# Removing any rows with HelpfulnessDenominator < HelpfulnessNumerator

food_reviews_final=food_reviews_final[food_reviews_final.HelpfulnessNumerator<=food_reviews_final.HelpfulnessDenominator]
print(food_reviews_final.shape)
# Observing total positive and negative reviews

food_reviews_final['Score'].value_counts()
# Set of stopwords in English

from nltk.corpus import stopwords

stop = set(stopwords.words('english'))

words_to_keep = set(('not'))

stop -= words_to_keep

# Initialising the snowball stemmer



import nltk

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer

sno = nltk.stem.SnowballStemmer('english')

 # Function to clean the word of any html-tags

def cleanhtml(sentence):

    cleanr = re.compile('<.*?>')

    cleantext = re.sub(cleanr, ' ', sentence)

    return cleantext
# Function to clean the word of any punctuation or special characters

def cleanpunc(sentence): 

    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)

    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)

    return  cleaned


# Importing libraries

import warnings

warnings.filterwarnings("ignore")



import sqlite3

import pandas as pd

import numpy as np

import nltk

import string

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer



from sklearn.feature_extraction.text import CountVectorizer

from nltk.stem.porter import PorterStemmer



import re



import string

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer



from gensim.models import Word2Vec

from gensim.models import KeyedVectors

import pickle







from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



from collections import Counter



from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.metrics import roc_curve, auc



# Code for removing 

# HTML tags , 

# punctuations , 

# stopwords . 

# Code for checking if word is not alphanumeric and also greater than 2 . 

# Code for stemmimg and also to converting them to lowercase.

i=0

str1=' '

final_string=[]

all_positive_words=[] # store words from +ve reviews here

all_negative_words=[] # store words from -ve reviews here.

s=''

for sent in food_reviews_final['Text'].values:

    filtered_sentence=[]

    #print(sent);

    sent=cleanhtml(sent) # remove HTMl tags

    for w in sent.split():

        for cleaned_words in cleanpunc(w).split():

            if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):    

                if(cleaned_words.lower() not in stop):

                    s=(sno.stem(cleaned_words.lower())).encode('utf8')

                    filtered_sentence.append(s)

                    if (food_reviews_final['Score'].values)[i] == 'positive': 

                        all_positive_words.append(s) #list of all words used to describe positive reviews

                    if(food_reviews_final['Score'].values)[i] == 'negative':

                        all_negative_words.append(s) #list of all words used to describe negative reviews reviews

                else:

                    continue

            else:

                continue 

    

    str1 = b" ".join(filtered_sentence) #final string of cleaned words

    

    

    final_string.append(str1)

    i+=1
#adding a column of CleanedText which displays the data after pre-processing of the review

food_reviews_final['CleanedText']=final_string  

food_reviews_final['CleanedText']=food_reviews_final['CleanedText'].str.decode("utf-8")

#below the processed review can be seen in the CleanedText Column 

print('Shape of final',food_reviews_final.shape)

food_reviews_final.head()
##Sorting data according to Time in ascending order for Time Based Splitting

time_sorted_data = food_reviews_final.sort_values('Time', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')



# We will collect different 40K rows without repetition from time_sorted_data dataframe

my_final = time_sorted_data.take(np.random.permutation(len(food_reviews_final))[:5000])



x = my_final['CleanedText'].values
#BoW

count_vect = CountVectorizer(min_df = 100) 

data = count_vect.fit_transform(x)

print("the type of count vectorizer :",type(data))

print("the shape of out text BOW vectorizer : ",data.get_shape())

print("the number of unique words :", data.get_shape()[1])
from sklearn.cluster import AgglomerativeClustering



model = AgglomerativeClustering(n_clusters=2).fit(data.toarray())



reviews = my_final['Text'].values

# Getting all the reviews in different clusters

cluster1 = []

cluster2 = []



for i in range(model.labels_.shape[0]):

    if model.labels_[i] == 0:

        cluster1.append(reviews[i])

    else :

        cluster2.append(reviews[i])

 

        

# Number of reviews in different clusters

print("No. of reviews in Cluster-1 : ",len(cluster1))

print("\nNo. of reviews in Cluster-2 : ",len(cluster2))
# Three Reviews of cluster 1

count=1

for i in range(3):

    if i < len(cluster1):

        print('Review-%d : \n %s\n'%(count,cluster1[i]))

        count +=1
# Three Reviews of cluster 2

count=1

for i in range(3):

    if i < len(cluster2):

        print('Review-%d : \n %s\n'%(count,cluster2[i]))

        count +=1
model = AgglomerativeClustering(n_clusters=5).fit(data.toarray())



# Getting all the reviews in different clusters

cluster1 = []

cluster2 = []

cluster3 = []

cluster4 = []

cluster5 = []



for i in range(model.labels_.shape[0]):

    if model.labels_[i] == 0:

        cluster1.append(reviews[i])

    elif model.labels_[i] == 1:

        cluster2.append(reviews[i])

    elif model.labels_[i] == 2:

        cluster3.append(reviews[i])

    elif model.labels_[i] == 3:

        cluster4.append(reviews[i])

    else :

        cluster5.append(reviews[i]) 

        

# Number of reviews in different clusters

print("No. of reviews in Cluster-1 : ",len(cluster1))

print("\nNo. of reviews in Cluster-2 : ",len(cluster2))

print("\nNo. of reviews in Cluster-3 : ",len(cluster3))

print("\nNo. of reviews in Cluster-4 : ",len(cluster4))

print("\nNo. of reviews in Cluster-5 : ",len(cluster5))
# Three Reviews of cluster 1

count=1

for i in range(3):

    if i < len(cluster1):

        print('Review-%d : \n %s\n'%(count,cluster1[i]))

        count +=1
# Three Reviews of cluster 2

count=1

for i in range(3):

    if i < len(cluster2):

        print('Review-%d : \n %s\n'%(count,cluster2[i]))

        count +=1
# Three Reviews of cluster 3

count=1

for i in range(3):

    if i < len(cluster3):

        print('Review-%d : \n %s\n'%(count,cluster3[i]))

        count +=1
# Three Reviews of cluster 4

count=1

for i in range(3):

    if i < len(cluster4):

        print('Review-%d : \n %s\n'%(count,cluster4[i]))

        count +=1
# Three Reviews of cluster 5

count=1

for i in range(3):

    if i < len(cluster5):

        print('Review-%d : \n %s\n'%(count,cluster5[i]))

        count +=1
model = AgglomerativeClustering(n_clusters=10).fit(data.toarray())

# Getting all the reviews in different clusters

cluster1 = []

cluster2 = []

cluster3 = []

cluster4 = []

cluster5 = []

cluster6 = []

cluster7 = []

cluster8 = []

cluster9 = []

cluster10 = []



for i in range(model.labels_.shape[0]):

    if model.labels_[i] == 0:

        cluster1.append(reviews[i])

    elif model.labels_[i] == 1:

        cluster2.append(reviews[i])

    elif model.labels_[i] == 2:

        cluster3.append(reviews[i])

    elif model.labels_[i] == 3:

        cluster4.append(reviews[i])

    elif model.labels_[i] == 4:

        cluster5.append(reviews[i])

    elif model.labels_[i] == 5:

        cluster6.append(reviews[i])

    elif model.labels_[i] == 6:

        cluster7.append(reviews[i])

    elif model.labels_[i] == 7:

        cluster8.append(reviews[i])

    elif model.labels_[i] == 8:

        cluster9.append(reviews[i])       

    else :

        cluster10.append(reviews[i])
# Heirarchial Clustering With TF-IDF

tf_idf_vect = TfidfVectorizer(min_df=100)

data = tf_idf_vect.fit_transform(x)

print("the type of count vectorizer :",type(data))

print("the shape of out text TFIDF vectorizer : ",data.get_shape())

print("the number of unique words :", data.get_shape()[1])
# Heirarchial Clustering With TF-IDF With 2 clusters

model = AgglomerativeClustering(n_clusters=2).fit(data.toarray())



reviews = my_final['Text'].values

# Getting all the reviews in different clusters

cluster1 = []

cluster2 = []



for i in range(model.labels_.shape[0]):

    if model.labels_[i] == 0:

        cluster1.append(reviews[i])

    else :

        cluster2.append(reviews[i])

 

        

# Number of reviews in different clusters

print("No. of reviews in Cluster-1 : ",len(cluster1))

print("\nNo. of reviews in Cluster-2 : ",len(cluster2))
# Three Reviews of cluster 1

count=1

for i in range(3):

    print('Review-%d : \n %s\n'%(count,cluster1[i]))

    count +=1
# Three Reviews of cluster 2

count=1

for i in range(3):

    print('Review-%d : \n %s\n'%(count,cluster2[i]))

    count +=1
# with 5 clusters

model = AgglomerativeClustering(n_clusters=5).fit(data.toarray())



# Getting all the reviews in different clusters

cluster1 = []

cluster2 = []

cluster3 = []

cluster4 = []

cluster5 = []



for i in range(model.labels_.shape[0]):

    if model.labels_[i] == 0:

        cluster1.append(reviews[i])

    elif model.labels_[i] == 1:

        cluster2.append(reviews[i])

    elif model.labels_[i] == 2:

        cluster3.append(reviews[i])

    elif model.labels_[i] == 3:

        cluster4.append(reviews[i])

    else :

        cluster5.append(reviews[i]) 

        

# Number of reviews in different clusters

print("No. of reviews in Cluster-1 : ",len(cluster1))

print("\nNo. of reviews in Cluster-2 : ",len(cluster2))

print("\nNo. of reviews in Cluster-3 : ",len(cluster3))

print("\nNo. of reviews in Cluster-4 : ",len(cluster4))

print("\nNo. of reviews in Cluster-5 : ",len(cluster5))
# With 10 clusters

model = AgglomerativeClustering(n_clusters=10).fit(data.toarray())

# Getting all the reviews in different clusters

cluster1 = []

cluster2 = []

cluster3 = []

cluster4 = []

cluster5 = []

cluster6 = []

cluster7 = []

cluster8 = []

cluster9 = []

cluster10 = []



for i in range(model.labels_.shape[0]):

    if model.labels_[i] == 0:

        cluster1.append(reviews[i])

    elif model.labels_[i] == 1:

        cluster2.append(reviews[i])

    elif model.labels_[i] == 2:

        cluster3.append(reviews[i])

    elif model.labels_[i] == 3:

        cluster4.append(reviews[i])

    elif model.labels_[i] == 4:

        cluster5.append(reviews[i])

    elif model.labels_[i] == 5:

        cluster6.append(reviews[i])

    elif model.labels_[i] == 6:

        cluster7.append(reviews[i])

    elif model.labels_[i] == 7:

        cluster8.append(reviews[i])

    elif model.labels_[i] == 8:

        cluster9.append(reviews[i])       

    else :

        cluster10.append(reviews[i])
# Hierarchial clustering with Average Word to vector

# List of sentence in X_train text

sent_x = []

for sent in x :

    sent_x.append(sent.split())

  

    

# Train your own Word2Vec model using your own train text corpus 

# min_count = 5 considers only words that occured atleast 5 times

w2v_model=Word2Vec(sent_x,min_count=5,size=50, workers=4)



w2v_words = list(w2v_model.wv.vocab)

print("number of words that occured minimum 5 times ",len(w2v_words))
# compute average word2vec for each review for sent_x .

train_vectors = []; 

for sent in sent_x:

    sent_vec = np.zeros(50) 

    cnt_words =0; 

    for word in sent: # 

        if word in w2v_words:

            vec = w2v_model.wv[word]

            sent_vec += vec

            cnt_words += 1

    if cnt_words != 0:

        sent_vec /= cnt_words

    train_vectors.append(sent_vec)

    

data = train_vectors
model = AgglomerativeClustering(n_clusters=2).fit(data)



reviews = my_final['Text'].values

# Getting all the reviews in different clusters

cluster1 = []

cluster2 = []



for i in range(model.labels_.shape[0]):

    if model.labels_[i] == 0:

        cluster1.append(reviews[i])

    else :

        cluster2.append(reviews[i])

 

        

# Number of reviews in different clusters

print("No. of reviews in Cluster-1 : ",len(cluster1))

print("\nNo. of reviews in Cluster-2 : ",len(cluster2))


# Three Reviews of cluster 1

count=1

for i in range(3):

    print('Review-%d : \n %s\n'%(count,cluster1[i]))

    count +=1


# Three Reviews of cluster 2

count=1

for i in range(3):

    print('Review-%d : \n %s\n'%(count,cluster2[i]))

    count +=1
# Heierarachial Clustering with TFIDF-Word2Vec

# TF-IDF weighted Word2Vec

tf_idf_vect = TfidfVectorizer()



# final_tf_idf1 is the sparse matrix with row= sentence, col=word and cell_val = tfidf

final_tf_idf1 = tf_idf_vect.fit_transform(x)



# tfidf words/col-names

tfidf_feat = tf_idf_vect.get_feature_names()



# compute TFIDF Weighted Word2Vec for each review for sent_x .

tfidf_vectors = []; 

row=0;

for sent in sent_x: 

    sent_vec = np.zeros(50) 

    weight_sum =0; 

    for word in sent: 

        if word in w2v_words:

            vec = w2v_model.wv[word]

            # obtain the tf_idfidf of a word in a sentence/review

            tf_idf = final_tf_idf1[row, tfidf_feat.index(word)]

            sent_vec += (vec * tf_idf)

            weight_sum += tf_idf

    if weight_sum != 0:

        sent_vec /= weight_sum

    tfidf_vectors.append(sent_vec)

    row += 1 

    

data = tfidf_vectors
# Heierarachial Clustering with TFIDF-Word2Vec with 2 clusters

model = AgglomerativeClustering(n_clusters=2).fit(data)



reviews = my_final['Text'].values

# Getting all the reviews in different clusters

cluster1 = []

cluster2 = []



for i in range(model.labels_.shape[0]):

    if model.labels_[i] == 0:

        cluster1.append(reviews[i])

    else :

        cluster2.append(reviews[i])

 

        

# Number of reviews in different clusters

print("No. of reviews in Cluster-1 : ",len(cluster1))

print("\nNo. of reviews in Cluster-2 : ",len(cluster2))
# Heierarachial Clustering with TFIDF-Word2Vec with 5 clusters



model = AgglomerativeClustering(n_clusters=5).fit(data)



# Getting all the reviews in different clusters

cluster1 = []

cluster2 = []

cluster3 = []

cluster4 = []

cluster5 = []



for i in range(model.labels_.shape[0]):

    if model.labels_[i] == 0:

        cluster1.append(reviews[i])

    elif model.labels_[i] == 1:

        cluster2.append(reviews[i])

    elif model.labels_[i] == 2:

        cluster3.append(reviews[i])

    elif model.labels_[i] == 3:

        cluster4.append(reviews[i])

    else :

        cluster5.append(reviews[i]) 

        

# Number of reviews in different clusters

print("No. of reviews in Cluster-1 : ",len(cluster1))

print("\nNo. of reviews in Cluster-2 : ",len(cluster2))

print("\nNo. of reviews in Cluster-3 : ",len(cluster3))

print("\nNo. of reviews in Cluster-4 : ",len(cluster4))

print("\nNo. of reviews in Cluster-5 : ",len(cluster5))
# Heierarachial Clustering with TFIDF-Word2Vec with 10 clusters



model = AgglomerativeClustering(n_clusters=10).fit(data)

# Getting all the reviews in different clusters

cluster1 = []

cluster2 = []

cluster3 = []

cluster4 = []

cluster5 = []

cluster6 = []

cluster7 = []

cluster8 = []

cluster9 = []

cluster10 = []



for i in range(model.labels_.shape[0]):

    if model.labels_[i] == 0:

        cluster1.append(reviews[i])

    elif model.labels_[i] == 1:

        cluster2.append(reviews[i])

    elif model.labels_[i] == 2:

        cluster3.append(reviews[i])

    elif model.labels_[i] == 3:

        cluster4.append(reviews[i])

    elif model.labels_[i] == 4:

        cluster5.append(reviews[i])

    elif model.labels_[i] == 5:

        cluster6.append(reviews[i])

    elif model.labels_[i] == 6:

        cluster7.append(reviews[i])

    elif model.labels_[i] == 7:

        cluster8.append(reviews[i])

    elif model.labels_[i] == 8:

        cluster9.append(reviews[i])       

    else :

        cluster10.append(reviews[i])
#  Hierarchial Clustering with TFIDF-Word2Vec

tf_idf_vect = TfidfVectorizer()



# final_tf_idf1 is the sparse matrix with row= sentence, col=word and cell_val = tfidf

final_tf_idf1 = tf_idf_vect.fit_transform(x)



# tfidf words/col-names

tfidf_feat = tf_idf_vect.get_feature_names()



# compute TFIDF Weighted Word2Vec for each review for sent_x .

tfidf_vectors = []; 

row=0;

for sent in sent_x: 

    sent_vec = np.zeros(50) 

    weight_sum =0; 

    for word in sent: 

        if word in w2v_words:

            vec = w2v_model.wv[word]

            # obtain the tf_idfidf of a word in a sentence/review

            tf_idf = final_tf_idf1[row, tfidf_feat.index(word)]

            sent_vec += (vec * tf_idf)

            weight_sum += tf_idf

    if weight_sum != 0:

        sent_vec /= weight_sum

    tfidf_vectors.append(sent_vec)

    row += 1 

    

data = tfidf_vectors
# Hierarchial Clustering with  TF-IDF weighted Word2Vec with 2 clusters



model = AgglomerativeClustering(n_clusters=2).fit(data)



reviews = my_final['Text'].values

# Getting all the reviews in different clusters

cluster1 = []

cluster2 = []



for i in range(model.labels_.shape[0]):

    if model.labels_[i] == 0:

        cluster1.append(reviews[i])

    else :

        cluster2.append(reviews[i])

 

        

# Number of reviews in different clusters

print("No. of reviews in Cluster-1 : ",len(cluster1))

print("\nNo. of reviews in Cluster-2 : ",len(cluster2))
# Hierarchial Clustering with  TF-IDF weighted Word2Vec with 5 clusters

model = AgglomerativeClustering(n_clusters=5).fit(data)



# Getting all the reviews in different clusters

cluster1 = []

cluster2 = []

cluster3 = []

cluster4 = []

cluster5 = []



for i in range(model.labels_.shape[0]):

    if model.labels_[i] == 0:

        cluster1.append(reviews[i])

    elif model.labels_[i] == 1:

        cluster2.append(reviews[i])

    elif model.labels_[i] == 2:

        cluster3.append(reviews[i])

    elif model.labels_[i] == 3:

        cluster4.append(reviews[i])

    else :

        cluster5.append(reviews[i]) 

        

# Number of reviews in different clusters

print("No. of reviews in Cluster-1 : ",len(cluster1))

print("\nNo. of reviews in Cluster-2 : ",len(cluster2))

print("\nNo. of reviews in Cluster-3 : ",len(cluster3))

print("\nNo. of reviews in Cluster-4 : ",len(cluster4))

print("\nNo. of reviews in Cluster-5 : ",len(cluster5))
# Hierarchial Clustering with  TF-IDF weighted Word2Vec with 10 clusters

model = AgglomerativeClustering(n_clusters=10).fit(data)

# Getting all the reviews in different clusters

cluster1 = []

cluster2 = []

cluster3 = []

cluster4 = []

cluster5 = []

cluster6 = []

cluster7 = []

cluster8 = []

cluster9 = []

cluster10 = []



for i in range(model.labels_.shape[0]):

    if model.labels_[i] == 0:

        cluster1.append(reviews[i])

    elif model.labels_[i] == 1:

        cluster2.append(reviews[i])

    elif model.labels_[i] == 2:

        cluster3.append(reviews[i])

    elif model.labels_[i] == 3:

        cluster4.append(reviews[i])

    elif model.labels_[i] == 4:

        cluster5.append(reviews[i])

    elif model.labels_[i] == 5:

        cluster6.append(reviews[i])

    elif model.labels_[i] == 6:

        cluster7.append(reviews[i])

    elif model.labels_[i] == 7:

        cluster8.append(reviews[i])

    elif model.labels_[i] == 8:

        cluster9.append(reviews[i])       

    else :

        cluster10.append(reviews[i])
# Three Reviews of cluster 1

count=1

for i in range(3):

    if i < len(cluster1):

        print('Review-%d : \n %s\n'%(count,cluster1[i]))

        count +=1
# Three Reviews of cluster 2

count=1

for i in range(3):

    if i < len(cluster2):

        print('Review-%d : \n %s\n'%(count,cluster2[i]))

        count +=1
# Three Reviews of cluster 3

count=1

for i in range(3):

    if i < len(cluster3):

        print('Review-%d : \n %s\n'%(count,cluster3[i]))

        count +=1