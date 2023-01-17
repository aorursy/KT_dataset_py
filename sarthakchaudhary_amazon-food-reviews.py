# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm #Process bar
tqdm.pandas()
con = sqlite3.connect('/kaggle/input/amazon-fine-food-reviews/database.sqlite') # connecting to database

raw_reviews = pd.read_sql_query("""SELECT * FROM Reviews WHERE Score != 3 LIMIT 6000""", con) # Loading only 6000 rows for fast computing
raw_reviews.head()
## Converting Score to positive (for Score 5 or 4) and negitive (for Score 1 or 2)

reviews = raw_reviews
reviews['Score'] = reviews['Score'].apply(lambda x : 'positive' if x>3 else 'negitive')
reviews.head()

reviews.shape
# removing Duplicated
# Some rows are duplicated as amazone use the same review for the different varient of the same product

reviews.drop_duplicates(subset={'UserId', 'ProfileName', 'Text', 'Summary', 'Score', 'Time'}, inplace=True)
reviews.shape
reviews.info()
reviews['Score'].value_counts()
# Histogram

sns.FacetGrid(reviews, height=5).map(plt.hist, 'Score').add_legend()
# Removing Garbage
import re
def rem(sen):
    sen = re.sub(r'<.*?>',' ',sen)#remove HTLM tags
    sen = re.sub(r'^https?:\/\/.*[\r\n\s]', ' ', sen)#remove links
    
    ##Changing decontracted words
    sen = re.sub(r"n\'t", " not", sen)
    sen = re.sub(r"\'re", " are", sen)
    sen = re.sub(r"\'s", " is", sen)
    sen = re.sub(r"\'d", " would", sen)
    sen = re.sub(r"\'ll", " will", sen)
    sen = re.sub(r"\'t", " not", sen)
    sen = re.sub(r"\'ve", " have", sen)
    sen = re.sub(r"\'m", " am", sen)
    
    sen = re.sub(r'\W',' ',sen)#remove special characters

    #### Removing alpha numeric words and non english words and len<=2 words
    
    new_sen = ""
    for word in sen.split():
        if word.isalpha():
            if len(word)>2:
                new_sen += word + " "
            else:
                continue
        else:
            continue
            
    return new_sen.lower() #Converting to Lowercase
# Testing our rem function
print(reviews['Text'][497])
print()
print()
print(rem(reviews['Text'][497]))
# performing rem() function on Text and Summary Coloumns
reviews['Text'] = reviews['Text'].progress_apply(rem)
reviews['Summary'] = reviews['Summary'].progress_apply(rem)
reviews.iloc[2]
from nltk.corpus import stopwords

#We are removing not, nor no from stopwords

s_words = set(stopwords.words('english'))
print('length of stopwordes before removing is {}'.format(len(s_words)))

s_words = s_words.difference({'no','not','nor'})
print('length of stopwordes after removing is {}'.format(len(s_words)))

def rem_swords(s, s_words = s_words):
    new_s = ""
    for word in s.split():
        if word not in s_words:
            new_s += word + " "
        else:
            continue
    return new_s

##Testing function
print(reviews['Text'][5])
print();print()
print(rem_swords(reviews['Text'][5]))
#Removing StopWords from Text and Summary

reviews['Text'] = reviews['Text'].progress_apply(rem_swords)
reviews['Summary'] = reviews['Summary'].progress_apply(rem_swords)
reviews['Text'][4]
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')

def stemming(sen, stemmer = stemmer):
    new_s = ""
    for word in sen.split():
        new_s += stemmer.stem(word) + " "
    return new_s
#Testing stemming()

print(reviews['Text'][5])
print("=> length before stemming {}".format(len(reviews['Text'][5].split())))
print()
print()
print(stemming(reviews['Text'][5]))
print("=> length after stemming {}".format(len(stemming(reviews['Text'][5]).split())))
#Stemming Text and Summary Coloumns
reviews['Text'] = reviews['Text'].progress_apply(stemming)
reviews['Summary'] = reviews['Summary'].progress_apply(stemming)
#This will be used later for vectorization, It will be a list of all document is corpus
preprocess_text = []
for sen in tqdm(reviews['Text'].values):
    preprocess_text.append(sen.strip())
preprocess_summary  = []
for sen in tqdm(reviews['Summary'].values):
    preprocess_summary.append(sen.strip())


from sklearn.feature_extraction.text import CountVectorizer #Class for generating Bag of words Vector
text_vectorizor = CountVectorizer()
summary_vectorizor = CountVectorizer()

# Creating Bag of Word vector for Text
text_sparse_matrix = text_vectorizor.fit_transform(preprocess_text)

# Creating Bag of Word vector for Summary
summary_sparse_matrix = summary_vectorizor.fit_transform(preprocess_summary)
print("Text vector")
print(text_vectorizor.get_feature_names()[:10]) #Some features of text Vector
print("Summary vector")
print(summary_vectorizor.get_feature_names()[:10]) #Some features of Summary vector
text_vector = text_sparse_matrix.toarray() #Converting text sparse matrix to numpy array

summary_vector = summary_sparse_matrix.toarray() # Converting summary sparse matrix to numpy array

print("Shape of text vector is {}".format(text_vector.shape))
print("Shape of summary vector is {}".format(summary_vector.shape))
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
# Standarization of Vector
std_text = StandardScaler()
standarized_text = std_text.fit_transform(text_vector)
print("shape of text_vector is {} and shape of standarized text is {}".format(text_vector.shape,standarized_text.shape))

std_summary = StandardScaler()
standarized_summary = std_summary.fit_transform(summary_vector)
print("shape of summary_vector is {} and shape of standarized summary is {}".format(summary_vector.shape,standarized_summary.shape))
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
#splitting training and test data 7:3
X_train, X_test, y_train, y_test = train_test_split(standarized_text, reviews['Score'], test_size=0.3, random_state=21)
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
#3 fold cross validation on train data to find the best K
neig = [i for i in range(1,len(X_train),2)]
cv_score = []
for i in neig:
    knn = KNeighborsClassifier(n_neighbors=i)
    score = cross_val_score(knn,X_train, y_train, cv=3, scoring='accuracy',n_jobs=-1)
    print(score)
    print(i)
    cv_score.append(score.mean())
cv_score.index(max(cv_score))
#Best K = 7
# Let's see accuracy on test data
from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,y_train)

#predicting test data
X_test_pred = knn.predict(X_test)
acc = accuracy_score(y_test ,X_test_pred, normalize=True)*float(100)
print("=> Accuracy on test data is: ",acc)
#splitting training and test data 7:3
X_train, X_test, y_train, y_test = train_test_split(standarized_summary, reviews['Score'], test_size=0.3, random_state=21)
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
#3 fold cross validation on train data to find the best K
neig = [i for i in range(1,22,2)]
cv_score = []
for i in neig:
    knn = KNeighborsClassifier(n_neighbors=i)
    score = cross_val_score(knn,X_train, y_train, cv=3, scoring='accuracy',n_jobs=-1)
    print(score)
    print(i)
    cv_score.append(score.mean())
print(cv_score.index(max(cv_score)),max(cv_score)*100)
#Best K = 13
# Let's see accuracy on test data
from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(X_train,y_train)

#predicting test data
X_test_pred = knn.predict(X_test)
acc = accuracy_score(y_test ,X_test_pred, normalize=True)*float(100)
print("=> Accuracy on test data is: ",acc)