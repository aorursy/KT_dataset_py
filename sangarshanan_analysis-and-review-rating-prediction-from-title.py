import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("../input/GrammarandProductReviews.csv")

data.head()
data.shape
len(data['brand'].unique())
len(data['categories'].unique())
sorted(data['reviews.rating'].unique())
Universalmusic = data.loc[data['brand'] == 'Universal Music',:]
ratings = list(Universalmusic['reviews.rating'])
print("Average Rating of UniversalMusic is:",sum(ratings)/len(ratings))
top = Universalmusic['reviews.rating'].value_counts().index.tolist()
value = Universalmusic['reviews.rating'].value_counts().values.tolist()
sns.barplot(top, value, alpha=0.8)
plt.xlabel('Rating of the product', fontsize=14)
plt.ylabel('Number of reviews with that given', fontsize=14)
plt.title("Rating for Universal Music", fontsize=16)
plt.show()
Lundberg = data.loc[data['brand'] == 'Lundberg',:]
ratings = list(Lundberg['reviews.rating'])
print("Average Rating of Lundberg is:",sum(ratings)/len(ratings))
junglebook = data[(data['brand'] == 'Disney') & (data['name'] == "The Jungle Book (blu-Ray/dvd + Digital)")]
top = junglebook['reviews.rating'].value_counts().index.tolist()
value = junglebook['reviews.rating'].value_counts().values.tolist()
sns.barplot(top, value, alpha=0.8)
plt.xlabel('Rating given for Jungle Book', fontsize=14)
plt.ylabel('Number of that rating given', fontsize=14)
plt.title("Rating for Jungle Book by disney", fontsize=16)
plt.show()
data['reviews.title'].unique()[:5]
totalreviews = list(data['reviews.text'])
length = []
for i in range(0,len(totalreviews)):
        totalreviews[i] = str(totalreviews[i])
        a = len(totalreviews[i].split(' '))
        length.append(a)

    
print("On average a review has about:", sum(length)/len(length),"words in them")
len(length)
ratings = list(data['reviews.rating'])
len(ratings)
dt = pd.DataFrame()
dt['length'] =  length
dt['ratings'] =  ratings
five_star = dt.loc[dt['ratings'] == 5,:]
five = sum(five_star['length'])/len(five_star['length'])
four_star = dt.loc[dt['ratings'] == 4,:]
four = sum(four_star['length'])/len(four_star['length'])
three_star = dt.loc[dt['ratings'] == 3,:]
three = sum(three_star['length'])/len(three_star['length'])
to_star = dt.loc[dt['ratings'] == 2,:]
to = sum(to_star['length'])/len(to_star['length'])
on_star = dt.loc[dt['ratings'] == 1,:]
on = sum(on_star['length'])/len(on_star['length'])
colors = ['gold', 'orange','yellowgreen', 'lightcoral', 'lightskyblue']
top = ['one','two','three','four','five']
value = [int(on), int(to),int(three),int(four),int(five)]
sns.barplot(top, value, alpha=0.8)
plt.xlabel('Rating of the product', fontsize=14)
plt.ylabel('Average number of words in the review', fontsize=14)
plt.title("Rating given vs Number of words used in review", fontsize=16)
plt.show()
f = data.loc[data['reviews.rating'] == 5,:]
ss = list(f['reviews.text'])
aa=[]
for i in range(0,len(ss)):
    ss[i] = str(ss[i])
    aa.append(ss[i].split(' '))
all_words = [j for i in aa for j in i]
all_words[:5]
import nltk
from nltk.corpus import stopwords
import string
exclude = set(string.punctuation)
for i in range(0,len(all_words)):
    all_words[i] = all_words[i].lower()
    all_words[i] = ''.join(ch for ch in all_words[i] if ch not in exclude)

stop = set(stopwords.words('english'))
stopwordsfree_words = [word for word in all_words if word not in stop]
from collections import Counter
counts = Counter(stopwordsfree_words)
counts.most_common(5)
f = data.loc[data['reviews.rating'] == 1,:]
ss = list(f['reviews.text'])
aa=[]
for i in range(0,len(ss)):
    ss[i] = str(ss[i])
    aa.append(ss[i].split(' '))
all_words = [j for i in aa for j in i]
all_words[:5]
exclude = set(string.punctuation)
for i in range(0,len(all_words)):
    all_words[i] = all_words[i].lower()
    all_words[i] = ''.join(ch for ch in all_words[i] if ch not in exclude)

stop = set(stopwords.words('english'))
stopwordsfree_words = [word for word in all_words if word not in stop]
from collections import Counter
counts = Counter(stopwordsfree_words)
counts.most_common(5)
df1 = data.replace(np.nan, 'Not Filled', regex=True)
X = list(df1['reviews.title'])
Y = list(df1['reviews.rating'])
(X[:5],Y[:5])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

from sklearn.feature_extraction.text import TfidfVectorizer
def tfidf_features(X_train, X_val, X_test):
    tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2),token_pattern='(\S+)')
    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_val = tfidf_vectorizer.transform(X_val)
    X_test = tfidf_vectorizer.transform(X_test)
    
    return X_train, X_val, X_test, tfidf_vectorizer.vocabulary_
X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(X_train, X_val, X_test)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
def train_classifier(X_train, y_train):    
    clf = OneVsRestClassifier(LogisticRegression())
    clf.fit(X_train, y_train)
    return clf
classifier_tfidf = train_classifier(X_train_tfidf, y_train)
y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
y_val_predicted_labels_tfidf  ==2
X_val[0]
y_val_predicted_labels_tfidf[0]
X_val[15]
y_val_predicted_labels_tfidf[15]
(X_val[101],y_val_predicted_labels_tfidf[101])
(X_val[68],y_val_predicted_labels_tfidf[68])
(X_val[42],y_val_predicted_labels_tfidf[42])
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
def print_evaluation_scores(y_val, predicted):
    print(accuracy_score(y_val, predicted))
    print(f1_score(y_val, predicted, average='weighted'))
print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)