# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
##### Phase 1 Cleaning:



    #reading from csv file

df_train=pd.read_csv('/kaggle/input/train.csv')

df_test=pd.read_csv('/kaggle/input/test.csv')



df_train.shape, df_test.shape



df_train.columns,df_test.columns



df_train=df_train.drop('Unnamed: 0',axis=1)

df_test=df_test.drop('Unnamed: 0',axis=1)



df_train.columns,df_test.columns



#removing stopwords, punctuations

import string

import re

import nltk

from nltk.tokenize import WordPunctTokenizer

from nltk.stem.porter import PorterStemmer

from nltk.tokenize import word_tokenize



stopword = nltk.corpus.stopwords.words('english')# All English Stopwords



# Function to clean data

ps = nltk.PorterStemmer()

def clean_text(text):

    text = "".join([word.lower() for word in text if word not in string.punctuation])

    text = re.sub(r"<.*>"," ",text)

    tokens = re.split('\W+', text)

    text = [ps.stem(word) for word in tokens if word not in stopword]

    return text



#stemming and lemmatization



wn = nltk.WordNetLemmatizer()



def lemmatizing(tokenized_text):

    text = [wn.lemmatize(word) for word in tokenized_text]

    return text





df_train['reviews'] = df_train['reviews'].apply(lambda x: re.sub(r"<.*>"," ",x))

df_train['reviews'] = df_train['reviews'].apply(lambda x: clean_text(x))

df_train['reviews_lemming'] = df_train['reviews'].apply(lambda x: lemmatizing(x))

df_train.head()



df_test['reviews'] = df_test['reviews'].apply(lambda x: re.sub(r"<.*>"," ",x))

df_test['reviews'] = df_test['reviews'].apply(lambda x: clean_text(x))

df_test['reviews_lemming'] = df_test['reviews'].apply(lambda x: lemmatizing(x))

df_test.head()



#feature selection





from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer(analyzer=clean_text)

X_train = vec.fit_transform(df_train['reviews'])



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

Y = le.fit_transform(df_train['labels'])



from sklearn.ensemble import ExtraTreesClassifier

tree_clf = ExtraTreesClassifier()

tree_clf.fit(X_train, Y)



importances = tree_clf.feature_importances_

feature_names = vec.get_feature_names()

feature_imp_dict = dict(zip(feature_names, importances))



from sklearn.feature_selection import SelectFromModel



model = SelectFromModel(tree_clf, prefit=True)

X_train_updated = model.transform(X_train)

print('Total features count', X_train.shape[1])

print('Selected features', X_train_updated.shape[1])



X_train_2 = vec.fit_transform(df_test['reviews'])



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

Y_2 = le.fit_transform(df_test['labels'])



from sklearn.ensemble import ExtraTreesClassifier

tree_clf = ExtraTreesClassifier()

tree_clf.fit(X_train_2, Y_2)



importances = tree_clf.feature_importances_

feature_names = vec.get_feature_names()

feature_imp_dict = dict(zip(feature_names, importances))



from sklearn.feature_selection import SelectFromModel



model = SelectFromModel(tree_clf, prefit=True)

X_train_updated_2 = model.transform(X_train_2)

print('Total features count', X_train_2.shape[1])

print('Selected features', X_train_updated_2.shape[1])



#### Phase 2 Exploration:



import operator

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import ExtraTreesClassifier



def print_features(df):

    tree_clf = ExtraTreesClassifier()

    vec2 = CountVectorizer(analyzer=clean_text) 

    X_trains_2 = vec2.fit_transform(df['reviews'])

    le = LabelEncoder()

    y = le.fit_transform(df['labels'])

    

    tree_clf.fit(X_trains_2,y)

    

    importances = tree_clf.feature_importances_

    feature_names = vec2.get_feature_names()

    feature_imp_dict = dict(zip(feature_names, importances))

    sorted_features = sorted(feature_imp_dict.items(),key=operator.itemgetter(1),reverse=True)

    indices = np.argsort(importances)[::-1]

    print("Feature ranking:")

    for f in range(20):

        print("Feature %d : %s (%f)" % (indices[f],sorted_features[f][0],sorted_features[f][1]))

    plt.figure(figsize = (20,20))

    plt.title("Feature Importances")

    plt.bar(range(100),importances[indices[:100]],color="r",align="center")

    plt.xticks(range(100),sorted_features[:100],rotation=90)

    plt.xlim([-1,100])

    plt.show()

    return()



print_features(df_train)



print_features(df_test)



df_unsup=pd.read_csv('/kaggle/input/unsup.csv')



df_unsup['reviews'] = df_unsup['reviews'].apply(lambda x: re.sub(r"<.*>"," ",x))



from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(analyzer=clean_text)

X = vectorizer.fit_transform(df_unsup.reviews)



# import modules

from sklearn.cluster import KMeans

# create an instance

kmeans = KMeans(n_clusters=2, random_state=0)

# fit the model

kmeans.fit(X)

# view the data labels after clustering

kmeans.labels_

# view the cluster centers

kmeans.cluster_centers_  #coordinate of centers





#### Phase 3 Visualization:





from wordcloud import WordCloud



df_train2=pd.read_csv('/kaggle/input/train.csv')



df_train2=df_train2.drop('Unnamed: 0',axis=1)

df_train2.shape



df_train2['reviews'] = df_train2['reviews'].apply(lambda x: re.sub(r"<.*>"," ",x))



neg_list = df_train2[df_train2["labels"] == 0]["reviews"].unique().tolist()

neg_list[:2]

pos_list = df_train2[df_train2["labels"] == 1]["reviews"].unique().tolist()

pos_list[:2]



#negative word cloud

neg = " ".join(neg_list)

neg[:100]

neg_wordcloud = WordCloud().generate(neg)

plt.figure()

plt.imshow(neg_wordcloud)

plt.show()



#positive word cloud

pos = " ".join(pos_list)

pos[:100]

pos_wordcloud = WordCloud().generate(neg)

plt.figure()

plt.imshow(pos_wordcloud)

plt.show()



df_train2['body_len'] = df_train2['reviews'].apply(lambda x: len(x) - x.count(" "))

bins = np.linspace(0, 200, 40)



plt.hist(df_train2[df_train2['labels']==0]['body_len'], bins, alpha=0.5, normed=True, label='neg')

plt.hist(df_train2[df_train2['labels']==1]['body_len'], bins, alpha=0.5, normed=True, label='pos')

plt.legend(loc='upper left')

plt.show()



#### Phase 4 Hypothesis testing :



# import the vectorizer

from sklearn.feature_extraction.text import CountVectorizer



# create an instance

count_vect = CountVectorizer(analyzer=clean_text)



# convert text to vectors

X = count_vect.fit_transform(df_train['reviews'])



# encode the target strings

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



y = le.fit_transform(df_train.labels)



#Naive Bayes

# import Nauve bayes classifier

from sklearn.naive_bayes import MultinomialNB

# split the data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

X_train.shape, X_test.shape



# fit the classifier model

clf = MultinomialNB()

clf.fit(X_train, y_train)

# predict the outcome for testing data

predictions = clf.predict(X_test)

predictions.shape





# check the accuracy of the model

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)

accuracy



from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=5)

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

accuracy





from sklearn.svm import SVC

clf = SVC()

# fit the classifier

clf.fit(X_train, y_train)

# predict the outcome for testing data

predictions = clf.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

accuracy

#### Phase 5 Model Building:



# convert text to vectors

X_2 = count_vect.fit_transform(df_test['reviews'])



#label encoding

y_2 = le.fit_transform(df_test.labels)



#splitting the data

X_train, X_test, y_train, y_test = train_test_split(X_2, y_2, test_size=0.25, random_state=0)





clf = RandomForestClassifier(n_estimators=5)

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

accuracy



clf = MultinomialNB()

clf.fit(X_train, y_train)

# predict the outcome for testing data

predictions = clf.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

accuracy





# import modules

from sklearn.cluster import KMeans

clust=2

# create an instance

kmeans = KMeans(n_clusters=clust, random_state=0)

# fit the model

kmeans.fit(X)

# Visualising the clusters

print("Top terms per cluster:")

order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

terms = vectorizer.get_feature_names()

for i in range(clust):

    print("Cluster %d:" % i),

    for vals in order_centroids[i, :10]:

        print(' %s' % terms[vals])