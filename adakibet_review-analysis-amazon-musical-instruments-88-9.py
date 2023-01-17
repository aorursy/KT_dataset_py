import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import spacy

from sklearn.model_selection import train_test_split

from nltk.tokenize import RegexpTokenizer

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

import string

from string import punctuation

import collections

from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/amazon-music-reviews/Musical_instruments_reviews.csv')

data.head()
data.shape
data = data.drop(['reviewerID', 'asin','reviewerName', 'unixReviewTime', 'helpful'], axis = 1)

data.head()
data.isna().sum()
data['reviewText'].fillna('Null', inplace = True)

data.isna().any()
data['overall'].unique()
def rating(overall):

    if (int(overall <= 3)):

        return 0

    else:

        return 1

        

data['rating'] = data['overall'].apply(rating)

data = data.drop(['overall'], axis = 1)

data.head()
data.rating.unique()
sns.countplot(data.rating)
data['reviewText'] = data['reviewText'] + data['summary']

data = data.drop(['summary'], axis = 1)

data.head()
x = pd.DataFrame(data['reviewText'])

y = pd.DataFrame(data.rating)

data.reviewText = data.reviewText.astype('str')
nlp = spacy.load("en_core_web_sm")

tokenizer = RegexpTokenizer(r'\w+')

lemmatizer = WordNetLemmatizer()

stop = set(stopwords.words('english'))

punctuation = list(string.punctuation)

stop.update(punctuation)



            

def furnished(text):

    final_text = []

    for i in text.split():

        if i.lower() not in stop:

            word = lemmatizer.lemmatize(i)

            final_text.append(word.lower())

    return " ".join(final_text)





            

data.reviewText = data.reviewText.apply(furnished)
data.reviewText.describe()
for i in data.reviewText:

    global text

    text = i.split()

    

counter=Counter(text)

most=counter.most_common()



x, y= [], []

for word,count in most[:20]:

    if (word not in stop):

        x.append(word)

        y.append(count)

plt.figure(figsize = (10,10))     

sns.barplot(x=y,y=x)
data.head()
x_train,x_test,y_train,y_test = train_test_split(data.reviewText,data.rating,test_size = 0.2 , random_state = 0)
#bow

cv=CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3))

bow_x_train = cv.fit_transform(x_train)

bow_x_test = cv.transform(x_test)



print('bow_x_train:',bow_x_train.shape)

print('bow_x_test:',bow_x_test.shape)
#tf-idf 

tv=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,3))



tfidf_x_train =tv.fit_transform(x_train)

tfidf_x_test =tv.transform(x_test)



print('tfidf_x_train:',tfidf_x_train.shape)

print('tfidf_x_test:',tfidf_x_test.shape)

#Naive Bayes

nb = MultinomialNB()



#fit

bow = nb.fit(bow_x_train, y_train)

tfidf = nb.fit(tfidf_x_train, y_train)



#predict

bow_predict = nb.predict(bow_x_test)

tfidf_predict = nb.predict(tfidf_x_test)



#accuracy

nb_bow = accuracy_score(y_test, bow_predict)

nb_tfidf = accuracy_score(y_test,tfidf_predict)



print('nb bow accuracy:', nb_bow)

print('tfidf accuracy:', nb_tfidf)
#random forest

rf = RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0)



#fit

bow = rf.fit(bow_x_train, y_train)

tfidf = rf.fit(tfidf_x_train, y_train)



#predict

bow_predict = rf.predict(bow_x_test)

tfidf_predict = rf.predict(tfidf_x_test)



#accuracy

rf_bow = accuracy_score(y_test, bow_predict)

rf_tfidf = accuracy_score(y_test,tfidf_predict)



print('rf bow accuracy:', rf_bow)

print('rf tfidf accuracy:', rf_tfidf)
#Linear SVC

ls =  LinearSVC()



#fit

bow = ls.fit(bow_x_train, y_train)

tfidf = ls.fit(tfidf_x_train, y_train)



#predict

bow_predict = ls.predict(bow_x_test)

tfidf_predict = ls.predict(tfidf_x_test)



#accuracy

ls_bow = accuracy_score(y_test, bow_predict)

ls_tfidf = accuracy_score(y_test,tfidf_predict)



print('ls bow accuracy:', ls_bow)

print('ls tfidf accuracy:', ls_tfidf)
#lr

lr = LogisticRegression(random_state=0)



#fit

bow = lr.fit(bow_x_train, y_train)

tfidf = lr.fit(tfidf_x_train, y_train)



#predict

bow_predict = lr.predict(bow_x_test)

tfidf_predict = lr.predict(tfidf_x_test)



#accuracy

lr_bow = accuracy_score(y_test, bow_predict)

lr_tfidf = accuracy_score(y_test,tfidf_predict)



print('lr bow accuracy:', lr_bow)

print('lr tfidf accuracy:', lr_tfidf)
data = {'accuracy': [nb_bow * 100, nb_tfidf * 100, rf_bow * 100, rf_tfidf * 100, lr_bow * 100, lr_tfidf * 100, ls_tfidf * 100, ls_bow * 100],

                   'model': ['naive bayes bow', 'naive bayes tfidf', 'random forest bow', 'random forest tfidf', 

                                'logit bow', 'logit tfidf', 'SVM bow', 'SVM tfidf']}

df = pd.DataFrame(data, columns = ['accuracy', 'model'])

df.head(8)
plt.figure(figsize = (17,7))

sns.barplot(y = df.accuracy, x = df.model)