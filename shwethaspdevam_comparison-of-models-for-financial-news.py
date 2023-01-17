import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer

import nltk

import string

import re
def load_data():

    data = pd.read_csv('../input/sentiment-analysis-for-financial-news/all-data.csv', sep=',', encoding='latin-1',names = ["category","comment"])

    return data
tweet_df = load_data()

df=load_data()

tweet_df.head()



print(tweet_df.shape)

print("COLUMN NAMES" , tweet_df.columns)



print(tweet_df.info())
#TEXT VISUALIZATION 

sns.countplot(x="category",data=tweet_df)
#remove punctuations

def remove_punct(text):

    text  = "".join([char for char in text if char not in string.punctuation])

    text = re.sub('[0-9]+', '', text)

    return text



tweet_df['comment'] = tweet_df['comment'].apply(lambda x: remove_punct(x))

tweet_df.head(10)
#stopwords removal

import nltk

nltk.download('stopwords')

stopword = nltk.corpus.stopwords.words('english')

print(stopword)



from nltk.corpus import stopwords

", ".join(stopwords.words('english'))

STOPWORDS = set(stopwords.words('english'))

def remove_stopwords(text):

    """custom function to remove the stopwords"""

    return " ".join([word for word in str(text).split() if word not in STOPWORDS])



tweet_df["text_wo_stop"] = tweet_df["comment"].apply(lambda text: remove_stopwords(text))

tweet_df.head()



#remove 

%matplotlib inline

pd.set_option('display.max_colwidth', 100)



tweet_df.head(20)
#stemming and lemmatization

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def stem_words(text):

    return " ".join([stemmer.stem(word) for word in text.split()])

tweet_df["text_stemmed"] = tweet_df["text_wo_stop"].apply(lambda text: stem_words(text))

tweet_df.head()



#remove frequent words - countvectorization

from collections import Counter

cnt = Counter()

for text in tweet_df["text_stemmed"].values:

    for word in text.split():

        cnt[word] += 1

        

cnt.most_common(20)
FREQWORDS = set([w for (w, wc) in cnt.most_common(10)])

def remove_freqwords(text):

    """custom function to remove the frequent words"""

    return " ".join([word for word in str(text).split() if word not in FREQWORDS])



tweet_df["text__stopfreq"] = tweet_df["text_stemmed"].apply(lambda text: remove_freqwords(text))

tweet_df.head()


from nltk.corpus import wordnet

from nltk.stem import WordNetLemmatizer



lemmatizer = WordNetLemmatizer()

wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}

def lemmatize_words(text):

    pos_tagged_text = nltk.pos_tag(text.split())

    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])



tweet_df["text_lemmatized"] = tweet_df["text__stopfreq"].apply(lambda text: lemmatize_words(text))

tweet_df.head()
#drop the columns

tweet_df=tweet_df.drop(["text_stemmed","text__stopfreq"],axis=1)
#label encoding

from sklearn.preprocessing import LabelEncoder

tweet_df['encoded_category'] = LabelEncoder().fit_transform(tweet_df['category'])

tweet_df[["category", "encoded_category"]] 
def clean_review(text):

    clean_text = []

    for w in word_tokenize(text):

        if w.lower() not in stop:

            pos = pos_tag([w])

            new_w = lemmatizer.lemmatize(w, pos=get_simple_pos(pos[0][1]))

            clean_text.append(new_w)

    return clean_text



def join_text(text):

    return " ".join(text)
tweet_df=tweet_df.drop(["category","text_wo_stop","comment"],axis=1)
tweet_df.head(10)
from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import BernoulliNB

from sklearn.naive_bayes import MultinomialNB

from sklearn.naive_bayes import MultinomialNB



from sklearn.ensemble import GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import RandomForestClassifier


from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import TfidfTransformer



from sklearn.metrics import accuracy_score, confusion_matrix,classification_report



x_train,x_test,y_train,y_test = train_test_split(tweet_df.text_lemmatized,tweet_df.encoded_category,test_size = 0.3 , random_state = 0)



x_train.shape,x_test.shape,y_train.shape,y_test.shape
pipe = Pipeline([('tfidf', TfidfVectorizer()),

                 ('model', LinearSVC())])



model = pipe.fit(x_train, y_train)

prediction = model.predict(x_test)

print("MODEL - LINEAR SVC")

print("accuracy score: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

print(confusion_matrix(y_test, prediction))

print(classification_report(y_test, prediction))
pipe = Pipeline([('vect', CountVectorizer()),

                 ('tfidf', TfidfTransformer()),

                 ('model', LogisticRegression())])



model = pipe.fit(x_train, y_train)

prediction = model.predict(x_test)

print("MODEL - LOGISTIC REGRESSION")

print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

print(confusion_matrix(y_test, prediction))

print(classification_report(y_test, prediction))
pipe = Pipeline([('vect', CountVectorizer()),

                 ('tfidf', TfidfTransformer()),

                 ('model', MultinomialNB())])



model = pipe.fit(x_train, y_train)

prediction = model.predict(x_test)

print("MULTINOMIAL NAIVE BAYES")

print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

print(confusion_matrix(y_test, prediction))

print(classification_report(y_test, prediction))
pipe = Pipeline([('vect', CountVectorizer()),

                 ('tfidf', TfidfTransformer()),

                 ('model', BernoulliNB())])

model = pipe.fit(x_train, y_train)

prediction = model.predict(x_test)

print("BERNOULLIS NAIVE BAYES")

print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

print(confusion_matrix(y_test, prediction))

print(classification_report(y_test, prediction))
pipe = Pipeline([('vect', CountVectorizer()),

                 ('tfidf', TfidfTransformer()),

                 ('model', GradientBoostingClassifier(loss = 'deviance',

                                                   learning_rate = 0.01,

                                                   n_estimators = 10,

                                                   max_depth = 5,

                                                   random_state=55))])

model = pipe.fit(x_train, y_train)

prediction = model.predict(x_test)

print("GRADIENT BOOST")

print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

print(confusion_matrix(y_test, prediction))

print(classification_report(y_test, prediction))
pipe = Pipeline([('vect', CountVectorizer()),

                 ('tfidf', TfidfTransformer()),

                 ('model', XGBClassifier(loss = 'deviance',

                                                   learning_rate = 0.01,

                                                   n_estimators = 10,

                                                   max_depth = 5,

                                                   random_state=2020))])



model = pipe.fit(x_train, y_train)

prediction = model.predict(x_test)

print("XGBOOST")

print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

print(confusion_matrix(y_test, prediction))

print(classification_report(y_test, prediction))
pipe = Pipeline([('vect', CountVectorizer()),

                 ('tfidf', TfidfTransformer()),

                 ('model', DecisionTreeClassifier(criterion= 'entropy',

                                           max_depth = 10, 

                                           splitter='best', 

                                           random_state=2020))])

model = pipe.fit(x_train, y_train)

prediction = model.predict(x_test)

print("DECISION TREE")

print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

print(confusion_matrix(y_test, prediction))

print(classification_report(y_test, prediction))
pipe = Pipeline([('vect', CountVectorizer()),

                 ('tfidf', TfidfTransformer()),

                 ('model', KNeighborsClassifier(n_neighbors = 10,weights = 'distance',algorithm = 'brute'))])

model = pipe.fit(x_train, y_train)

prediction = model.predict(x_test)

print("K NEAREST NEIGHBOR")

print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

print(confusion_matrix(y_test, prediction))

print(classification_report(y_test, prediction))