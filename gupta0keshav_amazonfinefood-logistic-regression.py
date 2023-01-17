#importing all the modules that might be used later on

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")







import pandas as pd

import numpy as np

import nltk

import string

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.metrics import roc_curve, auc

from nltk.stem.snowball import SnowballStemmer



import re

import string

from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer

from nltk.stem.wordnet import WordNetLemmatizer



from gensim.models import Word2Vec

from gensim.models import KeyedVectors

import pickle



from tqdm import tqdm

import os
df = pd.read_csv('../input/amazon-fine-food-reviews/Reviews.csv')
print(df.shape)

df.head()
df = df.drop_duplicates(subset = {'UserId', 'Time', 'Text'})

df.shape
df = df.loc[lambda df: df['Score'] != 3]

print(df.shape)

df['Score'].unique()
def scorer(x):

    if x > 3:

        return 1

    return 0



scores = df['Score']

scores_binary = scores.map(scorer)

df['Score'] = scores_binary

df['Score'].unique()
df = df[df.HelpfulnessDenominator >= df.HelpfulnessNumerator]

df.shape
df = df.sort_values('Time', axis = 0, inplace = False, ascending = True)

texts = df['Text']

texts.head()
# removing all the url from the text

def remove_url(s):

  return re.sub(r'http\S+', '', s)

test = "hello https://www.google.com/ world"

print(remove_url(test))

texts = texts.map(remove_url)
# removing all the tags from the text

def remove_tag(s):

  return re.sub(r'<.*?>', ' ', s)

test = "<p> hello world </p>"

print(remove_tag(test))

texts = texts.map(remove_tag)
#converting strings into only lowercase.

def lower_words(s):

   return s.lower()

test = "HELLO world"

print(lower_words(test))

texts = texts.map(lower_words)
# decontracting all contracted words



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

test = "it'll take time to complete this notebook"

print(decontracted(test))

texts = texts.map(decontracted)
# deleting all the words with numbers in them



def remove_words_with_nums(s):

  return re.sub(r"\S*\d\S*", "", s)

test = "hello0 world"

print(remove_words_with_nums(test))

texts = texts.map(remove_words_with_nums)
# deleting words with special character in them



def remove_special_character(s):

  return re.sub('[^A-Za-z0-9]+', ' ', s)

test = "hello-world"

print(remove_special_character(test))

texts = texts.map(remove_special_character)
# defining the set of stop words according to our problem basically we'll remove all the negations from the pre-defined set of stopwords

# i removed some stopwords from basic english language stopwords set, the removed elements are related to negations that generally express a negative emotion



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

            's', 't', 'can', 'will', 'just', 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \

            've', 'y', 'ain', 'aren'])
def remove_stopword(s):

    res = ' '.join([word for word in s.split(' ') if word not in stopwords])

    return res



test = "hello my world"

print(remove_stopword(test))

texts = texts.map(remove_stopword)
test_texts = texts[:10000:].copy()

test_texts.shape
#Stemming (Snowball Stemmer)

stemmer = SnowballStemmer('english')

def stemming(s):

    res = ' '.join([stemmer.stem(word) for word in s.split(' ')])

    return res

test = "running and walking"

print(stemming(test))

stemmed_texts = test_texts.map(stemming)
lemmatizer = WordNetLemmatizer()

def lemmatization(s):

    res = ' '.join([lemmatizer.lemmatize(word) for word in s.split(' ')])

    return res

test = "was running"

lemmatized_texts = test_texts.map(lemmatization)

print(lemmatization(test))
#texts

X = test_texts

y = df['Score'][:10000:].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

count_vect = CountVectorizer()

X_train = count_vect.fit_transform(X_train)

X_test = count_vect.transform(X_test)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)

from sklearn.metrics import roc_auc_score

predictions = model.predict(X_test)

print('AUC: ', roc_auc_score(y_test, predictions))
#texts

X = stemmed_texts

y = df['Score'][:10000:].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

count_vect = CountVectorizer()

X_train = count_vect.fit_transform(X_train)

X_test = count_vect.transform(X_test)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)

from sklearn.metrics import roc_auc_score

predictions = model.predict(X_test)

print('AUC: ', roc_auc_score(y_test, predictions))
#texts

X = lemmatized_texts

y = df['Score'][:10000:].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

count_vect = CountVectorizer()

X_train = count_vect.fit_transform(X_train)

X_test = count_vect.transform(X_test)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)

from sklearn.metrics import roc_auc_score

predictions = model.predict(X_test)

print('AUC: ', roc_auc_score(y_test, predictions))
text = texts[:100000:]

from nltk.stem import PorterStemmer

st = PorterStemmer()

stemmed_data = []

for review in text:

    stemmed_data.append(st.stem(review))

print('Done')
X = stemmed_data

y = df['Score'][:100000:].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

count_vect = CountVectorizer()

train_bow = count_vect.fit_transform(X_train)

test_bow = count_vect.transform(X_test)

print(train_bow.shape)
c_dist = []

for x in range(-2, 3):

    mul = 10 ** (-x + 1)

    center = 10 ** x

    for y in range(-5,6):

        c_dist.append(y/mul + center)

print(c_dist)

max_iter = []

for x in range (75, 130, 5):

    max_iter.append(x)

print(max_iter)

param_dist = {'C' : c_dist, 'max_iter' : max_iter}

print(param_dist)
from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LogisticRegression

random_model = RandomizedSearchCV(LogisticRegression(class_weight='balanced', penalty='l1'), param_dist, cv = 10, scoring = 'accuracy', n_jobs=-1)

random_model.fit(train_bow, y_train)
print(random_model.best_estimator_)

pred = random_model.predict(test_bow)

from sklearn.metrics import accuracy_score

print('Accuracy :', accuracy_score(y_test, pred)*100)
from sklearn.metrics import classification_report

print(classification_report(y_test,pred))
from sklearn.metrics import confusion_matrix

import seaborn as sns

confusion = confusion_matrix(y_test , pred)

print(confusion)

df_cm = pd.DataFrame(confusion , index = ['Negative','Positive'])

sns.heatmap(df_cm ,annot = True)

plt.xticks([0.5,1.5],['Negative','Positive'],rotation = 45)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()



from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LogisticRegression

random_model = RandomizedSearchCV(LogisticRegression(class_weight='balanced', penalty='l2'), param_dist, cv = 10, scoring = 'accuracy', n_jobs=-1)

random_model.fit(train_bow, y_train)
print(random_model.best_estimator_)

pred = random_model.predict(test_bow)

from sklearn.metrics import accuracy_score

print('Accuracy :', accuracy_score(y_test, pred)*100)
from sklearn.metrics import classification_report

print(classification_report(y_test,pred))
from sklearn.metrics import confusion_matrix

import seaborn as sns

confusion = confusion_matrix(y_test , pred)

print(confusion)

df_cm = pd.DataFrame(confusion , index = ['Negative','Positive'])

sns.heatmap(df_cm ,annot = True)

plt.xticks([0.5,1.5],['Negative','Positive'],rotation = 45)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()
count_vect = CountVectorizer(ngram_range = (1, 2))

train_bow = count_vect.fit_transform(X_train)

test_bow = count_vect.transform(X_test)

print(train_bow.shape)
from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LogisticRegression

random_model = RandomizedSearchCV(LogisticRegression(class_weight='balanced', penalty='l1'), param_dist, cv = 10, scoring = 'accuracy', n_jobs=-1)

random_model.fit(train_bow, y_train)
print(random_model.best_estimator_)

pred = random_model.predict(test_bow)

from sklearn.metrics import accuracy_score

print('Accuracy :', accuracy_score(y_test, pred)*100)
from sklearn.metrics import classification_report

print(classification_report(y_test,pred))
from sklearn.metrics import confusion_matrix

import seaborn as sns

confusion = confusion_matrix(y_test , pred)

print(confusion)

df_cm = pd.DataFrame(confusion , index = ['Negative','Positive'])

sns.heatmap(df_cm ,annot = True)

plt.xticks([0.5,1.5],['Negative','Positive'],rotation = 45)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()
from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LogisticRegression

random_model = RandomizedSearchCV(LogisticRegression(class_weight='balanced', penalty='l2'), param_dist, cv = 10, scoring = 'accuracy', n_jobs=-1)

random_model.fit(train_bow, y_train)
print(random_model.best_estimator_)

pred = random_model.predict(test_bow)

from sklearn.metrics import accuracy_score

print('Accuracy :', accuracy_score(y_test, pred)*100)
from sklearn.metrics import classification_report

print(classification_report(y_test,pred))
from sklearn.metrics import confusion_matrix

import seaborn as sns

confusion = confusion_matrix(y_test , pred)

print(confusion)

df_cm = pd.DataFrame(confusion , index = ['Negative','Positive'])

sns.heatmap(df_cm ,annot = True)

plt.xticks([0.5,1.5],['Negative','Positive'],rotation = 45)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer()

train_tfidf = tfidf_vect.fit_transform(X_train)

test_tfidf = tfidf_vect.transform(X_test)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler(with_mean = False)

x_train = sc.fit_transform(train_tfidf)

x_test = sc.transform(test_tfidf)
c_dist = []

for x in range(-2, 3):

    mul = 10 ** (-x + 1)

    center = 10 ** x

    for y in range(-5,6):

        c_dist.append(y/mul + center)

print(c_dist)

max_iter = []

for x in range (75, 130, 5):

    max_iter.append(x)

print(max_iter)

param_dist = {'C' : c_dist, 'max_iter' : max_iter}

print(param_dist)
from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LogisticRegression

random_model = RandomizedSearchCV(LogisticRegression(class_weight='balanced', penalty='l1'), param_dist, cv = 10, scoring = 'accuracy', n_jobs=-1)

random_model.fit(x_train, y_train)
print(random_model.best_estimator_)

pred = random_model.predict(x_test)

from sklearn.metrics import accuracy_score

print('Accuracy :', accuracy_score(y_test, pred)*100)
from sklearn.metrics import classification_report

print(classification_report(y_test,pred))
from sklearn.metrics import confusion_matrix

import seaborn as sns

confusion = confusion_matrix(y_test , pred)

print(confusion)

df_cm = pd.DataFrame(confusion , index = ['Negative','Positive'])

sns.heatmap(df_cm ,annot = True)

plt.xticks([0.5,1.5],['Negative','Positive'],rotation = 45)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()
from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LogisticRegression

random_model = RandomizedSearchCV(LogisticRegression(class_weight='balanced', penalty='l2'), param_dist, cv = 10, scoring = 'accuracy', n_jobs=-1)

random_model.fit(train_tfidf, y_train)
print(random_model.best_estimator_)

pred = random_model.predict(test_tfidf)

from sklearn.metrics import accuracy_score

print('Accuracy :', accuracy_score(y_test, pred)*100)
from sklearn.metrics import classification_report

print(classification_report(y_test,pred))
from sklearn.metrics import confusion_matrix

import seaborn as sns

confusion = confusion_matrix(y_test , pred)

print(confusion)

df_cm = pd.DataFrame(confusion , index = ['Negative','Positive'])

sns.heatmap(df_cm ,annot = True)

plt.xticks([0.5,1.5],['Negative','Positive'],rotation = 45)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()
tfidf_vect = TfidfVectorizer(ngram_range=(1,2))

train_tfidf = tfidf_vect.fit_transform(X_train)

test_tfidf = tfidf_vect.transform(X_test)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler(with_mean = False)

x_train = sc.fit_transform(train_tfidf)

x_test = sc.transform(test_tfidf)
from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LogisticRegression

random_model = RandomizedSearchCV(LogisticRegression(class_weight='balanced', penalty='l1'), param_dist, cv = 10, scoring = 'accuracy', n_jobs=-1)

random_model.fit(x_train, y_train)
print(random_model.best_estimator_)

pred = random_model.predict(x_test)

from sklearn.metrics import accuracy_score

print('Accuracy :', accuracy_score(y_test, pred)*100)
from sklearn.metrics import classification_report

print(classification_report(y_test,pred))
from sklearn.metrics import confusion_matrix

import seaborn as sns

confusion = confusion_matrix(y_test , pred)

print(confusion)

df_cm = pd.DataFrame(confusion , index = ['Negative','Positive'])

sns.heatmap(df_cm ,annot = True)

plt.xticks([0.5,1.5],['Negative','Positive'],rotation = 45)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()
from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LogisticRegression

random_model = RandomizedSearchCV(LogisticRegression(class_weight='balanced', penalty='l2'), param_dist, cv = 10, scoring = 'accuracy', n_jobs=-1)

random_model.fit(x_train, y_train)
print(random_model.best_estimator_)

pred = random_model.predict(x_test)

from sklearn.metrics import accuracy_score

print('Accuracy :', accuracy_score(y_test, pred)*100)
from sklearn.metrics import classification_report

print(classification_report(y_test,pred))
from sklearn.metrics import confusion_matrix

import seaborn as sns

confusion = confusion_matrix(y_test , pred)

print(confusion)

df_cm = pd.DataFrame(confusion , index = ['Negative','Positive'])

sns.heatmap(df_cm ,annot = True)

plt.xticks([0.5,1.5],['Negative','Positive'],rotation = 45)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()
list_of_sent_train = []

for i in X_train:

    sent = []

    for word in i.split():

        sent.append(word)

    list_of_sent_train.append(sent)
from gensim.models import Word2Vec

w2v_model = Word2Vec(list_of_sent_train,min_count = 5,size = 50,workers = 4)

sent_vectors_train = []

for sent in list_of_sent_train:

    sent_vec = np.zeros(50)

    cnt_word = 0

    for word in sent:

        try:

            vec = w2v_model.wv[word]

            sent_vec += vec

            cnt_word += 1

        except:

            pass

    sent_vec /= cnt_word

    sent_vectors_train.append(sent_vec)

print(len(sent_vectors_train))
list_of_sent_test = []

for i in X_test:

    sent = []

    for word in i.split():

        sent.append(word)

    list_of_sent_test.append(sent)
import warnings

warnings.filterwarnings("ignore")

from gensim.models import Word2Vec

w2v_model = Word2Vec(list_of_sent_test,min_count = 5,size = 50,workers = 4)

sent_vectors_test = []

for sent in list_of_sent_test:

    sent_vec = np.zeros(50)

    cnt_word = 0

    for word in sent:

        try:

            vec = w2v_model.wv[word]

            sent_vec += vec

            cnt_word += 1

        except:

            pass

    sent_vec /= cnt_word

    sent_vectors_test.append(sent_vec)

print(len(sent_vectors_test))
np.where(np.isnan(sent_vectors_test))
sent_vectors_train = pd.DataFrame(sent_vectors_train)

sent_vectors_test = pd.DataFrame(sent_vectors_test)

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

imputer = imputer.fit(sent_vectors_train)

sent_vectors_train = imputer.transform(sent_vectors_train)

sent_vectors_test = imputer.transform(sent_vectors_test)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

w2v_train = sc.fit_transform(sent_vectors_train)

w2v_test = sc.transform(sent_vectors_test)

from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LogisticRegression

random_model = RandomizedSearchCV(LogisticRegression(class_weight='balanced', penalty='l1'), param_dist, cv = 10, scoring = 'accuracy', n_jobs=-1)

random_model.fit(w2v_train, y_train)
print(random_model.best_estimator_)

pred = random_model.predict(w2v_test)

from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test,pred)

print('accuracy is',acc*100)
from sklearn.metrics import classification_report

print(classification_report(y_test,pred))
from sklearn.metrics import confusion_matrix

import seaborn as sns

confusion = confusion_matrix(y_test , pred)

print(confusion)

df_cm = pd.DataFrame(confusion , index = ['Negative','Positive'])

sns.heatmap(df_cm ,annot = True)

plt.xticks([0.5,1.5],['Negative','Positive'],rotation = 45)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

w2v_train = sc.fit_transform(sent_vectors_train)

w2v_test = sc.transform(sent_vectors_test)

from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LogisticRegression

random_model = RandomizedSearchCV(LogisticRegression(class_weight='balanced', penalty='l2'), param_dist, cv = 10, scoring = 'accuracy', n_jobs=-1)

random_model.fit(w2v_train, y_train)
print(random_model.best_estimator_)

pred = random_model.predict(w2v_test)

from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test,pred)

print('accuracy is',acc*100)
from sklearn.metrics import classification_report

print(classification_report(y_test,pred))
from sklearn.metrics import confusion_matrix

import seaborn as sns

confusion = confusion_matrix(y_test , pred)

print(confusion)

df_cm = pd.DataFrame(confusion , index = ['Negative','Positive'])

sns.heatmap(df_cm ,annot = True)

plt.xticks([0.5,1.5],['Negative','Positive'],rotation = 45)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()
tfidf_vect = TfidfVectorizer()

train_tfidf = tfidf_vect.fit_transform(X_train)

test_tfidf = tfidf_vect.transform(X_test)
tf_idf_feat = tfidf_vect.get_feature_names()

tfidf_sent_vec_train = []

row = 0

w2v_model = Word2Vec(list_of_sent_train,min_count = 5,size = 50,workers = 4)

for sent in list_of_sent_train:

    sent_vec = np.zeros(50)

    weight_sum = 0

    for word in sent:

        try:

            vec = w2v_model.wv[word]

            tfidf = train_tfidf[row,tf_idf_feat.index(word)]

            sent_vec += (vec*tfidf)

            weight_sum += tfidf

        except:

            pass

    sent_vec/= weight_sum

    tfidf_sent_vec_train.append(sent_vec)

    row += 1
tf_idf_feat = tfidf_vect.get_feature_names()

tfidf_sent_vec_test = []

row = 0

w2v_model = Word2Vec(list_of_sent_test,min_count = 5,size = 50,workers = 4)

for sent in list_of_sent_test:

    sent_vec = np.zeros(50)

    weight_sum = 0

    for word in sent:

        try:

            vec = w2v_model.wv[word]

            tfidf = test_tfidf[row,tf_idf_feat.index(word)]

            sent_vec += (vec*tfidf)

            weight_sum += tfidf

        except:

            pass

    sent_vec/= weight_sum

    tfidf_sent_vec_test.append(sent_vec)

    row += 1
np.where(np.isnan(tfidf_sent_vec_train))
tfidf_sent_vec_train = pd.DataFrame(tfidf_sent_vec_train)

tfidf_sent_vec_test = pd.DataFrame(tfidf_sent_vec_test)

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

imputer = imputer.fit(tfidf_sent_vec_train)

tfidf_sent_vec_train = imputer.transform(tfidf_sent_vec_train)

tfidf_sent_vec_test = imputer.transform(tfidf_sent_vec_test)
sc =  StandardScaler()

tfidf_w2v_train = sc.fit_transform(tfidf_sent_vec_train)

tfidf_w2v_test = sc.transform(tfidf_sent_vec_test)

from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LogisticRegression

random_model = RandomizedSearchCV(LogisticRegression(class_weight='balanced', penalty='l1'), param_dist, cv = 10, scoring = 'accuracy', n_jobs=-1)

random_model.fit(tfidf_w2v_train, y_train)
from sklearn.metrics import accuracy_score

print(random_model.best_estimator_)

pred = random_model.predict(tfidf_w2v_test)

acc = accuracy_score(y_test,pred)

print('accuracy is',acc*100)
from sklearn.metrics import classification_report

print(classification_report(y_test,pred))
from sklearn.metrics import confusion_matrix

import seaborn as sns

confusion = confusion_matrix(y_test , pred)

print(confusion)

df_cm = pd.DataFrame(confusion , index = ['Negative','Positive'])

sns.heatmap(df_cm ,annot = True)

plt.xticks([0.5,1.5],['Negative','Positive'],rotation = 45)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()
sc =  StandardScaler()

tfidf_w2v_train = sc.fit_transform(tfidf_sent_vec_train)

tfidf_w2v_test = sc.transform(tfidf_sent_vec_test)

from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LogisticRegression

random_model = RandomizedSearchCV(LogisticRegression(class_weight='balanced', penalty='l2'), param_dist, cv = 10, scoring = 'accuracy', n_jobs=-1)

random_model.fit(tfidf_w2v_train, y_train)
from sklearn.metrics import accuracy_score

print(random_model.best_estimator_)

pred = random_model.predict(tfidf_w2v_test)

acc = accuracy_score(y_test,pred)

print('accuracy is',acc*100)
from sklearn.metrics import classification_report

print(classification_report(y_test,pred))
from sklearn.metrics import confusion_matrix

import seaborn as sns

confusion = confusion_matrix(y_test , pred)

print(confusion)

df_cm = pd.DataFrame(confusion , index = ['Negative','Positive'])

sns.heatmap(df_cm ,annot = True)

plt.xticks([0.5,1.5],['Negative','Positive'],rotation = 45)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()