import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('../input/spam-text-message-classification/SPAM text message 20170820 - Data.csv')

#stopword_list = [k.strip() for k in open("E:/MaLearning/souhu/stopwords.txt", encoding='utf8').readlines() if k.strip() != '']

from nltk.corpus import stopwords

stopword_list = stopwords.words('english')
from pylab import *

warnings.filterwarnings("ignore") 

%matplotlib inline

%config InlineBackend.figure_format = 'svg'

sns.set_style('white') 
import itertools

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
data.head()
data["Category"] = data["Category"].map({'ham': 0,'spam':1})
data.head()
from nltk.stem import WordNetLemmatizer 

from nltk.tokenize import word_tokenize

import re

description_list = []

for article in data["Message"]:

    article = re.sub("[^a-zA-Z]"," ",article)

    article = article.lower()   # low case letter

    article = word_tokenize(article)

    lemma = WordNetLemmatizer()

    article = [ lemma.lemmatize(word) for word in article]

    article = " ".join(article)

    description_list.append(article) #we hide all word one section

    

    

def text_replace(text):

    '''some text cleaning method'''

    text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text
from sklearn.feature_extraction.text import CountVectorizer 

count_vectorizer = CountVectorizer(max_features = 100, stop_words = "english")

sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()

tokens = count_vectorizer.get_feature_names()
print(type(sparce_matrix))

sparce_matrix = pd.DataFrame(sparce_matrix, columns=tokens)

sparce_matrix.head()
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features = 100)

tfidfmatrix = vectorizer.fit_transform(description_list)

cname = vectorizer.get_feature_names()

tfidfmatrix = pd.DataFrame(tfidfmatrix.toarray(),columns=cname)

tfidfmatrix.head()
tfidfmatrix.columns
count_vectorizer = CountVectorizer(max_features = 100, stop_words = "english",ngram_range=(2, 2),)

sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()

tokens = count_vectorizer.get_feature_names()

gram2 = pd.DataFrame(sparce_matrix, columns=tokens)

gram2.head()
from sklearn.model_selection import train_test_split

y = data.iloc[:,0].values   

x = sparce_matrix

tfidfx = tfidfmatrix



x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 2019)

tf_x_train, tf_x_test, tf_y_train, tf_y_test = train_test_split(tfidfmatrix ,y,

                                                                test_size = 0.3,

                                                                random_state = 2019)



gm_x_train, gm_x_test, gm_y_train, gm_y_test = train_test_split(gram2 ,y,

                                                                test_size = 0.3,

                                                                random_state = 2019)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score 
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train, y_train)

print('CountVectorizer Accuracy Score',nb.score(x_test,y_test))

nb.fit(tf_x_train, tf_y_train)

print('TF-IDF Vectorizer Accuracy Score',nb.score(tf_x_test,tf_y_test))

nb.fit(gm_x_train, gm_y_train)

print('bi-gram Vectorizer Accuracy Score',nb.score(gm_x_test,gm_y_test))
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

nb.fit(x_train, y_train)

print('CountVectorizer Accuracy Score',nb.score(x_test,y_test))

nb.fit(tf_x_train, tf_y_train)

print('TF-IDF Vectorizer Accuracy Score',nb.score(tf_x_test,tf_y_test))

nb.fit(gm_x_train, gm_y_train)

print('bi-gram Vectorizer Accuracy Score',nb.score(gm_x_test,gm_y_test))
from sklearn.naive_bayes import BernoulliNB

nb = BernoulliNB()

nb.fit(x_train, y_train)

print('CountVectorizer Accuracy Score',nb.score(x_test,y_test))

nb.fit(tf_x_train, tf_y_train)

print('TF-IDF Vectorizer Accuracy Score',nb.score(tf_x_test,tf_y_test))

nb.fit(gm_x_train, gm_y_train)

print('bi-gram Vectorizer Accuracy Score',nb.score(gm_x_test,gm_y_test))
%%time

from sklearn import svm

svmmodel = svm.SVC(kernel='linear', C = 1)

svmmodel.fit(x_train, y_train)

print('CountVectorizer Accuracy Score',svmmodel.score(x_test,y_test))

svmmodel.fit(tf_x_train, tf_y_train)

print('TF-IDF Vectorizer Accuracy Score',svmmodel.score(tf_x_test,tf_y_test))

svmmodel.fit(gm_x_train, gm_y_train)

print('bi-gram Vectorizer Accuracy Score',svmmodel.score(gm_x_test,gm_y_test))
from sklearn import svm

svmmodel = svm.SVC(kernel='linear', C = 1)

svmmodel.fit(tf_x_train, tf_y_train)

print('TF-IDF Vectorizer Accuracy Score',svmmodel.score(tf_x_test,tf_y_test))
%%time

from sklearn.linear_model import LogisticRegression

logit = LogisticRegression(random_state=0, solver='lbfgs')

logit.fit(x_train, y_train)

print('CountVectorizer Accuracy Score',logit.score(x_test,y_test))

svmmodel.fit(tf_x_train, tf_y_train)

print('TF-IDF Vectorizer Accuracy Score',logit.score(tf_x_test,tf_y_test))

svmmodel.fit(gm_x_train, gm_y_train)

print('bi-gram Vectorizer Accuracy Score',logit.score(gm_x_test,gm_y_test))
%%time

from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators=50)

clf.fit(x_train, y_train)

print('CountVectorizer Accuracy Score',clf.score(x_test,y_test))

svmmodel.fit(tf_x_train, tf_y_train)

print('TF-IDF Vectorizer Accuracy Score',clf.score(tf_x_test,tf_y_test))

svmmodel.fit(gm_x_train, gm_y_train)

print('bi-gram Vectorizer Accuracy Score',clf.score(gm_x_test,gm_y_test))
#description_list
description_list = []

for article in data["Message"]:

    article = re.sub("[^a-zA-Z]"," ",article)

    article = article.lower() 

    cutWords = [k for k in word_tokenize(article) if k not in stopword_list]

    cutWords = [ lemma.lemmatize(word) for word in cutWords]

    description_list.append(cutWords)

#description_list
from gensim.models import Word2Vec

def getVector_v2(cutWords, word2vec_model):

    vector_list = [word2vec_model[k] for k in cutWords if k in word2vec_model]

    vector_df = pd.DataFrame(vector_list)

    cutWord_vector = vector_df.mean(axis=0).values

    return cutWord_vector



word2vec_model = Word2Vec(description_list, size=100, iter=10, min_count=20)
vector_list = []

for c in description_list:

    vec = getVector_v2(c, word2vec_model)

    vector_list.append(vec)
#vector_list
X = pd.DataFrame(vector_list)

X.shape
Y = data["Category"]

Y = pd.DataFrame(Y)

Y.shape
#X = X.fillna()

X = X.fillna(X.mean())

Y = Y.dropna()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size=0.3)

logistic_model = LogisticRegression()

logistic_model.fit(train_X, train_y)

y_predict = logistic_model.predict(test_X)



print('CountVectorizer Accuracy Score',accuracy_score(y_test, y_predict))

pd.DataFrame(confusion_matrix(y_test,y_predict))
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators=50)

gbdt = clf.fit(train_X, train_y)

y_predict = gbdt.predict(test_X)

print('CountVectorizer Accuracy Score',accuracy_score(y_test, y_predict))

pd.DataFrame(confusion_matrix(y_test,y_predict))