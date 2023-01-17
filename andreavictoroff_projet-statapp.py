# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from IPython.display import Markdown as md

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import re



from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



import nltk

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from nltk.stem.porter import *

from nltk.tokenize import word_tokenize

from nltk.probability import FreqDist

from nltk import ngrams, everygrams

import nltk.classify

from nltk.classify import NaiveBayesClassifier



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import naive_bayes

from sklearn.naive_bayes import MultinomialNB

from sklearn import svm

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from sklearn.metrics import precision_recall_fscore_support

from sklearn.feature_extraction.text import CountVectorizer 





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/traindataset/train-dataset.csv", sep=",",encoding = "ISO-8859-1")

data = pd.DataFrame({"DESCRIP" : data["DESCRIP"],"label" : data["label"]})

data.dropna(inplace = True)

labels = list(set(data["label"]))

print(labels)
data.head()
data['label'].describe()
data["label"].value_counts()[:10].plot.bar()
test = data.query(' label == "upccoo" ')["DESCRIP"]

for balade in test:

    if type(balade) != str :

        print (balade, type(balade))
stopword = set(STOPWORDS)

for i in range(len(labels)):

    lab = labels[i]

    test = data.query(' label == @lab ')["DESCRIP"]

    text = ''.join(test)

    wordcloud = WordCloud(stopwords=stopword,max_font_size=100, max_words=10, background_color="white").generate(text)

    plt.figure()

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis("off")

    plt.title("10 mots les plus fréquents dans la catégorie "+ lab)

    plt.show()
#mots les plus frequent par categorie sous forme d'une liste [serie des 10 mots ,id categorie]

Most_frequent = []

for i in range(len(labels)):

    lab = labels[i]

    test = data.query(' label == @lab ')["DESCRIP"]

    text = ''.join(test)

    serie = pd.DataFrame(pd.Series((text).lower().split(), name = "words"))

    serie = serie.query(' words not in @stopword ')

    serie = pd.Series(serie["words"])

    serie = [serie.value_counts()[:10], lab]

    Most_frequent.append(serie)

Most_frequent
data_med = data.query(' label == "upcana" ')

data_bois = data.query(' label == "upcsdr" ')

data_med

data_bois

train_2 = pd.concat([data_med[:(len(data_med)//3)], data_bois[:(len(data_bois)//3)] ])

test_2 = pd.concat([data_med[(len(data_med)//3):], data_bois[(len(data_bois)//3):] ])

train_2 = train_2.reset_index()

del train_2["index"]

test_2 = test_2.reset_index()

del test_2["index"]
corpus_train_2 = []

n = len(train_2)

for i in range(0,n):

    descrip = re.sub("<.*?>", "", train_2['DESCRIP'][i])

    descrip = re.sub('[^a-zA-Z]', ' ', descrip)

    descrip = descrip.lower()

    descrip = descrip.split()

    ps = PorterStemmer()

    lemmatizer = WordNetLemmatizer()

    descrip = [ps.stem(words) for words in descrip if not words in set(stopwords.words('english'))]

    descrip = [lemmatizer.lemmatize(words) for words in descrip if not words in set(stopwords.words('english'))]

    descrip = ' '.join(descrip)

    corpus_train_2.append(descrip)
corpus_test_2 = []

n = len(test_2)

for i in range(0,n):

    descrip = re.sub("<.*?>", "", test_2['DESCRIP'][i])

    descrip = re.sub('[^a-zA-Z]', ' ', descrip)

    descrip = descrip.lower()

    descrip = descrip.split()

    ps = PorterStemmer()

    lemmatizer = WordNetLemmatizer()

    descrip = [ps.stem(words) for words in descrip if not words in set(stopwords.words('english'))]

    descrip = [lemmatizer.lemmatize(words) for words in descrip if not words in set(stopwords.words('english'))]

    descrip = ' '.join(descrip)

    corpus_test_2.append(descrip)
cv = TfidfVectorizer(max_features=2000) 



X_train = cv.fit_transform(corpus_train_2).toarray()

X_test = cv.transform(corpus_test_2).toarray()
classifier = MultinomialNB(alpha = 1 , fit_prior=True, class_prior=None)

classifier.fit(X_train, train_2['label'])

y_pred = classifier.predict(X_test)
pd.DataFrame(confusion_matrix(test_2["label"] , y_pred),index=['predicted positive','predicted negative'],columns=['actual positive','actual negative'])
def accuracy(confusion_matrix):

    diagonal_sum = confusion_matrix.trace()

    sum_of_all_elements = confusion_matrix.sum()

    return diagonal_sum / sum_of_all_elements 

accuracy(confusion_matrix(test_2["label"] , y_pred))
data_m = data.sample(frac=1).reset_index(drop=True)

data_m
txt_train, txt_test, y_train, y_test = train_test_split(data_m["DESCRIP"],data_m["label"],test_size=0.33)

txt_train = txt_train.reset_index(drop=True)

txt_test = txt_test.reset_index(drop=True)

y_train = y_train.reset_index(drop=True)

y_test = y_test.reset_index(drop=True)

txt_train
corpus_train = []

n = len(txt_train)

for i in range(0,n):

    descrip = re.sub("<.*?>", "", txt_train[i])

    descrip = re.sub('[^a-zA-Z]', ' ', descrip)

    descrip = descrip.lower()

    descrip = descrip.split()

    ps = PorterStemmer()

    lemmatizer = WordNetLemmatizer()

    descrip = [ps.stem(words) for words in descrip if not words in set(stopwords.words('english'))]

    descrip = [lemmatizer.lemmatize(words) for words in descrip if not words in set(stopwords.words('english'))]

    descrip = ' '.join(descrip)

    corpus_train.append(descrip)
corpus_test = []

n = len(txt_test)

for i in range(0,n):

    descrip = re.sub("<.*?>", "", txt_test[i])

    descrip = re.sub('[^a-zA-Z]', ' ', descrip)

    descrip = descrip.lower()

    descrip = descrip.split()

    ps = PorterStemmer()

    lemmatizer = WordNetLemmatizer()

    descrip = [ps.stem(words) for words in descrip if not words in set(stopwords.words('english'))]

    descrip = [lemmatizer.lemmatize(words) for words in descrip if not words in set(stopwords.words('english'))]

    descrip = ' '.join(descrip)

    corpus_test.append(descrip)
# methode tfidf

cv = TfidfVectorizer(max_features=100000) 



X_train = cv.fit_transform(corpus_train).toarray()

X_test = cv.transform(corpus_test).toarray()
classifier = MultinomialNB(alpha = 0.1 , fit_prior=True, class_prior=None)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test , y_pred)

cm.trace()/cm.sum()
# Methode comptage, avec la possibilité de considérer les ngrams



for n in (1,2,3):

    cv = CountVectorizer(max_features=150000, ngram_range = (1,n), stop_words = "english") 



    X_train = cv.fit_transform(corpus_train).toarray()

    X_test = cv.transform(corpus_test).toarray()



    classifier = MultinomialNB(alpha = 0.01 , fit_prior=True, class_prior=None)

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)



    cm = confusion_matrix(y_test , y_pred)

    print(str(n)+"gram accuracy :" + str(cm.trace()/cm.sum()))
nb = MultinomialNB()

clf = GridSearchCV(nb, {"alpha" : [k/10 for k in range(11)]})

clf.fit(X_train, y_train)

clf.cv_results_
'''for n in (1,2,3):

    cv = CountVectorizer(max_features=250000, ngram_range = (1,n), stop_words = "english") 



    X_train = cv.fit_transform(corpus_train).toarray()

    X_test = cv.transform(corpus_test).toarray()



    clf = GridSearchCV(nb, {"alpha" : [k/100 for k in range(10,20)]})

    clf.fit(X_train, y_train)

    print(clf.cv_results_)'''
'''#long

classifier = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)'''
'''cm = confusion_matrix(y_test, y_pred)

cm.trace()/cm.sum()'''
'''classifier = LogisticRegression(solver = 'lbfgs',max_iter = 5000, multi_class = "auto")

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)'''
'''cm = confusion_matrix(y_test , y_pred)

cm.trace()/cm.sum()'''
'''classifier = RandomForestClassifier(n_estimators = 100)

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)'''
'''cm = confusion_matrix(y_test , y_pred)

cm.trace()/cm.sum()'''
m = 5 #pour voir les m labels avec lequel ont a confondu le plus chaque categorie

def table_precisions(y_test, y_pred,labels):

    n = len(labels)

    CM = confusion_matrix(y_test, y_pred, labels = labels)

    precision = [0 for k in range(n)]

    recall = [0 for k in range(n)]

    lbl_conf = [["" for i in range(m)] for k in range(n)]

    for k in range(n):

        precision[k] = float(CM[k,k])/(sum(CM[i,k] for i in range(n)))

        recall[k] = float(CM[k,k])/(sum(CM[k,j] for j in range(n)))

        L = CM[k,:]

        L[k] = 0

        for a in range(m):

            i_max = np.argmax(L)

            lbl_conf[k][a] = (labels[i_max], max(L))

            L[i_max]=0

    df = pd.DataFrame({"label":labels,"precision" : precision, "recall" : recall, "lab_conf" : lbl_conf})

    return df

            
table_precisions(y_test,y_pred,labels)
md('''|Stemming + Lemmatization|Max n-gram| Tfidf |     Méthode     | Accuracy | Paramètres |

|-----|-----| ----- | ----- | ----- |-----|

|oui|1| oui | Naive Bayes | .85 |alpha = 0.05|

|non|1| non | Naive Bayes | .79 |

|non|2|non|Naive Bayes| .72| 

|non|3|non|Naive Bayes| .70|

|oui|1| non | Naive Bayes | .83 |

|oui|2|non|Naive Bayes| .78|

|oui|3|non|Naive Bayes| .75|

|oui|1|oui|SVM|.85|

|oui|1|oui|Reglog|.84|

|oui|1|oui|RandomForest|.83|''')


"""plt.savefig('name.png')"""