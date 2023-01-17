import numpy as np

import pandas as pd

import os

import matplotlib

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as esw
dir = "/kaggle/input/jmlr1"

os.chdir(dir)

JMLR_docu = []



for file in os.listdir(dir):

    with open(file, "r", encoding="utf-8") as input:

        for string in input:

            JMLR_docu.append(string)
len(JMLR_docu) # list of 20 abstracts from Journal of Machine Learning Research
JMLR_docu[1] # just to check whether I correctly read the data
dir = "/kaggle/input/journalfe"

os.chdir(dir)

JFE_docu = []



for file in os.listdir(dir):

    with open(file, "r", encoding="cp949") as input:

        for string in input:

            JFE_docu.append(string)
len(JFE_docu) # list of 20 abstracts from Journal of Financial Economics
JFE_docu[1] # just to check whether I correctly read the data
dir = "/kaggle/input/journal_ijbs" 

os.chdir(dir)

IJBS_docu = []



for file in os.listdir(dir):

    with open(file, "r", encoding="cp949") as input:

        for string in input:

            IJBS_docu.append(string)
len(IJBS_docu) # list of 20 abstracts from International Journal of Biological Sciences
IJBS_docu[1] # just to check whether I correctly read the data
corpus = JMLR_docu + JFE_docu + IJBS_docu

label = 20 * ["ML"] + 20 * ["EC"] + 20 * ["BO"]

# ML label represents Machine Learning, EC label represents Economics, 

# and BO label represents Biological Sciences. 
# split the dataset into training and test set

corpus_train, corpus_test, label_train, label_test = train_test_split(corpus, label, test_size=0.2)
# building term matrices

vectorizer = CountVectorizer(tokenizer=str.split, stop_words=esw)

corpus_train_mat = vectorizer.fit_transform(corpus_train)

corpus_train_mat = corpus_train_mat.toarray()



corpus_test_mat = vectorizer.transform(corpus_test)

corpus_test_mat = corpus_test_mat.toarray()
# building Naive Bayes Classifier

def fit_NBclassifier(trainset, trainlabel):

    nbclassifier = MultinomialNB()

    nbclassifier.fit(trainset, trainlabel)

    

    return nbclassifier
NB_clf = fit_NBclassifier(corpus_train_mat, label_train) # train the classifier
 # predict a label of the documents in the test set using the trained classifier

label_predicted = NB_clf.predict(corpus_test_mat)
accuracy = accuracy_score(label_test, label_predicted) # accuracy rate of the classifier

accuracy 
# visualize a heat map of confusion matrix to evaluate the quality of the output of the classifier 

conf_mat = confusion_matrix(label_test, label_predicted)

labels = sorted(set(label_predicted))



plt.figure()

plt.title("Heat Map Confusion Matrix")

plt.imshow(conf_mat, interpolation="nearest", cmap=plt.cm.Reds)

plt.xticks(np.arange(len(labels)), labels)

plt.yticks(np.arange(len(labels)), labels)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.colorbar()

plt.show()