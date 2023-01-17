# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import nltk

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

import sklearn

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.





    
dosya= open("../input/7all.csv")

print("Reading from file ...")

i=0

cats=[]

articles=[]

vocab=[]

for line in dosya:

 i=i+1

 print ("\rComplete: ", i, "%", end="")

 lines=line.split(",")

 cat=lines[0]

 cats.append(cat)

 article=lines[1].split()

 articles.append(article)

 vocab= vocab+ article

fd=nltk.FreqDist(vocab)

print("Article Size ",len(articles))

print("initial vocab size ", len(vocab))

vocab2=[k for k in fd.keys() if fd[k]>3]

from sklearn.feature_extraction.text import CountVectorizer

K=2000

mc=fd.most_common(K) 

freqK=[e[0] for e in mc]

art2=[" ".join([k for k in a if k in freqK ]) for a in articles]
vectorizer = CountVectorizer()

X = vectorizer.fit_transform(art2)
data_labels=cats

#models_name = ["k-NN1", "k-NN2", "NB1", "LR-1", "LinearSVM",  "SGD", "DecisionTree", "RandomForest", "NeuralNet"]

models_name = ["Multi NB", "LR", "LinearSVM" ]



models = [ MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True), LogisticRegression(), LinearSVC()]

print("The number of Selected Feature (Frequency Based):", K)

for j in range(len(models)):

 print(models_name[j]+ " ")

 predicted = sklearn.model_selection.cross_val_predict(models[j], X, data_labels, cv=4)

 acc=sklearn.metrics.accuracy_score(data_labels, predicted)    

 print ("Accuracy ", acc)

 print("***")


