# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import nltk
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import cross_val_predict

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "-l", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Counting registers
dataset = pd.read_csv('../input/Tweets_Mg.csv',encoding='utf-8')
dataset.count()
# Separating tweets from its classes
tweets = dataset["Text"].values
tweets
classes = dataset["Classificacao"].values
classes
# Model training using Bag of Words approach and Naive Bayes Multinomial algorithm
#    - Bag of Words creates a vector with every word from the text, so it calculates the frequency
#      that these words appears in a given sentece, then classify/train the model.
#    - Hypothetical example of three sentences vectorized "by word" and classified in its "word frequency":
#         {0,3,2,0,0,1,0,0,0,1, Positivo}
#         {0,0,1,0,0,1,0,1,0,0, Negativo}
#         {0,1,1,0,0,1,0,0,0,0, Neutro}
#    - Looking at these vectors, my guess is that the words in positions 2 and 3 are the ones with the greatest 
#      weight in determining which class belongs to each of the three sentences evaluated
#    - The function fit_transform do exactly this proccess: adjust the model, learn the vocabulary,
#      and transforms the training data in feature vectors (vector with the words frequency)

vectorizer = CountVectorizer(analyzer = "word")
f_tweets = vectorizer.fit_transform(tweets)

model = MultinomialNB()
model.fit(f_tweets, classes)
# Sentences to test with the created model
tests = ["Esse governo está no início, vamos ver o que vai dar",
         "Estou muito feliz com o governo de São Paulo esse ano",
         "O estado de Minas Gerais decretou calamidade financeira!!!",
         "A segurança desse país está deixando a desejar",
         "O governador de Minas é do PT",
         "O prefeito de São Paulo está fazendo um ótimo trabalho"]

f_tests = vectorizer.transform(tests)
model.predict(f_tests)
# Model Cross Validation. In this case, the model is divided in ten parts, trained in 9 and tested in 1
results = cross_val_predict(model, f_tweets, classes, cv = 10)
results
# What's the accuracy of the model?
metrics.accuracy_score(classes, results)
# Model validation measurements
sentiments = ["Positivo", "Negativo", "Neutro"]
print(metrics.classification_report(classes, results, sentiments))

#    : precision = true positive / (true positive + false positive)
#    : recall    = true positive / (true positive + false negative)
#    : f1-score  = 2 * ((precision * recall) / (precision + recall))
# Confusion Matrix
print(pd.crosstab(classes, results, rownames = ["Real"], colnames=["Predict"], margins=True))

#    - Predict = The program classified Negativo, Neutro, Positivo and All
#    - Real    = What is in fact Negativo, Neutro, Positivo and All
#
# That is, only 9 tweets was in fact negative and the program classified positive. But the
# positives that the program classified negative was 45.
# With the Bigrams model, instead of vectorize the text "by word", the text is vectorizes
# by "two words", like: Eu gosto de São Paulo => { eu gosto, gosto de, de são, são paulo }
vectorizer = CountVectorizer(ngram_range = (1, 2))
f_tweets = vectorizer.fit_transform(tweets)

model = MultinomialNB()
model.fit(f_tweets, classes)
# New prediction, using Bigrams
results = cross_val_predict(model, f_tweets, classes, cv = 10)
results
# Checking the accuracy
metrics.accuracy_score(classes, results)
# A bit better than the last one
print(metrics.classification_report(classes, results, sentiments))

print(pd.crosstab(classes, results, rownames = ["Real"], colnames=["Predict"], margins=True))