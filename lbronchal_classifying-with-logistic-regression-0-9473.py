%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



np.random.seed(123456)
dataset = pd.read_csv("../input/uci-news-aggregator.csv")
dataset.head()
dataset['CATEGORY'].unique()
dataset['CATEGORY'].value_counts().plot(kind="bar")

plt.show()
import re

import string



def clean_text(s):

    s = s.lower()

    for ch in string.punctuation:                                                                                                     

        s = s.replace(ch, " ") 

    s = re.sub("[0-9]+", "||DIG||",s)

    s = re.sub(' +',' ', s)        

    return s



dataset['TEXT'] = [clean_text(s) for s in dataset['TITLE']]
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer

from sklearn.preprocessing import LabelEncoder

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.model_selection import cross_val_score



# pull the data into vectors

vectorizer = CountVectorizer()

x = vectorizer.fit_transform(dataset['TEXT'])



# for Tfidf (we have tried and the results aren't better)

#tfidf = TfidfVectorizer()

#x = tfidf.fit_transform(dataset['TEXT'].values)



encoder = LabelEncoder()

y = encoder.fit_transform(dataset['CATEGORY'])



# split into train and test sets

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)



# take a look at the shape of each of these

print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
%%time 



nb = MultinomialNB()

nb.fit(x_train, y_train)
%%time



results_nb_cv = cross_val_score(nb, x_train, y_train, cv=10)

print(results_nb_cv.mean())
nb.score(x_test, y_test)
x_test_pred = nb.predict(x_test)

confusion_matrix(y_test, x_test_pred)
print(classification_report(y_test, x_test_pred, target_names=encoder.classes_))
def predict_cat(title):

    cat_names = {'b' : 'business', 't' : 'science and technology', 'e' : 'entertainment', 'm' : 'health'}

    cod = nb.predict(vectorizer.transform([title]))

    return cat_names[encoder.inverse_transform(cod)[0]]
predict_cat("the economy goes up")
%%time 



from sklearn.linear_model import LogisticRegression

from sklearn.multiclass import OneVsRestClassifier



# Instantiate the classifier: clf

clf = OneVsRestClassifier(LogisticRegression())



# Fit the classifier to the training data

clf.fit(x_train, y_train)



# Print the accuracy

print("Accuracy: {}".format(clf.score(x_test, y_test)))
#%%time



#results_clf_cv = cross_val_score(clf, x_train, y_train, cv=10)

#print(results_clf_cv.mean())
x_test_clv_pred = clf.predict(x_test)

confusion_matrix(y_test, x_test_clv_pred)
print(classification_report(y_test, x_test_clv_pred, target_names=encoder.classes_))
clf_pred = clf.predict(x)

np_pred = nb.predict(x)



models_correlation = np.corrcoef(clf_pred, np_pred)

models_correlation[0,1]