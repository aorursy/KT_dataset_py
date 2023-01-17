
import re
import numpy as np 
import pandas as pd 

from sklearn.naive_bayes import MultinomialNB

from sklearn.cross_validation import train_test_split #for sklearn < v0.18


from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import LabelEncoder


news = pd.read_csv("../input/uci-news-aggregator.csv")
# let's take a look at our data
news.head()
def normalize_text(s):
    s = s.lower()
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W\s',' ',s)
    s = re.sub('\s+',' ',s)
    
    return s

news['TEXT'] = [normalize_text(s) for s in news['TITLE']]
news['TEXT'].head()
from nltk.corpus import stopwords 
import nltk
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r"\w+")
stopwords = nltk.corpus.stopwords.words("english")
stop_words=stopwords
vectorizer = CountVectorizer(tokenizer=tokenizer.tokenize, stop_words=stopwords)
x = vectorizer.fit_transform(news['TEXT'])

encoder = LabelEncoder()
y = encoder.fit_transform(news['CATEGORY'])

# split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
nb = MultinomialNB()
nb.fit(x_train, y_train)
nb.score(x_test, y_test)
from sklearn.metrics import confusion_matrix
x_test_pred = nb.predict(x_test)
confusion_matrix(y_test, x_test_pred)
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

clf = OneVsRestClassifier(LogisticRegression())
clf.fit(x_train, y_train)

# Print the accuracy
print("Accuracy: {}".format(clf.score(x_test, y_test)))
x_test_clv_pred = clf.predict(x_test)
confusion_matrix(y_test, x_test_clv_pred)
from sklearn.metrics import classification_report
print(classification_report(y_test, x_test_clv_pred, target_names=encoder.classes_))
from sklearn.ensemble import RandomForestClassifier
clf_1 = RandomForestClassifier()
# Fit the classifier to the training data
clf_1.fit(x_train, y_train)

# Print the accuracy
print("Accuracy: {}".format(clf_1.score(x_test, y_test)))
from sklearn import tree
clf_1 = tree.DecisionTreeClassifier()
clf_1.fit(x_train, y_train)

# Print the accuracy
print("Accuracy: {}".format(clf.score(x_test, y_test)))

def predict(title):
    cat_names = {'b' : 'business', 't' : 'science and technology', 'e' : 'entertainment', 'm' : 'health'}
    cod = clf.predict(vectorizer.transform([title]))
    return cat_names[encoder.inverse_transform(cod)[0]]
predict("this movie is very good")
predict("i am sick today")
predict("market gonna crash ")
predict("Machine learning and AI is the future ")
