import pandas as pd
df = pd.read_csv('/kaggle/input/spam-mails-dataset/spam_ham_dataset.csv')
df.head()
X = df.text
X.head()
y = df.label_num
y.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
X_train.head()
X_train.values
y_train.head()
y_train.values
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(X_train.values)
#vectorizer.vocabulary_
import itertools
dict(itertools.islice(vectorizer.vocabulary_.items(), 5))
targets = y_train.values
classifier.fit(counts, targets)
example_email = 'Hello Cassia, lets have our meetings on Tuesday at 7. Is that ok?'
example_spam = 'Buy more of the amazing clean everything product. Hi, there. We have a special offer for you. We can sell you 10 times more of our product. Just call 998-345-234 and order until tomorrow and you will get 30% discount on the price of 70 all cleaners.'
examples = [example_email, example_spam]
example_count = vectorizer.transform(examples)
example_count
example_predictions = classifier.predict(example_count)
example_predictions
counts_test = vectorizer.transform(X_test.values)
y_pred = classifier.predict(counts_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
from sklearn.metrics import f1_score
f1_score(y_test, y_pred)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))