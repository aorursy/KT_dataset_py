# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/shopee-sentiment-analysis/train.csv")
df.head()
df.shape
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score
X = df['review']
y = df['rating']
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
def text_processing(text):
    # split into words
    tokens = word_tokenize(text)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    return words
X = X.apply(lambda x: text_processing(x))
X = X.apply(lambda x: " ".join(x))
X.to_csv("X_processed.csv", index=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
print(len(X_train), len(X_test), len(y_train), len(y_test))
X_train.head()
y_train.head()
X.apply(lambda x:len(x.split(" "))).describe()
import seaborn as sns
sns.distplot(X.apply(lambda x:len(x.split(" "))))
classifier_sgd = Pipeline([
    ('vect', TfidfVectorizer(ngram_range=(1,1), analyzer='word')),
    ('clf-sgd', SGDClassifier()),
])

classifier_sgd.fit(X_train, y_train)
y_pred_sgd = classifier_sgd.predict(X_test)
print("--- Accuracy Report ---\n")
print("-- Balanced Accuracy Score --")
print(balanced_accuracy_score(y_test, y_pred_sgd))
print("")
print("-- F1 Score --")
print(f1_score(y_test, y_pred_sgd, average='macro'))
classifier_svm = Pipeline([
    ('vect', TfidfVectorizer(ngram_range=(1,1), analyzer='word')),
    ('clf-svm', LinearSVC()),
])

classifier_svm.fit(X_train, y_train)
y_pred_svm = classifier_svm.predict(X_test)
print("--- Accuracy Report ---\n")
print("-- Balanced Accuracy Score --")
print(balanced_accuracy_score(y_test, y_pred_svm))
print("")
print("-- F1 Score --")
print(f1_score(y_test, y_pred_svm, average='macro'))
classifier_mnb = Pipeline([
    ('vect', TfidfVectorizer(ngram_range=(1,1), analyzer='word')),
    ('clf-mnb', MultinomialNB()),
])

classifier_mnb.fit(X_train, y_train)
y_pred_mnb = classifier_mnb.predict(X_test)
print("--- Accuracy Report ---\n")
print("-- Balanced Accuracy Score --")
print(balanced_accuracy_score(y_test, y_pred_mnb))
print("")
print("-- F1 Score --")
print(f1_score(y_test, y_pred_mnb, average='macro'))
#### Unigram Bigram (SGD, SVM, MNB)
classifier_sgd = Pipeline([
    ('vect', TfidfVectorizer(ngram_range=(1,2), analyzer='word')),
    ('clf-sgd', SGDClassifier()),
])

classifier_sgd.fit(X_train, y_train)
y_pred_sgd = classifier_sgd.predict(X_test)
print("--- Accuracy Report ---\n")
print("-- Balanced Accuracy Score --")
print(balanced_accuracy_score(y_test, y_pred_sgd))
print("")
print("-- F1 Score --")
print(f1_score(y_test, y_pred_sgd, average='macro'))
classifier_svm = Pipeline([
    ('vect', TfidfVectorizer(ngram_range=(1,2), analyzer='word')),
    ('clf-svm', LinearSVC()),
])

classifier_svm.fit(X_train, y_train)
y_pred_svm = classifier_svm.predict(X_test)
print("--- Accuracy Report ---\n")
print("-- Balanced Accuracy Score --")
print(balanced_accuracy_score(y_test, y_pred_svm))
print("")
print("-- F1 Score --")
print(f1_score(y_test, y_pred_svm, average='macro'))
classifier_mnb = Pipeline([
    ('vect', TfidfVectorizer(ngram_range=(1,2), analyzer='word')),
    ('clf-mnb', MultinomialNB()),
])

classifier_mnb.fit(X_train, y_train)
y_pred_mnb = classifier_mnb.predict(X_test)
print("--- Accuracy Report ---\n")
print("-- Balanced Accuracy Score --")
print(balanced_accuracy_score(y_test, y_pred_mnb))
print("")
print("-- F1 Score --")
print(f1_score(y_test, y_pred_mnb, average='macro'))
classifier_sgd = Pipeline([
    ('vect', TfidfVectorizer(ngram_range=(2,2), analyzer='word')),
    ('clf-sgd', SGDClassifier()),
])

classifier_sgd.fit(X_train, y_train)
y_pred_sgd = classifier_sgd.predict(X_test)
print("--- Accuracy Report ---\n")
print("-- Balanced Accuracy Score --")
print(balanced_accuracy_score(y_test, y_pred_sgd))
print("")
print("-- F1 Score --")
print(f1_score(y_test, y_pred_sgd, average='macro'))
classifier_svm = Pipeline([
    ('vect', TfidfVectorizer(ngram_range=(2,2), analyzer='word')),
    ('clf-svm', LinearSVC()),
])

classifier_svm.fit(X_train, y_train)
y_pred_svm = classifier_svm.predict(X_test)
print("--- Accuracy Report ---\n")
print("-- Balanced Accuracy Score --")
print(balanced_accuracy_score(y_test, y_pred_svm))
print("")
print("-- F1 Score --")
print(f1_score(y_test, y_pred_svm, average='macro'))
classifier_mnb = Pipeline([
    ('vect', TfidfVectorizer(ngram_range=(2,2), analyzer='word')),
    ('clf-mnb', MultinomialNB()),
])

classifier_mnb.fit(X_train, y_train)
y_pred_mnb = classifier_mnb.predict(X_test)
print("--- Accuracy Report ---\n")
print("-- Balanced Accuracy Score --")
print(balanced_accuracy_score(y_test, y_pred_mnb))
print("")
print("-- F1 Score --")
print(f1_score(y_test, y_pred_mnb, average='macro'))
classifier_sgd = Pipeline([
    ('vect', TfidfVectorizer(ngram_range=(1,4), analyzer='word')),
    ('clf-sgd', SGDClassifier()),
])

classifier_sgd.fit(X_train, y_train)
y_pred_sgd = classifier_sgd.predict(X_test)
print("--- Accuracy Report ---\n")
print("-- Balanced Accuracy Score --")
print(balanced_accuracy_score(y_test, y_pred_sgd))
print("")
print("-- F1 Score --")
print(f1_score(y_test, y_pred_sgd, average='macro'))
classifier_svm = Pipeline([
    ('vect', TfidfVectorizer(ngram_range=(1,4), analyzer='word')),
    ('clf-svm', LinearSVC()),
])

classifier_svm.fit(X_train, y_train)
y_pred_svm = classifier_svm.predict(X_test)
print("--- Accuracy Report ---\n")
print("-- Balanced Accuracy Score --")
print(balanced_accuracy_score(y_test, y_pred_svm))
print("")
print("-- F1 Score --")
print(f1_score(y_test, y_pred_svm, average='macro'))
classifier_mnb = Pipeline([
    ('vect', TfidfVectorizer(ngram_range=(1,4), analyzer='word')),
    ('clf-mnb', MultinomialNB()),
])

classifier_mnb.fit(X_train, y_train)
y_pred_mnb = classifier_mnb.predict(X_test)
print("--- Accuracy Report ---\n")
print("-- Balanced Accuracy Score --")
print(balanced_accuracy_score(y_test, y_pred_mnb))
print("")
print("-- F1 Score --")
print(f1_score(y_test, y_pred_mnb, average='macro'))
test = pd.read_csv("../input/shopee-sentiment-analysis/test.csv")
test.head()
X_test_final = test['review']
X_test_final.head()
X_test_final = X_test_final.apply(lambda x: text_processing(x))
X_test_final = X_test_final.apply(lambda x: " ".join(x))
X_test_final.head()
X_train_combined = pd.concat([X_train, X_test])
y_train_combined = pd.concat([y_train, y_test])
classifier_svm = Pipeline([
    ('vect', TfidfVectorizer(ngram_range=(1,4), analyzer='word')),
    ('clf-svm', LinearSVC()),
])

classifier_svm.fit(X_train_combined, y_train_combined)
y_pred_svm = classifier_svm.predict(X_test_final)
y_pred_svm
sample_df = pd.read_csv("../input/shopee-sentiment-analysis/sampleSubmission.csv")
sample_df.head()
submission_df = pd.DataFrame({"review_id": test['review_id'], "rating": y_pred_svm})
submission_df.head()
submission_df.to_csv("prediction_processed_ngram_1_4_mnb.csv", index=False)