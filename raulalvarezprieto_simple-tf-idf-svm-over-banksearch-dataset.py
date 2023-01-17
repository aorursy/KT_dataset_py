import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import re

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
DATA_DIR = "/kaggle/input/banksearch-reduced/banksearch/"
lemmatizer = WordNetLemmatizer()
porter_stemmer = PorterStemmer()
def stemming_tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    words = [porter_stemmer.stem(word) for word in words]
    return words
def lemmatizer_tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words
# load text files with categories as subfolder names
data = load_files(DATA_DIR, encoding="utf-8", decode_error="replace")
my_parser = 'lxml'
for i, html in enumerate(data.data):
    soup = BeautifulSoup(html, my_parser)
    a = soup.get_text().split("\n")[6:]
    b = " "
    b = b.join(a)
    data.data[i] = b
print ("Used Parser: ", my_parser)
print("Parsed Files: ", i+1)
# calculate count of each category
labels, counts = np.unique(data.target, return_counts=True)
# convert data.target_names to np array for fancy indexing
labels_str = np.array(data.target_names)[labels]
print(dict(zip(labels_str, counts)))
# split the data into training 75% and testing set 25% 
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)
list(t[:80] for t in X_train[:10])
# Create pipeline for support vector classifier. max_df = 0.8, min_df= 0.01 
# Composite estimator, as a chain of transforms and estimators 
min_df= 0.01
max_df=0.70
my_idf = TfidfVectorizer(stop_words="english", min_df= min_df, max_df=max_df,
                                             tokenizer=lemmatizer_tokenizer, 
                                             ngram_range=(1, 1) )
svc_tfidf = Pipeline([
        ("tfidf_vectorizer", my_idf ),
        ("linear svc", SVC(kernel="linear", C=1)) ])
# train and evaluate pipeline SVC in the test dataset
model = svc_tfidf
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy Score: ", accuracy_score(y_test, y_pred))
print("Classification Report:\n ", classification_report(y_test, y_pred))
print("min_df = ", min_df, " max_df = ", max_df )
#Plot confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print (cm)
tdfif_matrix = svc_tfidf.named_steps['tfidf_vectorizer'].fit_transform(X_train, y_train)
X = tdfif_matrix.todense()
print (X.round(3))
print ("Shape: ", X.shape)
