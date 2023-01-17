import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import os
print(os.listdir("../input"))

df=pd.read_csv("../input/spam.csv",encoding = "ISO-8859-1")
df.info()
df.describe()
df.head()
df=df[["v1","v2"]]
df.head()
# All small letters
df["v2"]=df["v2"].apply(lambda x: x.lower())
df.head()
# delete punctuation
import string

def del_punctuation(text):
    answer=[]
    for letter in text:
        if letter not in string.punctuation:
            answer.append(letter)
    answer="".join(answer)
    return answer

df["v2"]=df["v2"].apply(lambda x: del_punctuation(x))
df.head()
# Now delete stopwords
from nltk.corpus import stopwords

def del_stopwords(text):
    answer=[]
    for word in text.split():
        if word not in stopwords.words("english"):
            answer.append(word)
    answer=" ".join(answer)
    return answer

df["v2"]=df["v2"].apply(lambda x: del_stopwords(x))
df.head()
"""#Translate words into numbers
#    * List of all words
#    * Translate texts into a countlist of words
list_of_words=[]
for item in df["v2"]:
    for word in item.split():
        if word not in list_of_words:
            list_of_words.append(word)
print(len(list_of_words))
print(list_of_words[0:4])"""

# This is being done by a sklearn feature (see below)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer().fit(df["v2"])
vectorizer.vocabulary_
len(vectorizer.vocabulary_)
# different from my number calculated above. It ignores 1-letter figures and "words" automatically
test_message=vectorizer.transform(df["v2"][0].split())
print(test_message)
# Construct a matrix with the 9376 features 
X=vectorizer.transform(df["v2"])
X
print(X[0])
X.shape
# How many non-zero items are in the 5572*9376 matrix? (52 mn entries) About 0,09% - indeed very sparse matrix
X.nnz
# These word-counts should be measured not in absolute terms but in relative terms within the message in relation to overall corpus frequency
# Tfidf = Term frequency inverse document frequency
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer=TfidfTransformer().fit(X)
X_relative=tfidf_transformer.transform(X)
X_relative
print(X_relative[0])
from sklearn.naive_bayes import MultinomialNB
spam_model=MultinomialNB().fit(X_relative, df["v1"])
spam_model.predict(X_relative)
# Now lets do a proper train_test_split to see if model predicts accurately
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df["v2"],df["v1"],test_size=0.3)

# Pipeline saves a process of instructions that can be used to different datasets
from sklearn.pipeline import Pipeline
pipeline = Pipeline([("bow",CountVectorizer()),          # Creates sparse matrix counting the words
                     ("tfidf",TfidfTransformer()),       # Relative weights
                    ("classifier",MultinomialNB())])     # Fit the model
pipeline.fit(X_train,y_train)
# Returns a fitted pipeline object
predictions=pipeline.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
