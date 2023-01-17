import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn import metrics
import nltk
import os
import gc
from keras.preprocessing import sequence,text
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Input, Dense,Dropout,Embedding,LSTM, CuDNNGRU, Conv1D,GlobalMaxPooling1D,Flatten,MaxPooling1D,GRU,GlobalMaxPool1D,SpatialDropout1D,Bidirectional
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

test = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")
train.head()
from nltk.corpus import stopwords
import string
punctuations = string.punctuation
stopword = stopwords.words("english")
def clean(text):
    
    lower_text = text.lower()
    
    text = "".join(w for w in lower_text if w not in punctuations)
    
    words = text.split()
    words = [w for w in words if w not in stopword]
    res = " ".join(words)
    return res
clean("this is a test!")

train['cleaned'] = train['question_text'].apply(clean)
test['cleaned'] = test['question_text'].apply(clean)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

cvz = CountVectorizer()
word_tfidf = TfidfVectorizer()
cvz.fit(train["cleaned"].values)
count_vector_train = cvz.transform(train["cleaned"].values)
count_vector_test = cvz.transform(test["cleaned"].values)

word_tfidf.fit(train["cleaned"].values)
word_vector_train = word_tfidf.transform(train["cleaned"].values)
word_vector_test = word_tfidf.transform(test["cleaned"].values)

train_vector = word_vector_train
test_vector = word_vector_test
target = train['target']
from sklearn.model_selection import train_test_split
trainx, valx, trainy, valy = train_test_split(train_vector,target)

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=777)
X_ROS, y_ROS = ros.fit_sample(trainx, trainy)
#X_ROS = trainx
#y_ROS = trainy
from sklearn import naive_bayes
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import ensemble
from sklearn.metrics import accuracy_score, f1_score
model_1 = naive_bayes.MultinomialNB()
model_1.fit(trainx,trainy)
pred1 = model_1.predict(valx)
print(accuracy_score(pred1,valy))
print(f1_score(pred1,valy))
model_2 = svm.SVC()
model_2.fit(trainx,trainy)
pred2 = model_2.predict(valx)
print(accuracy_score(pred2,valy))
print(f1_score(pred2,valy))
model_3 = LogisticRegression()
model_3.fit(trainx,trainy)
pred3 = model_3.predict(valx)
print(accuracy_score(pred3,valy))
print(f1_score(pred3,valy))
