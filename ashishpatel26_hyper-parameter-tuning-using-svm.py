import pandas as pd
import numpy as np

% matplotlib inline
import matplotlib.pyplot as plt

from collections import Counter
from pprint import pprint

pd.set_option('display.max_columns', None)
pd.set_option('display.max_row', None)

import re
from nltk.stem import WordNetLemmatizer

# Grid search for optimal parameters of the model
from sklearn.model_selection import GridSearchCV

# Model modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.neural_network import MLPClassifier

# modules for # estimate
from sklearn.model_selection import cross_val_score
from sklearn import cross_validation
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier

# modules for encoding features
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Modules for dividing a data set
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from gensim.models import word2vec
from contextlib import contextmanager

import gc
import time
import json
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))
# Dataset Preparation
print ("Read Dataset ... ")
def read_dataset(path):
    return json.load(open(path)) 
train = read_dataset('../input/train.json')
test = read_dataset('../input/test.json')
def read():
    train = read_dataset('../input/train.json')
    test = read_dataset('../input/test.json')
    return train,test
    
def analysis():
    train = pd.read_json("../input/train.json")
    test = pd.read_json("../input/test.json")
    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    df = train.copy()
    print(df.shape)
    df.head(2)
    df.count()
    df.isnull().sum()
    print('Cuisine is {}.'.format(len(df.cuisine.value_counts())))
    df.cuisine.value_counts()
    # cuisine type visualization
    plt.style.use('ggplot')
    df.cuisine.value_counts().plot(kind = 'bar',title='Cuisine Types',figsize=(20,5),legend=True,fontsize=12)
    plt.ylabel("Number of Recipes", fontsize=12)
    return plt.show()

analysis()
#prepare text data for traina and test
def generate_text(data):
    print("prepare text data for Train and Test....")
    text_data = [" ".join(doc['ingredients']).lower() for doc in data]
    return text_data
# count the word frequency for text data
tfidf = TfidfVectorizer(binary=True)
def tfidf_features(text, flag):
    print ("TF-IDF on text data ... ")
    if flag == "train":
        x = tfidf.fit_transform(text)
    else:
        x = tfidf.transform(text)
    x = x.astype('float16')
    print()
    return x 
lb = LabelEncoder()
# Label Encoding - Target 
def label_encoding(target):
    print ("Label Encode the Target Variable ... ")
    y = lb.fit_transform(target)
    return y
# Model Training 
def data_model(X,y,X_test):
    print ("Train the model ... ")
    classifier = SVC(C=200, # penalty parameter, setting it to a larger value 
                     kernel='rbf', # kernel type, rbf working fine here
                     degree=3, # default value, not tuned yet
                     gamma=4, # kernel coefficient, not tuned yet
                     coef0=1, # change to 1 from default value of 0.0
                     shrinking=True, # using shrinking heuristics
                     tol=0.001, # stopping criterion tolerance 
                     probability=False, # no need to enable probability estimates
                     cache_size=200, # 200 MB cache size
                     class_weight=None, # all classes are treated equally 
                     verbose=False, # print the logs 
                     max_iter=-1, # no limit, let it run
                     decision_function_shape=None, # will use one vs rest explicitly 
                     random_state=None)
    model = OneVsRestClassifier(classifier, n_jobs=4)
    model.fit(X, y)
    print("This Model Accuracy is :", model.score(X,y))
    # Predictions 
    print ("Predict on test data... ")
    y_test = model.predict(X_test)
    y_pred = lb.inverse_transform(y_test)
    return (y_pred, y_test)

def submission(test, y_pred):
    # Submission
    print ("Generate Submission File ... ")
    test_id = [doc['id'] for doc in test]
    sub = pd.DataFrame({'id': test_id, 'cuisine': y_pred}, columns=['id', 'cuisine'])
    sub.to_csv('Submission.csv', index=False)
    return sub
def main(debug = False):
    with timer("Process Read Data.."):
        analysis()
        train, test = read()
#         train = load_data("../input/train.json")
#         test = load_data("../input/test.json")
        print("Full Report of Train Data......")
        print("Report Completed...")
        gc.collect()
    with timer("Process of prepare text data for Train and Test...."):
        print("Start Data preparation...")
        train_text = generate_text(train)
        test_text = generate_text(test)
        target = [doc['cuisine'] for doc in train]
        print("Data Preparation Completed...")
        gc.collect()
    with timer("Process of TFIDF for train_text and test_text"):
        print("Start TFIDF...")
        X = tfidf_features(train_text, flag="train")
        X_test = tfidf_features(test_text, flag="test")
        print("TFIDF Completed...")
        gc.collect()
    with timer("Process Label Encoding"):
        print("Start Label Encoding...")
        y = label_encoding(target=target)
        print("Label Encoding Completed...")
        gc.collect()
    with timer("Run SVM Training"):
        print("Start Model Training...")
        y_pred, y_test = data_model(X,y,X_test)
        print("Model Training Completed...")
        gc.collect()
    with timer("Final Submission"):
        print("Start file Submission...")
        sub = submission(test,y_pred)
        sub.head(20)
        print("Run Compeleted.")

if __name__ == "__main__":
    with timer("Full model run"):
        main(debug= False)