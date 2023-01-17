import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time

#Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

#from sklearn.svm import SVC,NuSVC
#from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,GradientBoostingClassifier,VotingClassifier,BaggingClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
#import xgboost as xgb
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
#rom keras.layers import Conv1D, GlobalMaxPooling1D
#import keras
#from keras.datasets import imdb

# Text Processing
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import nltk
from sklearn.preprocessing import LabelEncoder
from nltk import FreqDist
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence


### Sentiment Analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer

### Dimensionality Reduction
from sklearn.decomposition import PCA
from sklearn import random_projection
from sklearn.decomposition import FastICA
from sklearn.feature_selection import VarianceThreshold

### Scaling
from sklearn.preprocessing import MinMaxScaler,StandardScaler,normalize

### Dataset Split
from sklearn.model_selection import train_test_split

# Evalution Metric
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#from keras import metrics

import os
import re
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# LOADING DATASET
training_set = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
testing_set = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

# LoADING STOPWORDS
stop_words = stopwords.words('english')

# LOADING SPACY ENGLISH LANGUAGE MODEL
nlp = spacy.load("en")

# LOADING WORD LIST FROM NLTK
words = set(nltk.corpus.words.words())

# DECLARING LEMMATIZER OBJECT AND SENTIMENT OBJECT
lemmatizer = WordNetLemmatizer()
analyser = SentimentIntensityAnalyzer()
#DECLARE PUNCTUATION STRING
string.punctuation = '!#"$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
def freq_words(df):    
    text_list = []
    less_frequent_word = []
    for rec in df["text"]:
        for words in rec.split():
            text_list.append(words)
    freqDist = FreqDist(text_list)
    words = list(freqDist.keys())
    for wrd in words:
        if freqDist[wrd] <= 3:
            less_frequent_word.append(wrd)
    return less_frequent_word

col_to_drop_train = []
col_to_drop_test = []
less_freq_wrd_train = freq_words(training_set)
less_freq_wrd_test = freq_words(testing_set)
less_freq_wrd = less_freq_wrd_test + less_freq_wrd_train
once_present_word = list(set(less_freq_wrd))
wrds_ignore = once_present_word+stop_words
def text_preprocessing(text):
    sentiment_value = 0
    sentence = ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    for wrd in text.split():
        if wrd not in wrds_ignore:
            #if wrd in words:
            if wrd.isalpha():
                wrd = lemmatizer.lemmatize(wrd)
                if wrd not in sentence:
                    sentence += " ".join(wrd.split())+" "

    sentence = sentence.strip()
    return sentence
training_set["text"] = training_set["text"].apply(text_preprocessing)
testing_set["text"] = testing_set["text"].apply(text_preprocessing)
dictionary = []
for dataframe in [training_set["text"],testing_set["text"]]:
    for sentence in dataframe:
        for words in sentence.split():
            dictionary.append(words)
            
unique_dictionary = list(set(dictionary))
# for text,iD in zip(training_set[training_set.keyword.isnull()]["text"],training_set[training_set.keyword.isnull()]["text"].index):
#     for wrd in text.split():
#         if wrd.startswith("#"):
#             training_set.loc[iD,"keyword"] = wrd[1:]
# for text,iD in zip(testing_set[testing_set.keyword.isnull()]["text"],testing_set[testing_set.keyword.isnull()]["text"].index):
#     for wrd in text.split():
#         if wrd.startswith("#"):
#             testing_set.loc[iD,"keyword"] = wrd[1:]
#training_set[training_set["target"]==0]["text"]
for text,iD in zip(testing_set[testing_set.keyword.isnull()]["text"],testing_set[testing_set.keyword.isnull()]["text"].index):
    doc = nlp(text)
    wrds_pos = []
    for ent in doc.ents: 
        if ent.label_ == "GPE":
            #testing_set.loc[iD,"keyword"] = ent.text
            #testing_set.loc[iD,"location"] = ent.text
            wrds_pos.append(ent.text)
        elif ent.label_ == "NORP":
            #testing_set.loc[iD,"keyword"] = ent.text
            wrds_pos.append(ent.text)
        elif ent.label_ == "LOC":
            #testing_set.loc[iD,"keyword"] = ent.text
            wrds_pos.append(ent.text)
        elif ent.label_ == "ORG":
            #testing_set.loc[iD,"keyword"] = ent.text
            wrds_pos.append(ent.text)
        elif ent.label_ =="EVENT":
            #testing_set.loc[iD,"keyword"] = ent.text
            wrds_pos.append(ent.text)
        elif ent.label_ == "FAC":
            #testing_set.loc[iD,"keyword"] = ent.text
            wrds_pos.append(ent.text)
        elif ent.label_ == "LANGUAGE":
            #testing_set.loc[iD,"keyword"] = ent.text
            wrds_pos.append(ent.text)
        elif ent.label_ == "PRODUCT":
            #testing_set.loc[iD,"keyword"] = ent.text
            wrds_pos.append(ent.text)
#         elif ent.label_ == 'WORK_OF_ART':
#             testing_set.loc[iD,"keyword"] = ent.text
#             wrds_pos.append(ent.text)
        else:
            wrds_pos.append("")
    for token in nlp(text):
        if token.pos_ == "NOUN":
            #testing_set.loc[iD,"keyword"] = token.text
            wrds_pos.append(token.text)
        elif token.pos_ == "VERB":
            #testing_set.loc[iD,"keyword"] = token.text
            wrds_pos.append(token.text)
#         elif token.pos_ == "PROPN":
#             testing_set.loc[iD,"keyword"] = token.text
#             wrds_pos.append(token.text)
#         elif token.pos_ == "ADJ":
#             testing_set.loc[iD,"keyword"] = token.text
#             wrds_pos.append(token.text)
#         if token.pos_ == "ADV":
#             testing_set.loc[iD,"keyword"] = token.text
#             wrds_pos.append(token.text)
    sent = " ".join(wrd for wrd in wrds_pos)
    testing_set.loc[iD,"text"] = sent
for text,iD in zip(training_set[training_set.keyword.isnull()]["text"],training_set[training_set.keyword.isnull()]["text"].index):
    doc = nlp(text)
    wrds_pos = []
    for ent in doc.ents: 
        if ent.label_ == "GPE":
            #training_set.loc[iD,"keyword"] = ent.text
            #training_set.loc[iD,"location"] = ent.text
            wrds_pos.append(ent.text)
        elif ent.label_ == "NORP":
            #training_set.loc[iD,"keyword"] = ent.text
            wrds_pos.append(ent.text)
        elif ent.label_ == "ORG":
            #training_set.loc[iD,"keyword"] = ent.text
            wrds_pos.append(ent.text)
        elif ent.label_ == "LOC":
            #training_set.loc[iD,"keyword"] = ent.text
            wrds_pos.append(ent.text)
        elif ent.label_ =="EVENT":
            #training_set.loc[iD,"keyword"] = ent.text
            wrds_pos.append(ent.text)
        elif ent.label_ == "FAC":
            #training_set.loc[iD,"keyword"] = ent.text
            wrds_pos.append(ent.text)
        elif ent.label_ == "LANGUAGE":
            #training_set.loc[iD,"keyword"] = ent.text
            wrds_pos.append(ent.text)
        elif ent.label_ == "PRODUCT":
            #training_set.loc[iD,"keyword"] = ent.text
            wrds_pos.append(ent.text)
#         elif ent.label_ == 'WORK_OF_ART':
#             training_set.loc[iD,"keyword"] = ent.text
#             wrds_pos.append(ent.text)
#         elif ent.label_ == 'PERSON':
#             training_set.loc[iD,"keyword"] = ent.text
#             wrds_pos.append(ent.text)
        else:
            wrds_pos.append("")
    for token in nlp(text):
        if token.pos_ == "NOUN":
            #training_set.loc[iD,"keyword"] = token.text
            wrds_pos.append(token.text)
        elif token.pos_ == "VERB":
            #training_set.loc[iD,"keyword"] = token.text
            wrds_pos.append(token.text)
#         elif token.pos_ == "PROPN":
#             training_set.loc[iD,"keyword"] = token.text
#             wrds_pos.append(token.text)
#         elif token.pos_ == "ADJ":
#             training_set.loc[iD,"keyword"] = token.text
#             wrds_pos.append(token.text)
#         if token.pos_ == "ADV":
#             training_set.loc[iD,"keyword"] = token.text
#             wrds_pos.append(token.text)
    sent = " ".join(wrd for wrd in wrds_pos)
    training_set.loc[iD,"text"] = sent
# creating label
y = training_set["target"]
# training_set["Sentiment"] = pd.Series()
# testing_set["Sentiment"] = pd.Series()
# training_set["Subjectivity"] = pd.Series()
# testing_set["Subjectivity"] = pd.Series()
# def calculate_sentiment(sentence):
#     text = TextBlob(sentence)
#     #print(text)
#     sentiment_value = text.sentiment.polarity
#     return sentiment_value
# def calculate_subjectivity(sentence):
#     text = TextBlob(sentence)
#     #print(text)
#     subject_value = text.sentiment.subjectivity
#     return subject_value
# training_set["Sentiment"] =  training_set["text"].apply(calculate_sentiment)
# training_set["Subjectivity"] =  training_set["text"].apply(calculate_subjectivity)
# testing_set["Sentiment"] =  testing_set["text"].apply(calculate_sentiment)
# testing_set["Subjectivity"] =  testing_set["text"].apply(calculate_subjectivity)
# print(training_set[training_set["target"]==1]["Sentiment"].sum())
# print(training_set[training_set["target"]==0]["Sentiment"].sum())
# print(training_set[training_set["target"]==1]["Subjectivity"].sum())
# print(training_set[training_set["target"]==0]["Subjectivity"].sum())
tfidf_vector = TfidfVectorizer(ngram_range=(1,2))
# training_set["keyword"] = training_set["keyword"].str.replace("%20"," ")
# testing_set["keyword"] = testing_set["keyword"].str.replace("%20"," ")
#training_set.loc[training_set["target"]==0,"keyword"] = "unknown"
# train_kw = pd.DataFrame(training_set["keyword"])
# test_kw = pd.DataFrame(testing_set["keyword"])
# training_set["text"] = training_set["text"]+" "+train_kw["keyword"]
# testing_set["text"] = testing_set["text"]+" "+test_kw["keyword"]
testing_set["text"]
text_vector_train = pd.DataFrame(tfidf_vector.fit_transform(training_set["text"].apply(lambda x: np.str_(x))).toarray(),columns = tfidf_vector.get_feature_names())
text_vector_test = pd.DataFrame(tfidf_vector.transform(testing_set["text"].apply(lambda x: np.str_(x))).toarray(),columns = tfidf_vector.get_feature_names())
# train_sent = pd.DataFrame(training_set,columns=["Sentiment","Subjectivity"])
# train_sent.loc[train_sent["Sentiment"]<= 0.5,"Sentiment"] = 0
#train_sent.loc[train_sent["Sentiment"]> 0.5,"Sentiment"] = 1
#train_sent["Sentiment"].value_counts()
#test_sent = pd.DataFrame(testing_set,columns = ["Sentiment","Subjectivity"])
#train_sent.corr(method ='pearson') 
#train_sent["Senti_Subj"] = train_sent["Sentiment"] + train_sent["Subjectivity"]
#train_sent.drop(columns=["Sentiment","Subjectivity"],inplace=True)
#X = pd.concat([text_vector_train, train_sent], axis=1)
#X_test = pd.concat([text_vector_test,test_sent],axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(text_vector_train, y,random_state=42, test_size=0.30)

print('Number of rows in the total set: {}'.format(text_vector_train.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_valid.shape[0]))
# Tried out - Models with worst performance
# default_classifiers = {'Gradient Boosting Classifier':GradientBoostingClassifier(),'Adaptive Boosting Classifier':AdaBoostClassifier(),'RadiusNN':RadiusNeighborsClassifier(radius=40.0),
#                'Linear Discriminant Analysis':LinearDiscriminantAnalysis(), 'GaussianNB': GaussianNB(), 'BerNB': BernoulliNB(), 'KNN': KNeighborsClassifier(),
#                'Random Forest Classifier': RandomForestClassifier(min_samples_leaf=10,min_samples_split=20,max_depth=4),'Decision Tree Classifier': DecisionTreeClassifier(),'Logistic Regression':LogisticRegression(), "XGBoost": xgb.XGBClassifier()}
#Dictionary of models
classifiers = {'BerNB': BernoulliNB(),'Logistic Regression':LogisticRegression()}
#Iterating through the dataset with all model declared in the dataset.
base_accuracy = 0
for Name,classify in classifiers.items():
    classify.fit(X_train,y_train)
    y_predictng = classify.predict(X_valid)
    print('Accuracy Score of '+str(Name) + " : " +str(accuracy_score(y_valid,y_predictng)))
    
#X_test_normalized= normalize(X_test, norm='l2')
pID = sample_submission["id"]
predicted_test = classify.predict(text_vector_test)
predicted_test_value = pd.DataFrame({ 'id': pID,
                        'target': predicted_test })
predicted_test_value.to_csv("PredictedTestScore.csv", index=False)
# predicted_test = []
# for x in model.predict_classes(X_test_ica):
#     predicted_test.append(x[:][0])
# predicted_test_value = pd.DataFrame({ 'PassengerId': pID,
#                         'Survived': predicted_test })
# predicted_test_value.to_csv("PredictedTestScore.csv", index=False)