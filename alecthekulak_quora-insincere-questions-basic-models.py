import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings("ignore")
import spacy
nlp = spacy.load('en')
training_data_raw = pd.read_csv("../input/train.csv")
training_data_raw.drop('qid', axis=1, inplace=True)
all_labels = training_data_raw.pop('target')
print(f"{len(training_data_raw):,} total training datapoints")
print(training_data_raw.values[:2])
print(all_labels.value_counts())
print(all_labels.value_counts(True))
# Preprocessing 
import re
NON_CHARACTER = re.compile(r'[^A-Za-z]+') #(?u)
NUMS = re.compile(r'\d+')
from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords 
# STOPS = stopwords.words('english')
from nltk.stem import WordNetLemmatizer
l = WordNetLemmatizer()
def process(text): 
    text = text.lower().replace('\\', '\\\\')
    text = NUMS.sub('XXX', text)
    text = NON_CHARACTER.sub(' ', text)
#     text = ' '.join([l.lemmatize(word) for word in word_tokenize(text) if word not in STOPS])
    text = ' '.join([l.lemmatize(word) for word in word_tokenize(text)])
    return text 
# for x in training_data_raw.values[:10]:
#     print(x[0])
#     print(process(x[0]))
all_texts = np.array([process(x[0]) for x in training_data_raw.values])
print("Done!")
train_x, test_x, train_y, test_y = train_test_split(
    all_texts, all_labels.values, test_size=0.2, random_state=0)
print("Test-train split done!")
vectorizer = CountVectorizer()
vectorizer2 = CountVectorizer(min_df=0.0001, max_df=0.999, max_features=5000, ngram_range=(1,2,)) 
bow_train = vectorizer.fit_transform(train_x) 
bow_train2 = vectorizer2.fit_transform(train_x) 
print(bow_train.shape)
print(bow_train2.shape)
bow_test = vectorizer.transform(test_x)
bow_test2 = vectorizer2.transform(test_x)
print("Done creating Bag-of-Words")
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
print(f"Results of logistic regression on full bag-of-words")
logistic = LogisticRegression(penalty="l1", C=3.5) 
logistic.fit(bow_train, train_y) 
train_predictions = logistic.predict(bow_train)
train_acc = accuracy_score(train_y, train_predictions)  #all_labels
train_f1 = f1_score(train_y, train_predictions) 
print(f"Training accuracy: {train_acc:.2%}, F1: {train_f1:.4f}, %1: {sum(train_predictions)/len(train_predictions):.2%}") 
test_predictions = logistic.predict(bow_test)
test_acc = accuracy_score(test_y, test_predictions) 
test_f1 = f1_score(test_y, test_predictions) 
print(f"Testing accuracy:  {test_acc:.2%}, F1: {test_f1:.4f}, %1: {sum(test_predictions)/len(test_predictions):.2%}")
# Training accuracy: 96.37%, F1: 0.6580, %1: 4.42%
# Testing accuracy:  95.33%, F1: 0.5504, %1: 4.22%
print(f"Results of logistic regression on simplified bag-of-words")
logistic2 = LogisticRegression(penalty="l1", C=3.5) 
logistic2.fit(bow_train2, train_y)
train_predictions = logistic2.predict(bow_train2)
train_acc = accuracy_score(train_y, train_predictions) 
train_f1 = f1_score(train_y, train_predictions) 
print(f"Training accuracy: {train_acc:.2%}, F1: {train_f1:.4f}, %1: {sum(train_predictions)/len(train_predictions):.2%}") 
test_predictions = logistic2.predict(bow_test2)
test_acc = accuracy_score(test_y, test_predictions) 
test_f1 = f1_score(test_y, test_predictions) 
print(f"Testing accuracy:  {test_acc:.2%}, F1: {test_f1:.4f}, %1: {sum(test_predictions)/len(test_predictions):.2%}")
# Training accuracy: 95.26%, F1: 0.5187, %1: 3.65%
# Testing accuracy:  95.14%, F1: 0.5051, %1: 3.66%
# print(f"Results of decision tree on full bag-of-words")
# dtc = DecisionTreeClassifier(max_depth=30) 
# dtc.fit(bow_train, train_y)
# train_predictions = dtc.predict(bow_train)
# train_acc = accuracy_score(train_y, train_predictions) 
# train_f1 = f1_score(train_y, train_predictions) 
# print(f"Training accuracy: {train_acc:.2%}, F1: {train_f1:.4f}, %1: {sum(train_predictions)/len(train_predictions):.2%}") 
# test_predictions = dtc.predict(bow_test)
# test_acc = accuracy_score(test_y, test_predictions) 
# test_f1 = f1_score(test_y, test_predictions) 
# print(f"Testing accuracy:  {test_acc:.2%}, F1: {test_f1:.4f}, %1: {sum(test_predictions)/len(test_predictions):.2%}")
# print(f"Results of decision tree on simplified bag-of-words")
# dtc2 = DecisionTreeClassifier(max_depth=30) 
# dtc2.fit(bow_train2, train_y)
# train_predictions = dtc2.predict(bow_train2)
# train_acc = accuracy_score(train_y, train_predictions) 
# train_f1 = f1_score(train_y, train_predictions) 
# print(f"Training accuracy: {train_acc:.2%}, F1: {train_f1:.4f}, %1: {sum(train_predictions)/len(train_predictions):.2%}") 
# test_predictions = dtc2.predict(bow_test2)
# test_acc = accuracy_score(test_y, test_predictions) 
# test_f1 = f1_score(test_y, test_predictions) 
# print(f"Testing accuracy:  {test_acc:.2%}, F1: {test_f1:.4f}, %1: {sum(test_predictions)/len(test_predictions):.2%}")
# print(f"Results of random forest on full bag-of-words")
# rfc = RandomForestClassifier(n_estimators=100, max_depth=3, class_weight='balanced') 
# rfc.fit(bow_train, train_y)
# train_predictions = rfc.predict(bow_train)
# train_acc = accuracy_score(train_y, train_predictions) 
# train_f1 = f1_score(train_y, train_predictions) 
# print(f"Training accuracy: {train_acc:.2%}, F1: {train_f1:.4f}, %1: {sum(train_predictions)/len(train_predictions):.2%}") 
# test_predictions = rfc.predict(bow_test)
# test_acc = accuracy_score(test_y, test_predictions) 
# test_f1 = f1_score(test_y, test_predictions) 
# print(f"Testing accuracy:  {test_acc:.2%}, F1: {test_f1:.4f}, %1: {sum(test_predictions)/len(test_predictions):.2%}")
# print(f"Results of random forest on simplified bag-of-words") 
# rfc2 = RandomForestClassifier(n_estimators=100, max_depth=3, class_weight='balanced') 
# rfc2.fit(bow_train2, train_y) 
# train_predictions = rfc2.predict(bow_train2)
# train_acc = accuracy_score(train_y, train_predictions) 
# train_f1 = f1_score(train_y, train_predictions) 
# print(f"Training accuracy: {train_acc:.2%}, F1: {train_f1:.4f}, %1: {sum(train_predictions)/len(train_predictions):.2%}") 
# test_predictions = rfc2.predict(bow_test2)
# test_acc = accuracy_score(test_y, test_predictions) 
# test_f1 = f1_score(test_y, test_predictions) 
# print(f"Testing accuracy:  {test_acc:.2%}, F1: {test_f1:.4f}, %1: {sum(test_predictions)/len(test_predictions):.2%}")
# print(f"Results of gradient boosting on full bag-of-words")
# gbc = GradientBoostingClassifier() 
# gbc.fit(bow_train, train_y) 
# train_predictions = gbc.predict(bow_train)
# train_acc = accuracy_score(train_y, train_predictions) 
# train_f1 = f1_score(train_y, train_predictions) 
# print(f"Training accuracy: {train_acc:.2%}, F1: {train_f1:.4f}, %1: {sum(train_predictions)/len(train_predictions):.2%}") 
# test_predictions = gbc.predict(bow_test)
# test_acc = accuracy_score(test_y, test_predictions) 
# test_f1 = f1_score(test_y, test_predictions) 
# print(f"Testing accuracy:  {test_acc:.2%}, F1: {test_f1:.4f}, %1: {sum(test_predictions)/len(test_predictions):.2%}")
# # Training accuracy: 94.56%, F1: 0.3035, %1: 1.61%
# # Testing accuracy:  94.58%, F1: 0.3014, %1: 1.61%
# print(f"Results of gradient boosting on simplified bag-of-words")
# gbc2 = GradientBoostingClassifier() 
# gbc2.fit(bow_train2, train_y)
# train_predictions = gbc2.predict(bow_train2)
# train_acc = accuracy_score(train_y, train_predictions) 
# train_f1 = f1_score(train_y, train_predictions) 
# print(f"Training accuracy: {train_acc:.2%}, F1: {train_f1:.4f}, %1: {sum(train_predictions)/len(train_predictions):.2%}") 
# test_predictions = gbc2.predict(bow_test2)
# test_acc = accuracy_score(test_y, test_predictions) 
# test_f1 = f1_score(test_y, test_predictions) 
# print(f"Testing accuracy:  {test_acc:.2%}, F1: {test_f1:.4f}, %1: {sum(test_predictions)/len(test_predictions):.2%}")
# # Training accuracy: 94.59%, F1: 0.3123, %1: 1.67%
# # Testing accuracy:  94.60%, F1: 0.3095, %1: 1.66%
validation_data = pd.read_csv("../input/test.csv")
final_vectorizer = CountVectorizer()
# final_vectorizer = CountVectorizer(min_df=0.0005, max_features=5000, ngram_range=(1,2,)) 
final_model = LogisticRegression(penalty="l1", C=3.5)
# final_model = RandomForestClassifier(n_estimators=100, max_depth=3, class_weight='balanced') 
### Code running 
print(f"Generating final model")
final_bow = final_vectorizer.fit_transform(all_texts) 
final_model.fit(final_bow, all_labels.values) 
final_train_predictions = final_model.predict(final_bow) 
final_acc = accuracy_score(all_labels.values, final_train_predictions) 
final_f1 = f1_score(all_labels.values, final_train_predictions) 
print(f"Final model accuracy:  {final_acc:.2%}, F1: {final_f1:.4f}, %1: {sum(final_train_predictions)/len(final_train_predictions):.2%}, %1 actual: {sum(all_labels.values)/len(all_labels.values):.2%}") 
# LogisticRegression(penalty="l1") Final model accuracy:  95.73%, F1: 0.5814, %1: 4.02%, %1 actual: 6.19%
# LogisticRegression(penalty="l1", C=2.5) Final model accuracy:  96.23%, F1: 0.6415, %1: 4.32%, %1 actual: 6.19%
# LogisticRegression(penalty="l1", C=3.0) Final model accuracy:  96.27%, F1: 0.6466, %1: 4.37%, %1 actual: 6.19%
# LogisticRegression(penalty="l1", C=3.5)
validation_texts = np.array([process(x) for x in validation_data['question_text']])
print(validation_texts[:3])
validation_bow = final_vectorizer.transform(validation_texts) 
validation_predictions = final_model.predict(validation_bow) 
print(validation_predictions[:3])
print(f"Submission %1: {sum(validation_predictions)/len(validation_predictions):.2%}")
validation_data.drop('question_text', axis=1, inplace=True)
validation_data['prediction'] = validation_predictions 
validation_data.to_csv('submission.csv', index=False)