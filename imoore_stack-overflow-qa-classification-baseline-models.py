import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import re

import random

import warnings



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score, confusion_matrix, plot_confusion_matrix, plot_precision_recall_curve



warnings.simplefilter("ignore")
def plot_metric(clf, testX, testY, name):

    """

    Small function to plot ROC-AUC values and confusion matrix

    """

    styles = ['bmh', 'classic', 'fivethirtyeight', 'ggplot']



    plt.style.use(random.choice(styles))

    plot_confusion_matrix(clf, testX, testY)

    plt.title(f"Confusion Matrix [{name}]")
data = pd.read_csv("../input/60k-stack-overflow-questions-with-quality-rate/train.csv")

data2 = pd.read_csv("../input/60k-stack-overflow-questions-with-quality-rate/valid.csv")

data.head()
data = data.drop(['Id', 'Tags', 'CreationDate'], axis=1)

data['Y'] = data['Y'].map({'LQ_CLOSE':0, 'LQ_EDIT': 1, 'HQ':2})



data2 = data2.drop(['Id', 'Tags', 'CreationDate'], axis=1)

data2['Y'] = data2['Y'].map({'LQ_CLOSE':0, 'LQ_EDIT': 1, 'HQ':2})



data.head()
labels = ['Open Questions', 'Low Quality Question - Close', 'Low Quality Question - Edit']

values = [len(data[data['Y'] == 2]), len(data[data['Y'] == 0]), len(data[data['Y'] == 1])]

plt.style.use('classic')

plt.figure(figsize=(16, 9))

plt.pie(x=values, labels=labels, autopct="%1.1f%%")

plt.title("Target Value Distribution")

plt.show()
data['text'] = data['Title'] + ' ' + data['Body']

data = data.drop(['Title', 'Body'], axis=1)



data2['text'] = data2['Title'] + ' ' + data2['Body']

data2 = data2.drop(['Title', 'Body'], axis=1)





data.head()
# Clean the data

def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^(a-zA-Z)\s]','', text)

    return text

data['text'] = data['text'].apply(clean_text)

data2['text'] = data2['text'].apply(clean_text)
# Training Sets

train = data

trainX = train['text']

trainY = train['Y'].values



# Validation Sets

valid = data2

validX = valid['text']

validY = valid['Y'].values



assert trainX.shape == trainY.shape

assert validX.shape == validY.shape



print(f"Training Data Shape: {trainX.shape}\nValidation Data Shape: {validX.shape}")
# Load the vectorizer, fit on training set, transform on validation set

vectorizer = TfidfVectorizer()

trainX = vectorizer.fit_transform(trainX)

validX = vectorizer.transform(validX)
# Define and fit the classifier on the data

lr_classifier = LogisticRegression(C=1.)

lr_classifier.fit(trainX, trainY)
# Print the accuracy score of the classifier

print(f"Validation Accuracy of Logsitic Regression Classifier is: {(lr_classifier.score(validX, validY))*100:.2f}%")
# Also plot the metric

plot_metric(lr_classifier, validX, validY, "Logistic Regression")
# Define and fit the classifier on the data

nb_classifier = MultinomialNB()

nb_classifier.fit(trainX, trainY)
# Print the accuracy score of the classifier

print(f"Validation Accuracy of Naive Bayes Classifier is: {(nb_classifier.score(validX, validY))*100:.2f}%")
# Also plot the metric

plot_metric(nb_classifier, validX, validY, "Naive Bayes")
# Define and fit the classifier on the data

rf_classifier = RandomForestClassifier()

rf_classifier.fit(trainX, trainY)
# Print the accuracy score of the classifier

print(f"Validation Accuracy of Random Forest Classifier is: {(rf_classifier.score(validX, validY))*100:.2f}%")
# Also plot the metric

plot_metric(nb_classifier, validX, validY, "Random Forest")
# Define and fit the classifier on the data

dt_classifier = DecisionTreeClassifier()

dt_classifier.fit(trainX, trainY)
# Print the accuracy score of the classifier

print(f"Validation Accuracy of Decision Tree Clf. is: {(dt_classifier.score(validX, validY))*100:.2f}%")
# Also plot the metric

plot_metric(dt_classifier, validX, validY, "Decision Tree Classifier")
# Define and fit the classifier on the data

kn_classifier = KNeighborsClassifier()

kn_classifier.fit(trainX, trainY)
# Print the accuracy score of the classifier

print(f"Validation Accuracy of KNN Clf. is: {(kn_classifier.score(validX, validY))*100:.2f}%")
# Also plot the metric

plot_metric(dt_classifier, validX, validY, "Decision Tree Classifier")
# Define and fit the classifier on the data

xg_classifier = XGBClassifier()

xg_classifier.fit(trainX, trainY)
# Print the accuracy score of the classifier

print(f"Validation Accuracy of XGBoost Clf. is: {(xg_classifier.score(validX, validY))*100:.2f}%")
# Also plot the metric

plot_metric(xg_classifier, validX, validY, "XGBoost Classifier")