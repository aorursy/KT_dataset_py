# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def preprocess(train_file_path="train.csv", test_file_path="test.csv"):
    """
        Load and split the data into train dataset and test dataset.

    params:
        train_file_path: Path of the file on which the model will be trained.
        test_file_path: Path of the file whose value will be predicted.

    return: different dataframes
    """
    train_file = pd.read_csv(train_file_path)
    test_file = pd.read_csv(test_file_path)

    train_features = train_file.drop(["author"], axis=1)
    train_labels = train_file["author"]

    features_train, features_test, labels_train, labels_test = \
        train_test_split(train_features, train_labels, test_size=0.1, random_state=42)
    
    return features_train, features_test, labels_train, labels_test, test_file


def sep_author_text(data: pd.DataFrame):
    """
        Make a dataframe having all text of each author in a row corresponding to author name.

    params:
    pandas_file: Dataframe containing text of different authors.

    return: Return two variables
            grouped_text: Concanated text of authors.
            authors: List of authors
    """
    authors = list(data["author"].unique())
    grouped_file = data.groupby(["author"])
    grouped_text = pd.DataFrame({})

    for author in authors:
        df = grouped_file.get_group(author)
        all_text = list(df["text"])
        all_text = " ".join(all_text)
        author_dict = {"text": all_text}
        author_df = pd.DataFrame(author_dict, index=[author])
        grouped_text = pd.concat([grouped_text, author_df])

    return grouped_text, authors

# Separating train and test data
train_data_path = "/kaggle/input/spooky-csv/train.csv"
test_data_path = "/kaggle/input/spooky-csv/test.csv"
features_train, features_test, labels_train, labels_test, test_file = preprocess(train_file_path=train_data_path, 
                                                                                 test_file_path=test_data_path)
# Concatanate the features and labels of training data
combined_data = pd.concat([features_train, labels_train], axis=1)

grouped_text, author_list = sep_author_text(combined_data)  # Concatanate the authors text
from nltk.corpus import stopwords

sw = stopwords.words("english")

def remove_stopwords(listOftexts: list):
    global sw
    
    updated_text = []

    for i in listOftexts:
        i = i.split(" ")
        for j in i:
            if j in sw:
                i.remove(j)
        i = " ".join(i)
        updated_text.append(i)
    
    return updated_text
    

grouped_text["text"] = remove_stopwords(grouped_text.text.values)

features_train["text"] = remove_stopwords(features_train.text.values)

features_test["text"] = remove_stopwords(features_test.text.values)

test_file["text"] = remove_stopwords(test_file.text.values)

from nltk.stem.snowball import SnowballStemmer

snowball = SnowballStemmer("english")

def stemSentence(sentence):
    word_list = sentence.split(" ")
    stemmed_words = []
    
    for word in word_list:
        stemmed_words.append(snowball.stem(word))
    
    stemmed_sentence = " ".join(stemmed_words)
    return stemmed_sentence


def stemListOfSentence(listOfsentence: list):
    stemmed_list = []
    
    for sentence in listOfsentence:
        stemmed_list.append(stemSentence(sentence))
    
    return stemmed_list


grouped_text["text"] = stemListOfSentence(grouped_text.text.values)

features_train["text"] = stemListOfSentence(features_train.text.values)

features_test["text"] = stemListOfSentence(features_test.text.values)

test_file["text"] = stemListOfSentence(test_file.text.values)
# convert the text into a matrix of TF-IDF features
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit(list(grouped_text["text"].values))
train_tfidf = vectorizer.transform(features_train.text.values)
test_tfidf = vectorizer.transform(features_test.text.values)
real_test = vectorizer.transform(test_file.text.values)

ytrain = labels_train.replace({"HPL": 1, "EAP": 2, "MWS": 3})
ytest = labels_test.replace({"HPL": 1, "EAP": 2, "MWS": 3})
# Naive bayes multinomial model
from sklearn.naive_bayes import MultinomialNB

clf_NB = MultinomialNB()
clf_NB.fit(train_tfidf, ytrain)
NB_predicted_labels = clf_NB.predict(test_tfidf)

NB_acc_score = accuracy_score(NB_predicted_labels, ytest)
NB_acc_score
NB_predicted_proba = clf_NB.predict_proba(real_test)
NB_predicted_proba_df = pd.DataFrame(NB_predicted_proba, columns=author_list, index=test_file.id.values)
NB_predicted_proba_df.to_csv("/kaggle/working/test_result_MultinomialNB.csv", index=True)

# kaggle submission score(multiclass loss) = 0.61052
# Decision Tree classifier
from sklearn.tree import DecisionTreeClassifier

clf_DT = DecisionTreeClassifier()
clf_DT.fit(train_tfidf, ytrain)
DT_predicted_labels = clf_DT.predict(test_tfidf)

DT_acc_score = accuracy_score(DT_predicted_labels, ytest)
DT_acc_score
clf_DT_impr1 = DecisionTreeClassifier(min_samples_split=400, criterion="entropy")
clf_DT_impr1.fit(train_tfidf, ytrain)
DT_impr1_predicted_labels = clf_DT_impr1.predict(test_tfidf)

DT_impr1_acc_score = accuracy_score(DT_impr1_predicted_labels, ytest)
DT_impr1_acc_score
DT_predicted_proba = clf_DT.predict_proba(real_test)
DT_predicted_proba_df = pd.DataFrame(DT_predicted_proba, columns=author_list, index=test_file.id.values)
# DT_predicted_proba_df.to_csv("/kaggle/working/test_result_DT.csv", index=True)

# kaggle submission score(multiclass loss) = 16.42547 (worst)
# Support Vector Machine
from sklearn.svm import SVC

clf_SVC = SVC(probability=True)
clf_SVC.fit(train_tfidf, ytrain)
# SVC_predicted_labels = clf_SVC.predict(test_tfidf)

# SVC_acc_score = accuracy_score(SVC_predicted_labels, ytest)
# SVC_acc_score  #0.7967313585291114
SVC_rbf_predicted_proba = clf_SVC.predict_proba(real_test)
SVC_rbf_predicted_proba_df = pd.DataFrame(SVC_rbf_predicted_proba, columns=author_list, 
                                          index=test_file.id.values)
SVC_rbf_predicted_proba_df.to_csv("/kaggle/working/test_result_SVC_rbf.csv", index=True)

# kaggle submission score(multiclass loss) = 0.47837
# linear kernel SVC
clf_SVC_linear = SVC(kernel="linear", probability=True)
clf_SVC_linear.fit(train_tfidf, ytrain)
# SVC_linear_predicted_labels = clf_SVC_linear.predict(test_tfidf)

# SVC_linear_acc_score = accuracy_score(SVC_linear_predicted_labels, ytest)
# SVC_linear_acc_score
SVC_linear_predicted_proba = clf_SVC_linear.predict_proba(real_test)
SVC_linear_predicted_proba_df = pd.DataFrame(SVC_linear_predicted_proba, columns=author_list, 
                                             index=test_file.id.values)
SVC_linear_predicted_proba_df.to_csv("/kaggle/working/test_result_SVC_linear.csv", index=True)

# kaggle submission score(multiclass loss) = 0.45481 (best)
# K-nearest neighbors classifier
from sklearn.neighbors import KNeighborsClassifier

clf_KNN = KNeighborsClassifier(n_neighbors=20)
clf_KNN.fit(train_tfidf, ytrain)
KNN_predicted_labels = clf_KNN.predict(test_tfidf)

KNN_acc_score = accuracy_score(KNN_predicted_labels, ytest)
KNN_acc_score
KNN_predicted_proba = clf_KNN.predict_proba(real_test)
KNN_predicted_proba_df = pd.DataFrame(KNN_predicted_proba, columns=author_list, 
                                             index=test_file.id.values)
KNN_predicted_proba_df.to_csv("/kaggle/working/test_result_KNN.csv", index=True)

# kaggle submission score(multiclass loss) = 0.93239
# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

clf_randomforest = RandomForestClassifier(min_samples_split=10)
clf_randomforest.fit(train_tfidf, ytrain)
randomforest_predicted_labels = clf_randomforest.predict(test_tfidf)

randomforest_acc_score = accuracy_score(randomforest_predicted_labels, ytest)
randomforest_acc_score
randomforest_predicted_proba = clf_randomforest.predict_proba(real_test)
randomforest_predicted_proba_df = pd.DataFrame(randomforest_predicted_proba, columns=author_list, 
                                             index=test_file.id.values)
randomforest_predicted_proba_df.to_csv("/kaggle/working/test_result_randomforest.csv", index=True)

# kaggle submission score(multiclass loss) = 0.73415
# AdaBoost classifier
from sklearn.ensemble import AdaBoostClassifier

clf_adaboost = AdaBoostClassifier(n_estimators=100, random_state=0)
clf_adaboost.fit(train_tfidf, ytrain)
adaboost_predicted_labels = clf_adaboost.predict(test_tfidf)

adaboost_acc_score = accuracy_score(adaboost_predicted_labels, ytest)
adaboost_acc_score
adaboost_predicted_proba = clf_adaboost.predict_proba(real_test)
adaboost_predicted_proba_df = pd.DataFrame(adaboost_predicted_proba, columns=author_list, 
                                             index=test_file.id.values)
adaboost_predicted_proba_df.to_csv("/kaggle/working/test_result_adaboost.csv", index=True)

# kaggle submission score(multiclass loss) = 1.07863
