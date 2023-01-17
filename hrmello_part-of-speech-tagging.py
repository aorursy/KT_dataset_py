import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

import re

import matplotlib.pyplot as plt

from collections import Counter



import os

print("File Folders:")

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")



train["isTrain"] = True

test["isTrain"] = False



full = pd.concat([train, test])

full
def get_at(row):

    return re.findall("@[\w]+", row["text"])



def get_http(row):

    return re.findall("http[\:\/\.\w]+", row["text"])



def get_hashtags(row):

    return re.findall("#[\w]+", row["text"])



def number_of_tags(row):

    return len(row["tags"])



def number_of_links(row):

    return len(row["links"])



def number_of_hashs(row):

    return len(row["hashtags"])



def clean_text(row):

    clean = row["text"]

    

    if len(row["tags"]) != 0:

        for word in row["tags"]:

            clean = clean.replace(word, "")

    

    if len(row["links"]) != 0:

        for word in row["links"]:

            clean = clean.replace(word, "")

    

    #only remove the # symbol

    clean = clean.replace("#", "").replace("/", "").replace("(", "").replace(")", "")

    

    return clean.strip()



full["tags"] = full.apply(lambda row: get_at(row), axis = 1)

full["links"] = full.apply(lambda row: get_http(row), axis = 1)

full["hashtags"] = full.apply(lambda row: get_hashtags(row), axis = 1)



full["number_of_tags"] = full.apply(lambda row: number_of_tags(row), axis = 1)

full["number_of_links"] = full.apply(lambda row: number_of_links(row), axis = 1)

full["number_of_hashs"] = full.apply(lambda row: number_of_hashs(row), axis = 1)



full["clean_text"] = full.apply(lambda row: clean_text(row), axis = 1)

full.sample(5)
from nltk.tokenize import word_tokenize



def get_tokens(row):

    return word_tokenize(row["clean_text"].lower())



full["tokens"] = full.apply(lambda row: get_tokens(row), axis = 1)

full.sample(5, random_state = 4)
s = ["screams", "in", "the", "distance"]



def get_postags(row):

    

    postags = nltk.pos_tag(row["tokens"])

    list_classes = list()

    for  word in postags:

        list_classes.append(word[1])

    

    return list_classes



full["postags"] = full.apply(lambda row: get_postags(row), axis = 1)

full.sample(5, random_state = 4)

# nltk.help.upenn_tagset('NNS')
def find_no_class(count, class_name = ""):

    total = 0

    for key in count.keys():

        if key.startswith(class_name):

            total += count[key]

            

            

    return total



def get_classes(row, grammatical_class = ""):

    count = Counter(row["postags"])

    return find_no_class(count, class_name = grammatical_class)/len(row["postags"])



full["freqAdverbs"] = full.apply(lambda row: get_classes(row, "RB"), axis = 1)

full["freqVerbs"] = full.apply(lambda row: get_classes(row, "VB"), axis = 1)

full["freqAdjectives"] = full.apply(lambda row: get_classes(row, "JJ"), axis = 1)

full["freqNouns"] = full.apply(lambda row: get_classes(row, "NN"), axis = 1)



full.sample(5)
training = full.loc[full["isTrain"] == True, :].copy()

testing = full.loc[full["isTrain"] == False, :].copy()
training.loc[training["target"] == 0.0, "freqNouns"].hist(alpha = 0.5);

training.loc[training["target"] == 1.0, "freqNouns"].hist(alpha = 0.5);
training.loc[training["target"] == 0.0, "freqVerbs"].hist(alpha = 0.5);

training.loc[training["target"] == 1.0, "freqVerbs"].hist(alpha = 0.5);
training.loc[training["target"] == 0.0, "freqAdjectives"].hist(alpha = 0.5);

training.loc[training["target"] == 1.0, "freqAdjectives"].hist(alpha = 0.5);
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.metrics import accuracy_score, confusion_matrix



x = training.loc[:, ["number_of_tags", "number_of_links", "freqAdverbs", "freqVerbs", "freqAdjectives", "freqNouns"]]

y = training.loc[:, "target"]



skf = StratifiedKFold(n_splits=5)

skf.get_n_splits(x, y)



for train_index, test_index in skf.split(x, y):

    print("TRAIN:", train_index, "TEST:", test_index)

    x_train, x_test = x.loc[train_index], x.loc[test_index]

    y_train, y_test = y.loc[train_index], y.loc[test_index]

    

    clf = GradientBoostingClassifier(learning_rate=0.1, max_depth= 5, max_features = 5,random_state = 42)

#     clf = RandomForestClassifier(random_state = 42)

    

    clf.fit(x_train, y_train)

    preds = clf.predict(x_test)

    

    print(accuracy_score(y_test, preds))

    

    print(confusion_matrix(y_test, preds))





total_preds = clf.predict(x)

print("Confusion Matrix:")

confusion_matrix(y,total_preds)
feature_importance = clf.feature_importances_

# make importances relative to max importance

feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)

pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(20,15))

plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, x.columns[sorted_idx])

plt.xlabel('Relative Importance')

plt.title('Variable Importance')

plt.show()
training.loc[training["target"] == 0.0, "number_of_links"].hist(alpha = 0.5);

training.loc[training["target"] == 1.0, "number_of_links"].hist(alpha = 0.5);
preds = clf.predict(testing.loc[:, ["number_of_tags", "number_of_links", "freqAdverbs", "freqVerbs", "freqAdjectives", "freqNouns"]])

testing["prediction"] = preds
submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

submission["target"] = preds.astype(int)

submission.to_csv("submission.csv", index = False)