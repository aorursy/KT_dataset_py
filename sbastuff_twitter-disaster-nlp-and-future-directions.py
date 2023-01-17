import numpy as np

import pandas as pd

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

print("Importing Training Data")

train_df = pd.read_csv("../input/nlp-getting-started/train.csv")

print("Import Done")

print("Importing Testing Data")

test_df = pd.read_csv("../input/nlp-getting-started/test.csv")

print("Import Done")

print("Counting Vectors")

count_vectorizer = feature_extraction.text.CountVectorizer()

print("Vector Count Done")

print("creating training vectors")

train_vectors = count_vectorizer.fit_transform(train_df["text"])

print("training vectors done")

print("creating testing vectors")

test_vectors = count_vectorizer.transform(test_df["text"])

print("testing vectors done")



print("initializing Ridge Linear Classifier")

clf = linear_model.RidgeClassifier()

print("initialized")



print("generating scores")

scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")

print(" scores generated")

print(scores)

print("Building Predictions")

clf.fit(train_vectors, train_df["target"])

train_vectors = count_vectorizer.fit_transform(train_df["text"])

sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

sample_submission["target"] = clf.predict(test_vectors)

print("Predictions Built")

print("Saving Predictions")

sample_submission.to_csv("results.csv", index=False)

print("Predictions Saved.")

saved_predictions = pd.read_csv("results.csv")

print("Showing First Few Predictions.")

print(saved_predictions)
import numpy as np

import sys

import pandas as pd

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

def predictioncore(userinput):

    train_df = pd.read_csv("datasets/train.csv")

    test_df = pd.read_csv("datasets/test.csv")

    count_vectorizer = feature_extraction.text.CountVectorizer()

    train_vectors = count_vectorizer.fit_transform(train_df["text"])

    herecomestheinput = userinput

    text=[herecomestheinput,]

    test_vectors = count_vectorizer.transform(text)

    clf = linear_model.RidgeClassifier()

    scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")

    clf.fit(train_vectors, train_df["target"])

    train_vectors = count_vectorizer.fit_transform(train_df["text"])

    bindas = clf.predict(test_vectors)

    status = ""

    if bindas==1:

        status = "This Tweet/Post Predicts There is a disaster."

    else:

        status = "No Disaster Predicted."

    return status

if len(sys.argv)==1:

    print("Error Expecting Text Parameter. Should Not Be Nulled.")

if len(sys.argv)==2:

    #print("Blay Blay")

    print(predictioncore(sys.argv[1]))

else:

    print("Error: More or Invalid Outputs. Try giving tweets in double quotes. like python commandline.py \"Hello I Crashed.\" ")