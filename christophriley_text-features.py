# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
trainData = pd.read_csv("../input/train.csv")

testData = pd.read_csv("../input/test.csv")
trainData.head()
trainData.describe()
trainData.describe(include=["O"])
trainData.Sex.value_counts().plot(kind="bar")
trainData.Pclass.value_counts().plot(kind="bar")
trainData.groupby(["Sex", "Survived"])["Survived"].count()
trainData[["PassengerId","Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]].corrwith(trainData.Survived)
#select the features we're interested in

#categorical data needs to be handled differently

numericFeatures = ["Pclass", "Age", "Fare", "Parch", "SibSp"]

catFeatures = ["Sex", "Embarked"]

textColumns = ["Name"]
#numeric features are pretty easy

X_num_feat = trainData[numericFeatures].fillna(0).as_matrix()

test_num_feat = testData[numericFeatures].fillna(0).as_matrix()

num_feat_labels = numericFeatures

print(num_feat_labels)
#transform the categorical features with one-hot, then concatenate with numeric features

X_cat_feat = pd.get_dummies(trainData[catFeatures]).fillna(0).as_matrix()

test_cat_feat = pd.get_dummies(testData[catFeatures]).fillna(0).as_matrix()

cat_feat_labels = list(pd.get_dummies(trainData[catFeatures]))

print(cat_feat_labels)
#transform the text features with SKLearn's CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer as Vectorizer

X_text_feat = np.ndarray(shape=(len(trainData), 0))

test_text_feat = np.ndarray(shape=(len(testData), 0))

text_feat_labels = []



for column in textColumns:

    vectorizer = Vectorizer(binary=True)

    #fit to the training data, transform the test data

    X_column_feat = vectorizer.fit_transform(trainData[column]).toarray()

    test_column_feat = vectorizer.transform(testData[column]).toarray()

    

    #append to the text features arrays

    X_text_feat = np.append(X_text_feat, X_column_feat, axis=1)

    test_text_feat = np.append(test_text_feat, test_column_feat, axis=1)

    text_feat_labels += vectorizer.get_feature_names()

print(X_text_feat.shape)

print(len(text_feat_labels))
#Build the full feature matrices

X_full = np.append(np.append(X_num_feat, X_cat_feat, axis=1), X_text_feat, axis=1)

test_full = np.append(np.append(test_num_feat, test_cat_feat, axis=1), test_text_feat, axis=1)

#and the feature labels

feature_labels_full = np.array(num_feat_labels + cat_feat_labels + text_feat_labels)

#and the observation labels

Y = trainData["Survived"].fillna(0).as_matrix()

print(X_full.shape)

print(Y.shape)
#Feature selection

from sklearn.feature_selection import VarianceThreshold

thresholdValue = 0.995

thresh = VarianceThreshold(threshold=(thresholdValue * (1 - thresholdValue)))

X = thresh.fit_transform(X_full)

test = thresh.transform(test_full)

feature_labels = feature_labels_full[thresh.get_support(indices=True)]

print(feature_labels)
from sklearn import tree



dtree = tree.DecisionTreeClassifier(max_depth=3)

dtree.fit(X, Y)
import graphviz

dot_data = tree.export_graphviz(dtree, out_file=None, feature_names=feature_labels)

graph = graphviz.Source(dot_data)

graph
testData["Survived"] = dtree.predict(test)
predictions = testData[["PassengerId", "Survived"]]

predictions.to_csv("dtree_predictions.csv", index=False)

print(check_output(["ls", "."]).decode("utf8"))
predictions.head()