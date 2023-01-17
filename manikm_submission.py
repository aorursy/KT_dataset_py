import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.svm import SVC

from sklearn.neighbors import *

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline
test_data = pd.read_csv('test_2.csv', usecols=["ID", "viewCount", "likeCount","dislikeCount","commentCount","title","description"])

train_data = pd.read_csv('train.csv', usecols=["class", "viewCount", "likeCount","dislikeCount","commentCount","title","description"])



Y_train = train_data["class"]



X_train = train_data.drop("class", axis=1)

X_test = test_data.drop("ID", axis=1)



def isupper(text):

    array = []

    for x in text:

        if x.isupper() or x=='!':

            array.append(x)

    return len(array)



def istrue(x):

    if x == True:

        return 1

    else:

        return 0



X_train["upper"] = X_train["title"].apply(lambda x: isupper(x))

X_test["upper"] = X_test["title"].apply(lambda x: isupper(x))

X_train["title_length"] = X_train["title"].apply(lambda x: len(x))

X_test["title_length"] = X_test["title"].apply(lambda x: len(x))

X_train["description_length"] = X_train["description"].apply(lambda x: len(x))

X_test["description_length"] = X_test["description"].apply(lambda x: len(x))





X_train["likeRatio"] = X_train["likeCount"] / (X_train["likeCount"]+X_train["dislikeCount"])

X_train.drop("dislikeCount", axis=1)

X_train.drop("likeCount", axis=1)

X_test["likeRatio"] = X_test["likeCount"] / (X_test["likeCount"]+X_test["dislikeCount"])

X_test.drop("dislikeCount", axis=1)

X_test.drop("likeCount", axis=1)





X_train, X_practice, Y_train, Y_practice = train_test_split(X_train, Y_train, test_size=.1)





vector = Pipeline(steps=[('coutvectorizer', CountVectorizer()),('test', MultinomialNB())])

vector = vector.fit(X_train["title"], Y_train)

X_train["title"] = vector.predict(X_train["title"])

X_test["title"] = vector.predict(X_test["title"])

X_practice["title"] = vector.predict(X_practice["title"])

X_train["title"] = X_train["title"].apply(lambda x: istrue(x))

X_test["title"] = X_test["title"].apply(lambda x: istrue(x))

X_practice["title"] = X_practice["title"].apply(lambda x: istrue(x))

vector = vector.fit(X_train["description"], Y_train)

X_train["description"] = vector.predict(X_train["description"])

X_test["description"] = vector.predict(X_test["description"])

X_practice["description"] = vector.predict(X_practice["description"])

X_train["description"] = X_train["description"].apply(lambda x: istrue(x))

X_test["description"] = X_test["description"].apply(lambda x: istrue(x))

X_practice["description"] = X_practice["description"].apply(lambda x: istrue(x))



X_train = X_train.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

X_test = X_test.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

X_practice = X_practice.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))



svclassifier = KNeighborsClassifier(p=1, n_neighbors=7)

svclassifier.fit(X_train,Y_train)

svclassifier.score(X_practice, Y_practice)
Y_pred = svclassifier.predict(X_test)



test_data["class"] = Y_pred

test_data["class"] = test_data["class"].map(lambda x: "True" if x==1 else "False")

result = test_data[["ID","class"]]

result.to_csv("submission.csv", index=False)

result.head()