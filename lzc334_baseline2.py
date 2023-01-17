from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.svm import SVC

import pandas as pd

import numpy as np
train_data = pd.read_csv("../input/clickbait-thumbnail-detection/train.csv", usecols=["class", "viewCount", "likeCount","dislikeCount","commentCount","title"]) 

test_data = pd.read_csv("../input/clickbait-thumbnail-detection/test_1.csv", usecols=["ID", "viewCount", "likeCount","dislikeCount","commentCount","title"])



Y_train = train_data["class"]



X_train = train_data.drop("class", axis=1)

X_test = test_data.drop("ID", axis=1)



X_train["title"] = X_train["title"].apply(lambda x: len(x))

X_test["title"] = X_test["title"].apply(lambda x: len(x))



X_train = X_train.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

X_test = X_test.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
svclassifier = SVC(kernel='rbf',C=3000,gamma=0.3, probability=True)

svclassifier.fit(X_train,Y_train)
Y_pred = svclassifier.predict(X_test)



test_data["class"] = Y_pred

test_data["class"] = test_data["class"].map(lambda x: "True" if x==1 else "False")

result = test_data[["ID","class"]]

result.to_csv("submission.csv", index=False)

result.head()