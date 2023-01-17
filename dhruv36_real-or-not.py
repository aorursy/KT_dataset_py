import numpy as np
import pandas as pd
from sklearn import feature_extraction, linear_model, preprocessing, model_selection
train = pd.read_csv("../input/nlp-getting-started/train.csv")
test = pd.read_csv("../input/nlp-getting-started/test.csv")
train.head()
train.drop(["location","keyword"],axis = 1,inplace = True)
train.head()
test.head()
test.drop(["location","keyword"],axis = 1,inplace = True)
test.head()
nw = []
for i in train["text"]:
    nw.append(len(i.split()))
train["NW"] = nw
train.head()
nw = []
for i in test["text"]:
    nw.append(len(i.split()))
test["NW"] = nw
test.head()
model = linear_model.LogisticRegression()
model.fit(train[["NW"]],train["target"])
pred = model.predict(test[["NW"]])
ss = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
ss["target"] = pred
ss.head()
ss.to_csv("submission.csv", index=False)
