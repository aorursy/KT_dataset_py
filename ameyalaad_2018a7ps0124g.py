import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
ogdf = pd.read_csv("../input/minor-project-2020/train.csv")

ogdf.head(5)

trainXdf = ogdf.drop(["id", "target"], axis=1)

trainXdf.sample(10)
trainydf = ogdf.target.copy()

trainydf.describe()
ogtestdf = pd.read_csv("../input/minor-project-2020/test.csv")

testXdf = ogtestdf.drop(["id"], axis = 1)

testXdf.shape
from sklearn.naive_bayes import GaussianNB



bnb = GaussianNB()

bnb.fit(trainXdf, trainydf)
bnb.predict_proba(testXdf)
bnb.predict(testXdf)
testydf = pd.DataFrame(bnb.predict_proba(testXdf)[:,1] , columns=["target"])

testydf["id"] = ogtestdf["id"]

testydf
testydf.to_csv("gaussian_naive_bayes_probs.csv", columns=["id", "target"], index=False)