import os

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
Simple_sampling = train.sample(20000)
from sklearn import tree

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
X = Simple_sampling.values[:, 2:203]

Y = Simple_sampling.values[:,1].astype(bool)
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 10000)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 10000,

                               max_depth=3, min_samples_leaf=5)

clf_gini.fit(X_train, y_train)
y_pred = clf_gini.predict(X_test)

y_pred
accuracy= accuracy_score(y_test,y_pred)*100
accuracy
#test_sample= test.sample(20000)



X = test.values[:, 1:202]

X.shape
test_pred = clf_gini.predict(X)
Target = test_pred.astype(int)
Target.shape
Target = pd.DataFrame(Target)
#Target.rename(columns={'0':'Target'}, inplace=True)

df = Target.rename({'0':'Targets'}, axis=1)
df
Sample_submission = pd.read_csv("../input/sample_submission.csv")

Output = Sample_submission.drop(['target'], axis=1)

Output
ProjectOutput = pd.concat([Output, df], axis=1)
ProjectOutput
ProjectOutput.to_csv("ProjectOutputFile.csv", index=False)