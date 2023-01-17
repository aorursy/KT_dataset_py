import pandas as pd

import numpy as np

from sklearn.metrics import accuracy_score

from sklearn.model_selection import StratifiedShuffleSplit
df = pd.read_csv("/kaggle/input/iris/Iris.csv")
df.shape
df.head()
df.info()
df.describe()
def transform(df):

    X = df.copy()

    y = X["Species"]

    X.drop(["Id", "Species"], axis=1, inplace=True)



    temp = list(df["Species"].unique())

    d = {}

    for i, j in zip(range(3), temp):

        d[j] = i 

    

    y = y.map(d)

    

    return [X, y]
X, y = transform(df)
# splitting the data

split = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)

for i, j in split.split(X, y):

    X_train, y_train = X.loc[i], y.loc[i]

    X_test, y_test = X.loc[j], y.loc[j]
# Naive Bayes Classifier

import scipy.stats

class NaiveBayesClassifier:



    def __init__(self):

        return



    def fit(self, X, y):

        self.X = X.copy()

        self.X["y"] = y



    def predict(self, test):

        preds = self.predict_proba(test)

        final_preds = []

        for i in range(len(test)):

            c = None

            max = 0

            for key in preds.keys():

                if preds[key][i] > max:

                    c = key

                    max = preds[key][i]

            final_preds.append(c)



        return final_preds



    def predict_proba(self, test):

        preds = []

        final = {}



        # looping over all labels

        for label in list(self.X["y"].unique()):

            probs = []



            # looping over row in the test set

            for row in range(len(test)):

                # num, deno = len(self.X[self.X["y"] == label]) / len(self.X), 1

                prob = len(self.X[self.X["y"] == label])/ len(self.X)

                

                # looping over cols in the test set

                for col in range(len(test.T)):

                    key = test.iloc[row, col]



                    if len(list(self.X.iloc[:, col].unique())) > 3:

                        

                        prob *= scipy.stats.norm(self.X[self.X["y"] == label].iloc[:, col].mean(), self.X[self.X["y"] == label].iloc[:, col].std()).pdf(key)



                    else:

                        temp = len(self.X[self.X.iloc[:, col] == key])

                        deno_temp = temp / len(self.X)

                        # deno *= deno_temp

                        num_temp = len(self.X[(self.X.iloc[:, col] == key) & (self.X["y"] == label)]) / len(self.X[self.X["y"] == label])

                        # num *= num_temp

                        prob *= (num_temp / deno_temp)



                

                probs.append(prob)

            final[label] = probs

        return final

model = NaiveBayesClassifier()

model.fit(X_train, y_train)
preds = model.predict(X_test)

print(accuracy_score(y_test, preds))