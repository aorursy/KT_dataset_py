# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import matplotlib.pyplot as plt # side-stepping mpl backend

import matplotlib.gridspec as gridspec # subplots

#import mpld3 as mpl



import numpy as np

from sklearn import preprocessing, metrics

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold, train_test_split, cross_val_score

import pandas as pd

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/creditcard.csv",header = 0)

df.head()
df.describe()
count_classes = pd.value_counts(df['Class'], sort = True).sort_index()

count_classes.plot(kind = 'bar')

plt.title("Fraud class histogram")

plt.xlabel("Class")

plt.ylabel("Frequency")
traindf, testdf = train_test_split(df, test_size = 0.3)
def classification_model(model, data, predictors, outcome):

    scores = cross_val_score(model, data[predictors],data[outcome],cv = 5, scoring = 'accuracy')

    print(scores)

    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

 

features = df.columns[0:30]

outcome = 'Class'

features
classification_model(LogisticRegression(penalty = 'l2'), traindf, features, outcome)
model = LogisticRegression(penalty = 'l2')

model.fit(traindf[features],traindf[outcome])
predictions = model.predict(testdf[features])

accuracy = metrics.accuracy_score(predictions,testdf[outcome])

accuracy