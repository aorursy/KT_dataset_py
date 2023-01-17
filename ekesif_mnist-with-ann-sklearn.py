# Libraries

import numpy as np

import pandas as pd
train = pd.read_csv('../input/digit-recognizer/train.csv')

test= pd.read_csv('../input/digit-recognizer/test.csv')





train.sample(4)
import matplotlib.pyplot as plt

%matplotlib inline
y = train['label']

X = train.drop('label', axis=1)

X.iloc[0].shape
X.shape, test.shape
import seaborn as sns

sns.countplot(y)
X.max().max()
#normalize

X = train / 255

test = test / 255 

X.max().max()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test  = train_test_split(X, y, random_state=5)
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(hidden_layer_sizes=(100, 10))

 
scores = cross_val_score(model, X_train, y_train, cv=5)

scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))