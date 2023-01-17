import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
data = pd.read_csv('../input/handson-pima/Hands on Exercise Feature Engineering_ pima-indians-diabetes (1).csv')

data.head()
#Select only the values in this dataset and create X,y

array = data.values

X = array[:,0:7]

y = array[:,8]



X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.30, random_state = 7)
type(X_train)
# it takes a list of tuples as parameter. The last entry is the call to the modelling algorithm

pipeline = Pipeline([

    ('scaler', StandardScaler()),

    ('logreg', LogisticRegression())

])



#use pipeline object as a regular classifier



pipeline.fit(X_train, y_train)
from sklearn import metrics



y_pred = pipeline.predict(X_test)

model_score = pipeline.score(X_test, y_test)

print(model_score)
print(metrics.confusion_matrix(y_test,y_pred))