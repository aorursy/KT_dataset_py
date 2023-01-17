import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from pandas import DataFrame
warnings.filterwarnings(action='ignore')

sns.set()
%matplotlib inline
red = pd.read_csv('../input/winequality-red.csv')

red.head()
red.shape
#showing the data type of each feature and if there is any missing value

red.info()
#Selecting all features for the base model

X = red.iloc[:,:11]
y = red['quality']

print(X.shape)
print(y.shape)
#Automatic feature selection using Recursive Feature Elimination wrapper method
logreg = LogisticRegression()

selector = RFE(logreg)

selector = selector.fit(X,y)

#the best features according to RFE have a ranking of 1, so we'll create a second model with those features.

selected_features = DataFrame({'Feature':list(X.columns),'Ranking':selector.ranking_})
selected_features.sort_values(by='Ranking')
X = red[['volatile acidity','chlorides','density','pH','sulphates']]
y = red['quality']

print(X.shape)
print(y.shape)
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=30)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
logreg = LogisticRegression()

y_pred = logreg.fit(X_train,y_train).predict(X_test)

print('The Accuracy of the model is {:.1f}%'.format(accuracy_score(y_test,y_pred)*100))
