# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', -1)
# Any results you write to the current directory are saved as output.
trainDF = pd.read_json("../input/train.json")
trainDF.size
trainDF.head()
trainDF['ingredients'] = trainDF['ingredients'].apply(lambda x: ' '.join(x))

trainDF.head()
X = trainDF.iloc[:,2]
y = trainDF.iloc[:,0]
trainDF.size
X.head()
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelenc_y = LabelEncoder()
y = labelenc_y.fit_transform(y).reshape(-1, 1)

# onehotencoder = OneHotEncoder(categorical_features=[0])
# y = onehotencoder.fit_transform(y).toarray()
ct = CountVectorizer()
X = ct.fit_transform(X)
len(ct.get_feature_names())
ct.get_feature_names()
X.shape
X.toarray()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier( n_estimators = 50)
classifier.fit(X_train, y_train)
classifier.score(X_test,y_test)
from sklearn.linear_model import LogisticRegression

# clf1 = LogisticRegression(C=10,dual=False)
regressor = LogisticRegression()
regressor.fit(X_train , y_train)
regressor.score(X_test, y_test)
y_pred = regressor.predict(X_test).reshape(-1,1)
labelenc_y.inverse_transform(y_pred)
testDF = pd.read_json("../input/test.json")
testDF['ingredients'] = testDF['ingredients'].apply(lambda x: ' '.join(x))
X_pred_final = testDF.iloc[:,1]
X_pred_final = X_pred_final.values
len(X_pred_final)

X_pred_final = ct.transform(X_pred_final)
y_pred_final = regressor.predict(X_pred_final).reshape(-1,1)
y_pred_final = labelenc_y.inverse_transform(y_pred_final)
predictions = pd.DataFrame({'cuisine' : np.ravel(y_pred_final) , 'id' : testDF.id })
predictions = predictions[[ 'id' , 'cuisine']]
predictions.to_csv('submit.csv', index = False)
test = np.ravel(y_pred_final)
test.shape

