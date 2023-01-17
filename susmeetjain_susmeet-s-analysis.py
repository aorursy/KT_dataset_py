import numpy as np # linear algebra

import pandas as pd # data processing
history = pd.read_csv('../input/Historical Product Demand.csv')
history[:5]
history.shape
len(history['Product_Category'].unique())
len(history['Product_Code'].unique())
dates = [pd.to_datetime(date) for date in history['Date']]

dates.sort()
dates[0]
dates[-1]
X = history[['Product_Code','Warehouse','Product_Category']]

Y = history[['Order_Demand']]
from sklearn import preprocessing

def encode(x):

    le = preprocessing.LabelEncoder()

    return le.fit_transform(x)
for column in X.columns:

    X[column] = encode(X[column])
from sklearn import tree



clf = tree.DecisionTreeClassifier()

clf = clf.fit(X, Y)