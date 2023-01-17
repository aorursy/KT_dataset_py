import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
raw_data = pd.read_csv(r'../input/mushroom-classification/mushrooms.csv')

raw_data
raw_data.describe(include = 'all')
raw_data['cap-shape'].value_counts()
raw_data.columns
raw_data['class']= raw_data['class'].map({'p':0 , 'e': 1})
x = raw_data[['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',

       'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',

       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',

       'stalk-surface-below-ring', 'stalk-color-above-ring',

       'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',

       'ring-type', 'spore-print-color', 'population', 'habitat']]

y = raw_data['class']
from sklearn.preprocessing import OneHotEncoder



enc = OneHotEncoder()

x_enc = enc.fit_transform(x)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x_enc,y, test_size = 0.2, random_state = 42)
from sklearn.tree import DecisionTreeClassifier



dtc = DecisionTreeClassifier()

dtc.fit(x_train,y_train)
dtc.predict(x_test)
dtc.score(x_test,y_test)
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(x_train,y_train)
logreg.predict(x_test)
logreg.score(x_test,y_test)