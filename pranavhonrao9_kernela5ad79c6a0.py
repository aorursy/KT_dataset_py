import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.linear_model import LogisticRegression
import json
train_df = pd.read_json('../input/train.json')

test_df =pd.read_json('../input/test.json')
print(train_df.info())
print(train_df.describe())
train_df
test_df
cuisine=train_df['cuisine']
train_df =train_df.drop("cuisine", axis=1)
train_df
train_df.ingredients.str.join(' ')

# 1-hot encoding
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#one_hot = MultiLabelBinarizer()

mlb = MultiLabelBinarizer()

X = mlb.fit_transform(train_df.ingredients)

train_df = pd.DataFrame(X, columns=mlb.classes_)

                          

dummies
train_df
# 1-hot encoding for test_data

dummies1 = mlb.transform(test_df.ingredients)
#test_df = pd.DataFrame(columns= mlb.classes_)
test_df = pd.DataFrame(dummies1, columns=mlb.classes_)
test_df
# Preparation for modeling
X=train_df
y=cuisine
cuisine_model = LogisticRegression(multi_class = 'ovr')
cuisine_model.fit(X,y)

result = cuisine_model.predict(test_df)

result_df = pd.Series(result).rename('cuisine')
print(result_df)
