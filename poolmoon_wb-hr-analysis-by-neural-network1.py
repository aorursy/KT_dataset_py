import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.manifold import TSNE



import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.models import load_model



import math



import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/HR_comma_sep.csv')

len(df)
df.head()
df.describe()
df.corr()
df.info()
df['sales'].unique()
df['salary'].unique()
y = df['left'].values

x = df.drop('left', axis=1).values
le1 = LabelEncoder()

x[:, 7] = le1.fit_transform(x[:, 7])



le2 = LabelEncoder()

x[:, 8] = le2.fit_transform(x[:, 8])



ohe1 = OneHotEncoder(categorical_features = [7, 8])

x = ohe1.fit_transform(x).toarray()
sc_x = StandardScaler()

x_std = sc_x.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

print('%s , %s , %s , %s' % (len(x_train), len(y_train), len(x_test), len(y_test)))
# TODO: Build Your Neural Network, your predition should be 'y_pred'
from sklearn.metrics import accuracy_score



accuracy_score(y_test, y_pred)