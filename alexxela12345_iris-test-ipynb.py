%matplotlib inline



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier



df = pd.read_csv("../input/Iris.csv")

df.isnull().any()
df.dtypes
df.describe()
df['PetalWidthCm'].plot.hist()

plt.show()
sns.pairplot(df, hue='Species')
all_inputs = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values

all_classes = df['Species'].values



(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.7, random_state=1)
dtc = DecisionTreeClassifier()

dtc.fit(train_inputs, train_classes)

dtc.score(test_inputs, test_classes)