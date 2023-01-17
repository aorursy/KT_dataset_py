import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



sns.set(font_scale=1)
titanic = pd.read_csv("../input/train.csv")

titanic.head().T
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=30000)

cv.fit_transform(titanic.Name)

print("Total Number of Names : {}".format(len(cv.vocabulary_)))

names = list(cv.vocabulary_.keys())

counts = list(cv.vocabulary_.values())

names_counts = np.array([ names , counts])

sorted_index = np.argsort(counts)

print(names_counts)

new = names_counts[0:1, sorted_index]

print(new[:,:5])

print("Sample Names:")

print(names_counts[:, -5:])
import numpy as np



c = np.array([5,2,8,2,4])    

a = np.array([[ 0,  1,  2,  3,  4],

              [ 5,  6,  7,  8,  9],

              [10, 11, 12, 13, 14],

              [15, 16, 17, 18, 19],

              [20, 21, 22, 23, 24]])



i = np.argsort(c)

print(i)

a = a[:,i]

print(a)
feature_names = cv.get_feature_names()

print("Number of features: {}".format(len(feature_names)))

print("First 20 features:\n{}".format(feature_names[:20]))

print("Features 20010 to 20030:\n{}".format(feature_names[20010:20030]))

print("Every 2000th feature:\n{}".format(feature_names[::2000]))