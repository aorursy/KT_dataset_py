import pandas as pd

import matplotlib as mpl

import sklearn as skl

import seaborn as sns

import numpy as np

import warnings



from matplotlib import pyplot as plt

from matplotlib.pyplot import  subplots as spl

from sklearn.preprocessing import LabelEncoder



%matplotlib inline

warnings.filterwarnings("ignore")



le = LabelEncoder()



data = pd.read_csv("../input/adult.csv")
cut_features = ["hours.per.week"]

for feature in cut_features:

    data[feature] = pd.cut(data[feature], 10)



transform_features = ["income"]

for feature in transform_features:

    le.fit(data[feature])

    data[feature] = le.transform(data[feature])
fig, ax = spl(figsize = (13, 8))

sns.barplot(data = data, x = "age", y = "income", hue = "sex")
x = np.arange(1, 800.1, 1)

y = np.exp(x)

plt.plot(x, y)