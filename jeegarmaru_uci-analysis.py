import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import scipy.stats as ss
df = pd.read_csv('../input/heart.csv')

df.head()
df.info()
df.describe()
for col in df.columns:

    print(col)

    print(df[col].unique())
categoricals = "sex cp fbs restecg exang slope ca thal target".split()

for col in categoricals:

    df[col] = df[col].astype('category')

df.dtypes
df['target'].value_counts()
corrmat = df.corr()

corrmat
sns.heatmap(corrmat, annot=True)
# Thanks to this article by Shaked Zychlinski!

# https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9

import math

from collections import Counter

def conditional_entropy(x, y):

    """

    Calculates the conditional entropy of x given y: S(x|y)

    Wikipedia: https://en.wikipedia.org/wiki/Conditional_entropy

    :param x: list / NumPy ndarray / Pandas Series

        A sequence of measurements

    :param y: list / NumPy ndarray / Pandas Series

        A sequence of measurements

    :return: float

    """

    # entropy of x given y

    y_counter = Counter(y)

    xy_counter = Counter(list(zip(x,y)))

    total_occurrences = sum(y_counter.values())

    entropy = 0.0

    for xy in xy_counter.keys():

        p_xy = xy_counter[xy] / total_occurrences

        p_y = y_counter[xy[1]] / total_occurrences

        entropy += p_xy * math.log(p_y/p_xy)

    return entropy



def theils_u(x, y):

    s_xy = conditional_entropy(x,y)

    x_counter = Counter(x)

    total_occurrences = sum(x_counter.values())

    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))

    s_x = ss.entropy(p_x)

    if s_x == 0:

        return 1

    else:

        return (s_x - s_xy) / s_x
for col in categoricals:

    print(f"Correlation of {col} with target : {theils_u(df[col], df['target'])}")
for col in categoricals:

    sns.countplot(x=col, hue='target', data=df)

    plt.show()
continuous = set(df.columns) - set(categoricals)

continuous
sns.pairplot(data=df[list(continuous)])