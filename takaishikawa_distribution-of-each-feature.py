import os; print(os.listdir("../input"))

import numpy as np

import pandas as pd

import warnings; warnings.filterwarnings("ignore")



%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

sns.set_palette(sns.color_palette('tab20', 20))
plt.rcParams['figure.figsize'] = [10, 5]

plt.rcParams['font.size'] = 12

C = ['#3D0553', '#4D798C', '#7DC170', '#F7E642']
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train.head(3)
%%time

for i in range(20):

    for j in range(10*i, 10*(i+1)):

        col = 'var_{}'.format(j)

        sns.distplot(train[col], label=col)

    plt.legend()

    plt.xlabel("value")

    plt.show()
%%time

for i in range(200):

    col = 'var_{}'.format(i)

    sns.distplot(train[col], label=col, color=C[0])

    sns.distplot(test[col], label=col, color=C[3])

    plt.legend()

    plt.xlabel(col)

    plt.show()