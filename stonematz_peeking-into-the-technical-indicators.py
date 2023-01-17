import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



# Load Data

with pd.HDFStore('../input/train.h5') as train:

    df = train.get('train')

    

df.head()
technical = df.filter(regex="technical").columns

technical[:5]
def crosscorr(x, y, lag=0):

    return y.corr(x.shift(lag))
def plot_lagged_correlation(id, lags, columns, color):

    xcov = {}

    for i in range(lags):

        xcov[i] = crosscorr(df[df.id == id]["y"], df[df.id == id][columns[feature]], lag=i)

    X = np.arange(len(xcov))

    plt.bar(X, xcov.values(), color = color)

    

for feature in range(len(technical)):

    plt.figure()

    plt.subplot(211)

    plot_lagged_correlation(70, 100, technical, "blue")

    plt.title("Feature : " + str(technical[feature]))

    plt.subplot(212)

    plot_lagged_correlation(150, 100, technical, "red")

    plt.tight_layout()
