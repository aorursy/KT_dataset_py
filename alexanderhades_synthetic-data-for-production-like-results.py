from sklearn.ensemble import RandomForestClassifier

from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt



import numpy as np

import pandas as pd

import os
df = pd.read_csv("../input/Iris.csv")

del df["Id"]

df.head()
for col in df.Species.unique():

    n = df.loc[df.Species == col].shape[0]

    print(n, col)
for col in df.Species.unique():

    print(col)

    df.loc[df.Species == col].hist()

    plt.show()
# Approcimate the PDF using the kernel density estimate (KDE) - a non-parametric way to estimate the probability density function of a random variable.



def resample_species(df, Species, sampleSize=50):

    subset = df.loc[df.Species == Species]

    del subset["Species"]

    kernel = gaussian_kde(subset.T)

    synt = kernel.resample(sampleSize)

    synt = pd.DataFrame(synt.T, columns=subset.columns)

    synt["Species"] = [Species for _ in range(sampleSize)]

    return synt
# Re-sampling helper function



def resample_dataframe(df, sampleSize=50):

    synt = pd.DataFrame()

    for col in df.Species.unique():

        print(col)

        synt = synt.append(resample_species(df, col, sampleSize))

    return synt
np.random.seed(42)

msk = np.random.rand(len(df)) < 0.75

train = df[msk]

test = df[~msk]
synthetic_train = resample_dataframe(train, 100)

synthetic_train.info()
# Helper function to set color for plot



def setC(clr):

    if clr=="Iris-setosa":

        return "red"

    elif clr=="Iris-versicolor":

        return "green"

    elif clr=="Iris-virginica":

        return "blue"
train.plot.scatter("SepalLengthCm", "SepalWidthCm", c=[setC(clr) for clr in train.Species])

plt.title("Original (train) data")

plt.show()
synthetic_train.plot.scatter("SepalLengthCm", "SepalWidthCm", c=[setC(clr) for clr in synthetic_train.Species])

plt.title("Synthetic data")

plt.show()
X_test = test.drop(['Species'], axis=1)

y_test = test.Species
X_train = train.drop(['Species'], axis=1)

y_train = train.Species
X_train_synt = synthetic_train.drop(['Species'], axis=1)

y_train_synt = synthetic_train.Species
X_train_synt.hist()

X_train.head()
X_train_synt.hist()

X_train_synt.head()
clf = RandomForestClassifier(random_state=0, n_estimators=10)

clf.fit(X_train_synt, y_train_synt)

synt_score = clf.score(X_test, y_test)

synt_score
clf = RandomForestClassifier(random_state=0, n_estimators=10)

clf.fit(X_train, y_train)

orig_score = clf.score(X_test, y_test)

orig_score