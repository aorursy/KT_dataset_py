# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy import stats
import sklearn.metrics as metrics

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/creditcard.csv")
df.head()
#Select only the anonymized features.
v_features = df.iloc[:,1:30].columns
plt.figure(figsize=(12,29*4))
gs = gridspec.GridSpec(29, 1)
for i, cn in enumerate(df[v_features]):
    ax = plt.subplot(gs[i])
    sns.distplot(df[cn][df.Class == 1], bins=50, color="red", label="fraud")
    sns.distplot(df[cn][df.Class == 0], bins=50, color="green", label="normal")
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(cn))
    plt.legend()
plt.show()

# Play with borders to see which feature is most skewed.

borders = [.0005, .9995]
minPercentFraud = 20

index = 1
print("fraction of fraudulent transactions lower and higher than 0.005 and 0.995 quantile normal transactions:")

for c in df.columns:
    quantile = df.loc[df.Class == 0, str(c)].quantile(borders)
    inverseQuantile = (
        stats.percentileofscore(df.loc[df.Class == 1, str(c)], quantile[borders[0]]),
        100 - stats.percentileofscore(df.loc[df.Class == 1, str(c)], quantile[borders[1]])
    )
    print(c, inverseQuantile)
    index += 1
# Feature V14, V12 or V17 have the most skewed distribution between the fraud and normal transactions.
# Therefore these features can be used to mark transactions as fraudulent or not.
# This simple classifier is introduced to compare detection performance with more advanced,
# but less effiecient algorithms.
for i in range(-20, 10):
    df.loc[df["V14"] < i, "cut-off_" + str(i)] = 1
    df.loc[df["V14"] > i, "cut-off_" + str(i)] = 0
df.head()
# calculate results based on different cut-off
columns = ["Cut-Off", "TruePositive", "FalsePositive", "FalseNegative", "TrueNegative", "Recall", "Precision", "F1-score"]
results = []
for i in range(-20, 10):
    TruePositive = len(df.loc[(df["cut-off_" + str(i)] == 1) & (df["Class"] == 1)])
    FalsePositive = len(df.loc[(df["cut-off_" + str(i)] == 1) & (df["Class"] == 0)])
    FalseNegative = len(df.loc[(df["cut-off_" + str(i)] == 0) & (df["Class"] == 1)])
    TrueNegative = len(df.loc[(df["cut-off_" + str(i)] == 0) & (df["Class"] == 0)])
    Recall = TruePositive / (TruePositive + FalseNegative)
    if TruePositive + FalsePositive != 0:
        Precision = TruePositive / (TruePositive + FalsePositive)
    else:
        Precision = 0
    if Recall + Precision != 0:
        F1score = 2 * Recall * Precision / (Recall + Precision)
    else:
        F1score = 0
    results.append([i, TruePositive, FalsePositive, FalseNegative, TrueNegative, Recall, Precision, F1score])
result = pd.DataFrame(results, columns=columns)
result
# Detailed search for best performance
for i in np.arange(-7, -5, 0.1):
    df.loc[df["V14"] < i, "cut-off_" + str(i)] = 1
    df.loc[df["V14"] > i, "cut-off_" + str(i)] = 0
df.head()
for i in np.arange(-7, -5, 0.1):
    TruePositive = len(df.loc[(df["cut-off_" + str(i)] == 1) & (df["Class"] == 1)])
    FalsePositive = len(df.loc[(df["cut-off_" + str(i)] == 1) & (df["Class"] == 0)])
    FalseNegative = len(df.loc[(df["cut-off_" + str(i)] == 0) & (df["Class"] == 1)])
    TrueNegative = len(df.loc[(df["cut-off_" + str(i)] == 0) & (df["Class"] == 0)])
    Recall = TruePositive / (TruePositive + FalseNegative)
    if TruePositive + FalsePositive != 0:
        Precision = TruePositive / (TruePositive + FalsePositive)
    else:
        Precision = 0
    if Recall + Precision != 0:
        F1score = 2 * Recall * Precision / (Recall + Precision)
    else:
        F1score = 0
    results.append([i, TruePositive, FalsePositive, FalseNegative, TrueNegative, Recall, Precision, F1score])
result = pd.DataFrame(results, columns=columns)
result.tail(20)
result = result.sort_values(by=["Cut-Off"])
plt.figure(figsize=(15,8))
plt.plot(result["Cut-Off"], result["Recall"], label="Recall")
plt.plot(result["Cut-Off"], result["Precision"], label="Precision")
plt.plot(result["Cut-Off"], result["F1-score"], label="F1-Score")
plt.legend()
plt.xlabel("Cut-Off")
plt.show()
plt.figure(figsize=(15,8))
fpr = result["FalsePositive"] / (result["FalsePositive"] + result["TrueNegative"])
tpr = result["Recall"]
plt.plot(fpr, tpr, label="AUROC=%.3f" % metrics.auc(fpr, tpr))
plt.legend()
plt.show()