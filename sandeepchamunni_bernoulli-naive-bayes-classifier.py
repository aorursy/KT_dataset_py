from sklearn import naive_bayes

from sklearn.metrics import confusion_matrix

from sklearn import preprocessing

import pandas as pd

import numpy as np

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns

import statistics
CancerData = pd.read_csv("../input/wdbc-data/wdbc.data").values
fig, axes = plt.subplots(4,3)

fig.set_size_inches(18.5, 6.5)

fig.tight_layout()

sns.distplot(CancerData[:,2],ax=axes[0][0]).set_title("Feature 1")

sns.distplot(CancerData[:,3],ax=axes[0][1]).set_title("Feature 2")

sns.distplot(CancerData[:,4],ax=axes[0][2]).set_title("Feature 3")

sns.distplot(CancerData[:,5],ax=axes[1][0]).set_title("Feature 4")

sns.distplot(CancerData[:,6],ax=axes[1][1]).set_title("Feature 5")

sns.distplot(CancerData[:,7],ax=axes[1][2]).set_title("Feature 6")

sns.distplot(CancerData[:,8],ax=axes[2][0]).set_title("Feature 7")

sns.distplot(CancerData[:,9],ax=axes[2][1]).set_title("Feature 8")

sns.distplot(CancerData[:,10],ax=axes[2][2]).set_title("Feature 9")

sns.distplot(CancerData[:,11],ax=axes[3][0]).set_title("Feature 10")

sns.distplot(CancerData[:,12],ax=axes[3][1]).set_title("Feature 11")

axes[3][2].hist(CancerData[:,1])

axes[3][2].set_title("Diagnosis")
CancerDataBinary = np.zeros((568,30))

for i in range(2,32,1):

    binarizer = preprocessing.Binarizer(threshold = statistics.mean(CancerData[:,i])).fit([CancerData[:,i]])  # fit does nothing

    CancerDataBinary[:,i-2] = binarizer.transform([CancerData[:,i]])
fig, axes = plt.subplots(4,3)

fig.set_size_inches(18.5, 6.5)

fig.tight_layout()

sns.distplot(CancerDataBinary[:,0],ax=axes[0][0]).set_title("Feature 1")

sns.distplot(CancerDataBinary[:,1],ax=axes[0][1]).set_title("Feature 2")

sns.distplot(CancerDataBinary[:,2],ax=axes[0][2]).set_title("Feature 3")

sns.distplot(CancerDataBinary[:,3],ax=axes[1][0]).set_title("Feature 4")

sns.distplot(CancerDataBinary[:,4],ax=axes[1][1]).set_title("Feature 5")

sns.distplot(CancerDataBinary[:,5],ax=axes[1][2]).set_title("Feature 6")

sns.distplot(CancerDataBinary[:,6],ax=axes[2][0]).set_title("Feature 7")

sns.distplot(CancerDataBinary[:,7],ax=axes[2][1]).set_title("Feature 8")

sns.distplot(CancerDataBinary[:,8],ax=axes[2][2]).set_title("Feature 9")

sns.distplot(CancerDataBinary[:,9],ax=axes[3][0]).set_title("Feature 10")

sns.distplot(CancerDataBinary[:,10],ax=axes[3][1]).set_title("Feature 11")

axes[3][2].hist(CancerData[:,1])

axes[3][2].set_title("Diagnosis")
X = CancerDataBinary

y = CancerData[:,1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
BernoulliNBModel = naive_bayes.BernoulliNB()

BernoulliNBModel.fit(X_train, y_train)
y_test_predicted = BernoulliNBModel.predict(X_test)
print(confusion_matrix(y_test, y_test_predicted))

accuracy_score(y_test, y_test_predicted)