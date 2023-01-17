import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression

import seaborn as sns

from sklearn import metrics
df = pd.read_csv('../input/mushrooms.csv')

 

X = df.iloc[:,1:] # feature values are characters

y = df.iloc[:,0] # labels. either 0 or 1

 

XVal = X.values

yVal = (y.values == 'e')*1 # 1 if it is edible, 0 otherwise



df.head()
def ConvCharVec2NumbVec(CharVec):

    # this function will take a vector of characters and convert it to a correspon

    # ding int vector. A range of int value will be determined by the length of

    # unique function value of given character vector.

 

    IntVec = np.zeros(CharVec.shape)

    CharVecUnique = np.unique(CharVec)

 

    # now, create the random vectors to assign values

    FirstRand = np.random.uniform(0,10,1)[0] # mean

    SecondRand = np.random.uniform(0.5,3,1)[0] # increment

    RandVal = np.zeros(CharVecUnique.shape)

 

    for i in range(0,CharVecUnique.shape[0]):

        RandVal[i] = FirstRand + (i*SecondRand)

 

    for i in range(0,CharVec.shape[0]):

        ThisChar = CharVec[i]

        CorrespInd = np.where(CharVecUnique==ThisChar)[0][0]

        IntVec[i] = RandVal[CorrespInd]

 

    return IntVec

 

FVerXVal = np.zeros(XVal.shape)

 

# convert XVal to float

for i in range(0,XVal.shape[1]):

    FVerXVal[:,i] = ConvCharVec2NumbVec(XVal[:,i])
sns.heatmap(np.corrcoef(np.transpose(FVerXVal)), square=True, cmap="RdYlGn", xticklabels=list(X), yticklabels=list(X), vmin=-1, vmax=1)

plt.show()

# split into train and test data

X_train, X_test, y_train, y_test = train_test_split(FVerXVal, yVal)

 

# data can be scaled depending on their minimum or maximum values

scaler = MinMaxScaler()

 

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

 

# exponent data for L1 regularization parameter

B = np.arange(-4,4).astype(float)

Power = np.zeros(B.shape)

for i in range(0,len(B)):

    Power[i] = 10**B[i]

 

TrainResult = np.zeros((len(B)))

TestResult = np.zeros((len(B)))

 

# logistic regression

for i in range(0,len(Power)):

    clf = LogisticRegression(C=Power[i], penalty='l1').fit(X_train_scaled, y_train)

    fpr, tpr, thresh = metrics.roc_curve(y_train, clf.predict(X_train_scaled))

    TrainResult[i] = metrics.auc(fpr, tpr)

    fpr, tpr, thresh = metrics.roc_curve(y_test, clf.predict(X_test_scaled))

    TestResult[i] = metrics.auc(fpr, tpr)

 

# plot the result; which C value should I use?

plt.plot(B,TrainResult, label='Training data')

plt.plot(B,TestResult, label='Test data')

plt.xlabel('log of penalty parameter (base 10)')

plt.ylabel('Performance')

plt.legend()

plt.show()

# it looks like acceptable C value is around 10**-1 to 10. I will use one of those

# and plot the plot importance

clf = LogisticRegression(C=1, penalty='l1').fit(X_train_scaled, y_train)

coefs = pd.Series(np.absolute(clf.coef_[0]), index=X.columns)

coefs = coefs.sort_values()

(coefs/np.max(coefs)).plot(kind="bar")

plt.ylabel('Normalized model parameter')

plt.show()