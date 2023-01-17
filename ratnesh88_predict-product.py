import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
file = "../input/Kaggle_Training_Dataset.csv"

trainData = pd.read_csv(file,sep=',')

trainData.info()
trainData.head()
trainData.describe()
trainData.shape
trainData.hist(figsize=(14,12))
trainData.groupby('went_on_backorder').size()
trainData.plot(kind='box',subplots=True,sharex=False,sharey=False,figsize=(12,8))
colx = trainData.columns[0: len(trainData.columns)-1]

colx
corr = trainData[trainData.columns].corr()
plt.figure(figsize=(13,10))

sns.heatmap(corr,annot=True)

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2
X = trainData.iloc[:,0:8]

Y = trainData.iloc[:,8]

print(len(trainData))

print(len(X))

print(len(Y))

select_top_4 = SelectKBest(score_func=chi2, k = 4)