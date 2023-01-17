# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")



import seaborn as sns



from collections import Counter



import warnings

warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dfTrain = pd.read_csv("/kaggle/input/titanic/train.csv")

dfTest = pd.read_csv("/kaggle/input/titanic/test.csv")

PassengerIdTest = dfTest["PassengerId"]
dfTrain.columns
dfTrain.head()
dfTrain.describe()
dfTrain.info()
def barPlot(variable):

    """

    input: variable ex: "Sex"

    output: bar plot & value count

    """

    # get feature

    var = dfTrain[variable]

    # count number of categorical variable(values/sample)

    varValue = var.value_counts()

    

    # visualize

    plt.figure(figsize=(9,3))

    plt.bar(varValue.index,varValue)

    plt.xticks(varValue.index,varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{} \n {}".format(variable,varValue))
category1 = ["Survived","Sex","Pclass","Embarked","SibSp","Parch"]

for c in category1:

    barPlot(c)
category2 = ["Cabin","Name","Ticket"]

for c in category2:

    print("{} \n".format(dfTrain[c].value_counts()))
def plotHist(variable):

    plt.figure(figsize=(9,3))

    plt.hist(dfTrain[variable], bins = 50)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} distribution with histogram".format(variable))

    plt.show()
numericVar = ["Fare","Age","PassengerId"]

for n in numericVar:

    plotHist(n)
dfTrain.corr()
# Pclass - Survived

dfTrain[["Pclass","Survived"]]
dfTrain[["Pclass","Survived"]].groupby(["Pclass"],as_index=False).mean().sort_values(by="Survived",ascending=False)
# Sex - Survived

dfTrain[["Sex","Survived"]]
dfTrain[["Sex","Survived"]].groupby(["Sex"],as_index=False).mean().sort_values(by="Survived",ascending=False)
# SibSp - Survived

dfTrain[["SibSp","Survived"]]
dfTrain[["SibSp","Survived"]].groupby(["SibSp"],as_index=False).mean().sort_values(by="Survived",ascending=False)
# Parch - Survived

dfTrain[["Parch","Survived"]]
dfTrain[["Parch","Survived"]].groupby(["Parch"],as_index=False).mean().sort_values(by="Survived",ascending=False)
# def detectOutlier(df,features):

    #outlierIndices = []

    

    # for c in features:

        # 1st quartile

        

        # 3rd qartile

        

        # IQR

        

        # outlier step

        

        # detect outlier and their ideces

        

        # store indeces
def detectOutlier(df,features):

    outlierIndices = []

    

    for c in features:

        Q1 = np.percentile(df[c],25)

        Q3 = np.percentile(df[c],75)

        IQR = Q3 - Q1

        outlierStep = IQR*1.5

        outlierListCol = df[(df[c]<Q1-outlierStep) | (df[c] > Q3 + outlierStep)].index

        outlierIndices.extend(outlierListCol)

        

    outlierIndices = Counter(outlierIndices)

    multipleOutliers = list(i for i, v in outlierIndices.items() if v > 2)

    

    return multipleOutliers
dfTrain.loc[detectOutlier(dfTrain,["Age","SibSp","Parch","Fare"])]
# drop outliers

dfTrain = dfTrain.drop(detectOutlier(dfTrain,["Age","SibSp","Parch","Fare"]),axis=0).reset_index(drop=True)
dfTreainLen = len(dfTrain)

dfTrain = pd.concat([dfTrain,dfTest],axis=0).reset_index(drop=True)
dfTrain.head()
dfTrain.columns[dfTrain.isnull().any()]
dfTrain.isnull().sum()
dfTrain[dfTrain["Embarked"].isnull()]
dfTrain.boxplot(column="Fare",by = "Embarked")

plt.show()
dfTrain["Embarked"] = dfTrain["Embarked"].fillna("C")

dfTrain[dfTrain["Embarked"].isnull()]
dfTrain[dfTrain["Fare"].isnull()]
fare3Mean = (dfTrain[dfTrain["Pclass"] == 3]["Fare"]).mean()
dfTrain["Fare"] = dfTrain["Fare"].fillna(fare3Mean)
dfTrain[dfTrain["Fare"].isnull()]