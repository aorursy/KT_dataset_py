

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")

import seaborn as sns



from collections import Counter



import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

test_PassengerId = test["PassengerId"]
train.columns
train.head()
train.describe()
train.info()
def barPlot(feature):

    # get feature

    temp = train[feature]

    # count number of categorical variable

    tempValue = temp.value_counts()

    # visualize

    plt.figure(figsize = (9,3))

    plt.bar(tempValue.index, tempValue)

    plt.xticks(tempValue.index, tempValue.index.values)

    plt.ylabel("Sıklık")

    plt.title(feature)

    plt.show()

    print("{}: \n {}".format(feature, tempValue))
category = ["Survived", "Sex", "Pclass", "Embarked", "SibSp", "Parch"]

for c in category:

    barPlot(c)
def plotHist(feature):

    plt.figure(figsize = (9,3))

    plt.hist(train[feature], bins = 50)

    plt.xlabel(feature)

    plt.ylabel("Sıklık")

    plt.title("{}: ".format(feature))

    plt.show()
numeric = ["Fare", "Age","PassengerId"]

for n in numeric:

    plotHist(n)
# Plcass ve Survived

train[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by="Survived",ascending = False)
# Sex ve Survived

train[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by="Survived",ascending = False)
# Sibsp ve Survived

train[["SibSp","Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values(by="Survived",ascending = False)
# Parch ve Survived

train[["Parch","Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by="Survived",ascending = False)
def detect_outliers(dataFrame,features):

    outlier_indices = []

    

    for c in features:

        # 1st quartile

        Q1 = np.percentile(dataFrame[c],25)

        # 3rd quartile

        Q3 = np.percentile(dataFrame[c],75)

        # IQR

        IQR = Q3 - Q1

        # Outlier step

        outlier_step = IQR * 1.5

        # Aykırı değerlerin index değerlerini bul

        outlier_list_col = dataFrame[(dataFrame[c] < Q1 - outlier_step) | (dataFrame[c] > Q3 + outlier_step)].index

        

        outlier_indices.extend(outlier_list_col)

    

    outlier_indices = Counter(outlier_indices)

    # Aykırı değer 2'den fazla alanda bulunuyor ise aykırı değerin index'ini yeni bir listeye at

    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)

    

    return multiple_outliers
# Counter kullanımı

temp = ["a","a","a","b","b"]

Counter(temp)
# Sayısal alanlar için aykırı değerleri bul

train.loc[detect_outliers(train, ["Age","SibSp","Parch","Fare"])]
# Aykırı değerleri sil

train = train.drop(detect_outliers(train, ["Age","SibSp","Parch","Fare"]),axis = 0).reset_index(drop = True)
new_train = pd.concat([train,test],axis = 0).reset_index(drop = True)

new_train.head()
new_train.columns[new_train.isnull().any()]

new_train.isnull().sum()
new_train[new_train["Embarked"].isnull()]
new_train.boxplot(column="Fare",by = "Embarked")

plt.show()
new_train["Embarked"] = new_train["Embarked"].fillna("C")

new_train[new_train["Embarked"].isnull()]
new_train[new_train["Fare"].isnull()]
new_train["Fare"] = new_train["Fare"].fillna(np.mean(new_train[new_train["Pclass"] == 3]["Fare"]))
new_train[new_train["Fare"].isnull()]