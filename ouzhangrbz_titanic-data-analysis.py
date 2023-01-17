import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use("fast") #plt.style.available





import seaborn as sns



from collections import Counter



import warnings

warnings.filterwarnings("ignore")





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data_train = pd.read_csv("/kaggle/input/titanic/train.csv")

data_test = pd.read_csv("/kaggle/input/titanic/test.csv")

data_test_PassengerId = data_test["PassengerId"]
data_train.columns
data_train.head()
data_train.tail()
data_train.describe()
data_train.info()
#percentage of NAN



NAN = [(each, data_train[each].isna().mean()*100) for each in data_train]

NAN = pd.DataFrame(NAN, columns=["name","rate"])

NAN
def bar_plot(variable):

    

    """

    input: variable ex; sex

    output: bar plot & value count

    

    """

    # get feature

    var = data_train[variable]

    

    # get number of categorical variable(value/sample)

    varValue = var.value_counts()

    

    #visualize

    plt.figure(figsize=(9,3))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index, varValue.index.values)

    plt.ylabel("Frequency")

    plt.title("Varible")

    plt.show()

    

    print("{}: \n {}".format(variable,varValue))



    
category1 = ["Survived","Sex","Pclass","Embarked","SibSp","Parch"]

for each in category1:

    bar_plot(each)
category2 = ["Name","Cabin","Ticket"]

for each in category2:

    print("{}: \n".format(data_train[each].value_counts()))
def plot_hist(variable):

    plt.figure(figsize = (9,3))

    plt.hist(data_train[variable], bins = 50)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} distribution with hist".format(variable))

    plt.show()
numericVar = ["Fare", "Age","PassengerId"]

for n in numericVar:

    plot_hist(n)
data_train[["Pclass","Survived"]].groupby(["Pclass"], as_index=False).mean().sort_values(by="Survived", ascending=False)
data_train[["Sex","Survived"]].groupby(["Sex"], as_index=False).mean().sort_values(by="Survived", ascending=False)
data_train[["SibSp","Survived"]].groupby(["SibSp"], as_index=False).mean().sort_values(by="Survived", ascending=False)
data_train[["Parch","Survived"]].groupby(["Parch"], as_index=False).mean().sort_values(by="Survived", ascending=False)
def detect_outliers(df,features):

    outlier_indices = []

    

    for c in features:

        # 1st quartile

        Q1 = np.percentile(df[c],25)

        # 3rd quartile

        Q3 = np.percentile(df[c],75)

        # IQR

        IQR = Q3 - Q1

        # Outlier step

        outlier_step = IQR * 1.5

        # detect outlier and their indeces

        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index

        # store indeces

        outlier_indices.extend(outlier_list_col)

    

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)

    

    return multiple_outliers
data_train.loc[detect_outliers(data_train,["Age","SibSp","Parch","Fare"])]

# drop outliers

data_train = data_train.drop(detect_outliers(data_train,["Age","SibSp","Parch","Fare"]),axis = 0).reset_index(drop = True)
data_train_len = len(data_train)

data_train = pd.concat([data_train,data_test],axis=0).reset_index(drop=True)
data_train.head()
data_train.columns[data_train.isnull().any()]
data_train.isnull().sum()
data_train["Embarked"].isnull()
data_train[data_train["Embarked"].isnull()]
data_train.boxplot(column="Fare",by = "Embarked")

plt.show()
data_train["Embarked"] = data_train["Embarked"].fillna("C")
data_train[data_train["Embarked"].isnull()]
data_train[data_train["Fare"].isnull()]
data_train["Fare"] = data_train["Fare"].fillna(np.mean(data_train[data_train["Pclass"]==3]["Fare"]))

data_train[1033:]