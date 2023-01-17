# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use("classic")



import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

sample_submission=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")



train.columns
train.info()
train.head()
train.describe()
train.corr()
plt.style.use("seaborn-whitegrid")

def bar_plot(variable):

    """

        input: variable :Street

        output: bar plot & value count

    """

    # get feature

    var = train[variable]

    # count number of categorical variable

    varValue = var.value_counts()

    

    # visualize

    plt.figure(figsize = (10,5))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index, varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{}: \n {}".format(variable,varValue))

    

    
categorical_variable=["MSZoning","Street","Alley","LotShape","MiscFeature","SaleType","SaleCondition","LotConfig"]



for i in categorical_variable:

    bar_plot(i)
def plot_hist(variable):

    #visualize

    plt.figure(figsize = (9,3))

    plt.hist(train[variable], bins = 50)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} distribution with hist".format(variable))

    plt.show()
Numerical_variable=["MSSubClass","LotFrontage","MoSold","YrSold","SalePrice"]



for n in Numerical_variable:

    plot_hist(n)
#YrSold vs SalePrice

train[["YrSold","SalePrice"]].groupby(["YrSold"],as_index=False).mean().sort_values("SalePrice",ascending=True)
#MSSubClass vs SalePrice

train[["MSSubClass","SalePrice"]].groupby(["MSSubClass"],as_index=False).mean().sort_values("SalePrice",ascending=False)
#YrSold vs MSSubClass

train[["YrSold","MSSubClass"]].groupby(["YrSold"],as_index=False).mean().sort_values("MSSubClass",ascending=True)
#MSSubClass vs LotArea

train[["MSSubClass","LotArea"]].groupby(["MSSubClass"]).mean().sort_values("LotArea",ascending=False)
train.head()
train.columns[train.isnull().any()]
train["LotFrontage"].astype("object")


train["LotFrontage"].isnull().sum()
train.head()
train[train["LotFrontage"].isnull()]
train.boxplot(column="MSSubClass",by="LotFrontage")

plt.show()
train["LotFrontage"]=train["LotFrontage"].fillna("55")
train[train["LotFrontage"].isnull()]