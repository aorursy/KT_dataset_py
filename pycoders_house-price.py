# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

import seaborn as sns

from collections import Counter

import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
train_df.columns
train_df.head()
train_df.describe()
train_df.info()
train_SalePrice = train_df["SalePrice"]
train_data = train_df.iloc[:, 0:27]
train_data.head()
train_data.drop(labels = ["Id"], axis = 1, inplace = True)

train_data.info()
train_data.select_dtypes(include=['object'])


list1 = ["MSSubClass","LotFrontage","LotArea","OverallQual","OverallCond","YearBuilt","YearRemodAdd","MasVnrArea","SalePrice"]
sns.heatmap(train_df[list1].corr(), annot = True, fmt = ".2f")
plt.show()

g = sns.factorplot(x = "MSSubClass", y = "SalePrice", kind = "bar", data = train_df, size = 6)
g.set_ylabels("Sales Price")
plt.show()
g = sns.factorplot(x = "MSZoning", y = "SalePrice", kind = "bar", data = train_df, size = 6)
g.set_ylabels("Sales Price")
plt.show()
def bar_plot(variable):
    """
        input: variable ex: "Sex"
        output: bar plot & value count
    """
    # get feature
    var = train_df[variable]
    # count number of categorical variable(value/sample)
    varValue = var.value_counts()
    
    # visualize
    plt.figure(figsize = (9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}: \n {}".format(variable,varValue))
category1 = ["MSZoning","Street","Alley","LotShape","LandContour","Utilities","LotConfig","LandSlope","Neighborhood","Condition1","Condition2","BldgType","HouseStyle","RoofStyle","RoofMatl","Exterior1st","Exterior2nd","MasVnrType"]
for c in category1:
    bar_plot(c)
train_data.select_dtypes(include=['integer'])

def plot_hist(variable):
    plt.figure(figsize = (9,3))
    plt.hist(train_df[variable], bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show()
numericVar = ["MSSubClass","LotFrontage","LotArea","OverallQual","OverallCond","YearBuilt","YearRemodAdd","MasVnrArea"]
for n in numericVar:
    plot_hist(n)
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
train_df.loc[detect_outliers(train_df,["MSSubClass","LotFrontage","LotArea","OverallQual","OverallCond","YearBuilt","YearRemodAdd","MasVnrArea","SalePrice"])]
# MSZoning vs SalePrice
train_df[["MSZoning","SalePrice"]].groupby(["MSZoning"], as_index = False).mean().sort_values(by="SalePrice",ascending = False)
category1 = ["MSZoning","Street","Alley","LotShape","LandContour","Utilities","LotConfig","LandSlope","Neighborhood","Condition1","Condition2","BldgType","HouseStyle","RoofStyle","RoofMatl","Exterior1st","Exterior2nd","MasVnrType"]
for c in category1:
    print(c," vs SalePrice")
    print("---------------------------------------")
    print(train_df[[c,"SalePrice"]].groupby([c], as_index = False).mean().sort_values(by="SalePrice",ascending = False))
    print("---------------------------------------")
    print("---------------------------------------")
train_data.columns[train_data.isnull().any()]
train_data.isnull().sum()