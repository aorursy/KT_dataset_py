import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")



import seaborn as sns

from collections import Counter



import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv("https://raw.githubusercontent.com/satyalytics/mini_projects/master/titanic/data/train.csv")

test = pd.read_csv("https://raw.githubusercontent.com/satyalytics/mini_projects/master/titanic/data/test.csv")
print("Train")

print(train.columns)

print()

print("Test")

print(test.columns)
train.head(3)
train.describe()
train.info()
def count_values(var):

    """

        Counts the unique variable to know the distribution and value counts.

        Args:

            var: The variable name, one of the categorical column from the dataframe.

        Returns:

            value_count: The count of unique variables.

    """

    var = train[var]

    return var.value_counts()
def plot_univariate(variable):

    """

        To plot the unique variables in the dataframe to visualize the distribution.

        Args:

            var: The variable name, one of the categorical columns in the dataframe.

        Return:

            None

    """

    varValue = count_values(variable)

    # visualize

    plt.figure(figsize = (9,3))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index, varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()
cat_columns = ["Survived","Sex","Pclass","Embarked","SibSp","Parch"]

for i in cat_columns:

    print(i)

    print("---------")

    print(count_values(i))

    plot_univariate(i)

    print()
category2 = ["Cabin", "Name", "Ticket"]

for c in category2:

    print(f"{train[c].value_counts()} \n")
def plot_hist(col):

    """

        Plots histogram for contineous values.

        Args:

            col, the numeric column names of the dataframe.

        Return:

            None

    """

    plt.figure(figsize=(6,6))

    plt.hist(train[col], bins=50)

    plt.xlabel("col")

    plt.ylabel("frequency")

    plt.title(f"Histogram distribution of {col}")

    plt.show()
numeric_col = ['Age','Fare','PassengerId']

for i in numeric_col:

    plot_hist(i)
## Basic Data Analysis - Bivariate analysis

def bivariate(col):

    """

        Prints the survived percentage with respect to other columns.

        Args:

            col: the name of the column on which we will compare the survival rate

        Return: 

            None

    """

    return train[[col,"Survived"]].groupby([col], as_index=False).mean().sort_values(by="Survived", ascending=False)
cols = ["Pclass","Sex","SibSp","Parch"]

for i in cols:

    print(bivariate(i))

    print()
def detect_outlier(df, col_ls):

    """

        Detects outlier in a dataframe in a given column.

        Args:

            df: The dataframe where the column is situated

            col_ls: The list of columns where we have to detect outlier

        Returns:

            out_idx: The index of the all the outliers present in the columns

            mul_out: index of multiple outlier column

    """

    out_idx = []

    

    for c in col_ls:

        # accessing 1st and 3rd quartile and IQR

        Q1 = np.percentile(df[c], 25)

        Q3 = np.percentile(df[c], 75)

        IQR = Q3 - Q1

        # step is distance from the 1st quartile and 3rd quartile

        step = IQR * 1.5

        # detecting outlier

        _ = df[(df[c]<Q1-step)|(df[c]>Q3+step)].index

        out_idx.extend(_)

        

    # couting which index comes how many times

    count = Counter(out_idx)

    # index of multiple outlier index

    mul_out = list(i for i,v in count.items() if v>2)

    return out_idx, mul_out
out_idx, mul_out = detect_outlier(train, ["Age","SibSp","Parch","Fare"])
train.loc[out_idx,:]
train.loc[mul_out,:]
df = train.drop(mul_out).reset_index(drop=True)

train.shape, df.shape
ls = ['SibSp', 'Parch', 'Age', 'Fare', 'Survived']

fig, ax = plt.subplots(figsize=(12,12))         # Sample figsize in inches

sns.heatmap(train[ls].corr(), annot=True, linewidths=2, ax=ax)

bottom, top = ax.get_ylim()

ax.set_ylim(bottom + 0.5, top - 0.5)