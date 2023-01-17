# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use("seaborn-whitegrid")



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
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')



test_PassengerId = test_df['PassengerId']
train_df.columns
train_df.head()
train_df.describe()
train_df.info()
def bar_plot(variable):

    """

    input: variable = ex:'Sex'

    output: bar plot & value count

    """

    # get feature

    var = train_df[variable]

    # count number of categorical variable(value)

    varValue = var.value_counts()

    

    #visualize

    plt.figure(figsize = (9,3))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index, varValue.index.values)

    plt.ylabel("Fre")

    plt.title("Variable")

    plt.show()

    

    print("{}: \n {}".format(variable,varValue))

    
category1 = ["Survived","Sex","Pclass","Embarked","Parch","SibSp"]



for i in category1:

    bar_plot(i)
category2 = ["Cabin","Name","Ticket"]



for c in category2:

    print("{} \n".format(train_df[c].value_counts()))
def plot_hist(variable):

    plt.figure(figsize = (9,3))

    plt.hist(train_df[variable],bins=50)

    plt.xlabel(variable)

    plt.ylabel("Fre")

    plt.title("{} distribution with hist".format(variable))

    plt.show()
numericVar = ['Fare','Age','PassengerId']



for i in numericVar:

    plot_hist(i)
#Plcass vs. Survived



train_df[["Pclass","Survived"]].groupby(["Pclass"],as_index=False).mean().sort_values(by='Survived',ascending=False)
#Sex vs Survived



train_df[["Sex","Survived"]].groupby(["Sex"],as_index=False).mean().sort_values(by='Survived',ascending=False)
#Sibsp vs Survived



train_df[["SibSp","Survived"]].groupby(["SibSp"],as_index=False).mean().sort_values(by='Survived',ascending=False)
# People who have 2 or more siblings and spouses have lower survival rates.

train1_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test1_df = pd.read_csv('/kaggle/input/titanic/test.csv')



train1_df['Living_Rate'] = ['Low' if i > 2 else 'High' for i in train1_df['SibSp']]



train1_df.loc[:10,['Living_Rate','SibSp']]
#Parch vs Survived



train_df[["Parch","Survived"]].groupby(["Parch"],as_index=False).mean().sort_values(by='Survived',ascending=False)
def detect_outlier(df,features):

    outlier_indices = []

    

    for c in features:

        # first quartile

        Q1 = np.percentile(df[c],25)

        # 3rd quartile

        Q3 = np.percentile(df[c],75)

        #iqr

        IQR = Q3 - Q1

        #outlier step

        outlier_step = IQR * 1.5

        #detect outlier and their indices

        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index

        #store indices

        outlier_indices.extend(outlier_list_col)

    

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)

    return multiple_outliers
train_df.loc[detect_outlier(train_df,['Age','SibSp','Parch','Fare'])]
#drop outliers



train_df = train_df.drop(detect_outlier(train_df,['Age','SibSp','Parch','Fare']), axis=0).reset_index(drop=True)
train_df_len = len(train_df)

train_df = pd.concat([train_df,test_df],axis=0).reset_index(drop=True)
train_df.head()
train_df.columns[train_df.isnull().any()] #missing value find
train_df.isnull().sum()
train_df[train_df['Embarked'].isnull()] #find missing value Embarked
train_df.boxplot(column='Fare',by = 'Embarked')

plt.show()
train_df['Embarked'] = train_df['Embarked'].fillna("C")

train_df[train_df['Embarked'].isnull()] #find missing value Embarked
train_df[train_df['Fare'].isnull()] #find missing value Fare
train_df[train_df['Pclass'] == 3]['Fare']
np.mean(train_df[train_df['Pclass'] == 3]['Fare'])
train_df['Fare'] = train_df['Fare'].fillna(np.mean(train_df[train_df['Pclass'] == 3]['Fare']))
train_df[train_df['Fare'].isnull()] #find missing value Fare