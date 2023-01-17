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
tran_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
passenger_id = test_df["PassengerId"]
tran_df.head()
tran_df.columns
tran_df.describe()
tran_df.describe().T
tran_df.info()
def bar_plot(variable):
    """
        input : variable ex: 'sex'
        output: bar plot & value count
    """
    #get feature
    var = tran_df[variable]
    #count number of categorical variable(value/sample)
    varValue = var.value_counts()
    
    #visualize
    plt.figure(figsize = (9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel('Frequency')
    plt.title(variable)
    plt.show()
    print("{} : \n {}".format(variable, varValue))

category1 = ["Survived","Sex","Pclass","Embarked","SibSp","Parch"]
for c in category1:
    bar_plot(c)
category2 = ["Cabin","Name","Ticket"]
for c in category2:
    print("{} \n".format(tran_df[c].value_counts()))
def plot_hist(variable):
    plt.figure(figsize = (9,3))
    plt.hist(tran_df[variable], bins=50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show()
numericVar = ["Fare", "Age", "PassengerId"]
for n in numericVar:
    plot_hist(n)
#Pclass - Survived
tran_df[["Pclass","Survived"]].groupby(["Pclass"],as_index = False ).mean().sort_values(by="Survived",ascending = False)
#Sex - Survived
tran_df[["Sex","Survived"]].groupby(["Sex"],as_index = False ).mean().sort_values(by="Survived",ascending = False)
#SibSp - Survived
tran_df[["SibSp","Survived"]].groupby(["SibSp"],as_index = False ).mean().sort_values(by="Survived",ascending = False)
#Parch - Survived
tran_df[["Parch","Survived"]].groupby(["Parch"],as_index = False ).mean().sort_values(by="Survived",ascending = False)
def detect_outlies(df,features):
    outliar_indices = []
    
    for c in features:
        #1st quartire
        Q1 = np.percentile(df[c],25)
        #3th quartire
        Q3 = np.percentile(df[c],75)
        #IQR
        IQR = Q3-Q1
        #Ourliar Step
        outliar_step = IQR * 1.5
        #Detect outliar their indices
        outliar_list_col = df[(df[c] < Q1-outliar_step) | (df[c] > Q3 + outliar_step)].index
        #Store Indices
        outliar_indices.extend(outliar_list_col)
        
    outliar_indices  = Counter(outliar_indices)
    multiple_outliar = list(i for i, v in outliar_indices.items() if v>2)
    return multiple_outliar
tran_df.loc[detect_outlies(tran_df,["Age","SibSp","Parch","Fare"])]
#drop Outliar
tran_df = tran_df.drop(detect_outlies(tran_df,["Age","SibSp","Parch","Fare"]),axis=0).reset_index(drop = True)
tran_df_len = len(tran_df)
tran_df = pd.concat([tran_df,test_df], axis = 0).reset_index(drop=True)
tran_df.head()
tran_df.columns[tran_df.isnull().any()]
tran_df.isnull().sum()
tran_df[tran_df["Embarked"].isnull()]
tran_df.boxplot(column="Fare",by="Embarked")
plt.show()
tran_df["Embarked"] = tran_df["Embarked"].fillna("C")
tran_df[tran_df["Embarked"].isnull()]
tran_df[tran_df["Fare"].isnull()]
tran_df["Fare"] = tran_df["Fare"].fillna(np.mean(tran_df[tran_df["Pclass"] == 3]["Fare"]))
tran_df[tran_df["Fare"].isnull()]
