# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt 

plt.style.use("seaborn-whitegrid") #use grid 

#plt.style.available , give all style



import seaborn as sns 

from collections import Counter



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv ("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv ("/kaggle/input/titanic/test.csv")

test_PassengerId = test_df["PassengerId"]
train_df.columns
train_df.head()
train_df.describe().T
train_df.info()
def bar_plot (variable):

    """

    input = variable ex : "sex"

    output = bar plot & value count

    """

    #get feature

    var = train_df[variable]

    #count number of categorical variable (value-sample)

    varValue =  var.value_counts()

    

    # visualize 

    plt.figure(figsize = (9,3))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index,varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{}: \n {}".format(variable,varValue))
category1=["Survived","Sex","Pclass","Embarked","SibSp","Parch"]

for i in category1:

    bar_plot(i)
category2=["Cabin","Name","Ticket"]

for i in category2 :

    print("{} \n".format(train_df[i].value_counts()))
def plot_histogram(variable):

    plt.figure(figsize =(9,3))

    plt.hist(train_df[variable], bins = 41)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} distribution with hist".format(variable))

    plt.show()
numericVar=["Fare","Age","PassengerId"]

for i in numericVar :

    plot_histogram(i)

  
#Pclass - Survived

train_df[["Pclass","Survived"]].groupby(["Pclass"], as_index = True).mean().sort_values(by="Survived", ascending= False)
#Sex - Survived

train_df[["Sex","Survived"]].groupby(["Sex"], as_index = True).mean().sort_values(by="Survived", ascending= False)
#SibSp - Survived

train_df[["SibSp","Survived"]].groupby(["SibSp"], as_index = True).mean().sort_values(by="Survived", ascending= False)
#Parch - Survived

train_df[["Parch","Survived"]].groupby(["Parch"], as_index = True).mean().sort_values(by="Survived", ascending= False)
#Pclass & Sex - Survived

train_df[["Sex","Pclass","Survived"]].groupby(["Sex","Pclass"], as_index = True).mean().sort_values(by="Survived", ascending= False)
#Data Correlation 



sns.heatmap(

    train_df.corr(), 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=120),

    square=True)
def detect_outlier (df,features) :

    outlier_indeces = []

    

    for i in features:

        #1st quartile 

        Q1 = np.percentile(df[i],25)

        #3rd quartile 

        Q3 = np.percentile(df[i],75)

        #IQR 

        IQR = Q3-Q1

        #Outlier step

        outlier_step = IQR *1.5

        # Detect outlier and their indeces

        outlier_list_col = df[(df[i] < Q1 - outlier_step) | (df[i] > Q3 - outlier_step)].index

        #store indeces 

        outlier_indeces.extend(outlier_list_col)

        

    outlier_indeces = Counter(outlier_indeces)

    multiple_outliers = list(c for c,v in outlier_indeces.items() if v>2)

    return multiple_outliers
train_df.loc[detect_outlier(train_df,["Age","SibSp","Parch","Fare"])]
#Drop Outliers 

tran_df = train_df.drop(detect_outlier(train_df,["Age","SibSp","Parch","Fare"]),axis = 0).reset_index(drop = True)
train_df_len=len(train_df)

data_df = pd.concat([train_df ,test_df ],axis= 0).reset_index(drop = True)
data_df.head()
data_df.columns[data_df.isnull().any()]
data_df.isnull().sum()
data_df[data_df["Embarked"].isnull()]
data_df.boxplot(column="Fare",by ="Embarked")

plt.show()
data_df["Embarked"] = data_df["Embarked"].fillna("C")

data_df[data_df["Embarked"].isnull()]
data_df[data_df["Fare"].isnull()]
data_df["Fare"] = data_df["Fare"].fillna(np.mean(data_df[data_df["Pclass"] == 3]["Fare"]))
data_df[data_df["Fare"].isnull()]