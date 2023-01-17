# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt # data visualition library

plt.style.use('seaborn-poster')



import seaborn as sns



from collections import Counter



import warnings

warnings.filterwarnings('ignore')







# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

test_PassenderId = test_df['PassengerId']
train_df.head()

train_df.columns
train_df.describe()
train_df.info()
def barPlot (variable):#defining barPlot function

    """

    input = variable

    output = Bar Plot & Count

    

    """

    #getting feature

    var = train_df[variable]

    

    #Counting value

    varValue = var.value_counts()

    

    #visualition

    plt.figure(figsize=(8,8))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index, varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{} \n {}".format(variable,varValue))

    
cat1 = ["Survived","Sex","Pclass","Embarked","SibSp","Parch"]



for i in cat1:

    barPlot(i)



cat2 = ["Cabin","Ticket","Name"]

for i in cat2:

    print("{} \n".format(train_df[i].value_counts()))
def plotHist(variable):

    plt.figure(figsize=(8,8))

    plt.hist(train_df[variable], bins= 30)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} disturbution with Histogram Plot ".format(variable))

    plt.show()
numericalVariable = ["Age","Fare","PassengerId"]

for i in numericalVariable:

    plotHist(i)
train_df[["Survived","Sex"]].groupby(["Sex"], as_index = False).mean().sort_values(by="Survived",ascending= False)



train_df[["Survived","Embarked"]].groupby(["Embarked"], as_index = False).mean().sort_values(by="Survived",ascending= False)
train_df[["Survived","Parch"]].groupby(["Parch"], as_index = False).mean().sort_values(by="Survived",ascending= False)
train_df[["Survived","SibSp"]].groupby(["SibSp"], as_index = False).mean().sort_values(by="Survived",ascending= False)
train_df[["Survived","Pclass"]].groupby(["Pclass"], as_index = False).mean().sort_values(by="Survived",ascending= False)
def detect_outliers(df,features):

    

    outliers_indices=[]

    

    for i in features:

        

    #Finding 1st quartile

        Q1 = np.percentile(df[i],25)

    

    #Finding 3rd quartile

        Q3 = np.percentile(df[i],75)

    

    #Calculating IQR

        IQR = Q3-Q1

    #Calculating Outlier Step

        outlier_step = IQR * 1.5

    #Finding Outliers and their indices

        outlier_list_col = df[(df[i] < Q1 - outlier_step) | (df[i] > Q3 + outlier_step)].index

    #Storing Outliers indices to the variable list

        outliers_indices.extend(outlier_list_col)

        

    outliers_indices = Counter(outliers_indices)

    multiple_outliers = list(i for i, c in outliers_indices.items() if c > 2)

        

    return multiple_outliers
train_df.loc[detect_outliers(train_df,["Age","Parch","SibSp","Fare"])]
#Drop Outliers



train_df = train_df.drop(detect_outliers(train_df,["Age","Parch","SibSp","Fare"]),axis=0).reset_index(drop = True)
train_df_len = len(train_df)

train_df = pd.concat([train_df,test_df],axis = 0).reset_index(drop=True)
train_df.columns[train_df.isnull().any()] #Finding Columns of the data contains null value
train_df.isnull().sum() #lookin for the null data total number for each feature
train_df[train_df["Embarked"].isnull()]
train_df.boxplot(column="Fare",by = "Embarked")

plt.show()
train_df.boxplot(column="Pclass",by = "Embarked")

plt.show()
train_df.boxplot(column="Age",by = "Embarked")

plt.show()
train_df["Embarked"] = train_df["Embarked"].fillna("C") 

train_df[train_df["Embarked"].isnull()]
train_df[train_df["Fare"].isnull()]


train_df.boxplot(column="Fare",by="Pclass")

plt.show()
np.mean(train_df[train_df["Pclass"]==3]["Fare"])
train_df["Fare"] = train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"]==3]["Fare"]))

train_df[train_df["Fare"].isnull()]
listCor=["Survived","Age","Parch","SibSp","Fare"]

sns.heatmap(train_df[listCor].corr(),annot=True,fmt =".2f")

plt.show()
a = sns.factorplot(x = "SibSp",y = "Survived", data = train_df, kind = "bar",size = 7)

a.set_ylabels("Survive Possiblity")

plt.show()
a = sns.factorplot(x = "Parch",y = "Survived", data = train_df, kind = "bar",size = 7)

a.set_ylabels("Survive Possiblity")

plt.show()
a = sns.factorplot(x = "Pclass",y = "Survived", data = train_df, kind = "bar",size = 7)

a.set_ylabels("Survive Possiblity")

plt.show()
sns.set_style("whitegrid")

a=sns.FacetGrid(train_df,col="Survived",size=8)



a.map(sns.distplot,"Age",bins = 30)

plt.show()
sns.set_style("whitegrid")

a=sns.FacetGrid(train_df,col="Survived",row ="Pclass",size=4)

a.add_legend()

a.map(plt.hist,"Age",bins = 30)

plt.show()
sns.set_style("whitegrid")

a = sns.FacetGrid(train_df, row ="Embarked",size=5)

a.map(sns.pointplot,"Pclass","Survived","Sex")

a.add_legend()

plt.show()
sns.set_style("whitegrid")

a = sns.FacetGrid(train_df, row ="Embarked",col="Survived",size=5)

a.map(sns.barplot,"Sex","Fare")

a.add_legend()

plt.show()
train_df[train_df["Age"].isnull()]
sns.factorplot(x="Sex",y="Age",kind="box",data=train_df,size=10)

plt.show()
sns.factorplot(x="Sex",y="Age", hue="Pclass",kind="box",data=train_df,size=10)

plt.show()
sns.factorplot(x="Parch",y="Age",kind="box",data=train_df,size=10)

sns.factorplot(x="SibSp",y="Age",kind="box",data=train_df,size=10)

plt.show()
train_df["Sex"] = [1 if i == "male" else 0 for i in train_df["Sex"]]

train_df["Embarked"] = [0 if i == "C"  else 1 if i == "Q" else 2 for i in train_df["Embarked"]]

sns.heatmap(train_df[["Age","Sex","Parch","SibSp","Pclass","Embarked"]].corr(),annot = True)

plt.show()
index_nan_age = list(train_df["Age"][train_df["Age"].isnull()].index)

for i in index_nan_age:

    age_predict = train_df["Age"][((train_df["SibSp"] == train_df.iloc[i]["SibSp"]) & (train_df["Parch"] == train_df.iloc[i]["Parch"]) & (train_df["Pclass"] == train_df.iloc[i]["Pclass"]))].median()

    age_median = train_df["Age"].median()

    if not np.isnan(age_predict):

        train_df["Age"].iloc[i] = age_predict

    else:

        train_df["Age"].iloc[i] = age_median
train_df[train_df["Age"].isnull()]