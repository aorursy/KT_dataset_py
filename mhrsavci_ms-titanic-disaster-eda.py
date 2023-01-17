# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt



plt.style.use("ggplot") # Using plot library with "ggplot" style.

# plt.style.available # Available using plot sytles



import plotly.graph_objects as go

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
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")



test_PassengerID = test_df["PassengerId"]
train_df.columns
train_df.head()
train_df.info() # Information about train_df dataframe.
train_df.describe() # Return statictics information about numeric column in train_df dataframe.
CategoryVar = ["Survived", "Pclass", "Name", "SibSp", "Sex", "Parch", "Cabin", "Embarked"]
def bar_plot(colName,dfName):

    """

        Input: Categorical data

        Output: Bar Plot & Value Count

    

    """

    # Get Feature

    var = dfName[colName]

    

    # Number of categorical variables

    varValue = var.value_counts()

    

    # Visualize

    plt.figure(figsize=(9,3))

    plt.bar(varValue.index,varValue)

    plt.xticks(varValue.index,varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(colName)

    plt.show()

    print(varValue)
CategoryVar
bar_plot("Survived",train_df)
bar_plot("Embarked",train_df)
NumericalVar=["PassengerId", "Age", "Fare"]
def hist_plot(colName,dfName):

    

    """

        Input : Numerical data

        Output: Histogram Plot & Value Counts of colName

    

    """

    

    plt.figure(figsize=(9,2))

    plt.hist(dfName[colName],bins=50)

    plt.xlabel(colName)

    plt.ylabel("Frequency")

    plt.title("{0} distribution with histogram".format(colName))

    plt.show()

    print(dfName[colName].value_counts())
hist_plot("Fare",train_df)
hist_plot("Age",train_df)
hist_plot("PassengerId",train_df)
## Pclass & Survived



train_df[["Pclass","Survived"]].groupby(["Pclass"],as_index=False).mean() # "Pclass" and "Survived" columns are groupped by "Pclass" column and than get mean value.
## Pclass & Survived 2



train_df[["Pclass","Survived"]].groupby(["Pclass"],as_index=False).mean().sort_values(by="Survived",ascending=False)
## Age & Survived



train_df[["Age","Survived"]].groupby(["Age"]).mean()
## Cabin & Survived



train_df[["Cabin","Survived"]].groupby(["Cabin"],as_index=False).mean()
## Sex & Survived



train_df[["Sex","Survived"]].groupby(["Sex"],as_index=False).mean()
## SibSp & Survived



train_df[["SibSp","Survived"]].groupby(["SibSp"],as_index=False).mean()
## Parch & Survived



train_df[["Parch","Survived"]].groupby(["Parch"],as_index=False).mean().sort_values(by="Survived",ascending=False)
train_df.head()
## Group by "Sex" column



train_df[["Sex","Survived","Pclass","Age","Parch","Fare"]].groupby(["Sex"]).mean()
## Group by "Embarked" column



train_df[["Embarked","Survived","Pclass","Age","Parch","Fare"]].groupby(["Embarked"]).mean()
train_df.head()
numlist = [1,2,3,5,6,8,12,15,22,27,33,39,45,56,59,100]



np.percentile(numlist,50)
def DetectOutlier(df,features):

    

    outlier_indices = []

    

    for c in features:

        

        # 1st quartile

        

        Q1 = np.percentile(df[c],25)

        

        # 3rd quartile

        

        Q3 = np.percentile(df[c],75)

        

        # IQR

        

        IQR = Q3 - Q1

        

        # Outlier Detection Step

        

        outlier_step = IQR * (1.5)

        

        # Outliers

        

        outliers = df[(df[c]<(Q1 - outlier_step)) | (df[c]>(Q3 + outlier_step))].index

        

        outlier_indices.extend(outliers)

        

        

    outlier_indices = Counter(outlier_indices)

    multiple_ouliers = list(i for i,v in outlier_indices.items() if v>2)

    

    return multiple_ouliers

        
# Detect Outliers



train_df.loc[DetectOutlier(train_df,["Age","SibSp","Parch","Fare"])]
# Drop Outliers



train_df = train_df.drop(DetectOutlier(train_df,["Age","SibSp","Parch","Fare"]),axis=0).reset_index(drop=True)
train_df
train_df_len = len(train_df)



train_df_all = pd.concat([train_df,test_df],axis=0).reset_index(drop=True)
train_df_all.head()
train_df_all.isnull().sum()
train_df_all.columns[train_df_all.isnull().any()] # Contains missing value in train_df columns
train_df_all.isnull().sum() # Null data counts by train_df columns
train_df_all[train_df_all["Embarked"].isnull()]
train_df_all.boxplot(column="Fare",by="Embarked")

plt.show()
train_df_all["Embarked"] = train_df_all["Embarked"].fillna("C") # Fill null value "C" 
train_df_all[train_df_all["Embarked"].isnull()]
train_df_all[train_df_all["Fare"].isnull()]
train_df_all.boxplot(column="Fare",by="Pclass")

plt.show()
train_df_all[["Fare","Pclass"]][train_df_all["Pclass"]==3].mean()
train_df_all["Fare"] = train_df_all["Fare"].fillna(12.74)
train_df_all[train_df_all["Fare"].isnull()]
train_df.head()
train_df_all.head()
# Correlation Matrix between Sibsp -- Parch -- Age -- Fare -- Survived



list1 = ["SibSp","Parch","Age","Fare","Survived"]



plt.figure(figsize=(15,15))



sns.heatmap(train_df[list1].corr(), annot = True,fmt=".2f")



plt.show()
# Fare feature seems to have correlation with "Survived" feature.
g = sns.factorplot(x="SibSp", y="Survived",data=train_df,kind = "bar",size = 9)

g.set_ylabels("Survived Probability")

plt.show()
s = sns.factorplot(x = "Parch",y="Survived", data = train_df,kind="bar",size=8)

s.set_ylabels("Survived Probability")

plt.show()
g = sns.factorplot(x="Pclass",y="SibSp",data=train_df,kind = "bar")



plt.show()
g = sns.factorplot(x="Pclass",y="Survived",data=train_df,kind = "bar")



g.set_ylabels("Survived Probability")



plt.show()
## Age Column Categorizing





train_df["AgeRange"] = ""

train_df["AgeRange"][train_df["Age"].isnull()] = "NAN"

train_df["AgeRange"][(train_df["Age"] > 0) & (train_df["Age"] <= 10)] = "[0-10]"

train_df["AgeRange"][(train_df["Age"] > 10) & (train_df["Age"] <= 25)] = "[10-25]"

train_df["AgeRange"][(train_df["Age"] > 25) & (train_df["Age"] <= 35)] = "[25-35]"

train_df["AgeRange"][(train_df["Age"] > 35) & (train_df["Age"] <= 50)] = "[35-50]"

train_df["AgeRange"][(train_df["Age"] > 50) & (train_df["Age"] <= 65)] = "[50-65]"

train_df["AgeRange"][(train_df["Age"] > 65) & (train_df["Age"] <= 75)] = "[65-75]"

train_df["AgeRange"][(train_df["Age"]>75)] = "[75-]"
train_df.head()
g = sns.factorplot(x = "AgeRange",y="Survived",data = train_df,kind = "bar")



plt.show()
g = sns.FacetGrid(train_df,col ="Survived")



g.map(sns.distplot,"Age",bins=15)



plt.show()
g = sns.FacetGrid(train_df,col="Survived",row="Pclass")



g.map(sns.distplot,"Age",bins=25)



plt.show()
g = sns.FacetGrid(train_df,row="Embarked",size=10)

g.map(sns.pointplot,"Survived","Pclass","Sex")

g.add_legend()

plt.show()
g = sns.FacetGrid(train_df,row="Embarked",col = "Survived",size=5)

g.map(sns.barplot,"Sex","Fare")

g.add_legend()

plt.show()

train_df[train_df["Age"].isnull()]
# Average Male Age 



train_df["Age"][train_df["Sex"]=="male"].describe()
# Average Female Age 



train_df["Age"][train_df["Sex"]=="female"].describe()
g = sns.factorplot(x="Pclass", y = "Age",data=train_df,kind="bar")



plt.show()
g = sns.factorplot(x="Pclass",y="AgeRange",data=train_df,kind="bar")
sns.factorplot(x = "Sex",y = "Age",data = train_df,kind = "bar")

plt.show()



# Sex feature is not benefical for Age feature missing fill.
sns.boxplot(x="Sex",y="Age",data=train_df)



plt.show()
sns.countplot(train_df["AgeRange"])
plt.figure(figsize=(10,10))

plt.pie(train_df["AgeRange"].value_counts().values,labels =train_df["AgeRange"].value_counts().keys() )

plt.show()
import plotly.express as px



fig = px.pie(train_df, values="Age",names="Embarked")

fig.show()
train_df[train_df["Embarked"]=="S"]["Age"].mean()
train_df.head()
import pandas as pd

import matplotlib.pyplot as plt



plt.figure(figsize=(15,10))

pd.plotting.parallel_coordinates(train_df.loc[:,["AgeRange","Pclass","Fare","SibSp","Survived","Age"]],"AgeRange")

plt.show()
import missingno as msno

import matplotlib.pyplot as plt



msno.matrix(train_df)

plt.show()
msno.bar(train_df)

plt.show()
sns.kdeplot(train_df["Age"],shade=True,cut=3)

plt.show()
sns.violinplot(data=train_df["Age"],inner="points")


plt.figure(figsize=(15,15))

sns.swarmplot(x="Sex",y="Age",hue="AgeRange",data=train_df)

plt.show()
plt.figure(figsize=(20,20))



sns.factorplot(x="Sex",y="Age",hue="Pclass",data=train_df,kind="box")



plt.show()
plt.figure(figsize=(20,20))



sns.factorplot(x="Sex",y="Age",hue="Embarked",data=train_df,kind="box")



plt.show()
plt.figure(figsize=(20,20))



sns.factorplot(x="SibSp",y="Age",data=train_df,kind="box")



plt.show()
plt.figure(figsize=(20,20))



sns.factorplot(x="Parch",y="Age",data=train_df,kind="box")



plt.show()
plt.figure(figsize=(20,20))



sns.factorplot(x="Pclass",y="Age",data=train_df,kind="box")



plt.show()
f,ax = plt.subplots(figsize=(15,15))



sns.heatmap(train_df.corr(),annot=True,linewidth=0.5,linecolor="white",fmt=".2f",ax=ax)



plt.show()
train_df[train_df["Age"].isnull()]
columns = {"SibSp":{0:29,1:30,2:24,3:8,4:7,5:11},"Parch":{0:30,1:24,2:15,3:35,4:44,5:39,6:44},"Pclass":{1:38,2:28,3:23}}
index_nan_Age = train_df[train_df["Age"].isnull()].index
for i in index_nan_Age:

    p = columns["Parch"].get(train_df["Parch"][i])

    s = columns["SibSp"].get(train_df["SibSp"][i])

    c = columns["Pclass"].get(train_df["Pclass"][i])

    train_df["Age"][i] = int((p+s+c) / 3)
train_df["Age"].head()