# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
def detect_outliers(df,n,features):

    """

    Takes a dataframe df of features and returns a list of the indices

    corresponding to the observations containing more than n outliers according

    to the Tukey method.

    """

    outlier_indices = []

    

    # iterate over features(columns)

    for col in features:

        # 1st quartile (25%)

        Q1 = np.percentile(df[col], 25)

        # 3rd quartile (75%)

        Q3 = np.percentile(df[col],75)

        # Interquartile range (IQR)

        IQR = Q3 - Q1

        

        # outlier step

        outlier_step = 1.5 * IQR

        

        # Determine a list of indices of outliers for feature col

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        

        # append the found outlier indices for col to the list of outlier indices 

        outlier_indices.extend(outlier_list_col)

        

    # select observations containing more than 2 outliers

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

    return multiple_outliers   



# detect outliers from Age, SibSp , Parch and Fare

Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])
train.loc[Outliers_to_drop]
train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
dataset =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
dataset = dataset.fillna(np.nan)

dataset.isnull().sum()
g = sns.heatmap(train[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
# Explore SibSp feature vs Survived

g = sns.factorplot(x="SibSp",y="Survived",data=train, kind="bar", height = 5 , 

palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
g  = sns.factorplot(x="Parch",y="Survived",data=train,kind="bar", height = 6)

g.despine(left=True)

g = g.set_ylabels("survival probability")
a = train["Parch"] 

ind = list(a.index[a==4])

train.loc[ind]
g = sns.FacetGrid(train, col='Survived')

g = g.map(sns.distplot, "Age")
g = sns.kdeplot(train["Age"][(train["Survived"] == 0) & (train["Age"].notnull())], color="Red", shade = True)

g = sns.kdeplot(train["Age"][(train["Survived"] == 1) & (train["Age"].notnull())], ax =g, color="Blue", shade= True)

g.set_xlabel("Age")

g.set_ylabel("Frequency")

g = g.legend(["Not Survived","Survived"])
g = sns.FacetGrid(train, col='Survived')

g = g.map(sns.distplot, "Fare")
g = sns.barplot(x="Sex",y="Survived",data=train)

g = g.set_ylabel("Survival Probability")
g = sns.factorplot(x="Pclass",y="Survived",data=train,kind="bar", height = 6 , 

palette = "muted")

g.despine(right=True)

g = g.set_ylabels("survival probability")
g = sns.factorplot(x = "Embarked", y = "Survived", data = train,kind = "bar", height = 6, palette = "muted")

g.despine(left = False)

g = g.set_ylabels("survival probability")
# Explore Pclass vs Embarked 

g = sns.factorplot("Pclass", col="Embarked",  data=train,

                   size=6, kind="count", palette="muted")

g.despine(left=True)

g = g.set_ylabels("Count")
g = sns.factorplot(y="Age",x="Sex",data=dataset,kind="bar")

g = sns.factorplot(y="Age",x="Sex",hue="Pclass", data=dataset,kind="bar")

g = sns.factorplot(y="Age",x="Parch", data=dataset,kind="bar")

g = sns.factorplot(y="Age",x="SibSp", data=dataset,kind="bar")
dataset["Sex"] = dataset["Sex"].map({"male": 0, "female":1})

g = sns.heatmap(dataset[["Age","Sex","SibSp","Parch","Pclass"]].corr(),cmap="BrBG",annot=True)
dataset["Name"].head()
# Get Title from Name

dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]

dataset["Title"] = pd.Series(dataset_title)

dataset["Title"].head()
g = sns.countplot(x="Title",data=dataset)

g = plt.setp(g.get_xticklabels(), rotation=45)
# Convert to categorical values Title 

dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})

dataset["Title"] = dataset["Title"].astype(int)

dataset["Title"].head()
g = sns.countplot(dataset["Title"])

g = g.set_xticklabels(["Master","Miss/Ms/Mme/Mlle/Mrs","Mr","Rare"])
g = sns.factorplot(x="Title",y="Survived",data=dataset,kind="bar")

g = g.set_xticklabels(["Master","Miss-Mrs","Mr","Rare"])

g = g.set_ylabels("survival probability")
# Create a family size descriptor from SibSp and Parch

dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1



g = sns.factorplot(x="Fsize",y="Survived",data = dataset)

g = g.set_ylabels("Survival Probability")
# Create new feature of family size

dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)

dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if  s == 2  else 0)

dataset['MedF'] = dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

dataset['LargeF'] = dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)
dataset[['Fsize','Single','SmallF','MedF','LargeF']].head(

                                                         )
g = sns.factorplot(x="Single",y="Survived",data=dataset,kind="bar")

g = g.set_ylabels("Survival Probability")

g = sns.factorplot(x="SmallF",y="Survived",data=dataset,kind="bar")

g = g.set_ylabels("Survival Probability")

g = sns.factorplot(x="MedF",y="Survived",data=dataset,kind="bar")

g = g.set_ylabels("Survival Probability")

g = sns.factorplot(x="LargeF",y="Survived",data=dataset,kind="bar")

g = g.set_ylabels("Survival Probability")
dataset.head()
# convert to indicator values Title and Embarked 

dataset = pd.get_dummies(dataset, columns = ["Title"])

dataset = pd.get_dummies(dataset, columns = ["Embarked"], prefix="Em")
dataset.head()
# Replace the Cabin number by the type of cabin 'X' if not

dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin'] ])
g = sns.countplot(dataset["Cabin"],order=['A','B','C','D','E','F','G','T','X'])
g = sns.factorplot(y="Survived",x="Cabin",data=dataset,kind="bar",order=['A','B','C','D','E','F','G','T','X'])

g = g.set_ylabels("Survival Probability")
#convert to indicators

dataset = pd.get_dummies(dataset, columns = ["Cabin"],prefix="Cabin")
dataset.head()
# Treat Ticket by extracting the ticket prefix. When there is no prefix it returns X. 



Ticket = []

for i in list(dataset.Ticket):

    if not i.isdigit() :

        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix

    else:

        Ticket.append("X")

        

dataset["Ticket"] = Ticket
dataset["Ticket"].head() 
dataset = pd.get_dummies(dataset, columns = ["Ticket"], prefix="T")



# Create categorical values for Pclass

dataset["Pclass"] = dataset["Pclass"].astype("category")

dataset = pd.get_dummies(dataset, columns = ["Pclass"],prefix="Pc")



# Drop useless variables 

dataset.drop(labels = ["PassengerId"], axis = 1, inplace = True)



dataset.head()
dataset.drop(labels = ["Name"], axis = 1, inplace = True)
dataset.head()