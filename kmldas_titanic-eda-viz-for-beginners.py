# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd

import statistics



# visualization

import seaborn as sns

sns.set(style='white', context='notebook', palette='deep')

import matplotlib.pyplot as plt

%matplotlib inline





# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve





from collections import Counter

import warnings

warnings.filterwarnings("ignore")





train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')

display("train data", train_df)

display("test data", test_df)

print(train_df.columns.values)

print(train_df.describe())
print(train_df.describe(include=['O']))
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Correlation matrix between numerical values (SibSp Parch Age and Fare values) and Survived 

g = sns.heatmap(train_df[["Survived","SibSp","Parch","Age","Fare"]].corr(),

                annot=True, fmt = ".2f", cmap = "coolwarm")
# Explore SibSp feature vs Survived

g = sns.catplot(x="SibSp",y="Survived",data=train_df,kind="bar", height = 6 , 

palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
#Explore Parch feature vs Survived

g  = sns.factorplot(x="Parch",y="Survived",data=train_df,kind="bar", height = 6 , 

palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
# Explore Age vs Survived

g = sns.FacetGrid(train_df, col='Survived')

g = g.map(sns.distplot, "Age")
# Explore Age distibution 

g = sns.kdeplot(train_df["Age"][(train_df["Survived"] == 0) & (train_df["Age"].notnull())], color="Red", shade = True)

g = sns.kdeplot(train_df["Age"][(train_df["Survived"] == 1) & (train_df["Age"].notnull())], ax =g, color="Blue", shade= True)

g.set_xlabel("Age")

g.set_ylabel("Frequency")

g = g.legend(["Not Survived","Survived"])
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')

grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', height=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
#Join train and test datasets in order to obtain the same number of features during categorical conversion



combined =  pd.concat(objs=[train_df, test_df], axis=0).reset_index(drop=True)

combined
# Fill empty and NaNs values with NaN

combined = combined.fillna(np.nan)



# Check for Null values

combined.isnull().sum()
#Fill Embarked nan values of dataset set with the most frequent value

embarked_mode=statistics.mode(combined["Embarked"])

combined["Embarked"] = combined["Embarked"].fillna(embarked_mode)
# Explore Age vs Sex, Parch , Pclass and SibSP

g = sns.catplot(y="Age",x="Sex",data=combined,kind="box")

g = sns.catplot(y="Age",x="Sex",hue="Pclass", data=combined,kind="box")

g = sns.catplot(y="Age",x="Parch", data=combined,kind="box")

g = sns.catplot(y="Age",x="SibSp", data=combined,kind="box")
# convert Sex into categorical value 0 for male and 1 for female

combined["Sex"] = combined["Sex"].map({"male": 0, "female":1})



g = sns.heatmap(combined[["Age","SibSp","Parch","Pclass"]].corr(),cmap="BrBG",annot=True)
# Filling missing value of Age 



## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp

# Index of NaN age rows

index_NaN_age = list(combined["Age"][combined["Age"].isnull()].index)



for i in index_NaN_age:

    age_med = combined["Age"].median()

    age_pred = combined["Age"][((combined['SibSp'] == combined.iloc[i]["SibSp"]) 

                                & (combined['Parch'] == combined.iloc[i]["Parch"]) 

                                & (combined['Pclass'] == combined.iloc[i]["Pclass"]))].median()

    if not np.isnan(age_pred) :

        combined['Age'].iloc[i] = age_pred

    else :

        combined['Age'].iloc[i] = age_med
# Get Title from Name

combined_title = [i.split(",")[1].split(".")[0].strip() for i in combined["Name"]]

combined["Title"] = pd.Series(combined_title)

combined["Title"].head()
g = sns.countplot(x="Title",data=combined)

g = plt.setp(g.get_xticklabels(), rotation=45) 
title=combined["Title"].unique()

title
# Convert to categorical values Title 

combined["Title"] = combined["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

combined["Title"] = combined["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})

combined["Title"] = combined["Title"].astype(int)



g = sns.catplot(x="Title",y="Survived",data=combined,kind="bar")

g = g.set_xticklabels(["Master","Miss-Mrs","Mr","Rare"])

g = g.set_ylabels("survival probability")

# Create a family size descriptor from SibSp and Parch

combined["Fsize"] = combined["SibSp"] + combined["Parch"] + 1
g = sns.catplot(x="Fsize",y="Survived",data=combined,kind="bar")

g = g.set_ylabels("survival probability")

# Create new feature of family size

combined['Single'] = combined['Fsize'].map(lambda s: 1 if s == 1 else 0)

combined['SmallF'] = combined['Fsize'].map(lambda s: 1 if 2 <= s <= 4 else 0)

combined['LargeF'] = combined['Fsize'].map(lambda s: 1 if s >= 5 else 0)

# convert to indicator values Title and Embarked 

combined = pd.get_dummies(combined, columns = ["Title"])

combined = pd.get_dummies(combined, columns = ["Embarked"], prefix="Em")

combined.head()

combined.columns.values
# Create categorical values for Pclass

combined["Pclass"] = combined["Pclass"].astype("category")

combined = pd.get_dummies(combined, columns = ["Pclass"],prefix="Pc")
# Drop variable

combined.drop(labels = ["Name",'SibSp', 'Parch','Fsize','Cabin',"Ticket"], axis = 1, inplace = True)



#you may decide some are useful and retain. Adding a few of the above will improve the final solution. 

# try it out. see if you can figure out what helps and what does not



combined.info()