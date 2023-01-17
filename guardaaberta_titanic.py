import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
df = pd.read_csv("../input/train.csv")
df.sample(5)
# check null values



null = pd.DataFrame(df.isnull().sum(), columns=["num_of_nulls"])

null["null_percentage"] = null["num_of_nulls"] / df.shape[0]

null
# check age really quickly



df[df.Age.isnull()].sample(5)
# check how many died and how many survived



print("Did not survived:", (df.Survived == 0).sum())

print("Survived:", (df.Survived == 1).sum())



sns.catplot(x="Survived", kind="count", data=df, height=3)
# check class distribution



# plot

sns.catplot(x="Pclass", col="Survived", kind="count", data=df, height=3)
# check Name



df.sample(5)
# extract some feature



last_name = df.Name.apply(lambda x: x.split(",")[0])

last_name.name = "last_name"



title = df.Name.apply(lambda x: x.split(",")[1]).apply(lambda x: x.split(".")[0])

title.name = "title"



name_ft_df = df.copy()

name_ft_df = pd.concat([name_ft_df, last_name, title], axis=1)
name_ft_df.groupby("last_name").count().sort_values(by="Survived", ascending=False).head()
name_ft_df[name_ft_df.last_name == "Andersson"]
# surival count per title



title_ft_df = name_ft_df.groupby("title").agg(["sum", "count"]).Survived.sort_values("count", ascending=False)

title_ft_df["survived_percentage"] = np.divide(title_ft_df[["sum"]], title_ft_df[["count"]])

title_ft_df
df.groupby("Sex").mean().Survived
# check gender distribution



# plot

sns.catplot(x="Sex", col="Survived", kind="count", data=df, height=3)
pd.get_dummies(df[["Survived", "Sex"]]).corr()
# check age distribution



#plot distributions of age of passengers who survived or did not survive

g = sns.FacetGrid(df, hue = 'Survived', height=2, aspect=5)

g.map(sns.kdeplot, 'Age', shade=True )

g.add_legend()
df[["Survived", "Age"]].corr()
new_age_feature = pd.cut(df["Age"], bins=[0, 10, 30, 50, np.inf], labels=["children", "young_adults", "adults", "old_adult"])

new_age_feature = pd.concat([df[["Survived", "Age"]], new_age_feature], axis=1)

pd.get_dummies(new_age_feature).corr()
df[df.Age.isnull()].sample(5)
# plot fare



# plot Fare distributions of who survived or did not survive

g = sns.FacetGrid(df, hue = 'Survived', height=3, aspect=5)

g.map(sns.kdeplot, 'Fare', shade=True )

g.add_legend()
# plot fare and Pclass - they should correlate

g = sns.FacetGrid(df, hue = 'Pclass', height=3, aspect=5)

g.map(sns.kdeplot, 'Fare', shade=True )

g.add_legend()
df[["Pclass", "Fare", "Survived"]].corr()
df[df.Pclass == 3].Fare.mean()
df[["Fare"]].describe()
# discretize fare



fare_ft = pd.cut(df["Fare"], bins=[0, 15, 30, np.inf], labels=["low_fare", "mid_fare", "high_fare"])

fare_ft = pd.concat([df.Survived, fare_ft], axis=1)

fare_ft = pd.get_dummies(fare_ft)

fare_ft.corr()
# SibSp



# check null

print(df.SibSp.isnull().sum())



# groupby SibSp and check how many survived

sibsp_df = df.groupby("SibSp").agg(["count", "sum"])["Survived"]

sibsp_df["survived_percentage"] = sibsp_df["sum"] / sibsp_df["count"]

sibsp_df
sns.catplot(x="Pclass", col="SibSp", kind="count", data=df)
# Parch



# check null

print(df.Parch.isnull().sum())



# groupby SibSp and check how many survived

parch_df = df.groupby("Parch").agg(["count", "sum"])["Survived"]

parch_df["survived_percentage"] = parch_df["sum"] / parch_df["count"]

parch_df
sns.catplot(x="Pclass", col="Parch", kind="count", data=df)
# A new feature that combines Parch and SibSp



# groupby SibSp and check how many survived

group_ft_df = df.copy()

group_ft_df["group_size"] = group_ft_df["SibSp"] + group_ft_df["Parch"]

group_ft_df["alone"] = group_ft_df["group_size"] == 0

group_ft_df["small_group"] = group_ft_df["group_size"].isin([1, 2])

group_ft_df["large_group"] = group_ft_df["group_size"] > 2



group_ft_df = group_ft_df[["Survived", "SibSp", "Parch", "group_size", "alone", "small_group", "large_group"]]
group_ft_df.corr()
# Embarked



print(df.Embarked.isnull().sum())



sns.catplot(x="Survived", col="Embarked", kind="count", data=df, height=3)



sns.catplot(x="Pclass", col="Embarked", kind="count", data=df, height=3)
# Ticket

print(df.Ticket.isnull().sum())



df.Ticket.unique()
df.groupby("Ticket").count().sort_values("Survived", ascending=False).tail(5)
df[df.Ticket == "343276"]
known_group_survival_rate = []



for row in df.iterrows():

    passenger_id = row[1].PassengerId

    ticket_id = row[1].Ticket

    temp_df = df.loc[df.PassengerId.apply(lambda x: x != passenger_id)]

    temp_df = temp_df[temp_df.Ticket == ticket_id]

    survival_rate = temp_df.Survived.mean()

    

    known_group_survival_rate.append(survival_rate)



known_group_survival_rate = pd.Series(known_group_survival_rate, name="known_group_survival_rate")
pd.concat([df.Survived, known_group_survival_rate], axis=1).corr()
# try replace large number of nan (group size of 1, so no known surival rate) with the mean survival rate of the entire dataset



pd.concat([df.Survived, known_group_survival_rate.fillna(df.Survived.mean())], axis=1).corr()
# Cabin

print(df.Cabin.isnull().sum())

print(df.Cabin.isnull().sum() / df.shape[0])
df.groupby("Cabin").agg(["mean", "count"]).Survived.sort_values(by="count", ascending=False).head()
df[df.Cabin == "C23 C25 C27"]
df[df.Ticket == "19950"]
# drop the irrelevant features (PassengerId) and features that require further parsing and engineering (Name, Ticket, and Cabin)

base_df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])



# drop nan

base_df = base_df.dropna(axis=0)



# assign y and X

y = base_df[["Survived"]]

X = base_df.drop(columns=["Survived"])



# one-hot encode

X = pd.get_dummies(X)
X.sample(5)
# A simple logistic regression model

# not the most appropriate since relationships are highly non-linear and tree-based model should perform well

# but for baseline it's fine



from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score



clf = LogisticRegression() #Initialize with whatever parameters you want to



# 5-fold cross validation

print(np.mean(cross_val_score(clf, X, y, cv=5)))
df.Age.median()
# a function that process dataframe

# written as a function so it's easier to process the test dataframe



def preprocess(df):

    frmt_df = df.copy()



    # Pclass

    frmt_df.Pclass = frmt_df.Pclass.replace({3: 1, 1: 3})



    # add features based on SibSp and Parch

    frmt_df["group_size"] = frmt_df["SibSp"] + frmt_df["Parch"] 

    frmt_df["alone"] = (frmt_df["group_size"] == 0).astype(int)

    frmt_df["small_group"] = (frmt_df["group_size"].isin([1, 2])).astype(int)

    frmt_df["large_group"] = (frmt_df["group_size"] > 2).astype(int)

    frmt_df = frmt_df.drop(columns=["SibSp", "Parch"]) # whether drop or not can be handled by TPOT



    # add feature based on ticket

    known_group_survival_rate = []

    for row in frmt_df.iterrows():

        passenger_id = row[1].PassengerId

        ticket_id = row[1].Ticket

        temp_df = frmt_df.loc[frmt_df.PassengerId.apply(lambda x: x != passenger_id)]

        temp_df = temp_df[temp_df.Ticket == ticket_id]

        survival_rate = temp_df.Survived.mean()

        known_group_survival_rate.append(survival_rate)

    known_group_survival_rate = pd.Series(known_group_survival_rate, name="known_group_survival_rate")

    frmt_df = pd.concat([frmt_df, known_group_survival_rate.fillna(df.Survived.mean())], axis=1)

    frmt_df = frmt_df.drop(columns=["Ticket"])



    # discretize fare

    fare_ft = pd.cut(frmt_df["Fare"], bins=[0, 15, 30, np.inf], labels=["low_fare", "mid_fare", "high_fare"])

    fare_ft.name = "discretized_fare"

    frmt_df = pd.concat([frmt_df, fare_ft], axis=1)

    frmt_df = frmt_df.drop(columns=["Fare"])



    # drop Cabin

    frmt_df = frmt_df.drop(columns=["Cabin"])



    # drop PassengerId

    frmt_df = frmt_df.drop(columns=["PassengerId"])



    # fill Age with median and discretize Age

    frmt_df.Age = frmt_df.Age.fillna(df.Age.median()) # null value imputation can be handled by TPOT

    age_ft = pd.cut(frmt_df["Age"], bins=[0, 10, 30, 50, np.inf], labels=["children", "young_adults", "adults", "old_adult"])

    age_ft.name = "discretized_age"

    frmt_df = pd.concat([frmt_df, age_ft], axis=1)

    frmt_df = frmt_df.drop(columns=["Age"]) # whether drop or not can be handled by TPOT



    # drop Name

    frmt_df = frmt_df.drop(columns=["Name"]) # whether drop or not can be handled by TPOT



    # one-hot_encode

    frmt_df = pd.get_dummies(frmt_df)

    

    return frmt_df
frmt_df = preprocess(df)

frmt_df.sample(5)
from tpot import TPOTClassifier

from sklearn.model_selection import train_test_split



y = frmt_df[["Survived"]]

X = frmt_df.drop(columns=["Survived"])

X_train, X_test, y_train, y_test = train_test_split(X, y,

                                                    train_size=0.75, test_size=0.25)



tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2)

tpot.fit(X_train, y_train)

print(tpot.score(X_test, y_test))
# load test df

test_df = pd.read_csv("../input/test.csv")



# extract test passenger id

test_passengerid = test_df.PassengerId

# this is necessary as "known group survival rate needs the original train set"

test_df["Survived"] = np.nan

test_df = test_df[df.columns]

all_df = pd.concat([df, test_df]).reset_index().iloc[:, 1:]

test_index = all_df[all_df.PassengerId.isin(test_passengerid)].index



# process test df

frmt_test_df = preprocess(all_df).loc[test_index]

frmt_test_df = frmt_test_df.drop(columns=["Survived"])
frmt_test_df.columns == X_train.columns
X_train.sample(3)
frmt_test_df.sample(3)
test_results = tpot.predict(frmt_test_df)
test_id = all_df.loc[frmt_test_df.index].PassengerId

my_submission = pd.DataFrame({'PassengerId': test_id, 'Survived': test_results})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)