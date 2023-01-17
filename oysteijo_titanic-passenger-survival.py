import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
titanic = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

full = pd.concat([titanic, test])
titanic.info()

print("-"*40)

test.info()
# just to show the numbers

full.Embarked.value_counts()
# or maybe even more convincing: A plot!

full.Embarked.value_counts().plot(kind='bar')
# See? It's pretty safe two assume they came for Southampton

titanic.Embarked.fillna(value='S', inplace=True)

# Let's update the full dataframe as well:

full.Embarked.fillna(value='S', inplace=True)
test[np.isnan(test["Fare"])]
# Mr. Storey from Southampton in third class.

test.loc[test["PassengerId"] == 1044, "Fare"] = full[(full["Embarked"]=='S') & (full["Pclass"]==3)].Fare.median()

test.loc[test["PassengerId"]==1044,:]
# So let's add a title column to each DataFrame

# we also make a list of all our frames, such that we can easily loop over them.

frames = [titanic, test, full]

for df in frames:

    df["Title"] = df.Name.str.replace('(.*, )|(\\..*)', '')
full.head()
# That went good. There are 18 unique titles.

full["Title"].value_counts()
# Let's check which titles that are missing age data

full[np.isnan(full["Age"])].Title.unique()
# Hmmm what's the difference of Miss and Ms?

full[full["Title"]=="Ms"]
# Let's just set the two Ms to Miss. Can't be that bad.

for df in frames:

    df.loc[df["Title"]=="Ms", "Title"] = "Miss"
# So here is the main juice. Assign the missing age to the median age with the given title.

for t in full[np.isnan(full["Age"])].Title.unique():

    for df in frames:

        df.loc[(df["Title"]==t) & np.isnan(df["Age"]), "Age" ] = full[full["Title"]==t].Age.median()
# Let's look at the cabin a bit. This might be important. The "Women and Children" policy was enforced differently on

# starboard and port side. Odd numbered cabins are starboard side, and even numbers are port side.

for df in [titanic, test]:

    df["CabinSide"] = "Unknown"

    df.loc[pd.notnull(df["Cabin"]) & df["Cabin"].str[-1].isin(["1", "3", "5", "7", "9"]),"CabinSide"] = "Starboard"

    df.loc[pd.notnull(df["Cabin"]) & df["Cabin"].str[-1].isin(["0", "2", "4", "6", "8"]),"CabinSide"] = "Port"
for df in [titanic, test]:

    df.loc[df["Ticket"]=="PC 17608", "CabinSide"] = "Starboard"
test.loc[test["Name"].str.contains("Bowen,"),"Cabin"] = "B68"
titanic.CabinSide.value_counts()
# Maybe the Deck is important? who knows?

for df in [titanic, test]:

    df["Deck"] = "Unknown"

    df.loc[pd.notnull(df["Cabin"]), "Deck"] = df["Cabin"].str[0]
titanic.loc[titanic.Cabin.str.len() == 5,:]
for df in [titanic, test]:

    df.loc[pd.notnull(df["Cabin"]) & (df.Cabin.str.len() == 5), "Deck"] = df["Cabin"].str[2]
test.loc[test.Cabin.str.len() == 5,:]
# Test if there is some strange decks as well. Deck T ??

titanic.Deck.value_counts()
titanic.loc[titanic["Deck"] == 'T',:]
titanic.loc[titanic["Deck"] == 'T',"Deck"] = "Unknown"
# Let's define another feature. FamilySize = Parch + SibSp + 1

for df in frames:

    df["FamilySize"] = df.Parch + df.SibSp + 1
# Ticket group size

# first we make a dictionary

ticket_dict = {}

for t in full.Ticket.unique():

    ticket_dict[t] = 0

for t in full.Ticket:

    ticket_dict[t] += 1



# Then we apply it to the dataframes

for df in frames:

    df["TicketGroupSize"] = df["Ticket"].apply( lambda x: ticket_dict[x])
for t in full.Ticket.unique():

    ticket_dict[t] = 0

for row in full.iterrows():

    t = row[1]["Ticket"]

    if row[1]["Survived"] > 0.1:

        ticket_dict[t] += 1

        

# Then we apply this to the dataframes

for df in [titanic, test]:

    df["TicketGroupSurvivors"] = df["Ticket"].apply( lambda x: ticket_dict[x])
# Here is the problem:

titanic.loc[titanic["TicketGroupSize"] == 1, ["Survived", "TicketGroupSurvivors"]]
# Here is the remedy:

titanic.loc[titanic["TicketGroupSize"] == 1, ["Survived", "TicketGroupSurvivors"]] = 0
# Normalizer

def normalize(feat):

    mean = full[feat].mean()

    stdv = full[feat].std()

    for df in [titanic,test]:

        df[feat + "_norm"] = (df[feat] - mean) / stdv
# Age, SibSp, ParCh, FamilySize, Fare and TicketGroupSize. Those are the ones.

[normalize(x) for x in ["Age", "SibSp", "Parch", "FamilySize", "Fare", "TicketGroupSize"]]
titanic.head()
full["Title"].value_counts()
# These can be discussed, of course.

titledict = {"Dr"   : "Mr",

             "Col"  : "Officer",

             "Mlle" : "Miss",

             "Major": "Officer",

             "Lady" : "Royal",

             "Dona" : "Royal",

             "Don"  : "Royal",

             "Mme"  : "Mrs",

             "the Countess": "Royal",

             "Jonkheer": "Royal",

             "Capt" : "Officer",

             "Sir"  : "Mr"

             }

#There is probably a pandas way to do this, however I do it the python way today.

for df in frames:

    for key,val in titledict.items():

        df.loc[df["Title"]==key, "Title"] = val
full["Title"].value_counts()
category_list = ["Pclass", "Sex", "Embarked", "Title", "CabinSide", "Deck"]

titanic = pd.get_dummies(titanic, columns=category_list)

test = pd.get_dummies(test, columns=category_list)
for df in [titanic, test]:

    df["TGS_norm"] = df["TicketGroupSurvivors"] / df["TicketGroupSize"]
titanic.columns
test.columns
# What should we expect if a predictor predicting all dead.

1 - titanic.Survived.mean()
# Let's see the real.

ds_submission = pd.DataFrame(test["PassengerId"])

ds_submission["Survived"] = 0  # All dead
# This is actually the simplest predictor I can imagine so for the fun of it, let's submit this

# and see how it scores

ds_submission.to_csv("all_dead.csv", index=False)
round(len(test) * 0.62679)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV
k_range = range(1, 31)

param_grid = dict(n_neighbors=list(k_range),weights = ["uniform", "distance"])
knn = KNeighborsClassifier(n_neighbors=5)

grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
titanic.columns
features = ['Age_norm', 'SibSp_norm', 'Parch_norm',

       'FamilySize_norm', 'Fare_norm', 'TicketGroupSize_norm', 'Pclass_1',

       'Pclass_2', 'Sex_female', 'Embarked_C',

       'Embarked_Q', 'Title_Master', 'Title_Miss', 

       'Title_Mrs', 'Title_Officer', 'Title_Rev', 'Title_Royal',

       'CabinSide_Port', 'CabinSide_Starboard', 'Deck_A',

       'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E', 'Deck_F', 'Deck_G',

       'TGS_norm']
len(features)
grid.fit(titanic[features], titanic.Survived)
grid.best_score_
grid.best_params_
# rerun fit() with the best parameters with the entire set (no CV)

knn = KNeighborsClassifier(n_neighbors=7, weights="distance")

knn.fit(titanic[features], titanic.Survived)
knn_predictions_with_TGS = knn.predict(test[features])
knn_predictions_with_TGS.sum()
knn_submission1 = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": knn_predictions_with_TGS

        })

knn_submission1.to_csv("knn_predictions_with_TGS.csv", index=False)
knn = KNeighborsClassifier(n_neighbors=5)

# Fit once more without TGS_norm

grid.fit(titanic[features[:-1]], titanic.Survived)

print(grid.best_score_ , grid.best_params_)
# rerun fit() with the best parameters with the entire set (no CV)

knn = KNeighborsClassifier(n_neighbors=8, weights='distance')

knn.fit(titanic[features[:-1]], titanic.Survived)
knn_predictions_wo_TGS = knn.predict(test[features[:-1]])

knn_predictions_wo_TGS.sum()
knn_submission2 = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": knn_predictions_wo_TGS

        })

knn_submission2.to_csv("knn_predictions_wo_TGS.csv", index=False)
knn_submission3 = pd.DataFrame({

    "PassengerId": test["PassengerId"],

    "Survived": np.logical_or(knn_predictions_with_TGS, knn_predictions_wo_TGS).astype('int') 

})

knn_submission3.to_csv("knn_predictions_or_TGS.csv", index=False)
df_tmp = pd.DataFrame({

    "PassengerId": test["PassengerId"],

    "TicketGroupSize": test["TicketGroupSize"],

    "s_with_tgs": knn_predictions_with_TGS,

    "s_wo_tgs": knn_predictions_wo_TGS

})

df_tmp.loc[df_tmp.TicketGroupSize == 1, "Survived"] = df_tmp["s_wo_tgs"]

df_tmp.loc[df_tmp.TicketGroupSize != 1, "Survived"] = df_tmp["s_with_tgs"]

df_tmp.drop(["TicketGroupSize", "s_with_tgs", "s_wo_tgs"], axis=1, inplace=True)

df_tmp.to_csv("knn_predictions_specified_TGS.csv", index=False, float_format='%.f')
from sklearn.ensemble import RandomForestClassifier
n_range = range(10, 100, 10)

param_grid = dict(n_estimators=list(n_range),criterion = ["gini", "entropy"])

rfc = RandomForestClassifier(n_estimators=20)

grid = GridSearchCV(rfc, param_grid, cv=10, scoring='accuracy')

grid.fit(titanic[features], titanic.Survived)

print(grid.best_score_ , grid.best_params_)
rfc = RandomForestClassifier(n_estimators=60, criterion='entropy')

rfc.fit(titanic[features], titanic.Survived)
pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": rfc.predict(test[features])

        }).to_csv("rfc_predictions.csv", index=False)