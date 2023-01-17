# An initial and the Titanic survival problem. I've gone through it step by step.

import pandas
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# load csv data
# load initial data:
initial_data = pandas.read_csv('../input/train.csv', header=0)

print ("Loaded data overview:")
print (initial_data.info())

print ("Loaded data description:")
print(initial_data.describe())

print("--- Ticket types")
print(initial_data["Ticket"].unique().size)

print("--- Fare values below 5:")
print(initial_data[initial_data["Fare"]<5][["Sex","Age","Pclass", "Fare"]])
# create clean data by removing values that don't have age
# alternatively we could try something like replacing missing values with the mean for each class/gender.
# since this only a look at the data we'll only remove invalid data this time around.
cleaned_data = initial_data[pandas.notnull(initial_data["Age"])]

# I think that embarked data is unimportant to survival so I'm going to drop it.
del cleaned_data["Embarked"]

# we also have very little information related to cabins, so we'll drop that too
del cleaned_data["Cabin"]

# it also seems that around 15 people have a fare of zero; this might have been acceptable if they were children
#   but by viewing the ages we can see that they are clearly adults;
#   so we'll assume that for these people fare information was unavailable and remove them too.
cleaned_data = cleaned_data[cleaned_data["Fare"] > 0]

# let's take a look at the ticket field
# print cleaned_data["Ticket"].unique()
# it seems that there's 538 different ticket types; I can't think any way for merging these into fewer groups so
#   we're going to delete this field too :)
del cleaned_data["Ticket"]

# we're going to change sex into a numeric value
cleaned_data["Gender"] = cleaned_data["Sex"].map({"female": 0, "male": 1}).astype(int)
del cleaned_data["Sex"]

# in order to reduce the total number of fields to make the ML process simpler, merge SibSP and Parch
cleaned_data["Family"] = cleaned_data["SibSp"] + cleaned_data["Parch"]
cleaned_data = cleaned_data.drop(["SibSp", "Parch"], axis=1)

# I'm guessing name and passenger id would be quite unrelated to survival as well,
#   so we're going to go ahead and delete that too
del cleaned_data["Name"]
del cleaned_data["PassengerId"]

print("Cleaned data overview:")
print(cleaned_data.info())
# make a final check to see we don't have any remaining NA values that could mess with our ML algorithm
print("Drop Na from cleaned data")
cleaned_data = cleaned_data.dropna()
print(cleaned_data.info())
train_data_df, test_data_df = train_test_split(cleaned_data, test_size=0.25)

# we have to turn our data frames into numpy arrays since that's what ML libs work with;

train_data = train_data_df.values
test_data = test_data_df.values

# we will try out three different methods for predicting and choose the best one

# the first method is the simple intuition that females survived

print("------")
print("simple score:")

res = test_data[0::,4] != test_data[0::,0]
print(sum(res)/float(res.size))

# next we will try a tree

print("------")

print("Tree score")
tree = DecisionTreeClassifier()
tree = tree.fit(train_data[0::, 1::], train_data[0::, 0])
print(tree.score(test_data[0::, 1::], test_data[0::, 0]))

# and finally we try the random forest
print("------")
print("Forest score:")

# create the random forest object which will include all the parameters for the fit
forest = RandomForestClassifier(n_estimators=100)

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(train_data[0::, 1::], train_data[0::, 0])

# now we will evaluate our forest by comparing the output with actual survival values in our test_data
print(forest.score(test_data[0::, 1::], test_data[0::, 0]))

# finally we'll print out the feature importance to see which fields had the highest impact
print("class", "         age", "         fare", "      gender", "     family")
print(forest.feature_importances_)
