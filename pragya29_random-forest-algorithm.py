import re
#numpy numerical calculation, matplotlib for visualization
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
#plt.style.use('ggplot')
train = pd.read_csv("train_titanic.csv")
test = pd.read_csv("test.csv")
train
#non numeric data

missing_val_df = pd.DataFrame(index=["Total", "Unique Cabin", "Missing Cabin"])
for name, df in zip(("Training data", "Test data"), (train, test)):
    total = df.shape[0]
    unique_cabin = len(df["Cabin"].unique())
    missing_cabin = df["Cabin"].isnull().sum()
    missing_val_df[name] = [total, unique_cabin, missing_cabin]
missing_val_df
train.drop("PassengerId", axis=1, inplace=True)
for df in train, test:
    df.drop("Cabin", axis=1, inplace=True)
non_empty_embarked = train["Embarked"].dropna()
unique_values, value_counts = non_empty_embarked.unique(), non_empty_embarked.value_counts()
X = np.arange(len(unique_values))
colors = ["brown", "grey", "purple"]

plt.bar(left=X,
        height=value_counts,
        color=colors,
        tick_label=unique_values)
plt.xlabel("Port of Embarkation")
plt.ylabel("Amount of embarked")
plt.title("Bar plot of embarked in Southampton, Queenstown, Cherbourg")
#Quantitative data

survived = train[train["Survived"] == 1]["Age"].dropna()
perished = train[train["Survived"] == 0]["Age"].dropna()

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
fig.set_size_inches(12, 6)
fig.subplots_adjust(hspace=0.5)
ax1.hist(survived, facecolor='green', alpha=0.75)
ax1.set(title="Survived", xlabel="Age", ylabel="Amount")
ax2.hist(perished, facecolor='brown', alpha=0.75)
ax2.set(title="Dead", xlabel="Age", ylabel="Amount")
survived = train[train["Survived"] == 1]["Fare"].dropna()
perished = train[train["Survived"] == 0]["Fare"].dropna()

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
fig.set_size_inches(12, 8)
fig.subplots_adjust(hspace=0.5)
ax1.hist(survived, facecolor='darkgreen', alpha=0.75)
ax1.set(title="Survived", xlabel="Age", ylabel="Amount")
ax2.hist(perished, facecolor='darkred', alpha=0.75)
ax2.set(title="Dead", xlabel="Age", ylabel="Amount")
for df in train, test:
    df["Embarked"].fillna("S", inplace=True)
    for feature in "Age", "Fare":
        df[feature].fillna(train[feature].mean(), inplace=True)
for df in train, test:
    df.drop("Ticket", axis=1, inplace=True)
for df in train, test:
    df["Embarked"] = df["Embarked"].map(dict(zip(("S", "C", "Q"), (0, 1, 2))))
    df["Sex"] = df["Sex"].map(dict(zip(("female", "male"), (0, 1))))
for df in train, test:
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
for df in train, test:
    titles = list()
    for row in df["Name"]:
        surname, title, name = re.split(r"[,.]", row, maxsplit=2)
        titles.append(title.strip())
    df["Title"] = titles
    df.drop("Name", axis=1, inplace=True)


title = train["Title"]
unique_values, value_counts = title.unique(), title.value_counts()
X = np.arange(len(unique_values))

fig, ax = plt.subplots()
fig.set_size_inches(18, 10)
ax.bar(left=X, height=value_counts, width=0.5, tick_label=unique_values)
ax.set_xlabel("Title")
ax.set_ylabel("Count")
ax.set_title("Passenger titles")
ax.grid(color='g', linestyle='--', linewidth=0.5)
#very rare surnames need to be replaced
for df in train, test:
    for key, value in zip(("Mr", "Mrs", "Miss", "Master", "Dr", "Rev"),
                          np.arange(6)):
        df.loc[df["Title"] == key, "Title"] = value
    df.loc[df["Title"] == "Ms", "Title"] = 1
    for title in "Major", "Col", "Capt":
        df.loc[df["Title"] == title, "Title"] = 6
    for title in "Mlle", "Mme":
        df.loc[df["Title"] == title, "Title"] = 7
    for title in "Don", "Sir":
        df.loc[df["Title"] == title, "Title"] = 8
    for title in "Lady", "the Countess", "Jonkheer":
        df.loc[df["Title"] == title, "Title"] = 9
test["Title"][414] = 0
nominal_features = ["Pclass", "Sex", "Embarked", "FamilySize", "Title"]
for df in train, test:
    for nominal in nominal_features:
        df[nominal] = df[nominal].astype(dtype="category")
train.sample(frac=1).head(5)
from sklearn.model_selection import train_test_split
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch",
              "Fare", "Embarked", "FamilySize", "Title"]
X_train, X_test, y_train, y_test = train_test_split(train[predictors], train["Survived"])
#Build a Random Forest model from the training set and evaluate the mean accuracy on the given test

forest = RandomForestClassifier(n_estimators=100,
                                criterion='gini',
                                max_depth=5,
                                min_samples_split=10,
                                min_samples_leaf=5,
                                random_state=0)
forest.fit(X_train, y_train)
print("Random Forest score: {0:.2}".format(forest.score(X_test, y_test)))
plt.bar(np.arange(len(predictors)), forest.feature_importances_)
plt.xticks(np.arange(len(predictors)), predictors, rotation='vertical')
predictors = ["Title", "Sex", "Fare", "Pclass", "Age", "FamilySize"]
clf = RandomForestClassifier(n_estimators=100,
                             criterion='gini',
                             max_depth=5,
                             min_samples_split=10,
                             min_samples_leaf=5,
                             random_state=0)
clf.fit(train[predictors], train["Survived"])
prediction = clf.predict(test[predictors])

submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": prediction})
submission.to_csv("submission.csv", index=False)






