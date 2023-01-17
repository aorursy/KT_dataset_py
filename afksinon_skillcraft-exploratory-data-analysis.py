# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# scikit-learn for naive-bayes classifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# random forest classifier
from sklearn.ensemble import RandomForestClassifier

# support vector machine
from sklearn import svm
starcraft = pd.read_csv("../input/SkillCraft.csv")
starcraft.head()
# missing data check
starcraft.apply(lambda x: sum(x.isnull()), axis=0)
# dataset summary
starcraft.describe()
# Age by League
sns.set(style = "whitegrid", rc = {"figure.figsize":(11.7, 8.27)})
ax = sns.boxplot(x = "LeagueIndex", y = "Age", data = starcraft).set_title("Age by League")
# HoursPerWeek by League
ax = sns.boxplot(x = "LeagueIndex", y = "HoursPerWeek", data = starcraft).set_title("HoursPerWeek by League")
# TotalHours by League
ax = sns.boxplot(x = "LeagueIndex", y = "TotalHours", data = starcraft).set_title("TotalHours by League")
# APM per league
ax = sns.boxplot(x = "LeagueIndex", y = "APM", data = starcraft).set_title("APM by League")
# SelectByHotkeys per league
ax = sns.boxplot(x = "LeagueIndex", y = "SelectByHotkeys", data = starcraft).set_title("SelectByHotkeys per League")
# AssignToHotkeys per League
ax = sns.boxplot(x = "LeagueIndex", y = "AssignToHotkeys", data = starcraft).set_title("AssignToHotkeys by League")
# UniqueHotkeys by League
ax = sns.boxplot(x = "LeagueIndex", y = "UniqueHotkeys", data = starcraft).set_title("UniqueHotkeys by League")
# MinimapAttacks by League
ax = sns.boxplot(x = "LeagueIndex", y = "MinimapAttacks", data = starcraft).set_title("MinimapAttakcs by League")
# MinimapRightClicks by League
ax = sns.boxplot(x = "LeagueIndex", y = "MinimapRightClicks", data = starcraft).set_title("MinimapRightClicks by League")
# NumberOfPACs by League
ax = sns.boxplot(x = "LeagueIndex", y = "NumberOfPACs", data = starcraft).set_title("NumberOfPACs by League")
# GapBetweenPACs by League
ax = sns.boxplot(x = "LeagueIndex", y = "GapBetweenPACs", data = starcraft).set_title("GapBetweenPACs by League")
# ActionLatency by League
ax = sns.boxplot(x = "LeagueIndex", y = "ActionLatency", data = starcraft).set_title("ActionLatency by League")
# ActionsInPAC by League
ax = sns.boxplot(x = "LeagueIndex", y = "ActionsInPAC", data = starcraft).set_title("ActionsInPAC by League")
# TotalMapExplored by League
ax = sns.boxplot(x = "LeagueIndex", y = "TotalMapExplored", data = starcraft).set_title("TotalMapExplored by League")
# WorkersMade by League
ax = sns.boxplot(x = "LeagueIndex", y = "WorkersMade", data = starcraft).set_title("WorkersMade by League")
# UniqueUnitsMade by League
ax = sns.boxplot(x = "LeagueIndex", y = "UniqueUnitsMade", data = starcraft).set_title("UniqueUnitsMade by League")
# ComplexUnitsMade by League
ax = sns.boxplot(x = "LeagueIndex", y = "ComplexUnitsMade", data = starcraft).set_title("ComplexUnitsMade by League")
# ComplexAbilitiesUsed by League
ax = sns.boxplot(x = "LeagueIndex", y = "ComplexAbilitiesUsed", data = starcraft).set_title("ComplexAbilitiesUsed by League")
# feature cleanup
drops = ["GameID", "Age", "TotalHours", "UniqueUnitsMade", "ComplexUnitsMade", "ComplexAbilitiesUsed"]

starcraft.drop(drops, axis = 1, inplace = True)
starcraft.head()
# split into training and test sets
y = starcraft.LeagueIndex
X_train, X_test, y_train, y_test = train_test_split(starcraft, y, test_size = 0.2)

# remove the target from the training data
X_train.drop("LeagueIndex", axis = 1, inplace = True)
X_test.drop("LeagueIndex", axis = 1, inplace = True)

# easy to read statements
print("X_train: ", X_train.shape) 
print("y_train: ", y_train.shape)
print("X_test: ", X_test.shape) 
print("y_test: ", y_test.shape)
# fit the model
clf = MultinomialNB()

# train the model
model_1 = clf.fit(X_train, y_train)
# predictions
predictions_1 = clf.predict(X_test)

# view predictions
predictions_1[:10]
# compare results
print("Classification Accuracy: ", round(accuracy_score(y_test, predictions_1), 2))
# build the model
clf2 = RandomForestClassifier(n_estimators = 64, random_state = 123)

# fit the model
model_2 = clf2.fit(X_train, y_train)
# predictions
predictions_2 = clf2.predict(X_test)

# view them
predictions_2[:10]
# compare results
print("Classification Accuracy: ", round(accuracy_score(y_test, predictions_2), 2))
# build the model
clf3 = svm.SVC(gamma = 0.00001, decision_function_shape = "ovr")

# fit the model
model_3 = clf3.fit(X_train, y_train)
# predictions
predictions_3 = clf3.predict(X_test)

predictions_3[:10]
# compare results
print("Classification Accuracy: ", round(accuracy_score(y_test, predictions_3), 2))