# Importing important libraries for EDA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
sns.set_style('whitegrid')

from IPython.display import display
%matplotlib inline

# Ignore warnings

import warnings
warnings.filterwarnings('ignore')

rand = 7
# Import the datasets and have a peek at the data

train_df = pd.read_csv("./../input/train.csv")
test_df = pd.read_csv("./../input/test.csv")
passenger_ids = test_df["PassengerId"]

display(train_df.sample(n = 3))
display(test_df.sample(n = 3))
# Are there any missing values?, any variables that need to be converted to another type?

print("Training Data Info\n")
display(train_df.info())

print("Testing Data Info\n")
display(test_df.info())
train_df['Survived'].value_counts().plot(kind = 'pie', explode = [0, 0.1], figsize = (4, 4), autopct = '%1.1f%%', shadow = True)
plt.ylabel("Survival Percentage")
plt.legend(["Did not Survive", "Survived"])
plt.show()
survival_by_class = train_df.groupby("Pclass")["Survived"].mean()
display(survival_by_class)

f, ax = plt.subplots(1, 2, figsize = (10, 4))


sns.barplot(x = "Pclass", y = "Survived", data = train_df, ax = ax[0])
ax[0].set_ylabel("Survived Fraction")

sns.countplot(y = 'Pclass', hue='Survived', data = train_df, ax = ax[1])
ax[1].set_xlabel("Survived Count")
ax[1].set_ylabel("Pclass")

plt.show()
sns.barplot(x = "Pclass", y = "Fare", data = train_df)
plt.ylabel("Average Fare")
plt.show()
survival_by_sex = train_df.groupby("Sex")["Survived"].mean()
display(survival_by_sex)

f, ax = plt.subplots(1, 2, figsize = (10, 4))

sns.barplot(x = "Sex", y = "Survived", data = train_df, ax = ax[0])
ax[0].set_ylabel("Survived Fraction")

sns.countplot(y = "Sex", hue = "Survived", data = train_df, ax = ax[1])
ax[1].set_xlabel("Survived Count")

plt.show()
survival_by_port = train_df.groupby("Embarked")["Survived"].mean()
display(survival_by_port)

f, ax = plt.subplots(1, 2, figsize = (10, 4))

sns.barplot(x = "Embarked", y = "Survived", data = train_df, ax = ax[0])
ax[0].set_ylabel("Survived Fraction")

sns.countplot(y = "Embarked", hue = "Survived", data = train_df, ax = ax[1])
ax[1].set_ylabel("Survived Count")

plt.show()
pd.crosstab([train_df["Sex"], train_df["Survived"]], train_df["Pclass"], margins = True).style.background_gradient(cmap = 'summer_r')
sns.factorplot(x = "Pclass", y = "Survived", hue = "Sex", data = train_df)
plt.ylabel("Survived Fraction")
plt.show()
pd.crosstab([train_df["Pclass"], train_df["Survived"]],train_df["Embarked"], margins = True).style.background_gradient(cmap = 'summer_r')
f, ax = plt.subplots(1, 2, figsize = (10, 4))

sns.factorplot(x = "Embarked", y = "Survived", hue = "Pclass", data = train_df, ax = ax[0])
ax[0].set_ylabel("Survived Fraction")

sns.countplot(y = "Embarked", hue = "Pclass", data = train_df, ax = ax[1])
ax[1].set_xlabel("People Count")

plt.show()
sns.factorplot(x = 'Pclass', y = 'Survived', hue = 'Sex', col = 'Embarked', data = train_df)
plt.ylabel("Survived Fraction")

plt.show()
print(train_df.groupby("SibSp")["Survived"].mean())

f, ax = plt.subplots(1, 2, figsize = (10, 4))

sns.barplot(x = "SibSp", y = "Survived", data = train_df, ax = ax[0])
ax[0].set_ylabel("Survived Fraction")

sns.countplot(y = "SibSp", hue = "Survived", data = train_df, ax = ax[1])
ax[1].set_xlabel("People Count")

plt.show()
print(train_df.groupby("Parch")["Survived"].mean())

f, ax = plt.subplots(1, 2, figsize = (10, 4))

sns.barplot(x = "Parch", y = "Survived", data = train_df, ax = ax[0])
ax[0].set_ylabel("Survived Fraction")

sns.countplot(y = "Parch", hue = "Survived", data = train_df, ax = ax[1])
ax[1].set_xlabel("People Count")

plt.show()
import math

# Has a cabin ?

def hasCabin(x):
    return int((x["Cabin"] is not np.nan))

train_df["HasCabin"] = train_df.apply(hasCabin, axis = 1)
test_df["HasCabin"] = test_df.apply(hasCabin, axis = 1)

# Combine SibSp & Parch to create a new variable FamilySize

train_df["FamilySize"] = train_df["SibSp"] + train_df["Parch"] + 1
test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1

# Use family size to create a new feature isAlone.

def isAlone(x):
    familySize = x["FamilySize"]
    return int(familySize == 1)

train_df["IsAlone"] = train_df.apply(isAlone, axis = 1)
test_df["IsAlone"] = test_df.apply(isAlone, axis = 1)

display(train_df.sample(n = 3))
f, ax = plt.subplots(1, 2, figsize = (10, 4))

sns.barplot(x = "HasCabin", y = "Survived", data = train_df, ax = ax[0])
ax[0].set_ylabel("Survived Fraction")

sns.countplot(y = "HasCabin", hue = "Survived", data = train_df, ax = ax[1])
ax[1].set_ylabel("People Count")

plt.show()
sns.barplot(x = "Pclass", y = "HasCabin", data = train_df)
plt.ylabel("Fraction of people who had cabins")
plt.show()
sns.barplot(x = "IsAlone", y = "Survived", data = train_df)
plt.ylabel("Survived Fraction")
plt.show()
import re

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""    
    
train_df["Title"] = train_df["Name"].apply(get_title)
test_df["Title"] = test_df["Name"].apply(get_title)

train_df['Title'] = train_df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
test_df['Title'] = test_df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

train_df['Title'] = train_df['Title'].replace('Mlle', 'Miss')
train_df['Title'] = train_df['Title'].replace('Ms', 'Miss')
train_df['Title'] = train_df['Title'].replace('Mme', 'Mrs')

test_df['Title'] = test_df['Title'].replace('Mlle', 'Miss')
test_df['Title'] = test_df['Title'].replace('Ms', 'Miss')
test_df['Title'] = test_df['Title'].replace('Mme', 'Mrs')
sns.barplot(x = "Title", y = "Survived", data = train_df)
plt.ylabel("Survived Fraction")
plt.show()
train_df["Age"] = train_df.groupby("Title")["Age"].transform(lambda x: x.fillna(x.mean()))
test_df["Age"] = test_df.groupby("Title")["Age"].transform(lambda x: x.fillna(x.mean()))
f,ax=plt.subplots(1,2,figsize=(15, 7))

sns.violinplot("Pclass", "Age", hue = "Survived", data = train_df, split = True, ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,110,10))

sns.violinplot("Sex", "Age", hue="Survived", data = train_df, split=True, ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,110,10))
plt.show()
bins = [0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

train_df['AgeGroup'] = pd.cut(train_df["Age"], bins, labels = labels)
test_df['AgeGroup'] = pd.cut(test_df["Age"], bins, labels = labels)

sns.barplot(x = "AgeGroup", y = "Survived", data = train_df)
train_df[train_df["Embarked"].isnull()]
# Since the ticket is first class and the persons survived, lets fill it with C
train_df["Embarked"] = train_df["Embarked"].fillna("C")
test_df["Fare"] = test_df.groupby("Pclass")["Fare"].transform(lambda x: x.fillna(x.mean()))
train_df['FareRange'] = pd.qcut(train_df['Fare'], 4)

train_df['Farecat'] = 0
train_df.loc[train_df['Fare'] <= 7.91,'Farecat'] = 0
train_df.loc[(train_df['Fare'] > 7.91) & (train_df['Fare']<=14.454),'Farecat'] = 1
train_df.loc[(train_df['Fare'] > 14.454) & (train_df['Fare']<=31),'Farecat'] = 2
train_df.loc[(train_df['Fare'] > 31) & (train_df['Fare']<=513),'Farecat'] = 3


test_df['FareRange'] = pd.qcut(test_df['Fare'], 4)

test_df['Farecat'] = 0
test_df.loc[test_df['Fare'] <= 7.91,'Farecat'] = 0
test_df.loc[(test_df['Fare'] > 7.91) & (test_df['Fare']<=14.454),'Farecat'] = 1
test_df.loc[(test_df['Fare'] > 14.454) & (test_df['Fare']<=31),'Farecat'] = 2
test_df.loc[(test_df['Fare'] > 31) & (test_df['Fare']<=513),'Farecat'] = 3
train_df = train_df.drop(labels=["PassengerId", "Name", "Ticket", "Cabin", "Fare", "FareRange", "Age"], axis=1)
test_df = test_df.drop(labels=["PassengerId", "Name", "Ticket", "Cabin", "Fare", "FareRange", "Age"], axis=1)
# Lets also solidify our findings using the correlation matrix
corr = train_df.corr()
sns.heatmap(corr, annot = True, cmap = 'RdYlGn', linewidths = 0.2)
fig = plt.gcf()
fig.set_size_inches(12, 12)
plt.show()
train_df = pd.get_dummies(train_df)
test_df = pd.get_dummies(test_df)
display(train_df.sample(n = 5))
# Split into features and target variable
features = train_df.iloc[:, 1:]
target = train_df.iloc[:, 0]
# Create a simple decision tree to see important features

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(min_samples_split = 10)
classifier.fit(features, target)

pd.Series(classifier.feature_importances_, features.columns).sort_values(ascending = True).plot.barh(width = 0.6)
fig = plt.gcf()
fig.set_size_inches(12, 12)
plt.show()
# split data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.2, stratify = target, random_state = rand)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
modelResults = pd.DataFrame(columns = ['Model_Name', 'Model', 'Params', 'Test_Score', 'CV_Mean', 'CV_STD'])

def save(grid, modelName, calFI):
    global modelResults
    cv_scores = cross_val_score(grid.best_estimator_, X_train, y_train, cv = 10, scoring = 'accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    test_score = grid.score(X_test, y_test)
    
    print("Best model parameter are\n", grid.best_estimator_)
    print("Saving model {}\n".format(modelName))
    print("Mean Cross validation score is {} with a Standard deviation of {}\n".format(cv_mean, cv_std))
    print("Test Score for the model is {}\n".format(test_score))
    
    if calFI:
        pd.Series(grid.best_estimator_.feature_importances_, features.columns).sort_values(ascending = True).plot.barh(width = 0.6)
        fig = plt.gcf()
        fig.set_size_inches(12, 12)
        plt.title("{} Feature Importance".format(modelName))
        plt.show()
    
    
    cm = confusion_matrix(y_test, grid.best_estimator_.predict(X_test))
    
    cm_df = pd.DataFrame(cm, index = ["Not Survived", "Survived"], columns = ["Not Survived", "Survived"])
    sns.heatmap(cm_df, annot = True)
    plt.show()
        
    
    modelResults = modelResults.append({'Model_Name' : modelName, 'Model' : grid.best_estimator_, 'Params' : grid.best_params_, 'Test_Score' : test_score, 'CV_Mean' : cv_mean, 'CV_STD' : cv_std}
                                       , ignore_index=True)

def norm_save(model, modelName):
    global modelResults
    cv_scores = cross_val_score(model, X_train, y_train, cv = 10, scoring = 'accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    y_pred = model.predict(X_test)
    test_score = accuracy_score(y_test, y_pred)
    
    print("Saving model {}\n".format(modelName))
    print("Mean Cross validation score is {} with a Standard deviation of {}\n".format(cv_mean, cv_std))
    print("Test Score for the model is {}\n".format(test_score))
    
    cm = confusion_matrix(y_test, y_pred)
    
    cm_df = pd.DataFrame(cm, index = ["Not Survived", "Survived"], columns = ["Not Survived", "Survived"])
    sns.heatmap(cm_df, annot = True)
    plt.show()
        
    
    modelResults = modelResults.append({'Model_Name' : modelName, 'Model' : model, 'Params' : None, 'Test_Score' : test_score, 'CV_Mean' : cv_mean, 'CV_STD' : cv_std}
                                       , ignore_index=True)
def doGridSearch(classifier, params):
    cv = StratifiedShuffleSplit(n_splits = 10, test_size = 0.2, random_state = rand)
    score_fn = make_scorer(accuracy_score)
    grid = GridSearchCV(classifier, params, scoring = score_fn, cv = cv)
    grid = grid.fit(X_train, y_train)
    
    return grid
from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier()
params = {"n_neighbors" : np.arange(5, 21, 1),
         "weights" : ["uniform", "distance"]}

grid = doGridSearch(KNN, params)

save(grid, "K-Nearest Neighbor", False)
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state = rand)

params = {"min_samples_split" : np.arange(5, 20, 1),
         "max_features" : np.arange(3, 25, 1)}

grid = doGridSearch(tree, params)
save(grid, "Decision Tree", False)
from sklearn.ensemble import AdaBoostClassifier

adaBoostModel = AdaBoostClassifier(random_state = rand)
params = {"n_estimators" : [50, 75, 100, 125, 150, 200],
         "learning_rate" : [0.5, 0.75, 1, 1.25, 1.5]}

grid = doGridSearch(adaBoostModel, params)
save(grid, 'ADABoost', True)
from sklearn.ensemble import RandomForestClassifier

randomForestModel = RandomForestClassifier(random_state = rand)
params = {"n_estimators" : [50, 75, 100, 125, 150, 200],
         "max_features" : [3, 4, 5, 6, 7, 8],
         "min_samples_split" : [2, 4, 6, 8, 10]}

grid = doGridSearch(randomForestModel, params)
save(grid, 'RandomForest', True)
from sklearn.svm import SVC

svc = SVC(random_state = rand)
params = {"C" : [0.1, 1, 1.1, 1.2], "gamma" : [0.01, 0.02, 0.03, 0.04, 0.08, 0.1, 1], 
          "kernel" : ["linear", "poly", "rbf", "sigmoid"]}

grid = doGridSearch(svc, params)
save(grid, 'SVC', False)
from sklearn.ensemble import GradientBoostingClassifier

gradientModel = GradientBoostingClassifier(random_state = 0)
params = {"learning_rate" : [0.03, 0.035, 0.04, 0.45], 
          "n_estimators" : [90, 100, 110], 
          "max_depth" : [2, 3],
          "min_samples_split" : [7, 8, 9]}

grid = doGridSearch(gradientModel, params)
save(grid, "GradientBoost", True)
from sklearn.linear_model import LogisticRegression

logisticModel = LogisticRegression(random_state = 0)
params = {}

grid = doGridSearch(logisticModel, params)
save(grid, 'LogisticRegression', False)
from sklearn.naive_bayes import GaussianNB

naiveModel = GaussianNB()
params = {}

grid = doGridSearch(naiveModel, params)
save(grid, 'NaiveBayes', False)
from sklearn.ensemble import VotingClassifier
votingClassifier = VotingClassifier(estimators = [(modelResults.loc[2]["Model_Name"], modelResults.loc[2]["Model"]),
                                                  (modelResults.loc[4]["Model_Name"], modelResults.loc[4]["Model"]), 
                                                  (modelResults.loc[5]["Model_Name"], modelResults.loc[5]["Model"])], voting = 'hard')

votingClassifier.fit(X_train, y_train)
norm_save(votingClassifier, "Voting-Top-3")
from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(base_estimator = modelResults.loc[4]["Model"], n_estimators = 10)
bag.fit(X_train, y_train)

norm_save(bag, "Bagged KNN")
sns.barplot(x = "Model_Name", y = "Test_Score", data = modelResults.sort_values(by = "Test_Score", ascending = False))
fig = plt.gcf()
fig.set_size_inches(18, 10)
plt.ylabel("Test Score")
plt.xlabel("Model")
plt.show()
display(modelResults)
# Submission File
# get the adaboost model
submissionModel = modelResults.loc[2]["Model"]
submissions = submissionModel.predict(test_df)
submissions = pd.Series(submissions, name="Survived")

submission = pd.concat([passenger_ids, submissions],axis = 1)

submission.to_csv("titanic.csv",index=False)



print("Done")
