# Loading libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib import cm as cm
import string
import seaborn as sns
import os
# Reading train.csv and taking a look!
print(os.listdir("../input"))
train_ori = pd.read_csv("../input/train.csv")
train_ori.head(5)
# Read test.csv
test_ori = pd.read_csv("../input/test.csv")
test_ori.head(5)
train_ori.info()
#We will check NA values in training set
print(train_ori.isnull().sum())
test_ori.info()
#We will check NA values in test set
print(test_ori.isnull().sum())
train_ori.Survived.value_counts(normalize=True)
#The number of passengers for each Pclass are:  
sns.countplot(x="Pclass", data=train_ori)
# The percentage of passengers survived for each Pclass:
sns.catplot(y="Survived", col="Pclass", data=train_ori, kind="bar", ci=None, aspect=.5)
#sns.catplot(x="Sex", y="Survived", col="Pclass", data=train_ori, kind="bar", ci=None, aspect=.8)
train_ori["Title"]=(train_ori["Name"].str.split(', ').str[1]).str.split('. ').str[0]
train_ori.head(5)
# The initial different values for Title column are:
print(np.unique(np.array(train_ori["Title"])))
# Factorize Title column as ["Mr","Mrs","Miss"]
# We will factorize those values for getting only 3 values: Man, married Woman or unmarried Woman
train_ori.loc[(train_ori["Title"] == "Capt") | (train_ori["Title"] == "Col") | (train_ori["Title"] == "Don") | 
              (train_ori["Title"] == "Jonkheer") 
    | (train_ori["Title"] == "Major") | (train_ori["Title"] == "Master") | (train_ori["Title"] == "Mr") |
              (train_ori["Title"] == "Rev") | (train_ori["Title"] == "Sir")
    | (train_ori["Title"] == "th"),"Title"]= "Mr"
train_ori.loc[(train_ori["Title"] == "Lady") | (train_ori["Title"] == "Mme"),"Title"] = "Mrs"
train_ori.loc[(train_ori["Title"] == "Mlle") | (train_ori["Title"] == "Ms") ,"Title"] = "Miss"
train_ori.loc[(train_ori["Title"] == "Dr") & (train_ori["Sex"] == "female") ,"Title"] = "Mrs"
train_ori.loc[(train_ori["Title"] == "Dr") & (train_ori["Sex"] == "male") ,"Title"] = "Mr"

unique_elements, counts_elements = np.unique(np.array(train_ori["Title"]), return_counts=True)
print(unique_elements, counts_elements)
#The number of passengers for each Title are:  
sns.countplot(x="Title", data=train_ori)
# The percentage of passengers survived for each Title:
sns.catplot(y="Survived", col="Title", data=train_ori, kind="bar", ci=None, aspect=.5)
#sns.catplot(x="Sex", y="Survived", col="Title", data=train_ori, kind="bar", ci=None, aspect=.8)
#The number of passengers for each Sex are:  
sns.countplot(x="Sex", data=train_ori)
# The percentage of passengers survived for each Sex:
sns.catplot(y="Survived", col="Sex", data=train_ori, kind="bar", ci=None, aspect=.5)
#sns.catplot(x="Sex", y="Survived", col="Title", data=train_ori, kind="bar", ci=None, aspect=.8)
# We calculate the median age per Title column and we will complete with these ages
mrage=train_ori[train_ori["Title"] == "Mr"]["Age"].mean()
mrsage=train_ori[train_ori["Title"] == "Mrs"]["Age"].mean()
missage=train_ori[train_ori["Title"] == "Miss"]["Age"].mean()
print("mean age for Mr: ",mrage)
print("mean age for Mrs: ",mrsage)
print("mean age for Miss: ",missage)

train_ori.loc[train_ori["Title"] == "Mr","Age"] = train_ori.loc[train_ori["Title"] == "Mr","Age"].fillna(mrage)
train_ori.loc[train_ori["Title"] == "Mrs","Age"] = train_ori.loc[train_ori["Title"] == "Mrs","Age"].fillna(mrsage)
train_ori.loc[train_ori["Title"] == "Miss","Age"] = train_ori.loc[train_ori["Title"] == "Miss","Age"].fillna(missage)
train_ori.loc[(train_ori["Age"] >= -0.001) & (train_ori["Age"] < 15),"Age"] = 0
train_ori.loc[(train_ori["Age"] >= 15) & (train_ori["Age"] < 18),"Age"] = 1
train_ori.loc[(train_ori["Age"] >= 18) ,"Age"] = 2
#The number of passengers for each Age are:  
sns.countplot(x="Age", data=train_ori)
# The percentage of passengers survived for each Age:
sns.catplot(y="Survived", col="Age", data=train_ori, kind="bar", ci=None, aspect=.5)
#sns.catplot(x="Sex", y="Survived", col="Title", data=train_ori, kind="bar", ci=None, aspect=.8)
train_ori["FamilySize"]= train_ori["SibSp"] + train_ori["Parch"] + 1
#The number of passengers for each FamilySize are:  
sns.countplot(x="FamilySize", data=train_ori)
# The percentage of passengers survived for each FamilySize:
sns.catplot(y="Survived", col="FamilySize", data=train_ori, kind="bar", ci=None, aspect=.8)
#sns.catplot(x="Sex", y="Survived", col="Title", data=train_ori, kind="bar", ci=None, aspect=.8)
print("#of differents Fares:",len(train_ori["Fare"].unique()))
print("Max Fare:",train_ori["Fare"].max())
print("Min Fare:",train_ori["Fare"].min())
print("Mean Fare:",train_ori["Fare"].mean())
sns.distplot(train_ori['Fare'],kde=False)
plt.scatter(train_ori['Fare'], train_ori['FamilySize'],c=train_ori['Survived'])
train_ori['CabinAssigned'] = np.where(train_ori.Cabin.isnull(), 0, 1)
train_ori.head(2)
sns.countplot(x="CabinAssigned", data=train_ori)
# The percentage of passengers survived for each CabinAssigned:
sns.catplot(y="Survived", col="CabinAssigned", data=train_ori, kind="bar", ci=None, aspect=.5)
#sns.catplot(x="Sex", y="Survived", col="CabinAssigned", data=train_ori, kind="bar", ci=None, aspect=.8)
# For Embarked column NA we will apply logic for avoid the NA value.
train_ori[train_ori["Embarked"].isnull()]
train_ori.loc[(train_ori["Age"] == 2.0) & (train_ori["Sex"] == 'female') & (train_ori["FamilySize"] == 1) & (train_ori["Pclass"] == 1)& (train_ori["CabinAssigned"] == 1) & (train_ori["Fare"] >= 75) & (train_ori["Fare"] <= 85)]
train_ori = train_ori[pd.notnull(train_ori['Embarked'])]
sns.countplot(x="Embarked", data=train_ori)
# The percentage of passengers survived for each Embarked:
sns.catplot(y="Survived", col="Embarked", data=train_ori, kind="bar", ci=None, aspect=.5)
#sns.catplot(x="Sex", y="Survived", col="Embarked", data=train_ori, kind="bar", ci=None, aspect=.8)
train = train_ori.drop(['Name','PassengerId','Ticket','SibSp','Cabin','Parch'],axis=1)
train.head(10)
# Factorize columns 
print(train.head(10))
train["Embarked"], uniques = pd.factorize(train["Embarked"])
train["Sex"], uniques = pd.factorize(train["Sex"])
train["Title"], uniques = pd.factorize(train["Title"])
# Get intervals for factoring Fares column
pd.qcut(train["Fare"], 4).value_counts().sort_index()
train.loc[(train["Fare"] >= -0.001) & (train["Fare"] < 7.896),"Fare"] = 0
train.loc[(train["Fare"] >= 7.896) & (train["Fare"] < 14.454),"Fare"] = 1
train.loc[(train["Fare"] >= 14.454) & (train["Fare"] < 31.0),"Fare"] = 2
train.loc[(train["Fare"] >= 31.0) ,"Fare"] = 3

train['Fare'] = train['Fare'].astype(int)
train.head(10)
corr = train.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
# cmap=cmap,
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
train.corr()["Survived"]
train.head(4)
# extract most important features and target for cross validation
features = train.drop(('Survived'), axis=1)
target = train['Survived'].values
from sklearn import model_selection, ensemble, svm
import xgboost as xgb

# initialise classifiers
rf_clf = ensemble.RandomForestClassifier(n_estimators=100, random_state=0)
et_clf = ensemble.ExtraTreesClassifier(n_estimators=100, random_state=0)
gb_clf = ensemble.GradientBoostingClassifier(n_estimators=100, random_state=0)
ada_clf = ensemble.AdaBoostClassifier(n_estimators=100, random_state=0)
svm_clf = svm.LinearSVC(C=0.1,random_state=0)
xgb_clf = xgb.XGBClassifier(n_estimators=100)

e_clf = ensemble.VotingClassifier(estimators=[('xgb', xgb_clf), ('rf',rf_clf),
                                              ('et',et_clf), ('gbc',gb_clf), ('ada',ada_clf), ('svm',svm_clf)])

# score using cross validation
clf_list = [xgb_clf, rf_clf, et_clf, gb_clf, ada_clf, svm_clf, e_clf]
name_list = ['XGBoost', 'Random Forest', 'Extra Trees', 'Gradient Boosted', 'AdaBoost', 'Support Vector Machine', 'Ensemble']

for clf, name in zip(clf_list,name_list) :
    scores = model_selection.cross_val_score(clf, features, target, cv=10)
    print("Accuracy: %0.2f +/- %0.2f (%s 95%% CI)" % (scores.mean(), scores.std()*2, name))
# fit ensemble classifier
svm_clf = svm_clf.fit(features,target)
# Process test dataset
# Create Title column
test_ori.head(5)
test_ori["Title"]=(test_ori["Name"].str.split(', ').str[1]).str.split('. ').str[0]

test_ori.loc[(test_ori["Title"] == "Capt") | (test_ori["Title"] == "Col") | (test_ori["Title"] == "Don") | 
              (test_ori["Title"] == "Jonkheer") 
    | (test_ori["Title"] == "Major") | (test_ori["Title"] == "Master") | (test_ori["Title"] == "Mr") |
              (test_ori["Title"] == "Rev") | (test_ori["Title"] == "Sir")
    | (test_ori["Title"] == "th"),"Title"]= "Mr"
test_ori.loc[(test_ori["Title"] == "Lady") | (test_ori["Title"] == "Mme"),"Title"] = "Mrs"
test_ori.loc[(test_ori["Title"] == "Mlle") | (test_ori["Title"] == "Ms") ,"Title"] = "Miss"
test_ori.loc[(test_ori["Title"] == "Dr") & (test_ori["Sex"] == "female") ,"Title"] = "Mrs"
test_ori.loc[(test_ori["Title"] == "Dr") & (test_ori["Sex"] == "male") ,"Title"] = "Mr"

test_ori["FamilySize"]= test_ori["SibSp"] + test_ori["Parch"]
test_ori['CabinAssigned'] = np.where(test_ori.Cabin.isnull(), 0, 1)
test_ori = test_ori[pd.notnull(test_ori['Embarked'])]


test = test_ori.drop(['Name','PassengerId','Ticket','Cabin','SibSp','Parch'],axis=1)

print(test.isnull().sum())

# We calculate the mean age per Title column and we will complete with these ages
mrage=test[test["Title"] == "Mr"]["Age"].mean()
mrsage=test[test["Title"] == "Mrs"]["Age"].mean()
missage=test[test["Title"] == "Miss"]["Age"].mean()
print("median age for Mr: ",mrage)
print("median age for Mrs: ",mrsage)
print("median age for Miss: ",missage)

test.loc[test["Title"] == "Mr","Age"] = test.loc[test["Title"] == "Mr","Age"].fillna(mrage)
test.loc[test["Title"] == "Mrs","Age"] = test.loc[test["Title"] == "Mrs","Age"].fillna(mrsage)
test.loc[test["Title"] == "Miss","Age"] = test.loc[test["Title"] == "Miss","Age"].fillna(missage)

test["Fare"] = test["Fare"].fillna(0)

print(test.isnull().sum())

test["Embarked"], uniques = pd.factorize(test["Embarked"])
test["Sex"], uniques = pd.factorize(test["Sex"])
test["Title"], uniques = pd.factorize(test["Title"])
test.loc[(test["Fare"] >= -0.001) & (test["Fare"] < 7.896),"Fare"] = 0
test.loc[(test["Fare"] >= 7.896) & (test["Fare"] < 14.454),"Fare"] = 1
test.loc[(test["Fare"] >= 14.454) & (test["Fare"] < 31.0),"Fare"] = 2
test.loc[(test["Fare"] >= 31.0) ,"Fare"] = 3

test['Fare'] = test['Fare'].astype(int)

test.loc[(test["Age"] >= -0.001) & (test["Age"] < 15),"Age"] = 0
test.loc[(test["Age"] >= 15) & (test["Age"] < 18),"Age"] = 1
test.loc[(test["Age"] >= 18) ,"Age"] = 2

test['Age'] = test['Age'].astype(int)
# make prediction
prediction = svm_clf.predict(test)
print("From the",prediction.size, "passengers, we have found", prediction.sum(),"survivals")
psgid = np.array(range(892,1310)).astype(int)
output = pd.DataFrame(prediction, index=psgid, columns = ['Survived'])
output.to_csv('submission.csv', index_label = 'PassengerId')