# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import sys

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier

from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, GridSearchCV

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt 

from sklearn.neural_network import MLPClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

train_df = train.drop("Survived", axis=1) 

test_df = pd.read_csv("../input/test.csv")



print(train_df.columns)

print(test_df.columns)



#Combine the two datasets for data processing

comb = pd.concat([train_df,test_df])

comb = comb.reset_index()

comb.head()
#Extract Title from Name

comb["Title"] = comb.Name.str.extract('^.* ([A-Z][a-z]+)\..*')

#comb.Title.value_counts()

pd.crosstab(comb.Sex, comb.Title)

comb.Title.loc[(comb.Title.isin(['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir', 'Dr'])) & (comb.Sex=='male')]  = 'Mr'

comb.Title.loc[(comb.Title.isin(['Countess', 'Dona', 'Lady', 'Mlle', 'Mme', 'Dr'])) & (comb.Sex=='female')] = 'Mrs'

comb.Title.loc[comb.Title.isin(['Ms'])] = 'Miss'

print(comb.Title.value_counts())



# Check that all titles match with sex

print(comb[["Title", "Sex"]].groupby(["Title", "Sex"]).size())
#Look for NaN Embarked samples

print(comb[comb.Embarked.isnull()])
#Look for appropriate Embarked based on Fare(80) and Pclass(1), assumes Fare is highly correlated to class and embarked

data = comb[["Fare","Pclass","Embarked"]]

fig = data.boxplot(by=["Pclass","Embarked"], grid=True)

fig.set_yticks([80])

fig.set_ylim(0,100)
# Set to C because (PClass, Embarked) = (1,C) is closest to Fare

comb.Embarked.loc[comb.Embarked.isnull()] = 'C'

comb.info()
comb[comb.Fare.isnull()]
plt.figure(figsize=(9,2))

mean = comb.Fare[(comb.Pclass==3) & (comb.Embarked=='S')].mean()

med = comb.Fare[(comb.Pclass==3) & (comb.Embarked=='S')].median()

data=comb["Fare"][(comb.Pclass==3) & (comb.Embarked=='S')]

data.plot(kind="kde")

plt.axvline(med, color='red')

plt.axvline(mean, color='green')
#Set Fare to median of corresponding Pclass and Embarked

comb.Fare[comb.Fare.isnull()] = comb.Fare[(comb.Pclass==3) & (comb.Embarked=='S')].median()

comb.info()
#Create FamilySize from Parch and SibSp + 1 (incl. themselves)

comb["FamilySize"] = comb.Parch + comb.SibSp + 1



#Change categorical variables to factors

gender_dummies = pd.get_dummies(comb.Sex, drop_first=True)

embarked_dummies = pd.get_dummies(comb.Embarked)

class_dummies = pd.get_dummies(comb.Pclass)

comb = comb.join(gender_dummies)

comb = comb.join(embarked_dummies)

comb = comb.join(class_dummies)

comb = comb.drop(["Sex", "Embarked", "Pclass"], axis=1)

comb.info()
#Fill in Age NaN values by regressing Age ~ other predictors

age_model = DecisionTreeRegressor(min_samples_split=5, random_state=42)

test = comb

comb = comb.join(pd.get_dummies(comb.Title))



# Drop irrelevant columns

comb = comb.drop(["Title", "Name", "PassengerId", "Ticket", "Cabin", "SibSp", "Parch"], axis=1)

print (comb.info())

x_train = comb.drop("Age",axis=1)[comb.Age.notnull()]

x_test = comb.drop("Age",axis=1)[comb.Age.isnull()]

y_train = comb["Age"][comb.Age.notnull()]

age_model.fit(x_train,y_train)

print(age_model.score(x_train, y_train))

y_test = age_model.predict(x_test)



#Convert np.array to Series

y_test_series = pd.Series(y_test)



#Plot new ages compared to old ages to find any changes in the shape of the age distribution

age_reg = y_train.append(y_test_series)

age_reg.plot(kind="hist", alpha=0.55, bins=70, legend=True, label='Predicted')

y_train.plot(kind="hist", alpha=0.55, bins=70, legend=True, label='Known')



print(comb.info())
comb.Age[comb.Age.isnull()]=y_test

comb.info()
# first 891 records are used in training set



x = comb.loc[:890].reset_index(drop=True)

x_test = comb.loc[891:].reset_index(drop=True)

x = x.drop("index", axis=1)

x_test = x_test.drop("index", axis=1)

x.info()
y = train.Survived

x_train, x_cv, y_train, y_cv = train_test_split(x, y, test_size=0.25, random_state=0)
model = DecisionTreeClassifier(min_samples_split = 30)

model.fit(x_train,y_train)

print("Decision Tree")

print(model.score(x_train,y_train))

print(model.score(x_cv,y_cv))



model2 = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 1, min_samples_split = 30))

model2.fit(x_train,y_train)

print("AdaBoost")

print(model2.score(x_train,y_train))

print(model2.score(x_cv,y_cv))



model3 = ExtraTreesClassifier(n_estimators = 100, min_samples_split = 5)

model3.fit(x_train,y_train)

print("ExtraTrees")

print(model3.score(x_train,y_train))

print(model3.score(x_cv,y_cv))



model4 = LogisticRegression()

model4.fit(x_train,y_train)

print("LogisticRegression")

print(model4.score(x_train,y_train))

print(model4.score(x_cv,y_cv))





imp = pd.DataFrame({"Features":x_train.columns})

imp["DecTree"] = model.feature_importances_



print("----------------------------")



rf = RandomForestClassifier(min_samples_split =30, n_estimators=100)

rf.fit(x_train,y_train)

print("Random Forest")

print(rf.score(x_train,y_train))

print(rf.score(x_cv,y_cv))



print("----------------------------")



imp["RandForest"] = rf.feature_importances_

print(imp)
# Grid Search to tune algorithm parameters

params = {"min_samples_split": [2, 5, 10, 20, 30, 40, 50, 100],

         "n_estimators": [1, 20, 50, 100, 250] }

clf = GridSearchCV(rf, params, iid=False, cv=5)

clf.fit(x_train, y_train)

print("Best parameter (CV score=%0.3f):" % clf.best_score_)

print(clf.best_params_)



clf_extra = GridSearchCV(model3, params, iid=False, cv=5)

clf_extra.fit(x_train, y_train)

print("Extra Trees Search")

print("Best parameter (CV score=%0.3f):" % clf_extra.best_score_)

print(clf_extra.best_params_)



optim_rf = RandomForestClassifier(**clf.best_params_)

optim_rf.fit(x,y)



optim_extra = ExtraTreesClassifier(**clf_extra.best_params_)

optim_extra.fit(x,y)
# Ensembling models that have low correlation can yield better results



model.fit(x,y)

model2.fit(x,y)

model3.fit(x,y)

model4.fit(x,y)





results = pd.DataFrame({"model1": model.predict(x_test), "model2":model2.predict(x_test), "model3":model3.predict(x_test), "model4":model4.predict(x_test), "optim_rf":optim_rf.predict(x_test), "optim_extra":optim_extra.predict(x_test)})

pd.set_option('display.max_rows', results.shape[0])



results["final"] = results.mean(axis=1).round().astype(int)



print(results.corr())

print(results)



y_test = optim_rf.predict(x_test)

# submission = pd.DataFrame({"PassengerId":test_df.PassengerId, "Survived":y_test})

# submission.to_csv("titanic_submission.csv", index=False)

submission = pd.DataFrame({"PassengerId":test_df.PassengerId, "Survived":results["final"]})

submission.to_csv("titanic_submission.csv", index=False)

fig = imp.plot(kind="barh", alpha = 0.50)

fig.set_yticklabels(imp.Features)
#Plot a learning curve to find if the model is overfitting or underfitting

#Because the two curves are so close when m os large, the model has high bias (underfitting)

#A solution to this is to add more features to increase complexity



train_sizes, train_scores, cv_scores = learning_curve(rf, x, y, train_sizes=range(51,751,50), cv=5)

#data = pd.DataFrame({"train_sizes":train_sizes, "train_scores":train_scores, "cv_scores":cv_scores})

train_scores = 1-train_scores.mean(axis=1)

cv_scores = 1-cv_scores.mean(axis=1)

data = pd.DataFrame({"train_sizes":train_sizes})

data["train_scores"] = train_scores

data["cv_scores"] = cv_scores

fig = data.plot(x="train_sizes", kind="line")

fig.grid(True)