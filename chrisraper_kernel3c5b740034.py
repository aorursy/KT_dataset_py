import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np



#import data

train_df = pd.read_csv('/kaggle/input/titanic/train.csv', index_col=0)

test_df = pd.read_csv('/kaggle/input/titanic/test.csv', index_col=0)
train_df.head()
train_df.info()
#converting objects data types into categorical types

train_df["Pclass"] = pd.Categorical(train_df["Pclass"], categories=[1,2,3], ordered=True)

train_df["Sex"] = pd.Categorical(train_df["Sex"], categories=["male", "female"], ordered=False)

train_df["Embarked"] = pd.Categorical(train_df["Embarked"], categories=["C", "Q", "S"], ordered=False)

test_df["Pclass"] = pd.Categorical(test_df["Pclass"], categories=[1,2,3], ordered=True)

test_df["Sex"] = pd.Categorical(test_df["Sex"], categories=["male", "female"], ordered=False)

test_df["Embarked"] = pd.Categorical(test_df["Embarked"], categories=["C", "Q", "S"], ordered=False)

train_df.info()
train_df.Name.head(50)
#extract beginning of string until comma is found

train_df["LastName"] = train_df.Name.str.extract("([^,]+)", expand=True)

test_df["LastName"] = test_df.Name.str.extract("([^,]+)", expand=True)



#extract beginning after the comma and ending at the period

train_df["Title"] = train_df.Name.str.extract(r", (.*?)\.", expand=True)

train_df["Title"] = pd.Categorical(train_df["Title"],

                                    categories=train_df.Title.unique(),

                                    ordered=False)

test_df["Title"] = test_df.Name.str.extract(r", (.*?)\.", expand=True)

test_df["Title"] = pd.Categorical(test_df["Title"],

                                    categories=test_df.Title.unique(),

                                    ordered=False)



#extract all text following the title word

train_df["FirstName"] = train_df.Name.str.extract(r"\. (.*?)$", expand=True)

test_df["FirstName"] = test_df.Name.str.extract(r"\. (.*?)$", expand=True)



#drop the original Name field since its information is now fully duplicated by the new fields.

train_df = train_df.drop("Name", axis=1)

test_df = test_df.drop("Name", axis=1)



train_df.loc[:,["LastName", "Title", "FirstName"]].head(50)
sns.set()

sns.catplot(x="Title", kind="count", data=train_df, aspect=2)

plt.xlabel("Titles")

plt.ylabel("Count")

plt.title("Passenger Name Title Histogram")

plt.xticks(rotation=30, ha='right')

plt.show()
train_df["AgeNotExact"] = np.where((train_df["Age"] > 1) & (round(train_df["Age"]) != train_df["Age"]), True, False)

train_df["AgeNotExact"] = pd.Categorical(train_df["AgeNotExact"], categories=[False, True], ordered=False)

test_df["AgeNotExact"] = np.where((test_df["Age"] > 1) & (round(test_df["Age"]) != test_df["Age"]), True, False)

test_df["AgeNotExact"] = pd.Categorical(test_df["AgeNotExact"], categories=[False, True], ordered=False)



sns.catplot(x='AgeNotExact', y='Age', data=train_df, kind='swarm', orient='v', aspect=2)

plt.show()
train_df = train_df.drop(["Ticket", "Cabin", "FirstName", "LastName"], axis=1)

test_df = test_df.drop(["Ticket", "Cabin", "FirstName", "LastName"], axis=1)

train_df.info()
from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder



#create a pipeline for transforming numeric variables

num_pipeline = Pipeline([

    ('imputer', SimpleImputer(strategy='median')),

    ('std_scaler', StandardScaler())

])



cat_pipeline = Pipeline([

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('cat_encode', OneHotEncoder())

])



#create lists which define the numeric and categorical variables

num_attr = ["Age", "SibSp", "Parch", "Fare"]

cat_attr = ["Pclass", "Sex", "Embarked", "AgeNotExact"]



full_pipeline = ColumnTransformer([

    ("num", num_pipeline, num_attr),

    ("cat", cat_pipeline, cat_attr)

])



y_train = train_df["Survived"]

prepped_X_train = full_pipeline.fit_transform(train_df.drop("Survived", axis=1))

prepped_X_test = full_pipeline.fit_transform(test_df)
pclass_agg = train_df.groupby("Pclass")["Survived"].agg("mean")

pclass_agg.plot(x=pclass_agg.index, y="Survived")

plt.xlabel("Passenger Class")

plt.ylabel("Survival Rate")

plt.ylim(bottom=0)

plt.show()
title_filter_agg = train_df.groupby("Title")["Survived"].agg("count")

title_filter = title_filter_agg > 5 #acts as a boolean filter to drop groups with small samples

title_agg = train_df.groupby("Title")["Survived"].agg("mean")

title_agg = title_agg[title_filter]

title_agg.plot.bar(x=title_agg.index, y="Survived")

plt.xlabel("Passenger Name Title")

plt.xticks(rotation=30, ha='right')

plt.ylabel("Survival Rate")

plt.ylim(bottom=0)

plt.show()
sex_agg = train_df.groupby("Sex")["Survived"].agg("mean")

sex_agg.plot.bar(x=sex_agg.index, y="Survived")

plt.xlabel("Passenger Sex")

plt.xticks(rotation=0, ha='center')

plt.ylabel("Survival Rate")

plt.ylim(bottom=0)

plt.show()
age_filter_agg = train_df.groupby("Age")["Survived"].agg("count")

age_filter = age_filter_agg > 5 #acts as a boolean filter to drop groups with small samples

age_agg = train_df.groupby("Age")["Survived"].agg("mean")

age_agg = age_agg[age_filter]

age_agg.plot(x=age_agg.index, y="Survived")

plt.xlabel("Passenger Age")

plt.ylabel("Survival Rate")

plt.ylim(bottom=0)

plt.show()
sibs_filter_agg = train_df.groupby("SibSp")["Survived"].agg("count")

sibs_filter = sibs_filter_agg > 5 #acts as a boolean filter to drop groups with small samples

sibs_agg = train_df.groupby("SibSp")["Survived"].agg("mean")

sibs_agg = sibs_agg[sibs_filter]

sibs_agg.plot(x=sibs_agg.index, y="Survived")

plt.xlabel("Passenger Siblings Count")

plt.ylabel("Survival Rate")

plt.ylim(bottom=0)

plt.show()
parch_filter_agg = train_df.groupby("Parch")["Survived"].agg("count")

parch_filter = parch_filter_agg > 5 #acts as a boolean filter to drop groups with small samples

parch_agg = train_df.groupby("Parch")["Survived"].agg("mean")

parch_agg = parch_agg[parch_filter]

parch_agg.plot(x=parch_agg.index, y="Survived")

plt.xlabel("Passenger Parent Count")

plt.xticks(ticks=[0,1,2])

plt.ylabel("Survival Rate")

plt.ylim(bottom=0)

plt.show()
train_df["RndFare"] = round(train_df["Fare"], -1)

fare_filter_agg = train_df.groupby("RndFare")["Survived"].agg("count")

fare_filter = fare_filter_agg > 5 #acts as a boolean filter to drop groups with small samples

fare_agg = train_df.groupby("RndFare")["Survived"].agg("mean")

fare_agg = fare_agg[fare_filter]

fare_agg.plot(x=fare_agg.index, y="Survived")

plt.xlabel("Passenger Fare (Aggregated by Rounding to Nearest 10 Unit)")

plt.ylabel("Survival Rate")

plt.ylim(bottom=0)

plt.show()
survived_agg = train_df.groupby("Survived").agg("count")/len(train_df)

survived_agg = survived_agg.iloc[:,1]

print(survived_agg.head())
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.model_selection import cross_validate

from graphviz import Source

from IPython.display import SVG



survived_agg = train_df.groupby("Survived").agg("count")/len(train_df)

survived_agg = survived_agg.iloc[:,1]

print(survived_agg.head())



tree_clf = DecisionTreeClassifier(max_depth=3, criterion="entropy", presort=True, random_state=123)

tree_clf.fit(prepped_X_train, y_train)



#cross validate model

k = 5

dt_scores = cross_validate(tree_clf, prepped_X_train, y_train, scoring=["accuracy","roc_auc"], cv=k)

acc_mean = dt_scores["test_accuracy"].mean()

acc_rng = dt_scores["test_accuracy"].max() - dt_scores["test_accuracy"].min()

auc_mean = dt_scores["test_roc_auc"].mean()

auc_rng = dt_scores["test_roc_auc"].max() - dt_scores["test_roc_auc"].min()

print(str().join([str(k), "-fold Accuracy Score Mean:"]), round(acc_mean, 5), "Range:", round(acc_rng, 5), "\n")

print(str().join([str(k), "-fold ROC AUC Score Mean:"]), round(auc_mean, 5), "Range:", round(auc_rng, 5), "\n")



graph = Source(export_graphviz(tree_clf, out_file=None))

SVG(graph.pipe(format='svg'))
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import roc_auc_score



rf_param_grid = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],

    'max_features': ['auto', 'sqrt'],

    'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],

    'min_samples_split': [2, 5, 10],

    'min_samples_leaf': [1, 2, 4],

    'bootstrap': [True, False]

}



#rf_clf = RandomForestClassifier(max_leaf_nodes=16, 

#                n_estimators=500, bootstrap=True, 

#                n_jobs=-1, oob_score=True, random_state=123)

rf_clf = RandomForestClassifier(random_state=123)

rf_rand = RandomizedSearchCV(rf_clf, param_distributions=rf_param_grid, n_iter=100, cv = k, verbose=2, random_state=123, n_jobs = -1, scoring=["accuracy","roc_auc"], refit="roc_auc")



rf_rand.fit(prepped_X_train, y_train)

acc_mean = rf_rand.cv_results_["mean_test_accuracy"].max()

auc_mean = rf_rand.cv_results_["mean_test_roc_auc"].max()

print(str().join([str(k), "-fold Accuracy Score Mean:"]), round(acc_mean, 5))

print(str().join([str(k), "-fold ROC AUC Score Mean:"]), round(auc_mean, 5))

y_test_pred_df = pd.DataFrame(rf_rand.predict(prepped_X_test))

y_test_pred_df.columns = ["Survived"]

test_pred_df = test_df.reset_index().join(y_test_pred_df)

submit_df = test_pred_df.loc[:,["PassengerId", "Survived"]]

submit_df.to_csv('submit.csv', index=False)
