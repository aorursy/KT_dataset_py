# Restart Runtime after installing libraries.

#!pip install eli5



# Importing all required libraries



import eli5 



import time

import numpy as np

import pandas as pd

import missingno as msno

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectKBest

from sklearn.pipeline import Pipeline

from sklearn.metrics import make_scorer

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold
# Load the datasets.

train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")



print("Train size: {0}\nTest size: {1}".format(train.shape,test.shape))
train.head()
test.head()
train.info()
train.Survived.value_counts()
# Calculate the percentage of missing values for each column in the training set.

train.isnull().sum()/train.shape[0]
msno.matrix(train, figsize=(10,10))
test.isnull().sum()/train.shape[0]
msno.matrix(test, figsize=(10,10))
# Showing the distribution of age.

train.Age.hist(bins=20)

train.Age.describe()
# Showing the distribution of fare.

train.Fare.hist(bins=20)

train.Fare.describe()
# Calculate the mean of other fields based on the class.

temp = train.groupby(["Pclass"]).mean()

temp

g = sns.catplot(x="Sex", hue="Pclass", col="Survived", data=train, kind="count", height=4, aspect=.7)
# Make a copy of the original dataset.

train_aux = train.copy()



# Drop passenger id (useless).



train_aux = train_aux.drop('PassengerId', axis=1)



# Define the mean age separated by class.

c1_age_mean = train_aux.loc[train_aux.Pclass == 1, "Age"].mean()

c2_age_mean = train_aux.loc[train_aux.Pclass == 2, "Age"].mean()

c3_age_mean = train_aux.loc[train_aux.Pclass == 3, "Age"].mean()



# Fixing null values in column Age using the mean for each class.

train_aux.loc[train_aux.Pclass == 1, "Age"] = train_aux.loc[train_aux.Pclass == 1, "Age"].fillna(c1_age_mean)

train_aux.loc[train_aux.Pclass == 2, "Age"] = train_aux.loc[train_aux.Pclass == 2, "Age"].fillna(c2_age_mean)

train_aux.loc[train_aux.Pclass == 3, "Age"] = train_aux.loc[train_aux.Pclass == 3, "Age"].fillna(c3_age_mean)



# Fixing null values in column Cabin (call the missing Cabin as Unknown).

cabin_inputer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value="Unknown")

train_aux.Cabin = cabin_inputer.fit_transform(train_aux[["Cabin"]])



# Fixing null values in column Embarked (set the missing values using the mode).

emb_inputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")

train_aux.Embarked = emb_inputer.fit_transform(train_aux[["Embarked"]])



# Verifying the columns don't have any remaining null value.

train_aux.isnull().sum()
# Use the first char of the string to represent the Cabin.

train_aux.Cabin = train_aux.Cabin.str[0:1]

train_aux.head()
sns.catplot(x="Cabin", hue="Survived", data=train_aux, kind="count")
train_aux.Name = train_aux.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

train_aux.Name.value_counts()
# Replace class other than "Mr", "Miss", "Mrs", "Master" as Other.

train_aux.loc[~train_aux.Name.isin(["Mr", "Miss", "Mrs", "Master"]), "Name"] = "Other"
sns.catplot(x="Name", hue="Survived", data=train_aux, kind="count")
def ticket_type(ticket):

  result = ticket[0:1]

  # Try to convert to a number. If fails, it's a string.

  try:

    int(result)

    result = 'NUM'

  except:

    pass

  return result



train_aux.Ticket = train_aux.Ticket.apply(ticket_type)
sns.catplot(x="Ticket", hue="Survived", data=train_aux, kind="count")
label_enc = LabelEncoder()



# Converting Sex colum to categorical.

train_aux.Sex = label_enc.fit_transform(train_aux.Sex.to_list())



# Converting Embarked colum to categorical.

train_aux.Embarked = label_enc.fit_transform(train_aux.Embarked.to_list())



# Converting Cabin to categorial.

train_aux.Cabin = label_enc.fit_transform(train_aux.Cabin.to_list())



# Converting Name to categorial.

train_aux.Name = label_enc.fit_transform(train_aux.Name.to_list())



# Converting Ticket to categorial.

train_aux.Ticket = label_enc.fit_transform(train_aux.Ticket.to_list())



train_aux.head()
plt.figure(figsize = (8, 8))

sns.heatmap(train_aux.corr(), cmap=("RdBu_r"), annot=True, fmt='.2f')

plt.xticks(rotation=45) 

plt.yticks(rotation=0) 

plt.show()
# These are the datasets to be used in the model.

train_p = train.copy()

test_p = test.copy()



# Drop passenger id (useless).

train_p = train_p.drop('PassengerId', axis=1)

test_p = test_p.drop('PassengerId', axis=1)



# Fixing null values in column Age using the mean for each Pclass.

train_p.loc[train_p.Pclass == 1, "Age"] = train_p.loc[train_p.Pclass == 1, "Age"].fillna(int(train_p.loc[train_p.Pclass == 1, "Age"].mean()))

train_p.loc[train_p.Pclass == 2, "Age"] = train_p.loc[train_p.Pclass == 2, "Age"].fillna(int(train_p.loc[train_p.Pclass == 2, "Age"].mean()))

train_p.loc[train_p.Pclass == 3, "Age"] = train_p.loc[train_p.Pclass == 3, "Age"].fillna(int(train_p.loc[train_p.Pclass == 3, "Age"].mean()))



test_p.loc[test_p.Pclass == 1, "Age"] = test_p.loc[test_p.Pclass == 1, "Age"].fillna(int(test_p.loc[test_p.Pclass == 1, "Age"].mean()))

test_p.loc[test_p.Pclass == 2, "Age"] = test_p.loc[test_p.Pclass == 2, "Age"].fillna(int(test_p.loc[test_p.Pclass == 2, "Age"].mean()))

test_p.loc[test_p.Pclass == 3, "Age"] = test_p.loc[test_p.Pclass == 3, "Age"].fillna(int(test_p.loc[test_p.Pclass == 3, "Age"].mean()))



# Fixing null values in column Cabin.

cabin_inputer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value="Unknown")

train_p.Cabin = cabin_inputer.fit_transform(train_p[["Cabin"]])

test_p.Cabin = cabin_inputer.fit_transform(test_p[["Cabin"]])



# Fixing null values in column Embarked (missing only in train).

emb_inputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")

train_p.Embarked = emb_inputer.fit_transform(train_p[["Embarked"]])



# Fixing null values in Fare (missing only in test).

fare_inputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")

test_p.Fare = fare_inputer.fit_transform(test_p[["Fare"]])



# Processing colums with text.

train_p.Cabin = train_p.Cabin.str[0:1]

test_p.Cabin = test_p.Cabin.str[0:1]



train_p.Name = train_p.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

train_p.loc[~train_p.Name.isin(["Mr", "Miss", "Mrs", "Master", "Dr", "Rev"]), "Name"] = "Other"

test_p.Name = test_p.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

test_p.loc[~test_p.Name.isin(["Mr", "Miss", "Mrs", "Master", "Dr", "Rev"]), "Name"] = "Other"



train_p.Ticket = train_p.Ticket.apply(ticket_type)

test_p.Ticket = test_p.Ticket.apply(ticket_type)
train_p.head()
# Dealing with number of siblings/ spouses and parentes/ children.

train_p["NumRelatives"] = train_p["SibSp"] + train_p["Parch"]

test_p["NumRelatives"] = test_p["SibSp"] + test_p["Parch"]



# Defining if the passenger was travelling alone.

train_p["Alone"] = "Yes"

train_p.loc[train_p["NumRelatives"] > 0, "Alone"] = "No"

test_p["Alone"] = "Yes"

test_p.loc[test_p["NumRelatives"] > 0, "Alone"] = "No"



# Creating categories for Age and Fare.

age_points = [0, 18, 30, 60, 100]

age_names = ["Child", "YoungAdult", "Adult", "Senior"]

train_p["AgeCat"] = pd.cut(train_p["Age"], age_points, labels=age_names)

test_p["AgeCat"] = pd.cut(test_p["Age"], age_points, labels=age_names)



fare_points = [-1, 15, 60, 300, 1000]

fare_names = ["Cheap",  "Regular", "Business", "First"]

train_p["FareCat"] = pd.cut(train_p["Fare"], fare_points, labels=fare_names)

test_p["FareCat"] = pd.cut(test_p["Fare"], fare_points, labels=fare_names)
train_p.AgeCat.value_counts()
sns.catplot(x="AgeCat", hue="Survived", data=train_p, kind="count")
train_p.FareCat.value_counts()
sns.catplot(x="FareCat", hue="Survived", data=train_p, kind="count")
train_p.head()
train_p['Pclass'] = train_p['Pclass'].map({1: 'First', 2: 'Second', 3: 'Third'})

test_p['Pclass'] = test_p['Pclass'].map({1: 'First', 2: 'Second', 3: 'Third'})



# Viewing the new Pclass.

train_p[["Pclass"]].head()
# Separating features of the target column.

target = train_p["Survived"]

train_p = train_p.drop("Survived", axis=1)



# Removing redundant columns.

train_p = train_p.drop(labels=["SibSp", "Parch",], axis=1)

test_p = test_p.drop(labels=["SibSp", "Parch",], axis=1)
# Encode the Ticket and Cabin using regular LabelEncode bacause they have so many categories.

train_p[["Ticket"]] = train_p[["Ticket"]].apply(lambda col: label_enc.fit_transform(col))

test_p[["Ticket"]] = test_p[["Ticket"]].apply(lambda col: label_enc.fit_transform(col))



train_p[["Cabin"]] = train_p[["Cabin"]].apply(lambda col: label_enc.fit_transform(col))

test_p[["Cabin"]] = test_p[["Cabin"]].apply(lambda col: label_enc.fit_transform(col))



# Join datasets to create the all the columns properly (without doing this, the test dataset would have less columns).

temp = pd.concat([train_p, test_p])



# Get all categorical and numerical columns.

categorical_features = temp.select_dtypes(include=["object", "category"]).columns

numerical_features = temp.select_dtypes(include=["int64","float64"]).columns



# Convert the categorical columns using one hot encoding.

temp = pd.get_dummies(temp, drop_first=True, columns=categorical_features)



# le = LabelEncoder()

# temp[categorical_features] = temp[categorical_features].apply(lambda col: le.fit_transform(col))



# Separate the datasets again.

train_p = temp.iloc[:891].copy()

test_p = temp.iloc[891:].copy()



# Normalize the numerical columns.

scaler = StandardScaler()

train_p[train_p.columns] = scaler.fit_transform(train_p[train_p.columns])

test_p[test_p.columns] = scaler.fit_transform(test_p[test_p.columns])
train_p.head()
test_p.head()
seed = 42

num_folds = 10

scoring = {'Accuracy': make_scorer(accuracy_score)}



kfold = StratifiedKFold(n_splits=num_folds,random_state=seed)

rfc = RandomForestClassifier(oob_score = True, bootstrap = True)



# Creating the traininig and testing sets.

X_train, X_test, y_train, y_test = train_test_split(train_p,

                                                    target,

                                                    test_size=0.20,

                                                    random_state=seed,

                                                    shuffle=True,

                                                    stratify=target)



search_space = {

    'n_estimators': [20, 50, 100],

    'min_samples_split': [2, 4, 8],

    'min_samples_leaf': [1, 2, 4],

}



grid = GridSearchCV(estimator=rfc, 

                    param_grid=search_space,

                    cv=kfold,

                    scoring=scoring,

                    return_train_score=True,

                    n_jobs=-1,

                    refit="Accuracy")





# Train the RandomForest.

best_model = grid.fit(X_train, y_train)



estimator = best_model.best_estimator_

print(estimator)



# Showing first result.

predict = estimator.predict(X_test)

print(accuracy_score(y_test, predict))

print(confusion_matrix(y_test, predict))

print(classification_report(y_test, predict))



# # Utilizando a biblioteca ELI5, exibe a feature importance da RandomForest.

eli5.show_weights(estimator, show_feature_values=True, feature_names=X_train.columns.to_list())
train_final = train_p

test_final = test_p



# Split the training dataset for the validation of the model.

X_train, X_test, y_train, y_test = train_test_split(train_final,

                                                    target,

                                                    test_size=0.20,

                                                    random_state=seed,

                                                    shuffle=True,

                                                    stratify=target)

# Piple line using the SelectKBest and classifier.

pipe = Pipeline(

    steps=[

           ('fs', SelectKBest()),

           ('clf', RandomForestClassifier())

    ]

)



# Search space.

search_space = [

  {"clf":[RandomForestClassifier()],

    "clf__n_estimators": [50, 100],

    "clf__criterion": ["entropy"],

    "clf__max_leaf_nodes": [64],

    "clf__min_samples_split": [4],

    "clf__random_state": [seed],

    "fs__k": [4, 10, 'all'],

    },

  {"clf":[XGBClassifier()],

    "clf__n_estimators": [50, 100],

    "clf__max_depth": [4],

    "clf__learning_rate": [0.001, 0.01, 0.1],

    "clf__random_state": [seed],

    "clf__subsample": [1.0],

    "clf__colsample_bytree": [1.0],

    "fs__k": [4, 10, 'all'],

    },

]



# Create the grid search.

grid = GridSearchCV(estimator=pipe, 

                    param_grid=search_space,

                    cv=kfold,

                    scoring=scoring,

                    return_train_score=True,

                    n_jobs=-1,

                    refit="Accuracy")



# Run the grid search.

tmp = time.time()

best_model = grid.fit(X_train, y_train)

print("CPU Training Time: %s seconds" % (str(time.time() - tmp))) 
print("Best model: %f using %s" % (best_model.best_score_, best_model.best_params_))



result = pd.DataFrame(best_model.cv_results_)

result.head()
prediction = best_model.best_estimator_.predict(X_test)

print(accuracy_score(y_test, prediction))

print(confusion_matrix(y_test, prediction))

print(classification_report(y_test, prediction))
predict_final = best_model.best_estimator_.predict(test_final)



holdout_ids = test["PassengerId"]

submission_df = {"PassengerId": holdout_ids,

                 "Survived": predict_final}

submission = pd.DataFrame(submission_df)



submission.to_csv("submission.csv", index=False)
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")