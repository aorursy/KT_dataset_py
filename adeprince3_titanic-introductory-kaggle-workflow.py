import matplotlib.pyplot as plt

import numpy as np

import os

import pandas as pd

import seaborn as sns

import warnings

import xgboost as xgb



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.multiclass import OneVsOneClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
# This magic method displays matplotlib plots inside of Jupyter notebooks instead of in a separate window

%matplotlib inline



# Set plot style using the Seaborn library

sns.set_style("whitegrid")



# Supress the oh so common pandas warnings; use at own risk

warnings.filterwarnings("ignore")
train = pd.read_csv("../input/train.csv")
# The head() method shows the first rows of a DataFrame; tail() shows the last rows

train.head()
test = pd.read_csv("../input/test.csv")
test.head()
passenger_id = test["PassengerId"]
all_data = pd.concat([train, test], keys=["Train", "Test"], names=["Dataset", "Id"], sort=False)
all_data.head()
all_data.loc["Train"].head()
all_data.loc["Test"].head()
all_data.loc["Train"].info()
all_data.loc["Test"].info()
all_data.loc["Train"].describe()
# The 'bins' parameter sets the number of vertical bars

# The 'figsize' parameter sets the size of the chart

all_data.loc["Train"].hist(bins=25, figsize=(20, 15))
# The dropna() method excludes missing values, so if the most common value in the column is null, we don't return that as the mode

embarked_mode = all_data.loc["Train"]["Embarked"].dropna().mode()

embarked_mode
# The mode() method returns a pandas Series, but we only want the value, so we look at index 0

embarked_mode = embarked_mode[0]

embarked_mode
# The fillna() method fills in missing values with the value indicated in the first parameter position

all_data["Embarked"].fillna(embarked_mode, inplace=True)
# The isnull() method returns a pandas Series with the row index and a Boolean (True or False) indicating if the value is null

# The sum() method sums up the Boolean values, with True = 1 and False = 0

all_data["Embarked"].isnull().sum()
# Perform the same process as with 'Embarked' only combined into one line of code

all_data["Fare"].fillna(all_data.loc["Train"]["Fare"].dropna().mode()[0], inplace=True)
all_data["Fare"].isnull().sum()
# The corr() method returns a correlation matrix for all numerical features

correlation_matrix = all_data.loc["Train"].corr()

correlation_matrix["Age"].sort_values(ascending=False)
# We can return multiple columns by passing a list to the second indexer

# The groupby() method groups the results by the values in the 'Sex' column, so male and female

# The mean() method returns the average

# The sort_values() method sorts the resulting DataFrame

all_data.loc["Train"][["Age", "Sex"]].groupby(["Sex"]).mean().sort_values(by="Age", ascending=False)
all_data.loc["Train"][["Age", "Embarked"]].groupby(["Embarked"]).mean().sort_values(by="Age", ascending=False)
for value in ["male", "female"]:

    for i in range(0, 3):

        median_age = all_data.loc["Train"][(all_data.loc["Train"]["Sex"] == value) & (all_data.loc["Train"]["Pclass"] == i+1)]["Age"].dropna().median()

        all_data.loc[(all_data["Age"].isnull()) & (all_data["Sex"] == value) & (all_data["Pclass"] == i+1), "Age"] = median_age
all_data["Age"].isnull().sum()
#missing_age_index = list(all_data.loc["train"][all_data.loc["train"]["Age"].isnull()].index)

#missing_age_index
# Get the index of any row missing an Age in the dataset

#missing_age_index = list(all_data.loc["train"][dataset["Age"].isnull()].index)

#for i in missing_age_index:

#    age_average = all_data.loc["train"]["Age"].median()

#    age_predict = all_data.loc["train"][(all_data.loc["train"]["SibSp"] == all_data.iloc[i]["SibSp"]) & (all_data.loc["train"]["Parch"] == all_data.iloc[i]["Parch"]) & (all_data.loc["train"]["Pclass"] == all_data.iloc[i]["Pclass"])]["Age"].median()

#    if np.isnan(age_predict):

#        all_data["Age"].iloc[i] = age_average

#    else:

#        all_data["Age"].iloc[i] = age_predict
all_data["Cabin"].fillna("None", inplace=True)
all_data["Cabin"].isnull().sum()
# The extract() method uses a regular expression to extract the title from the 'Name' column

# The expand=False parameter returns a pandas Series

# Assigning values to an index that does not yet exist ('Title') will create it

all_data["Title"] = all_data["Name"].str.extract("([A-Za-z]+)\.", expand=False)
# The unique() method returns all the unique values in the Series

all_data["Title"].sort_values().unique()
pd.crosstab(all_data.loc["Train"]["Title"], all_data.loc["Train"]["Sex"])
all_data["Title"].replace(["Capt", "Col", "Countess", "Don", "Dona", "Dr", "Jonkheer", "Lady", "Major", "Rev", "Sir"], "Rare", inplace=True)

all_data["Title"].replace("Mlle", "Miss", inplace=True)

all_data["Title"].replace("Ms", "Miss", inplace=True)

all_data["Title"].replace("Mme", "Mrs", inplace=True)
all_data["Title"].sort_values().unique()
all_data.loc["Train"][["Title", "Survived"]].groupby(["Title"]).mean().sort_values(by="Survived", ascending=False)
# The pandas cut function splits the data into equal sized value ranges

pd.cut(all_data.loc["Train"]["Age"], bins=5).dtype
all_data.loc[all_data["Age"] <= 16, "Age"] = 0

all_data.loc[(all_data["Age"] > 16) & (all_data["Age"] <= 32), "Age"] = 1

all_data.loc[(all_data["Age"] > 32) & (all_data["Age"] <= 48), "Age"] = 2

all_data.loc[(all_data["Age"] > 48) & (all_data["Age"] <= 64), "Age"] = 3

all_data.loc[all_data["Age"] > 64, "Age"] = 4

# Since the category values are integers, we set the column type to int

# Notice that reassignment is used here as the astype() method does not have an inplace parameter

all_data["Age"] = all_data["Age"].astype(int)
all_data["Age"].sort_values().unique()
all_data.loc["Train"][["Age", "Survived"]].groupby(["Age"]).mean().sort_values(by="Survived", ascending=False)
all_data["FamilySize"] = all_data["SibSp"] + all_data["Parch"] + 1
all_data["FamilySize"].sort_values().unique()
all_data.loc["Train"][["FamilySize", "Survived"]].groupby(["FamilySize"]).mean().sort_values(by="Survived", ascending=False)
all_data["IsAlone"] = 0

all_data.loc[all_data["FamilySize"] == 1, "IsAlone"] = 1
all_data["IsAlone"].sort_values().unique()
all_data.loc["Train"][["IsAlone", "Survived"]].groupby(["IsAlone"]).mean()
# The qcut() method splits the data into equal sized bins where each bin has the same number of records

pd.qcut(train["Fare"], 4).dtype
all_data.loc[all_data["Fare"] <= 7.91, "Fare"] = 0

all_data.loc[(all_data["Fare"] > 7.91) & (all_data["Fare"] <= 14.45), "Fare"] = 1

all_data.loc[(all_data["Fare"] > 14.45) & (all_data["Fare"] <= 31.00), "Fare"] = 2

all_data.loc[all_data["Fare"] > 31.00, "Fare"] = 3

all_data["Fare"] = all_data["Fare"].astype(int)
all_data["Fare"].sort_values().unique()
all_data.loc["Train"][["Fare", "Survived"]].groupby(["Fare"], as_index=False).mean().sort_values(by="Survived", ascending=False)
all_data["Cabin"] = all_data["Cabin"].loc[all_data["Cabin"].isnull() == False].str.extract("([A-Za-z]+)", expand=False)
all_data["Cabin"].sort_values().unique()
all_data.loc["Train"][["Cabin", "Survived"]].groupby(["Cabin"], as_index=False).mean().sort_values(by="Survived", ascending=False)
all_data.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)
all_data.head()
# Instantiate a LabelEncoder object

label_encoder = LabelEncoder()

# The apply() method applies a function across a column or row

# Here we apply the LabelEncoder across the entire DataFrame, affecting only categorical (text) features

all_data_encoded = all_data.apply(label_encoder.fit_transform)

all_data_encoded.head()
correlation_matrix = all_data_encoded.loc["Train"].corr()

correlation_matrix["Survived"].sort_values(ascending=False)
plt.figure(figsize=(10, 8))

plt.title("Pearson Correlation of Features", y=1.05, size=15)

sns.heatmap(

    correlation_matrix,

    linewidths=0.1,

    vmax=1.0,

    square=True,

    cmap=plt.cm.jet,

    linecolor="white",

    annot=True

)
y_train = all_data_encoded.loc["Train"]["Survived"].astype(int)

X_train = all_data_encoded.loc["Train"].drop(["Survived"], axis=1)
# Instantiate a Random Forest Classifier object

random_forest_classifier = RandomForestClassifier()

# Train (fit) the Random Forest Classifier on the training data

random_forest_classifier.fit(X_train, y_train)
# The zip() function takes two iterables and joins them together into an iterable of tuples, in this case with the column name matched up to its feature importance

feature_importances = zip(list(X_train.columns.values), random_forest_classifier.feature_importances_)

# Sort the list by feature importance using a lambda function

feature_importances = sorted(feature_importances, key=lambda feature: feature[1], reverse=True)

# Iterate over the feature_importances list

for name, score in feature_importances:

    # The format() method replaces any set of curly brackets in a string with the specified arguments

    # The :<12 inside the first set of curly brackets aligns the text to the left and sets the character length to 12 characters, making everything print neatly

    print("{:<12} | {}".format(name, score))
all_data.drop(["FamilySize", "Cabin"], axis=1, inplace=True)
all_data.head()
all_data = pd.get_dummies(all_data)
all_data.loc["Train"].head()
y_train = all_data.loc["Train"]["Survived"].astype(int)

X_train = all_data.loc["Train"].drop(["Survived"], axis=1)

X_test = all_data.loc["Test"].drop(["Survived"], axis=1)
y_train.head()
X_train.head()
X_test.head()
#standard_scaler = StandardScaler()

#X_train = standard_scaler.fit_transform(X_train)

#X_test = standard_scaler.transform(X_test)



robust_scaler = RobustScaler()

X_train = robust_scaler.fit_transform(X_train)

X_test = robust_scaler.fit_transform(X_test)
X_train[:3]
X_test[:3]
sgd_classifier = SGDClassifier()

cross_val_score(sgd_classifier, X_train, y_train, cv=10, scoring="accuracy").mean()
one_vs_one_classifier = OneVsOneClassifier(SGDClassifier())

cross_val_score(one_vs_one_classifier, X_train, y_train, cv=10, scoring="accuracy").mean()
random_forest_classifier = RandomForestClassifier(min_samples_split=10, min_samples_leaf=2)

cross_val_score(random_forest_classifier, X_train, y_train, cv=10, scoring="accuracy").mean()
extra_trees_classifier = ExtraTreesClassifier(min_samples_split=7, min_samples_leaf=5)

cross_val_score(extra_trees_classifier, X_train, y_train, cv=10, scoring="accuracy").mean()
svm_classifier = SVC(probability=True, C=4.5)

cross_val_score(svm_classifier, X_train, y_train, cv=10, scoring="accuracy").mean()
adaboost_classifier = AdaBoostClassifier(n_estimators=200, learning_rate=0.1)

cross_val_score(adaboost_classifier, X_train, y_train, cv=10, scoring="accuracy").mean()
gradient_boost_classifier = GradientBoostingClassifier(learning_rate=0.03)

cross_val_score(gradient_boost_classifier, X_train, y_train, cv=10, scoring="accuracy").mean()
logistic_regression_classifier = LogisticRegression()

cross_val_score(logistic_regression_classifier, X_train, y_train, cv=5, scoring="accuracy").mean()
lda_classifier = LinearDiscriminantAnalysis()

cross_val_score(lda_classifier, X_train, y_train, cv=5, scoring="accuracy").mean()
xgboost_classifier = xgb.XGBClassifier(gamma=0.7)

cross_val_score(xgboost_classifier, X_train, y_train, cv=5, scoring="accuracy").mean()
parameter_grid = [

    {

        "kernel": ["rbf"],

        "C": [4, 4.5, 5],

        "shrinking": [True, False],

        "tol": [0.00001, 0.00003, 0.00005, 0.00008],

        "class_weight": ["balanced", None],

        "gamma": ["auto_deprecated", "scale"],

        "probability": [True]

    },

    {

        "kernel": ["poly"],

        "degree": [1, 3, 5],

        "gamma": ["auto_deprecated", "scale"]

    }

]
grid_search = GridSearchCV(

    svm_classifier,

    parameter_grid,

    cv=5,

    scoring="accuracy",

    n_jobs=2,

    verbose=1,

    return_train_score=True

)
grid_search.fit(X_train, y_train)
grid_search.best_params_
grid_search.best_estimator_
svm_classifier = grid_search.best_estimator_
cross_val_score(svm_classifier, X_train, y_train, cv=5, scoring="accuracy").mean()
svm_classifier.fit(X_train, y_train)

gradient_boost_classifier.fit(X_train, y_train)

logistic_regression_classifier.fit(X_train, y_train)

lda_classifier.fit(X_train, y_train)

xgboost_classifier.fit(X_train, y_train)
voting_classifier = VotingClassifier(

    estimators=[

        ("svc", svm_classifier),

        ("gradient_boost", gradient_boost_classifier),

        ("logistic_regression", logistic_regression_classifier),

        ("lda", lda_classifier),

        ("xgboost", xgboost_classifier)

    ],

    voting="soft"

)
voting_classifier.fit(X_train, y_train)
cross_val_score(voting_classifier, X_train, y_train, cv=10, scoring="accuracy").mean()
predictions = voting_classifier.predict(X_test)
submission = pd.DataFrame(

    {

        "PassengerId": passenger_id,

        "Survived": predictions

    }

)
submission.head(10)
# Write the submission DataFrame to a CSV file using the constructed filename

submission.to_csv("submission.csv", index=False)