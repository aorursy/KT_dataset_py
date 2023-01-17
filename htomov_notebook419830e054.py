# The model to use.

from sklearn.ensemble import RandomForestRegressor



# The error metric (c-stat aka ROC/AUC)

from sklearn.metrics import roc_auc_score



# Data structure.

import pandas as pd



# Import data

X = pd.read_csv("../input/train.csv")

y = X.pop("Survived")

X.describe()
# Fill missing NA for 'Age'

X["Age"].fillna(X.Age.mean(), inplace=True)

X.describe()
# Get only the numeric variables

numeric_variables = list(X.dtypes[X.dtypes != "object"].index)

X[numeric_variables].head()
# Build initial model.

model = RandomForestRegressor(n_estimators = 100, oob_score=True, random_state=1980)



# Fit the numeric data

model.fit(X[numeric_variables], y)
# c-stat

y_oob = model.oob_prediction_

print("c-stat: ", roc_auc_score(y,y_oob))
# Define function similar to describe(), but returns the results for categorical variables only.

def describe_categorical(X):

    from IPython.display import display, HTML

    display(HTML(X[X.columns[X.dtypes == 'object']].describe().to_html()))
# Drop some variables

X.drop(["Name", "Ticket", "PassengerId"], axis=1, inplace=True)

# Defines a function which change the cabin variable to be only the first letter or None

def clean_cabin(x):

    try:

        return x[0]

    except TypeError:

        return "None"
X["Cabin"] = X.Cabin.apply(clean_cabin)
categorical_variables = ['Sex', 'Cabin', 'Embarked']



for variable in categorical_variables:

    # Fill missing data with word "Missing"

    X[variable].fillna("Missing", inplace=True)

    #Crate array of dummies

    dummies = pd.get_dummies(X[variable], prefix=variable)

    # Update X to include dummies and drop the main variable

    X = pd.concat([X, dummies], axis=1)

    X.drop([variable], axis=1, inplace=True)
# Look at all the columns in the dataset

def printall(X, max_rows=10):

    from IPython.display import display, HTML

    display(HTML(X.to_html(max_rows=max_rows)))

printall(X)
# Build and fit a model with the new dataset

model = RandomForestRegressor(100, oob_score = True, n_jobs=-1, random_state=1980)

model.fit(X,y)

print("C-stat", roc_auc_score(y, model.oob_prediction_))
# Some Exploratory Data Analysis

model.feature_importances_

# Shows all of the variables

%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt



feature_importances = pd.Series(model.feature_importances_, index=X.columns)

feature_importances.sort_values(inplace=True)

feature_importances.plot(kind="barh", figsize=(7,6))
# Complex version 

def graph_feature_importance(model, feature_names, autoscale=True, headroom=0.05, width=10, summarized_columns=None):

    if autoscale:

        x_scale = model.feature_importances_.max() + headroom

    else:

        x_scale=1

    

    feature_dict = dict(zip(feature_names, model.feature_importances_))

    

    if summarized_columns:

        for col_name in summarized_columns:

            sum_value = sum(v for k, v in feature_dict.items() if col_name in k)

            keys_to_remove = [k for k in feature_dict.keys() if col_name in k]

            

            for i in keys_to_remove:

                feature_dict.pop(i)

            feature_dict[col_name] = sum_value





    results = pd.Series(list(feature_dict.values()), index=list(feature_dict.keys()))

    results.sort_values(ascending=1, inplace=True)

    results.plot(kind="barh", figsize=(width,len(results)/4), xlim=(0,x_scale))
%%timeit

model = RandomForestRegressor(1000, oob_score=True, n_jobs=1, random_state=1980)

model.fit(X,y)
%%timeit

model = RandomForestRegressor(1000, oob_score=True, n_jobs=-1, random_state=1980)

model.fit(X,y)
#n_estimators

results = []

n_estimator_options = [30,50,100,200,500,1000,2000]



for trees in n_estimator_options:

    model = RandomForestRegressor(trees, oob_score=True, n_jobs=-1, random_state=1980)

    model.fit(X,y)

    print("trees", trees)

    roc = roc_auc_score(y, model.oob_prediction_)

    print("C-stat", roc)

    results.append(roc)

    print("-----")



pd.Series(results, n_estimator_options).plot()
#"max_features"

results = []

max_features_options = ["auto", None, "sqrt", "log2", 0.9, 0.2]



for max_features in max_features_options:

    model = RandomForestRegressor(n_estimators=1000, oob_score=True, n_jobs=-1, random_state=1980, max_features=max_features)

    model.fit(X,y)

    print("max_features", max_features)

    roc = roc_auc_score(y, model.oob_prediction_)

    print("C-stat", roc)

    results.append(roc)

    print("-----")



pd.Series(results, max_features_options).plot(kind="barh", xlim=(.85,.88))
#min_samples_leaf

results = []

min_samples_leaf_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]



for min_samples in min_samples_leaf_options:

    model = RandomForestRegressor(n_estimators=1000, oob_score=True, n_jobs=-1, random_state=1980, max_features="auto", 

                                 min_samples_leaf=min_samples)

    model.fit(X,y)

    print("min_samples", min_samples)

    roc = roc_auc_score(y, model.oob_prediction_)

    print("C-stat", roc)

    results.append(roc)

    print("-----")



pd.Series(results, min_samples_leaf_options).plot()
# Final model

model = RandomForestRegressor(n_estimators=1000, oob_score=True, n_jobs=-1, 

                              random_state=1980, max_features="auto", min_samples_leaf=5)

model.fit(X,y)

roc = roc_auc_score(y, model.oob_prediction_)

print("C-stat:", roc)