from __future__ import print_function

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import roc_auc_score

import pandas as pd

X = pd.read_csv("../input/train.csv")

y = X.pop("Survived")



X["Age"].fillna(X.Age.mean(), inplace=True)

X.drop(["Name", "Ticket", "PassengerId"],axis=1,inplace=True)



def clean_cabin(x):

    try:

        return x[0]

    except TypeError:

        return "None"



X["Cabin"] = X.Cabin.apply(clean_cabin)



categorical_variables = ['Sex', 'Cabin', 'Embarked']



for variable in categorical_variables:

    X[variable].fillna("Missing", inplace=True)

    dummies = pd.get_dummies(X[variable], prefix=variable)

    X = pd.concat([X,dummies], axis=1)

    X.drop([variable], axis=1, inplace=True)



model = RandomForestRegressor(n_estimators=100, oob_score=True, n_jobs=-1,random_state=42)

model.fit(X,y)

print ("roc auc score ", roc_auc_score(y,model.oob_prediction_))
%%timeit

model = RandomForestRegressor(1000, oob_score=True, n_jobs=1,random_state=42)

model.fit(X,y)
%%timeit

model = RandomForestRegressor(1000, oob_score=True, n_jobs=-1,random_state=42)

model.fit(X,y)
results = []

max_features_options = ["auto", None, "sqrt", "log2", 0.9, 0.2]



for max_features in max_features_options:

    model = RandomForestRegressor(n_estimators=1000, oob_score=True, n_jobs=-1, random_state=42, max_features=max_features)

    model.fit(X,y)

    print (max_features, "max features")

    roc = roc_auc_score(y, model.oob_prediction_)

    print ("Roc :", roc)

    results.append(roc)

    print ("")



pd.Series(results, max_features_options).plot()
results = []

min_samples_leaf_options = list(range(1,11))



for min_samples in min_samples_leaf_options:

    model = RandomForestRegressor(n_estimators=1000, oob_score=True, n_jobs=-1, random_state=42, max_features="auto", min_samples_leaf=min_samples)

    model.fit(X,y)

    print (min_samples, "min samples")

    roc = roc_auc_score(y, model.oob_prediction_)

    print ("Roc :", roc)

    results.append(roc)

    print ("")



pd.Series(results, min_samples_leaf_options).plot() 
model = RandomForestRegressor(n_estimators=1000, 

                              oob_score=True, 

                              n_jobs=-1, 

                              random_state=42, 

                              max_features="auto", 

                              min_samples_leaf=5)

model.fit(X,y)

print (min_samples, "min samples")



roc = roc_auc_score(y, model.oob_prediction_)

print ("Roc :", roc)