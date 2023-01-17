# set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.ml_intermediate.ex7 import *

print("Setup Complete")
# check your answer (Run this code cell to receive credit!)

q_1.solution()
# check your answer (Run this code cell to receive credit!)

q_2.solution()
# check your answer (Run this code cell to receive credit!)

q_3.solution()
# check your answer (Run this code cell to receive credit!)

q_4.solution()
# fill in the line below with one of 1, 2, 3 or 4.

potential_leakage_feature = 2



# check your answer

q_5.check()
# q_5.hint()

# q_5.solution()
# load data

import pandas as pd

data = pd.read_csv(

    "../input/aer-credit-card-data/AER_credit_card_data.csv",

    true_values=["yes"],

    false_values=["no"],

)



# separate target (y) from features (X)

y = data["card"]

X = data.drop(["card"], axis=1)



# since there is no preprocessing, we don't need a pipeline (used anyway as best practice!)

from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier

my_pipeline = make_pipeline(RandomForestClassifier(n_estimators=100))



# evalutae model

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(my_pipeline, X, y, cv=5, scoring="accuracy")

cv_scores.mean()
# fraction of those who received a card and had no expenditures

print((X["expenditure"][y] == 0).mean())



# fraction of those who did not receive a card and had no expenditures

print((X["expenditure"][~y] == 0).mean())
# drop leaky features from dataset

potential_leaks = ["expenditure", "share", "active", "majorcards"]

X2 = X.drop(potential_leaks, axis=1)



# evaluate the model, with leaky predictors removed

cv_scores = cross_val_score(my_pipeline, X2, y, cv=5, scoring="accuracy")

cv_scores.mean()