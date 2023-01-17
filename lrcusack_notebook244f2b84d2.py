import pandas as pd

import numpy as np

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.cluster import KMeans

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from time import time



# Read pokemon table (don't use pokemon # as index, it is non-unique

pokemon = pd.read_csv("../input/Pokemon.csv", na_filter=False, index_col=False)





print(pokemon.loc[0:10])
# Don't show warnings tripped by adding columns to a dataframe

pd.options.mode.chained_assignment = None





# save type strings for convenience

t1 = 'Type 1'

t2 = 'Type 2'

t = 'Type'





def set_type(typefcn, df):

    """Sets pokemon type using typefcn and clears the Type 1 and Type 2 fields

    typefcn should take a dataframe as an arugument and return a series that will fit into that dataframe"""

    newdf = df

    newdf[t] = typefcn(df)

    return newdf.filter(items=list(set(list(df)).difference([t1, t2])))





# get pokemon with only one type

singletypepokemon = set_type(lambda df: df[t1], pokemon.loc[lambda p: p[t2] == '', :])





# get pokemon with 2 types

dualtypepokemon = pokemon.loc[lambda p: p[t2] != '', :]





# copy dual typed pokemon and set type to first type

firsttypepokemon = set_type(lambda df: df[t1], dualtypepokemon)

# copy dual typed pokemon and set type to second type

secondtypepokemon = set_type(lambda df: df[t2], dualtypepokemon)

# copy dual typed pokemon and set type to first+second type

mergedtypepokemon = set_type(lambda df: df[t1] + df[t2], dualtypepokemon)





# combine pokemon into single table and create a type number

newpokemon = singletypepokemon.append(firsttypepokemon).append(secondtypepokemon).append(mergedtypepokemon)

typelookup = list(set(list(newpokemon[t])))

typenumlookup = dict([(poketype, idx) for idx, poketype in enumerate(typelookup)])

newpokemon['typenum'] = pd.Series([typenumlookup[poketype] for poketype in list(newpokemon[t])])





# Print bulbasaur and charmander examples

print(newpokemon.loc[0].append(newpokemon.loc[4]))
predictorcolumns = ['Speed', 'Attack', 'Sp. Atk', 'Defense', 'Sp. Def', 'HP']

responsecolumn = 'typenum'



testsize = .2



trainingset, testset = train_test_split(newpokemon, test_size=testsize)



xtrain = trainingset.loc[:, predictorcolumns]

ytrain = trainingset.typenum



xtest = testset.loc[:, predictorcolumns]

ytest = testset.typenum
def try_classifier(modelname, model, scorefcn):

    t0 = time()

    model.fit(xtrain, ytrain)

    t1 = time()

    print(modelname)

    print("Accuracy on training set: {:.3f}".format(scorefcn(model,xtrain, ytrain)))

    print("Accuracy on test set: {:.3f}".format(scorefcn(model, xtest, ytest)))

    print("Run time: {:.3f}".format(t1-t0))

    print("------------------------------------\n")

    

models = {"Decision Tree Classifier": DecisionTreeClassifier(),

          "Random Forest Classifier": RandomForestClassifier(),

          "Gradient Boosting Classifier": GradientBoostingClassifier(),

          "K Nearest Neighbors Classifier": KNeighborsClassifier(),

          "K Means Cluster Classifier": KMeans(),

          "Naive Bayes Classifier": GaussianNB(),

          "Scaled Vector Classifier": SVC()}



for modelname in models:

    try_classifier(modelname, models[modelname], lambda model, x, y: model.score(x,y))