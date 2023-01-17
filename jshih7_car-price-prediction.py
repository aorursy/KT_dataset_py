%matplotlib inline



import sklearn



from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt



import csv

import numpy as np

import pandas as pd

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, make_scorer, mean_squared_error



# If true, include make and model in Random Forest

# Shows how much make and model come into play, but when when we calculate prices

# we should omit to see if the models are overpriced

includeMakeAndModel = True



# Number of trees in forest

nEstimators = 500



def GetDataMatrix():

    

    # Data frame with make and model

    Xmodelmake = pd.read_csv("../input/data.csv",header=0, usecols=(0,1,2,3,4,5,6,7,8,9,10,11,13,14,));

    

    # Excluding make and model

    if not includeMakeAndModel:

        X = pd.read_csv("../input/data.csv",header=0, usecols=(2,3,4,5,6,7,8,9,10,11,13,14,));

    else:

        X = Xmodelmake

    Y = pd.read_csv("../input/data.csv",header=0, usecols=(15,));



    X, Y, Xmodelmake = shuffle(X, Y, Xmodelmake)

    Xmake = Xmodelmake['Make']

    Xmodel = Xmodelmake['Model']

    

    # Turns categorical data into binary values across many columns

    if not includeMakeAndModel:

        X = pd.get_dummies(X, dummy_na = False, columns=['Engine Fuel Type', 'Transmission Type', 'Driven_Wheels', 'Market Category', 'Vehicle Size', 'Vehicle Style'] );

    else:    

        X = pd.get_dummies(X, dummy_na = False, columns=['Make', 'Model', 'Engine Fuel Type', 'Transmission Type', 'Driven_Wheels', 'Market Category', 'Vehicle Size', 'Vehicle Style'] );

    

    X.insert(0, 'ModelRef', Xmodel);

    X.insert(0, 'MakeRef', Xmake);

    

    # Fill the null values with zeros

    X.fillna(0, inplace=True);

    return (X, Y, Xmodelmake)



##########



(X, Y, Xmodelmake) = GetDataMatrix() #Gets the X,Y



# Turn into a proper one D arrayY = numpy.ravel(Y);

Y_unraveled = np.ravel(Y);



# Split dataset into training and testing

print('Splitting into training and testing...')

X_train, X_test, Y_train, y_test = train_test_split(X, Y_unraveled, test_size=0.10, random_state=32)

MSE_Scorer = make_scorer(mean_squared_error);



# Model/Make columns are only used later on to relate indices to Model/Makes

X_train2 = X_train.drop('MakeRef', axis = 1).drop('ModelRef', axis = 1)

X_test2 = X_test.drop('MakeRef', axis = 1).drop('ModelRef', axis = 1)



# Train using Random Forest

print('Training classifier...')

clf = RandomForestRegressor(n_estimators=nEstimators, max_features="sqrt");

# The gradient boosting classifier didnt finish running

# clf = GradientBoostingClassifier(n_estimators=5)

clf = clf.fit(X_train2, Y_train);

print("Done training best classifier.")



print('Calculating error...')

y_pred = clf.predict(X_test2);

scores = cross_val_score(clf,X_test2,y_test, cv = 5)

print()



print("Scores:")

print(scores);

print("Mean absolute error:");

mean_error = sum(abs(y_test-y_pred))/len(y_test);

print(mean_error);

print("Mean percent error: ")

print(mean_error/np.mean(y_test))

print()



print("ypred:");

print(y_test);

print(y_pred);

np.savetxt("ypred_test.csv",(y_pred,y_test),delimiter=",");

print()
# If we used JSON file, this would've been easier

# This code is trying to get the data off Kaggle and making the make and popularities unique

# since the data can list them multiple times



# Make elements in cars unique and return in same order

cars = np.asarray(Xmodelmake['Make'])

uniquecarindices = np.unique(cars, return_index=True)[1]

cars = np.asarray([cars[index] for index in sorted(uniquecarindices)])



# Make elements in popularities unique and return in same order

popularities = np.asarray(Xmodelmake['Popularity'])

uniquepopularityindices = np.unique(popularities, return_index=True)[1]

popularities = np.asarray([popularities[index] for index in sorted(uniquepopularityindices)])



# Get the indices sorted on popularities from highest to lowest

popindices = np.argsort(popularities)[::-1]



# Data range

totalN = popindices.shape[0]



figsize = (8,6)

plt.figure(figsize=figsize)

plt.title("Car Popularities")

plt.bar(range(totalN), popularities[popindices], color="b", align="center")

plt.xticks(rotation=90)

plt.xticks(range(totalN), cars[popindices])

plt.xlim([-1, totalN])

plt.xlabel('Car Models')

plt.ylabel('Popularity')

plt.show()
# Important questions to answer



# 1. What features most predict price?



# Get the importances and calculate standard deviations for each

importances = clf.feature_importances_

std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)

indices = np.argsort(importances)[::-1]



# Get the feature names

features = X_test2.columns.values



# Want the top 20 features, so limit the indices and labels

topLimit = 20 # limit to show up to, ex. top 10

indices = indices[0: topLimit] # indices for features

topLabels = features[indices[0: topLimit]] # actual feature labels, we want to print these



# Plot the feature importances of the forest (top 20)

figsize = (8,6)

plt.figure(figsize=figsize)

plt.title("Top 20 Important Features")

ax = plt.bar(range(topLimit), importances[indices], color="r", yerr=std[indices], align="center")

plt.xticks(rotation=90)

plt.xticks(range(topLimit), topLabels)

plt.xlim([-1, topLimit])

plt.xlabel('Features')

plt.ylabel('Importance')

plt.show()
# 2. What cars are the most over-priced for their feature set?



# Get the errors from the prediction and sort from greatest to least

y_error = y_test-y_pred

old_indices = np.argsort(y_error)[::-1] # returns the old indices



# Put top 10 overpriced cars into a list

modelmakelist = []

N = 10 # number of top values to extract

for i in range(N):

    modelmakelist.append(X_test['MakeRef'].iloc[old_indices[i]]

                         + ' ' + X_test['ModelRef'].iloc[old_indices[i]]

                         + ' ' + str(X['Year'].iloc[old_indices[i]]))

modelmakelist = np.asarray(modelmakelist) # don't index into original



# Plot the top 10 overpriced cars against their price

figsize = (8,6)

plt.figure(figsize=figsize)

plt.title("Top 10 Overpriced Cars")

plt.bar(range(N), y_error[old_indices[0:N]], color="b", align="center")

plt.xticks(rotation=90)

plt.xticks(range(N), modelmakelist)

plt.xlim([-1, N])

plt.xlabel('Car Make and Model')

plt.ylabel('Price')

plt.show()



# Put top 10 overpriced brands into a list

# Scan all entries, if maker already exists, go to next entry, else add maker to list

existingmakers = []

pricelist = []

for i in range(old_indices.shape[0]):

    currentmaker = X_test['MakeRef'].iloc[old_indices[i]]

    if currentmaker not in existingmakers:

        existingmakers.append(currentmaker)

        pricelist.append(y_error[old_indices[i]])

        if len(existingmakers) == N:

            break



existingmakers = np.asarray(existingmakers)

pricelist = np.asarray(pricelist)

    

figsize = (8,6)

plt.figure(figsize=figsize)

plt.title("Top 10 Overpriced Car Brands:")

plt.bar(range(N), pricelist, color="g", align="center")

plt.xticks(rotation=90)

plt.xticks(range(N), existingmakers)

plt.xlim([-1, N])

plt.xlabel('Car Brand')

plt.ylabel('Price')

plt.show()