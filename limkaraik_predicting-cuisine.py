# Basic Libraries

import numpy as np

import pandas as pd

import seaborn as sb

import matplotlib.pyplot as plt # we only need pyplot

sb.set() # set the default Seaborn style for graphics
#import test and train file

everything = pd.read_json("../input/whats-cooking/train.json")

test = pd.read_json("../input/whats-cooking/test.json")
#making a dict of ingredients with their total used times in the data set

ingredientData = pd.DataFrame(everything[['ingredients']])

ingredientDict = {}

for i in range(0,39774):

    for ingredient in ingredientData.iloc[i][0]:

        if ingredient not in ingredientDict.keys():

            ingredientDict[ingredient] = 1

        else:

            ingredientDict[ingredient]+=1
#sorting the dictionary according to its value

sorted_dict={}

sorted_keys=sorted(ingredientDict, key=ingredientDict.get, reverse=True)

for r in sorted_keys:

    sorted_dict[r]= ingredientDict[r]
#removing the common ingredients

useless=[]

for key in sorted_dict.keys():

    useless.append(key)

    if len(useless)==12:

        break

for i in useless:

    del sorted_dict[i]
#getting the top 100 ingredients

top100=[]

for key in sorted_dict.keys():

    top100.append(key)

    if len(top100)==100:

        break
top100
# extracted into csv

#creating a new dataset for Machine Learning

# mlDict = {}

# for top_ingredient in top100:

#     mlDict[top_ingredient]=[]

#     for i in range(0,39774):

#         if top_ingredient in everything.iloc[i]['ingredients']:

#             mlDict[top_ingredient].append(1)

#         else:

#             mlDict[top_ingredient].append(0)
# cuisine = []

# id_=[]

# for i in range(0,39774):

#         cuisine.append(everything.iloc[i]['cuisine'])

#         id_.append(everything.iloc[i]['id'])

# mlDict['id']=id_

# mlDict['cuisine']= cuisine

# mlDF=pd.DataFrame(mlDict)

# mlDF['cuisine'] = mlDF['cuisine'].astype('category')

# mlDF[top100] = mlDF[top100].astype('category')
# extract into CSV file

#machine_learning_csv = mlDF.to_csv (r'C:\Users\limka\OneDrive\Documents\NTU\Y1S2\CZ1015\Mini Project\machine_learning_csv.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path
mlDF = pd.read_csv('../input/cooking/kaggle cooking/machine_learning_csv.csv')

mlDF
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.tree import export_graphviz

import graphviz



# Recall the Legendary-Total Dataset

y_train = pd.DataFrame(mlDF['cuisine'])   # Response

X_train = pd.DataFrame(mlDF[top100])       # Predictor



# Decision Tree using Train Data

dectree = DecisionTreeClassifier(max_depth = 2)  # create the decision tree object

dectree.fit(X_train, y_train)                    # train the decision tree model



# Predict Response corresponding to Predictors

y_train_pred = dectree.predict(X_train)



# Check the Goodness of Fit (on Train Data)

print("Goodness of Fit of Model \tTrain Dataset")

print("Classification Accuracy \t:", dectree.score(X_train, y_train))

print()



# Plot the Decision Tree

treedot = export_graphviz(dectree,                                      # the model

                          feature_names = X_train.columns,              # the features 

                          out_file = None,                              # output file

                          filled = True,                                # node colors

                          rounded = True,                               # make pretty

                          special_characters = True)                    # postscript



graphviz.Source(treedot)

# Decision Tree using Train Data

dectree = DecisionTreeClassifier(max_depth = 100)  # create the decision tree object

dectree.fit(X_train, y_train)                    # train the decision tree model



# Predict Response corresponding to Predictors

y_train_pred = dectree.predict(X_train)



# Check the Goodness of Fit (on Train Data)

print("Goodness of Fit of Model \tTrain Dataset")

print("Classification Accuracy \t:", dectree.score(X_train, y_train))

print()

cuisine_pred = mlDF[mlDF["id"].isin(["11462", "40989", "27976", '22213','6487','25557','27976','1299'])]

cuisine_pred
# Extract Predictors for Prediction

X_pred = pd.DataFrame(cuisine_pred[top100])



# Predict Response corresponding to Predictors

y_pred = dectree.predict(X_pred)



# Summarize the Actuals and Predictions

y_pred = pd.DataFrame(y_pred, columns = ["PredType"], index = cuisine_pred.index)

predictedDF = pd.concat([cuisine_pred[['id','cuisine']], y_pred], axis = 1)



#predicting the cusine frome some of the recipe

predictedDF
from sklearn.ensemble import RandomForestClassifier



# Random Forest using Train Data

clf = RandomForestClassifier(n_jobs=2,n_estimators=100, random_state=0)  # create the  object

clf.fit(X_train, y_train)                    # train the model



y_train_pred = clf.predict(X_train)



print("Goodness of Fit of Model \tTrain Dataset")

print("Classification Accuracy \t:", clf.score(X_train, y_train))

print()

# Extract Predictors for Prediction

X_pred = pd.DataFrame(cuisine_pred[top100])



# Predict Response corresponding to Predictors

y_pred = clf.predict(X_pred)



# Summarize the Actuals and Predictions

y_pred = pd.DataFrame(y_pred, columns = ["PredType"], index = cuisine_pred.index)

predictedDF = pd.concat([cuisine_pred[['id','cuisine']], y_pred], axis = 1)



#predicting the cusine frome some of the recipe

predictedDF
#creating a test dataset for Machine Learning

testdict = {}

for top_ingredient in top100:

    testdict[top_ingredient]=[]

    for i in range(0,9944):

        if top_ingredient in test.iloc[i]['ingredients']:

            testdict[top_ingredient].append(1)

        else:

            testdict[top_ingredient].append(0)
id_=[]

for i in range(0,9944):

        id_.append(test.iloc[i]['id'])

testdict['id']=id_
testDF=pd.DataFrame(testdict)

testDF[top100] = testDF[top100].astype('category')
#classification Tree #predicting the cusine frome some of the recipe

cuisine_pred = testDF[testDF["id"].isin([ 36914,2280,14729,4594,2237,45631,45523,4977,7124,])]

X_pred = pd.DataFrame(cuisine_pred[top100])



# Predict Response corresponding to Predictors

y_pred = dectree.predict(X_pred)



# Summarize the Actuals and Predictions

y_pred = pd.DataFrame(y_pred, columns = ["PredType"], index = cuisine_pred.index)

predictedDF = pd.concat([cuisine_pred[['id']], y_pred], axis = 1)



predictedDF
#random forest #predicting the cusine frome some of the recipe

cuisine_pred = testDF[testDF["id"].isin([ 36914,2280,14729,4594,2237,45631,45523,4977,7124,])]

X_pred = pd.DataFrame(cuisine_pred[top100])



# Predict Response corresponding to Predictors

y_pred = clf.predict(X_pred)



# Summarize the Actuals and Predictions

y_pred = pd.DataFrame(y_pred, columns = ["PredType"], index = cuisine_pred.index)

predictedDF = pd.concat([cuisine_pred[['id']], y_pred], axis = 1)



predictedDF