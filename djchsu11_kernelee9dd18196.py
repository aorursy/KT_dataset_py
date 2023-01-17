import pandas as pd

import numpy as np



#Import and sample the data

X = pd.read_csv('../input/recipeData.csv', encoding='latin1')

y = pd.read_csv('../input/results.csv', encoding='latin1')



X.sample()
X.describe()
y.sample()
percent_y_nan = y.isna().sum()



print("{0:.2f}%".format(percent_y_nan['ratings'] / y.shape[0] * 100))
y = y['ratings']



#Get null indices

indices = y[y.isna() == True]

indices = indices.index.values



#Drop NA ratings

X = X.drop(indices, axis=0)

y = y.drop(indices, axis=0)

print(y.shape[0])

print("Number of NA remaining in y: {}".format(y.isna().sum()))
import matplotlib.pyplot as plt

import numpy as np

#Plot OG and ABV

plt.scatter(X['OG'].values, X['ABV'].values, c='r')

plt.xlabel('Original Gravity')

plt.ylabel('Alcohol By Volume')

plt.show()
print("Rows of X using plato gravity: \n{}".format(X.loc[X['SugarScale'] == 'Plato', 'OG']))
#Convert plato OG and FG to specific gravity



X.loc[X['SugarScale'] == 'Plato', 'OG'] = 259/(259 - X.loc[X['SugarScale'] == 'Plato', 'OG'])

X.loc[X['SugarScale'] == 'Plato', 'FG'] = 259/(259 - X.loc[X['SugarScale'] == 'Plato', 'FG'])

X.loc[X['SugarScale'] == 'Plato', 'BoilGravity'] = 259/(259 - X.loc[X['SugarScale'] == 'Plato', 'BoilGravity'])



print("Converted OG -> specific gravity: \n{}".format(X.loc[X['SugarScale'] == 'Plato', 'OG']))

print("Converted FG -> specific gravity: \n{}".format(X.loc[X['SugarScale'] == 'Plato', 'FG']))

print("Converted BG -> specific gravity: \n{}".format(X.loc[X['SugarScale'] == 'Plato', 'BoilGravity']))
#Remake plot

plt.scatter(X['OG'].values, X['ABV'].values, c='r')

plt.xlabel('Original Gravity')

plt.ylabel('Alcohol By Volume')

plt.show()



plt.scatter(X['FG'].values, X['ABV'].values, c='b')

plt.xlabel('Final Gravity')

plt.ylabel('Alcohol By Volume')

plt.show()



plt.scatter(X['OG'].values, X['FG'].values, c='g')

plt.xlabel('Original Gravity')

plt.ylabel('Final Gravity')

plt.show()
y = y.drop(X.loc[X['ABV'] > 30].index, axis=0)

X = X.drop(X.loc[X['ABV'] > 30].index, axis=0)

print(X.loc[X['ABV'] > 30])
#Remake plot

plt.scatter(X['OG'].values, X['ABV'].values, c='r')

plt.xlabel('Original Gravity')

plt.ylabel('Alcohol By Volume')

plt.show()



plt.scatter(X['FG'].values, X['ABV'].values, c='b')

plt.xlabel('Original Gravity')

plt.ylabel('Alcohol By Volume')

plt.show()



plt.scatter(X['OG'].values, X['FG'].values, c='g')

plt.xlabel('Original Gravity')

plt.ylabel('Alcohol By Volume')

plt.show()
#Histograms for all numeric columns in X

X[X.dtypes[(X.dtypes=="float64")|(X.dtypes=="int64")]

                        .index.values].hist(figsize=[11,11])
#Drop unneeded columns and outliers

X = X.drop(['UserId', 'StyleID', 'BeerID','Name', 'URL', 'Style', 'MashThickness', 'PitchRate', 'PrimingMethod', 'PrimingAmount', 'BrewMethod', 'SugarScale'], axis=1)



y = y.drop(X.loc[X['BoilSize'] > 60].index, axis=0)

X = X.drop(X.loc[X['BoilSize'] > 60].index, axis=0)



y = y.drop(X.loc[X['ABV'] > 20].index, axis=0)

X = X.drop(X.loc[X['ABV'] > 20].index, axis=0)



y = y.drop(X.loc[X['IBU'] > 150].index, axis=0)

X = X.drop(X.loc[X['IBU'] > 150].index, axis=0)



X = X.drop(['Size(L)'], axis=1)



y = y.drop(X.loc[X['MashThickness'] > 10].index, axis=0)

X = X.drop(X.loc[X['MashThickness'] > 10].index, axis=0)

X.shape
X = X.fillna(X.mean())

X.isna().sum()
#Histograms for all numeric columns in X

X[X.dtypes[(X.dtypes=="float64")|(X.dtypes=="int64")]

                        .index.values].hist(figsize=[11,11])
X.to_csv('sanitizedRecipeData.csv')

y.to_csv('sanitizedResults.csv', header=['results'])