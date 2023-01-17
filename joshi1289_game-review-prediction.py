import pandas 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression



# Read in the data.

games = pandas.read_csv("../input/games.csv")

# Print the names of the columns in games.

print(games.columns)

print(games.shape)



# Make a histogram of all the ratings in the average_rating column.

plt.hist(games["average_rating"])



# Show the plot.

plt.show()
import numpy as np


games.info()
# finding the nulll values 

games.isnull().sum()
# finding the colums that had average _rating=0 it means these games were never played 

data=games.loc[games['average_rating']==0]

data[["average_rating","yearpublished"]]
#first we need to remove the nan values b replacing it with -1

games=games.replace(np.nan,-1)

games.isnull().sum()
# removing the null values data from datset

games=games.loc[games['yearpublished']>=0]

games=games.loc[games['minplayers']>=0]

games=games.loc[games['maxplayers']>=0]

games=games.loc[games['playingtime']>=0]

games=games.loc[games['minplaytime']>=0]

games=games.loc[games['maxplaytime']>=0]

games=games.loc[games['minage']>=0]
# now we need to remove the enteries of year published which is 0 becuase it means game is never published

print(games['yearpublished'].unique())

games=games.loc[games['yearpublished']>0]

print(games['yearpublished'].unique())

# as you can see the enteries with yearpublished=0 has been removed
# mapping the type of games

print(games['type'].unique())

games['type']=games['type'].map({"boardgame":1,"boardgameexpansion":2})

print(games['type'].unique())

print(games.shape)
# playing with various averages and seeing them for any discrepancy

# lets find out if all the  averages provided=0,maxplayng time=0 and user rated=0 that means game is never played

d=games.loc[(games['bayes_average_rating']==0.0) & (games['average_rating']==0) & (games['users_rated']==0) & (games['playingtime']==0)]

print(d.shape)

d[['bayes_average_rating','average_rating','users_rated','playingtime','maxplaytime']]

# so u can clearly see that our dataset has such 7000+ enteries that must be removed to acheive generalisation 
games.describe()

print(games.shape)
# removing thee inconsistency of data

games=games.loc[((games['average_rating']>0) & (games['playingtime']>0))]

games.shape
games.describe()
plt.scatter(games['users_rated'],games['average_rating'],color='g')

plt.plot(games['average_rating'],games['bayes_average_rating'],color='r')

plt.show()
# Make a histogram of all the ratings in the average_rating column to ensure that most of rating with 0 have been removed and 

#only genuine cases have been left

plt.hist(games["average_rating"])



# Show the plot.

plt.show()
#correlation matrix

corrmat = games.corr()

fig = plt.figure(figsize = (12, 9))

sns.heatmap(corrmat, square=True);

plt.show()
# Get all the columns from the dataframe.

columns = games.columns.tolist()

# Filter the columns to remove ones we don't want.

columns = [c for c in columns if c not in ["average_rating", "name", "id"]]



# Store the variable we'll be predicting on.

target = "average_rating"
print(columns)
# Import a convenience function to split the sets.

from sklearn.model_selection import train_test_split



# Generate the training set.  Set random_state to be able to replicate results.

train = games.sample(frac=0.8, random_state=1)

# Select anything not in the training set and put it in the testing set.

test = games.loc[~games.index.isin(train.index)]

# Print the shapes of both sets.

print(train.shape)

print(test.shape)
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score
# Import the linear regression model.

from sklearn.linear_model import LinearRegression



# Initialize the model class.

model = LinearRegression()

# Fit the model to the training data.

model.fit(train[columns], train[target])



# Import the scikit-learn function to compute error.

from sklearn.metrics import mean_squared_error



# Generate our predictions for the test set.

predictions = model.predict(test[columns])



# Compute error between our test predictions and the actual values.

print("Mean square Error : ",mean_squared_error(predictions, test[target]))

# SInce is predicted data is continuous so we cant find the accuracy and classification _report 

# Therefore calculating the mean squared error is best measure
# Import the random forest model.

from sklearn.ensemble import RandomForestRegressor



# Initialize the model with some parameters.

model = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)

# Fit the model to the data.

model.fit(train[columns], train[target])

# Make predictions.

predictions = model.predict(test[columns])

# Compute the error.

mean_squared_error(predictions, test[target])
x=pandas.DataFrame({'Prediction':predictions})

x
# If u find it helpful please do upvote this kernel