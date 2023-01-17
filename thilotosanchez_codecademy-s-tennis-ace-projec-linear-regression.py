#import codecademylib3_seaborn

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
# load and investigate the data here:



players = pd.read_csv("../input/tennis_stats.csv")

players.head()
players.columns
players.describe()
# perform exploratory analysis here:

corrPlayers = players.corr()

corrPlayers.to_csv('corrPlayers.csv',index=True)

corrPlayers
plt.scatter(players['BreakPointsOpportunities'], players['Winnings'])

plt.title('BreakPointsOpportunities vs Winnings')

plt.xlabel('BreakPointsOpportunities')

plt.ylabel('Winnings')

plt.show()

plt.clf()



# Let's also play a little bit around to explore the data.



plt.scatter(players['FirstServePointsWon'], players['Winnings'])

plt.title('FirstServePointsWon vs Winnings')

plt.xlabel('FirstServePointsWon')

plt.ylabel('Winnings')

plt.show()

plt.clf()



plt.scatter(players['FirstServeReturnPointsWon'], players['Winnings'])

plt.title('FirstServeReturnPointsWon vs Winnings')

plt.xlabel('FirstServeReturnPointsWon')

plt.ylabel('Winnings')

plt.show()

plt.clf()
## single feature linear regression (FirstServeReturnPointsWon):



# select features and values to predict

features = players['FirstServeReturnPointsWon']

outcome = players['Winnings']



# train, test, split the data using scikit-learn's train_test_split function



features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size=0.8)

features_train = features_train.values.reshape(-1,1)

features_test = features_test.values.reshape(-1,1)



# create and train model on training data

model = LinearRegression()

model.fit(features_train, outcome_train)



# score model on test data

print('Predicted Winnings with FirstServeReturnPointsWon Test Score:', model.score(features_test, outcome_test))



# make predictions with model

prediction = model.predict(features_test)



# plot predictions against actual winnings

plt.scatter(outcome_test, prediction, alpha=0.4)

plt.title('Predicted Winnings vs. Actual Winnings - 1 Feature')

plt.xlabel('Actual Winnings')

plt.ylabel('Predicted Winnings')

plt.show()

plt.clf()
## perform another single feature linear regressions here (BreakPointsOpportunities):



# select features and value to predict

features = players['BreakPointsOpportunities']

outcome = players['Winnings']



# train, test, split the data using scikit-learn's train_test_split function



features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size=0.8)

features_train = features_train.values.reshape(-1,1)

features_test = features_test.values.reshape(-1,1)



# create and train model on training data



model = LinearRegression()

model.fit(features_train, outcome_train)



# score model on test data

print('Predicting Winnings with BreakPointsOpportunities Test Score:', model.score(features_test, outcome_test))



# make predictions with model

prediction = model.predict(features_test)



# plot predictions against actual winnings

plt.scatter(outcome_test, prediction, alpha=0.4)

plt.title('Predicted Winnings vs. Actual Winnings - 1 Feature')

plt.xlabel('Actual Winnings')

plt.ylabel('Predicted Winnings')

plt.show()

plt.clf()
## perform two feature linear regressions here:



# select features and values to predict



features = players[['BreakPointsOpportunities', 'FirstServeReturnPointsWon']]

outcome = players['Winnings']



# train, test, split the data



features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size=0.8)



# create and train model on data



model = LinearRegression()

model.fit(features_train, outcome_train)



# score model on test data



print('Predicted Winnings with 2 Features Test Score:', model.score(features_test, outcome_test))



# make predictions with model



prediction = model.predict(features_test)



# plot predictions against actual winnings



plt.scatter(outcome_test, prediction, alpha=0.4)

plt.title('Predicted Winnings vs. Actual Winnings - 2 Features')

plt.xlabel('Actual Winnings')

plt.ylabel('Predicted Winnings')

plt.show()

plt.clf()
## perform multiple feature linear regressions here:



# select features and values to predict



features = players[['FirstServe','FirstServePointsWon','FirstServeReturnPointsWon','SecondServePointsWon','SecondServeReturnPointsWon','Aces','BreakPointsConverted','BreakPointsFaced','BreakPointsOpportunities','BreakPointsSaved','DoubleFaults','ReturnGamesPlayed','ReturnGamesWon','ReturnPointsWon','ServiceGamesPlayed','ServiceGamesWon','TotalPointsWon','TotalServicePointsWon']]

outcome = players['Winnings']



# train, test, split the data



features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size=0.8)



# create and train model on training data



model = LinearRegression()

model.fit(features_train, outcome_train)



# score model on test data



print('Predicted Winnings vs. Multiple Features Test Score:', model.score(features_test, outcome_test))



# make predictions with model



prediction = model.predict(features_test)



# plot predictions against actual winnings



plt.scatter(outcome_test, prediction, alpha=0.4)

plt.title('Predicted Winnings vs. Actual Winnings - Multiple Features')

plt.xlabel('Acutal Winnings')

plt.ylabel('Predicted Winnings')

plt.show()

plt.clf()