# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

seed = 5
np.random.seed(seed)
tf.set_random_seed(seed)

# Any results you write to the current directory are saved as output.
#Import necessary dataframes
Teams = pd.read_csv('../input/Teams.csv')
Seasons = pd.read_csv('../input/Seasons.csv')
RegSeasResults = pd.read_csv('../input/RegularSeasonDetailedResults.csv')
TournResults = pd.read_csv('../input/NCAATourneyDetailedResults.csv')
TournSeeds = pd.read_csv('../input/NCAATourneySeeds.csv')
#List the variables of each
print(Teams.columns)
print(Seasons.columns)
print(RegSeasResults.columns)
print(TournResults.columns)
print(TournSeeds.columns)
#For each season with regular season statistics, aggregate all of the game statistics for the winning and losing team for each game played.
#These Statistics will be combined in later steps.
RegSeasStats1 = RegSeasResults.groupby(['WTeamID', 'Season']).agg({'WScore': ["mean", "count"], 'WFGM': ["mean", "count"], 'WFGA': ["mean", "count"], 'WFGM3': ["mean", "count"], 'WFGA3': ["mean", "count"], 
                                                       'WFTM': ["mean", "count"], 'WFTA': ["mean", "count"], 'WOR': ["mean", "count"], 'WDR': ["mean", "count"], 'WAst': ["mean", "count"], 'WTO': ["mean", "count"],
                                                      'WTO': ["mean", "count"], 'WStl': ["mean", "count"], 'WBlk': ["mean", "count"], 'WPF': ["mean", "count"]})
RegSeasStats2 = RegSeasResults.groupby(['LTeamID', 'Season']).agg({'LScore': ["mean", "count"], 'LFGM': ["mean", "count"], 'LFGA': ["mean", "count"], 'LFGM3': ["mean", "count"], 'LFGA3': ["mean", "count"], 
                                                       'LFTM': ["mean", "count"], 'LFTA': ["mean", "count"], 'LOR': ["mean", "count"], 'LDR': ["mean", "count"], 'LAst': ["mean", "count"], 'LTO': ["mean", "count"],
                                                      'LTO': ["mean", "count"], 'LStl': ["mean", "count"], 'LBlk': ["mean", "count"], 'LPF': ["mean", "count"]})
#Rename the index names so that they can be merged
RegSeasStats1.index.names = ['TeamID', 'Season']
RegSeasStats2.index.names = ['TeamID', 'Season']
#Join the aggregated dataframes by 'TeamID' and 'Season' so that each row is the winning/losing game statistics for each team for each season.
RegSeasStats = RegSeasStats1.join(RegSeasStats2, how = 'outer')
#Include calculated win percentage. 'GamesPlayed' will be used to compute the weighted averages of the winning/losing game statistics for each team for each season.
RegSeasStats['WinPercent'] = RegSeasStats['WScore']['count']/(RegSeasStats['WScore']['count'] + RegSeasStats['LScore']['count'])
RegSeasStats['GamesPlayed'] = RegSeasStats['WScore']['count'] + RegSeasStats['LScore']['count']

#Game statistics names for the combined list.
Stats = ['Score', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']
#Combine winning/losing game stats into season stats for each team.
for stat in Stats:
    RegSeasStats[stat] = (RegSeasStats['W' + stat]['mean']*RegSeasStats['W' + stat]['count'] + RegSeasStats['L' + stat]['mean']*RegSeasStats['L' + stat]['count'])/RegSeasStats['GamesPlayed']
    del RegSeasStats['W' + stat]
    del RegSeasStats['L' + stat]
RegSeasStats[0:10]
TournResults[0:10]
#The find function returns the index for each element of 'elem' in 'searchList'. This way we can extract the regular season statistics for every winning and losing team in the tournament and stack them.
find = lambda searchList, elem: [[i for i, x in enumerate(searchList) if x == e] for e in elem]
InputStats = {}
for season in TournResults.Season.unique(): 
    #The regular season statistics for all teams that won at least one game in the tournament for the current season. The same for teams that appeared in a tournament game and lost.
    WTeamStats = RegSeasStats.loc[(TournResults[TournResults.Season == season].WTeamID, [season]),]
    LTeamStats = RegSeasStats.loc[(TournResults[TournResults.Season == season].LTeamID, [season]),]
    #Use the 'find' function to find the regular season statistics index for each winning/losing team in the tournament (possibly indexing the same winning team multiple times that is not eliminated immediately).  
    idx1 = find(WTeamStats.index.get_level_values(level = 0), list(TournResults[TournResults.Season == season].WTeamID))
    idx2 = find(LTeamStats.index.get_level_values(level = 0), list(TournResults[TournResults.Season == season].LTeamID))   
    #Convert dataframes to matrices for concatenation into one 3-d array
    WTeamStats = WTeamStats.reset_index().as_matrix()
    LTeamStats = LTeamStats.reset_index().as_matrix()
    #Concatenate the winning and losing game statistics matrices so that every row corresponds to a particular matchup in the tournament.
    #Transpose the resulting matrix so that the first two dimensions are the team and statistics dimensions and the third dimension is for each game of the tournament.
    InputStats[season] = np.transpose(np.concatenate((WTeamStats[idx1,:], LTeamStats[idx2,:]), axis = 1))

    
#Example for the first game of the 2003 tournament between UNC Asheville (TeamID = 1421) and Texas Southern (TeamID = 1411)
InputStats[2003][:,:,0]
#Dictionary for the generated tournament matchup examples, vectors for the reshaped original and generated tournament matchup examples.
InputStats_alt = {} 
X = np.empty([1, 32])
X_alt = np.empty([1, 32])

for season in TournResults.Season.unique():
    #Delete the first two rows of every example which contain 'TeamID' and 'Season' values. These will not be fed into our machine learning models. 
    InputStats[season] = np.delete(InputStats[season], (0, 1), axis = 0)
    #Create the alternative dictionary with the position of Team 1 and Team 2 in each array reversed. These will be our examples labeled '0'.
    InputStats_alt[season] = InputStats[season]
    InputStats_alt[season] = np.flip(InputStats_alt[season], 1)
    #Stack the two team regular season statistics one on top of the other for both the original and alternative array
    InputStats[season] = np.transpose(np.reshape(InputStats[season], (32, InputStats[season].shape[2]), order = 'F'))
    InputStats_alt[season] = np.transpose(np.reshape(InputStats_alt[season], (32, InputStats_alt[season].shape[2]), order = 'F'))
    #Add these examples to the new input matrices. 
    X = np.concatenate((X, InputStats[season]), axis = 0)
    X_alt = np.concatenate((X_alt, InputStats_alt[season]), axis = 0)
X = np.delete(X, (0), axis = 0)
X_alt = np.delete(X_alt, (0), axis = 0)
#Create label vectors for the original and alternative examples.
y = np.ones((X.shape[0], 1))
y_alt = np.zeros((X_alt.shape[0], 1))
#Concatenate the original and alternative examples.
X = np.concatenate((X, X_alt), axis = 0)
y = np.concatenate((y, y_alt), axis = 0)

#Eliminate any examples with any 'nan' values.
index_complete = np.sum(np.isfinite(X), axis = 1) == 32
X = X[index_complete,:]
y = y[index_complete,:]

#Shuffle the examples and partition them into train and test sets on a 80:20 split.
train_index = np.random.choice(X.shape[0], round(X.shape[0] * 0.8), replace=False)
test_index = [i for i in range(X.shape[0]) if i not in train_index]
train_X = X[train_index,:]
train_y = y[train_index,:]
test_X = X[test_index,:]
test_y = y[test_index,:]
logisticRegr = LogisticRegression()
logisticRegr.fit(train_X, np.ravel(train_y))
predictions = logisticRegr.predict(test_X)
#Training accuracy
trainscore = logisticRegr.score(train_X, np.ravel(train_y))
print(trainscore)
# Testing accuracy
testscore = logisticRegr.score(test_X, np.ravel(test_y))
print(testscore)

from sklearn import metrics
cm = metrics.confusion_matrix(np.ravel(test_y), predictions)
print(cm)
learning_rate = 0.01
num_iter = 50000
Xtens = tf.placeholder(tf.float64, shape = (None, 32))
ytens = tf.placeholder(tf.float64, shape = (None, 1))
W = tf.Variable(tf.random_normal([32, 1], dtype = tf.float64)*0.01)
b = tf.zeros(shape = (1,1), dtype = tf.float64)

logits = tf.matmul(Xtens, W) + b

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = ytens, logits = logits))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)

prediction = tf.round(tf.sigmoid(logits))
correct = tf.cast(tf.equal(prediction, ytens), dtype=tf.float32)
accuracy = tf.reduce_mean(correct)

init = tf.global_variables_initializer()
cost_trace = []
train_acc = []
test_acc = []
with tf.Session() as sess:
    sess.run(init)
    
    for it in range(num_iter):
        _, temp_cost = sess.run([optimizer, cost], feed_dict = {Xtens: train_X, ytens: train_y})
        temp_train_acc = sess.run(accuracy, feed_dict={Xtens: train_X, ytens: train_y})
        temp_test_acc = sess.run(accuracy, feed_dict={Xtens: test_X, ytens: test_y})
        
        cost_trace.append(temp_cost)
        train_acc.append(temp_train_acc)
        test_acc.append(temp_test_acc)
        
        if (it + 1) % 100 == 0:
            print('iteration: {:4d} loss: {:5f} train_acc: {:5f} test_acc: {:5f}'.format(it + 1, temp_cost, temp_train_acc, temp_test_acc))

        
plt.plot(cost_trace)
plt.title('Cross Entropy Loss')
plt.xlabel('iteration of gradient descent')
plt.ylabel('cost')
plt.show()
plt.plot(train_acc, 'b-', label='train accuracy')
plt.plot(test_acc, 'k-', label='test accuracy')
plt.xlabel('iteration of gradient descent')
plt.ylabel('accuracy')
plt.title('Train and Test Accuracy')
plt.legend(loc='best')
plt.show()
learning_rate_nn = 5
num_epochs = 50000
batch_size = 30
Xtens = tf.placeholder(tf.float64, shape = (None, 32))
ytens = tf.placeholder(tf.float64, shape = (None, 1))

W1 = tf.Variable(tf.random_normal([32, 16], dtype = tf.float64)*0.01)
b1 = tf.zeros(shape = (1,16), dtype = tf.float64)
W2 = tf.Variable(tf.random_normal([16, 16], dtype = tf.float64)*0.01)
b2 = tf.zeros(shape = (1,16), dtype = tf.float64)
W3 = tf.Variable(tf.random_normal([16, 1], dtype = tf.float64)*0.01)
b3 = tf.zeros(shape = (1,1), dtype = tf.float64)

Z1 = tf.matmul(Xtens, W1) + b1
A1 = tf.nn.relu(Z1)

Z2 = tf.matmul(A1, W2) + b2
A2 = tf.nn.relu(Z2)

logits_nn = tf.matmul(A2, W3) + b3

cost_nn = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = ytens, logits = logits_nn))
optimizer_nn = tf.train.AdamOptimizer(learning_rate = learning_rate_nn).minimize(cost_nn)

prediction_nn = tf.round(tf.sigmoid(logits_nn))
correct_nn = tf.cast(tf.equal(prediction_nn, ytens), dtype=tf.float32)
accuracy_nn = tf.reduce_mean(correct_nn)

init_nn = tf.global_variables_initializer() 
cost_trace_nn = []
train_acc_nn = []
test_acc_nn = []
with tf.Session() as sess2:
    sess2.run(init_nn)
    
    for epoch in range(num_epochs):
#         for minibatch in range(math.ceil((train_X.shape[1] + 1)/batch_size)):
#             minibatch_X = train_X[:, minibatch*batch_size:min((minibatch + 1)*batch_size,train_X.shape[1])]
#             minibatch_y = train_y[:, minibatch*batch_size:min((minibatch + 1)*batch_size,train_X.shape[1])]
        _, temp_cost_nn = sess2.run([optimizer_nn, cost_nn], feed_dict = {Xtens: train_X, ytens: train_y})
        temp_train_acc_nn = sess2.run(accuracy_nn, feed_dict={Xtens: train_X, ytens: train_y})
        temp_test_acc_nn = sess2.run(accuracy_nn, feed_dict={Xtens: test_X, ytens: test_y})
        
        cost_trace_nn.append(temp_cost_nn)
        train_acc_nn.append(temp_train_acc_nn)
        test_acc_nn.append(temp_test_acc_nn)
        
        if (epoch + 1) % 100 == 0:
            print('epoch: {:4d} loss: {:5f} train_acc: {:5f} test_acc: {:5f}'.format(epoch + 1, temp_cost_nn, temp_train_acc_nn, temp_test_acc_nn))
from keras.models import Sequential
from keras.layers import Dense
# fix random seed for reproducibility
np.random.seed(7)

# create model
model = Sequential()
model.add(Dense(16, input_dim=32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(train_X, train_y, epochs=100, batch_size=10)

# evaluate the model
scores = model.evaluate(train_X, train_y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

testscores = model.evaluate(test_X, test_y)
print("\n%s: %.2f%%" % (model.metrics_names[1], testscores[1]*100))