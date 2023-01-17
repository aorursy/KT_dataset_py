import numpy as np

import pandas as pd
train_x = pd.DataFrame([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])

train_x.columns = ["X1", "X2"]

train_x
train_y = pd.Series(["A","A","B","B"])

train_y
train_y = train_y.map({"A":0, "B":1})

train_y
test_x = pd.DataFrame([[0.2,0.1],[0.0,0.1],[1,1],[1.2,1.0]])

test_x.columns = ["X1", "X2"]

test_x
test_y = pd.Series(["B","B","A","A"])

test_y
import operator

def predict(train_x, train_y, test_x, k):

    distance = ((train_x - test_x)**2).sum(axis=1)**0.5

    sorted_index = distance.argsort()

    

    voting_result = {}

    # Voting process: every number under k, cast a vote according to its label

    for i in range(k):

        vote = train_y[sorted_index[i]]

        if vote in voting_result:

            voting_result[vote] += 1

        else:

            voting_result[vote] = 1

            

    # Sort result and take the largest        

    voting_result = sorted(voting_result.items(), key=operator.itemgetter(1))

    return voting_result[0][0]



def predict_all(train_x, train_y, test_x, k):

    result = []

    for index, row in test_x.iterrows():

        result.append(predict(train_x, train_y, row, k))

    return result   

#predict(train_x, train_y, test_x.iloc[1,:], 4)

predict_all(train_x, train_y, test_x, 4)