# importing pandas for data pre-processing

import sys

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
def process(x):

    if x == "male": return 1

    else: return 2
# loading trainig data

train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = train.sample(frac=1).reset_index(drop=True)

test = test.sample(frac=1).reset_index(drop=True)

train.head()
#extra columns are removed

train = train.drop(["PassengerId", "Name", "Ticket", "Cabin", "Fare"], axis = 1)

test_x = test.drop(["PassengerId", "Name", "Ticket", "Cabin", "Fare"], axis = 1)

test_x.tail()
#replace all NaN with mean

train = train.fillna(train.mean())

test_x = test_x.fillna(test_x.mean())
#Encoding Sex column

#DataFrame.apply(self, func, axis=0, raw=False, result_type=None, args=(), **kwds)

train["Sex"] = train["Sex"].apply(process)

test_x["Sex"] = test_x["Sex"].apply(process)
#Encoding Embarked column

train["Embarked"] = train["Embarked"].apply(lambda x: 0 if x == "C" else (1 if x == "Q" else 2))

test_x["Embarked"] = test_x["Embarked"].apply(lambda x: 0 if x == "C" else (1 if x == "Q" else 2))
#encoding age

train["Age"] = train["Age"].apply(lambda x : 1 if x in range(0, 11) else (2 if x in range(11, 21) else (3 if x in range(21, 31) else(4 if x in range(31, 41) else(5 if x in range(41, 51) else (6 if x in range(51, 61) else 7))))))

test_x["Age"] = test_x["Age"].apply(lambda x : 1 if x in range(0, 11) else (2 if x in range(11, 21) else (3 if x in range(21, 31) else(4 if x in range(31, 41) else(5 if x in range(41, 51) else (6 if x in range(51, 61) else 7))))))
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1

test_x["FamilySize"] = test_x["SibSp"] + test_x["Parch"] + 1
train["IsAlone"] = 0

train["IsAlone"] = train["FamilySize"].apply(lambda x: 1 if x == 1 else 0)

test_x["IsAlone"] = 0

test_x["IsAlone"] = test_x["FamilySize"].apply(lambda x: 1 if x == 1 else 0)

train.head()
# # applying min-max normalization

# train = (train - train.min()) / (train.max() - train.min())

# test_x = (test_x - test_x.min()) / (test_x.max() - test_x.min())
# separating  x, y

train_x = train.iloc[ : , 1 : ]

train_y = train.iloc[ : , 0 : 1]

train_x, validation_x = train_x.iloc[ : int(.9 * len(train_x)), : ], train_x.iloc[int(.9 * len(train_x)): , : ]

train_y, validation_y = train_y.iloc[ : int(.9 * len(train_y)), : ], train_y.iloc[int(.9 * len(train_y)): , : ]

train_x.tail()
# exporting to numpy

train_x = train_x.values

train_y = train_y.values

test_x = test_x.values

validation_x = validation_x.values

validation_y = validation_y.values
# prediction hypothesis

def predict(feature, weight, bias = 0):

    """

    features: numpy array

    shape: (712, 8)

    

    weight: numpy array

    shape: (8, 1)

    

    """

    

    """

    return: numpy array

    shape: np.dot((712, 8), (8, 1))

         : (712, 1)

    """

    return sigmoid(np.dot(feature, weight) + bias)
# simoid function

def sigmoid(z):

    """

    z: numpy array

    shape:(712, 1)

    returns the scalar which is in between 0 and 1

    """

    return (1 / (1 + np.exp(-z)))
# cost function

def log_cost(prediction, label, weight):

    """

    prediction: numpy array

    shape: (712, 1)

    

    label: numpy array

    shape: (712, 1)

    """

    

    """

    return: scalar

    shape: np.sum((712, 1) * (712, 1) - (712, 1) * (712, 1))

         : np.sum(712, 1)

         : 0

    """

    lmd = 10

    return (np.sum((-label * (np.log(prediction))) - ((1 - label) * np.log(1 - prediction))) / len(label)) + lmd/(2*len(label)) * sum(weight**2)
# gradient decent

def update_weight(feature, weight, prediction, label, learning_rate = 0.001):

    """

    feature: numpy array

    shape: (712, 8)

    

    weight: numpy array

    shape: (8, 1)

    

    learning_rate: number

    shape: 0

    

    prediction: numpy array

    shape: (712, 1)

    

    label: numpy array

    shape: (712, 1)

    """

    

    """

    weight: numpy array

    shape: 

       : (8, 1) - 0.001 * (np.dot((8, 712), ((712, 1) - (712, 1))))

       : (8, 1) - 0.001 * (np.dot((8, 712), (712, 1)))

       : (8, 1) - 0.001 * (8, 1)

       : (8, 1)

    """

    weight -= (learning_rate * ((np.dot(feature.T, (prediction - label))) / len(feature)))

    return weight
# training function

def train(feature, weight, label, epoc = 30):

    """

    feature: numpy array

    shape: (712, 8)

    

    weight: numpy array

    shape: (8, 1)

    

    label: numpy array

    shape: (712, 1)

    """

    

    cost  = 0

    for i in range(0, epoc):

        

        #prediction: numpy array

        #shape: (712, 1)

        prediction = predict(feature, weight)

        

        #weight: numpy array

        #shape: (1, 8)

        weight = update_weight(feature, weight, prediction, label, learning_rate)

        

    #prediction: numpy array

    #shape: (712, 1)

    prediction = predict(feature, weight)

    

    #cost: scalar

    #shape: 0

    cost = log_cost(prediction, label, weight)

    

    return weight, cost

        

    
#initializing learning rate

learning_rate = 0.4



epoc = 40



#initalizing weight in the range of (-1, 1)

#weight: numpy array

#shape: (8, 1)

weight = np.random.rand(8, 1)



validation_cost_history = []

training_cost_history = []

#feature:shape: (712, 8)

#label:shape: (712, 1)

#weight:shape: (8, 1)



for each in [i * 100 for i in range(1, 9)]:

    weight, cost = train(train_x[ : each, :], weight, train_y[ : each, :], epoc)

    training_cost_history.append(cost)

    prediction = predict(validation_x, weight)

    cost = log_cost(prediction, validation_y, weight)

    validation_cost_history.append(cost)

    

plt.plot([i * 100 for i in range(1, 9)], training_cost_history, color = "orange", label="training cost")

plt.plot([i * 100 for i in range(1, 9)], validation_cost_history, color = "blue", label="validation cost")

plt.title("Learning Curve")

plt.xlabel("Training Set Size"), plt.ylabel("Cost"), plt.legend(loc="best")

plt.tight_layout()

plt.show()

print("training completed")
prediction = predict(test_x, weight)

prediction = pd.Series(prediction[ : , 0])

prediction = prediction.apply(lambda x: 1 if x >= 0.5 else 0)

pId = test.iloc[ : , 0]



frame = {"PassengerId": pId, "Survived": prediction}



gender_submission = pd.DataFrame(frame)



gender_submission.head()



gender_submission.to_csv("./gender_submission.csv", index = False, header = True)



print(weight.reshape(8,))