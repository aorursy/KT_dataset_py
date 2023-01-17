import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
def sigmoid(z):

    return 1/(1+np.e**(-z))



def predict(X, theta):

    z = np.dot(X, theta)

    return sigmoid(z)



def cost_function(X, Y, theta):

    m = len(Y)

    h = predict(X,theta)

    

    cost = (np.dot((-Y.T), np.log(h)) - np.dot((1-Y).T, np.log(1-h))) / m

    return cost



def gradient_descent(X, Y, theta, rate):

    m = len(Y)

    h = predict(X, theta)



    gradient = rate * np.dot(X.T, (h-Y)) / m

    theta -= gradient

    return theta



def decision_boundary(probability):

    classification = 1 if probability >= 0.5 else 0

    return classification



def classify(probabilities):

    return np.vectorize(decision_boundary)(probabilities).flatten()



def train(X, Y, theta, rate, iters):

    cost_history = [2, 1]

    

    for i in range(iters):

        theta = gradient_descent(X, Y, theta, rate)

        

        cost = cost_function(X, Y, theta)

        cost_history.append(cost)

        

        if i % 10000 == 0:

            print("iter: " + str(i) + " cost: "+str(cost))

    return theta, cost_history



def normalize(df, column):

    df[column] = (df[column] -df[column].mean()) / np.std(df[column]) # zscore
passengers = pd.read_csv('../input/train.csv')

passengers.head()
passengers.isna().sum()
# convert categorical variables in factors

X = pd.get_dummies(passengers[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']], drop_first=True)

Y = passengers[['Survived']]



# median fill for na values

X['Age'] = X['Age'].fillna(X['Age'].mean())

X['Embarked_S'] = X['Embarked_S'].fillna(X['Embarked_S'].mean())

X['Embarked_Q'] = X['Embarked_Q'].fillna(X['Embarked_Q'].mean())



# normalize values (zscore)

for column in X.columns:

    normalize(X, column)



# add constant for matrix multiplication

constant = np.ones((X.shape[0],1))

X = np.concatenate((constant, X), axis=1)



# initialize theta to zeros vector

theta = np.zeros((X.shape[1],1))
X # design matrix
trained_theta, cost_history = train(X, Y, theta, 0.0001, 100001)
probabilities = predict(X, trained_theta)

predictions = classify(probabilities)

passengers['prediction'] = predictions
 # percent correct in training

len(passengers[passengers['Survived'] == passengers['prediction']]) / len(passengers)
passengers_test = pd.read_csv('../input/test.csv')
passengers_test.isna().sum()
# convert categorical variables in factors

X_test = pd.get_dummies(passengers_test[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']], drop_first=True)



# median fill for na values

X_test['Age'] = X_test['Age'].fillna(X_test['Age'].mean())

X_test['Fare'] = X_test['Fare'].fillna(X_test['Fare'].mean())



# normalize values (zscore)

for column in X_test.columns:

    normalize(X_test, column)



# add constant for matrix multiplication

constant = np.ones((X_test.shape[0],1))

X_test = np.concatenate((constant, X_test), axis=1)
X_test
test_probabilities = predict(X_test, trained_theta)

test_predictions = classify(test_probabilities)

passengers_test['Survived'] = test_predictions
passengers_test.head()