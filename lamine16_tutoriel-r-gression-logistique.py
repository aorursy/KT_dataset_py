import pandas as pd
import numpy as np
import math
X_data = pd.read_csv('../input/train.csv')
X_data.head(3)
# Select only easy to use columns
X_data = X_data[['PassengerId','Survived', 'Pclass','Sex','Age','SibSp','Parch','Fare', 'Embarked']]
X_data = X_data.dropna()

Y_data = X_data['Survived']
passenger_ids = X_data['PassengerId']

X_data = X_data.drop(['PassengerId', 'Survived'], axis=1)
X_data.head(3)
gender_map = {'male': 1, 'female':0}
embarked_map = { 'S': -1, 'C': 0, 'Q': 1}

X_data = X_data.replace({"Sex": gender_map, "Embarked": embarked_map})
X_data.head(3)
X = X_data.values
X
X.shape
(rows, cols) = X.shape
Y = Y_data.values
Y = Y.reshape(rows,1)
Y.shape
def sigmoid(x):
  return 1 / (1 + np.exp(-x))
iterations = 10000
alpha = 0.005
W = np.random.randn(1,cols)
b = np.random.randn(1,1)

for i in range(0,iterations):
#     print("Weights : ", W)
#     print("Bias : ", b)
    A = sigmoid(np.dot(X,W.T)+ b)
    Z = A-Y
    if i%1000 == 999 :print("Iteration :", i, " Total loss :", np.sum(Z**2), "Total wrong : ", np.sum((Y - np.rint(A))**2))
    W -= alpha*(X.T.dot(Z).T)/rows
    b -= alpha*np.sum(Z)/(rows*cols)
X_test = pd.read_csv('../input/test.csv')
pass_ids_train = X_test[['PassengerId']]
X_test = X_test[['Pclass','Sex','Age','SibSp','Parch','Fare', 'Embarked']]

gender_map = {'male': 1, 'female':0}
embarked_map = { 'S': -1, 'C': 0, 'Q': 1}

X_test = X_test.replace({"Sex": gender_map, "Embarked": embarked_map})
X_test.head(6)
Y_test = sigmoid(np.dot(X_test,W.T)+ b)
Y_test
Y_test = np.nan_to_num(np.rint(Y_test))
Y_test
data_to_submit = pd.DataFrame(np.hstack((np.rint(pass_ids_train), np.rint(Y_test))),
                             columns=['PassengerId', 'Survived'])
data_to_submit.astype(int)
data_to_submit.astype(int).to_csv('csv_to_submit.csv', index = False)