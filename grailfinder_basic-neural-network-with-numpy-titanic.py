'''how to basic neural network with numpy and titanic'''

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale

# Input data files are available in the "../input/" directory.
df_train_full = pd.read_csv("../input/train.csv", index_col="PassengerId")
df_test = pd.read_csv("../input/test.csv", index_col="PassengerId")


# don't really care about these
df_train = df_train_full.drop(['Name', 'Ticket', 'Cabin', 'Embarked', 'Survived'], axis=1)
df_train.info()
df_train.head()
# we need sex to be int
for dataset in (df_train, df_test):
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    
# and what about missing values?
for col in df_train.columns.values:
    print(f"any nan in{col}?:", df_train[col].isnull().any())
# looks like age is our only problem
df_train.Age.fillna(df_train.Age.mean(), inplace=True) # its a lazy example, so mean is fine
print(df_train['Age'].isnull().any())
# https://en.wikipedia.org/wiki/Sigmoid_function
def sigmoid(x, derivative=False):
    if derivative:
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))
    
X = scale(df_train)  # always scale
y = df_train_full['Survived'].values
print(X.shape, y.shape)

# nice, but we need to reshape y, to make it dot productable
y = y.reshape(y.size, 1)

# split the data to get accuracy
eighty = round(len(X) * .8)
X_train, X_test = X[:eighty], X[eighty:]
y_train, y_test = y[:eighty], y[eighty:]
print(X_train.shape, y_test.shape)


# weight
np.random.seed(1) # for repeatable outcome
weight = np.random.random((6, 1)) # only part that changes over time

for i in range(1000):
    inp_layer = X_train
    # dot production on input data and weights and use activation function
    out_layer = sigmoid(np.dot(inp_layer, weight))
    # get a err size
    err = y_train - out_layer
    # backpropagation, use err to correct weights
    d_out = err * sigmoid(out_layer, derivative=True)
    weight += np.dot(inp_layer.T, d_out)
    
# now our nn is trained (we got expirienced weights)
y_pred = sigmoid(np.dot(X_test, weight))

print(y_test.shape, y_pred.shape)
# softmax
y_pred[y_pred > 0.5] = 1
y_pred[y_pred < 0.5] = 0

print("accuracy on test data: ", round(accuracy_score(y_test[:, 0], y_pred[:, 0]), 2))
# yay, prediction level more than 50%, time to celebrate