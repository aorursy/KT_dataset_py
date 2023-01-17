import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



# Import dataset

data = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

# Define good quality as >5



x = data.drop('quality',axis=1).values

y = data['quality'].values

y[y<=5] = 0

y[y>5] = 1

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
weights_size = len(data.columns)-1

W = np.random.uniform(-0.01, 0.01, weights_size)

#W = np.zeros(weights_size)

W
def sign(x):

    return (1 if x>=0 else 0)



def predict(x,W):

    y = W.T.dot(x)

    return sign(y)



def train(X,Y):

    global W

    epochs = 15

    rate = 0.1

    for _ in range(epochs):

        for i in range(len(X)):

            x = X[i]

            y = predict(x,W)

            error = Y[i]-y

            W = W + rate * error * x



train(X_train,y_train)

accuracy_score(y_test,[predict(x,W) for x in X_test])