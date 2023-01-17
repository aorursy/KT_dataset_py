import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

from sklearn.metrics import accuracy_score, mean_squared_error, log_loss

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from matplotlib.colors import LinearSegmentedColormap
data = pd.read_csv('../input/heart.csv')
data.head()
X = data.drop(columns='target')

Y = data['target']
X.head()
Y.head()
print(X.shape, Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, stratify=Y)
print(X_train.shape, Y_train.shape)

print(X_test.shape, Y_test.shape)
X_train = X_train.values

X_test = X_test.values

Y_train = Y_train.values

Y_test = Y_test.values
std_scaler = StandardScaler()
X_scaled_train = std_scaler.fit_transform(X_train)
X_scaled_test = std_scaler.transform(X_test)
minmax_scaler = MinMaxScaler()
Y_scaled_train = minmax_scaler.fit_transform(Y_train.reshape(-1, 1))
Y_scaled_test = minmax_scaler.transform(Y_test.reshape(-1, 1))
print(Y_scaled_train.min(), Y_scaled_train.max())

print(Y_scaled_test.min(), Y_scaled_test.max())
class SigmoidNeuron:

    def __init__(self):

        self.w = None

        self.b = None

    

    def perceptron(self, x):

        return np.dot(x, self.w.T) + self.b

    

    def sigmoid(self, x):

        return 1.0/(1.0 + np.exp(-x))

    

    def grad_w_mse(self, x, y):

        y_pred = self.sigmoid(self.perceptron(x))

        return (y_pred - y) * y_pred * (1 - y_pred) * x

    

    def grad_b_mse(self, x, y):

        y_pred = self.sigmoid(self.perceptron(x))

        return (y_pred - y) * y_pred * (1 - y_pred)

    

    def grad_w_ce(self, x, y):

        y_pred = self.sigmoid(self.perceptron(x))

        if y == 0:

            return y_pred * x

        elif y == 1:

            return -1 * (1 - y_pred) * x

        else:

            raise ValueError("y should be 0 or 1")

        

    def grad_b_ce(self, x, y):

        y_pred = self.sigmoid(self.perceptron(x))

        return (-1 * (1 - y_pred)) if y ==1 else y_pred

            

    def fit(self, X, Y, epochs=1, learning_rate=1, initialise=True, loss_fn="mse", display_loss=False):

    

        # initialise w, b

        if initialise:

            self.w = np.random.randn(1, X.shape[1])

            self.b = 0

            

        if display_loss:

            loss = {}

        

        for i in tqdm_notebook(range(epochs), total=epochs, unit="epoch"):

            dw = 0

            db = 0

            for x, y in zip(X, Y):

                if loss_fn == "mse":

                    dw += self.grad_w_mse(x, y)

                    db += self.grad_b_mse(x, y) 

                elif loss_fn == "ce":

                    dw += self.grad_w_ce(x, y)

                    db += self.grad_b_ce(x, y)



                self.w -= learning_rate * dw

                self.b -= learning_rate * db

                

            if display_loss:

                Y_pred = self.sigmoid(self.perceptron(X))

                if loss_fn == "mse":

                    loss[i] = mean_squared_error(Y, Y_pred)

                elif loss_fn == "ce":

                    loss[i] = log_loss(Y, Y_pred)

        

        if display_loss:

            plt.plot(loss.values())

            plt.xlabel('Epochs')

            if loss_fn == "mse":

                plt.ylabel('Mean Squared Error')

            elif loss_fn == "ce":

                plt.ylabel('Log Loss')

            plt.show()

            

    def predict(self, X):

        Y_pred = [self.sigmoid(self.perceptron(x)) for x in X]

        return np.array(Y_pred)

    

    def predict_accuracy(self, X, Y, threshold=0.5):

        Y_pred = [self.sigmoid(self.perceptron(x)) for x in X]

        Y_pred = np.array(Y_pred)

        Y_pred = (Y_pred >= threshold).astype(int)

        

        Y = (Y >= threshold).astype(int)

        

        return accuracy_score(Y_pred, Y)

        
# Training the model using Mean Squared Error loss function

sn_mse = SigmoidNeuron()

sn_mse.fit(X_scaled_train, Y_scaled_train, epochs=30000, learning_rate=0.015, loss_fn="mse", display_loss=True)
# Training the model usign Cross Entropy loss function

sn_ce = SigmoidNeuron()

sn_ce.fit(X_scaled_train, Y_scaled_train, epochs=30000, learning_rate=0.015, loss_fn="ce", display_loss=True)
print('MSE Training Accuracy-->', sn_mse.predict_accuracy(X_scaled_train, Y_scaled_train))

print('CE Training Accuracy-->', sn_ce.predict_accuracy(X_scaled_train, Y_scaled_train))
print('MSE Training Accuracy-->', sn_mse.predict_accuracy(X_scaled_train, Y_scaled_train))

print('MSE Test Accuracy-->', sn_mse.predict_accuracy(X_scaled_test, Y_scaled_test))

print('--'*40)

print('CE Training Accuracy-->', sn_ce.predict_accuracy(X_scaled_train, Y_scaled_train))

print('CE Test Accuracy-->', sn_ce.predict_accuracy(X_scaled_test, Y_scaled_test))