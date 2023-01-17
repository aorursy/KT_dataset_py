import numpy as np
class LogisticRegression:

    EPS = 1e-5

    def __ols_solve(self, x, y):

        # uses the closed-form formula

        rows, cols = x.shape

        if rows >= cols == np.linalg.matrix_rank(x):

            y = np.maximum(self.EPS, np.minimum(y.astype(np.float32), 1-self.EPS))

            ols_y = -np.log(np.divide(1, y) - 1)

            self.weights = np.matmul(

                np.matmul(

                    np.linalg.inv(

                        np.matmul(x.transpose(), x)

                    ),

                    x.transpose()),

                ols_y)

        else:

            print('Error! X has not full column rank.')

    

    def __sgd(self, x, y, grad_fn, learning_rate, iterations, batch_size):

        rows, cols = x.shape

        self.weights = np.random.normal(scale=1.0/cols, size=(cols, 1))

        num_batches = int(np.ceil(rows/batch_size))

        

        for i in range(iterations):

            xy = np.concatenate([x, y], axis=1)

            np.random.shuffle(xy)

            x, y = xy[:, :-1], xy[:, -1:]

            for step in range(num_batches):

                start, end = batch_size*step, np.min([batch_size*(step+1), rows])

                xb, yb = x[start:end], y[start:end]

                

                grads = grad_fn(xb, yb)

                

                self.weights -= learning_rate*grads

    

    def __sse_grad(self, xb, yb):

        # computes the gradient of the Sum of Squared Errors loss

        yb = np.maximum(self.EPS, np.minimum(yb.astype(np.float32), 1-self.EPS))

        ols_yb = -np.log(np.divide(1, yb) - 1)

        

        grads = 2*np.matmul(

            xb.transpose(),

            np.matmul(xb, self.weights) - ols_yb)

        

        return grads

    

    def __mle_grad(self, xb, yb):

        # computes the gradient of the MLE loss

        term1 = np.matmul(xb.transpose(), 1-yb)

        exw = np.exp(-np.matmul(xb, self.weights))

        term2 = np.matmul(

            (np.divide(exw, 1+exw)*xb).transpose(),

            np.ones_like(yb))

        return term1-term2

    

    def fit(self, x, y, method, learning_rate=0.001, iterations=500, batch_size=32):

        x = np.concatenate([x, np.ones_like(y, dtype=np.float32)], axis=1)

        if method == "ols_solve":

            self.__ols_solve(x, y)

        elif method == "ols_sgd":

            self.__sgd(x, y, self.__sse_grad, learning_rate, iterations, batch_size)

        elif method == "mle_sgd":

            self.__sgd(x, y, self.__mle_grad, learning_rate, iterations, batch_size)

        else:

            print(f'Unknown method: \'{method}\'')

        

        return self

    

    def predict(self, x):

        if not hasattr(self, 'weights'):

            print('Cannot predict. You should call the .fit() method first.')

            return

        

        x = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)

        

        if x.shape[1] != self.weights.shape[0]:

            print(f'Shapes do not match. {x.shape[1]} != {self.weights.shape[0]}')

            return

        

        xw = np.matmul(x, self.weights)

        return np.divide(1, 1+np.exp(-xw))

    

    def accuracy(self, x, y):

        y_hat = self.predict(x)

        

        if y.shape != y_hat.shape:

            print('Error! Predictions don\'t have the same shape as given y')

            return

        

        zeros, ones = np.zeros_like(y), np.ones_like(y)

        y = np.where(y >= 0.5, ones, zeros)

        y_hat = np.where(y_hat >= 0.5, ones, zeros)

        

        return np.mean((y == y_hat).astype(np.float32))
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv('../input/heart-disease-uci/heart.csv')

df
x, y = df.iloc[:, 0:-1].values, df.iloc[:, -1].values.reshape((-1, 1))
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
def print_acc(model):

    print(f'Train accuracy = {model.accuracy(x_train, y_train)} ; '+

          f'Test accuracy = {model.accuracy(x_test, y_test)}')
scaler = MinMaxScaler().fit(x_train)

x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
lr_ols_solve = LogisticRegression().fit(x_train, y_train, 'ols_solve')

print_acc(lr_ols_solve)
lr_ols_sgd = LogisticRegression().fit(x_train, y_train, 'ols_sgd')

print_acc(lr_ols_sgd)
lr_mle_sgd = LogisticRegression().fit(x_train, y_train, 'mle_sgd')

print_acc(lr_mle_sgd)