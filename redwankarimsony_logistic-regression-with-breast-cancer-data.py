# Reading the Libraries

import pandas as pd

import numpy as np



# Reading the Data

data = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')

data.head(20)
data.info()
# feature names as a list

# .columns gives columns names in data 

col = data.columns       

print(col)





data.drop(['Unnamed: 32',"id"], axis=1, inplace=True)

data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

y_data = data.diagnosis.values

x_data = data.drop(['diagnosis'], axis=1)

x_data
# Using transformer from sklearn library

from sklearn.preprocessing import MinMaxScaler

scalar = MinMaxScaler()

output = scalar.fit_transform(x_data)



# Manual Implementation of the normalization process

X_data = (x_data -np.min(x_data))/ (np.max(x_data)-np.min(x_data)).values
class LogisticRegression(object):

    """

    Logistic Regression Classifier

    Parameters

    ----------

    learning_rate : int or float, default=0.1

        The tuning parameter for the optimization algorithm (here, Gradient Descent) 

        that determines the step size at each iteration while moving toward a minimum 

        of the cost function.

    max_iter : int, default=100

        Maximum number of iterations taken for the optimization algorithm to converge

    

    penalty : None or 'l2', default='l2'.

        Option to perform L2 regularization.

    C : float, default=0.1

        Inverse of regularization strength; must be a positive float. 

        Smaller values specify stronger regularization. 

    tolerance : float, optional, default=1e-4

        Value indicating the weight change between epochs in which

        gradient descent should terminated. 

    """



    def __init__(self, learning_rate=0.1, max_iter=100, regularization='l2', lambda_ = 10 , tolerance = 1e-4):

        self.learning_rate  = learning_rate

        self.max_iter       = max_iter

        self.regularization = regularization

        self.lambda_        = lambda_

        self.tolerance      = tolerance

        self.loss_log       = []

    

    def fit(self, X, y, verbose = False):

        """

        Fit the model according to the given training data.

        Parameters

        ----------

        X : {array-like, sparse matrix} of shape (n_samples, n_features)

            Training vector, where n_samples is the number of samples and

            n_features is the number of features.

        y : array-like of shape (n_samples,)

            Target vector relative to X.

        Returns

        -------

        self : object

        """

        self.theta = np.random.rand(X.shape[1] + 1)

        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

        self.loss_log = []



        for iteration in range(self.max_iter):

            Z = np.matmul(X,  self.theta)

            y_hat = self.__sigmoid(Z)

            

            errors = y_hat - y



            N = X.shape[1] 

            

            cost = (-1.0/N) * np.sum( y*np.log(y_hat) + (1.0 - y)*np.log(1.0-y_hat))

            self.loss_log.append(cost)

            

            if verbose:

                print(f'Iteration {iteration} Loss: {cost}')



            if self.regularization is not None:

                delta_grad = (1./N) *(np.matmul(errors.T, X)+ self.lambda_ * self.theta)

            else:

                delta_grad = (1./N) *(np.matmul(errors.T, X))

                

            self.theta -= self.learning_rate * delta_grad



#             if np.all(abs(delta_grad) >= self.tolerance):

#                 self.theta -= self.learning_rate * delta_grad

#             else:

#                 break

                

        return self



    def predict_proba(self, X):

        """

        Probability estimates for samples in X.

        Parameters

        ----------

        X : array-like of shape (n_samples, n_features)

            Vector to be scored, where `n_samples` is the number of samples and

            `n_features` is the number of features.

        Returns

        -------

        probs : array-like of shape (n_samples,)

            Returns the probability of each sample.

        """

        return self.__sigmoid((X @ self.theta[1:]) + self.theta[0])

    

    def predict(self, X):

        """

        Predict class labels for samples in X.

        Parameters

        ----------

        X : array_like or sparse matrix, shape (n_samples, n_features)

            Samples.

        Returns

        -------

        labels : array, shape [n_samples]

            Predicted class label per sample.

        """

        return np.round(self.predict_proba(X))

        

    def __sigmoid(self, z):

        """

        The sigmoid function.

        Parameters

        ------------

        z : float

            linear combinations of weights and sample features

            z = w_0 + w_1*x_1 + ... + w_n*x_n

        Returns

        ---------

        Value of logistic function at z

        """

        return (1.0 / (1.0 + np.exp(-z)))



    def get_params(self):

        """

        Get method for models coeffients and intercept.

        Returns

        -------

        params : dict

        """

        try:

            params = dict()

            params['intercept'] = self.theta[0]

            params['coef'] = self.theta[1:]

            return params

        except:

            raise Exception('Fit the model first!')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.20, random_state=42)



# Train and Test Data Summary

import plotly.graph_objects as go

split = ['Train','Test']



fig = go.Figure()

fig.add_trace(go.Bar(x=split, y=[np.sum(y_train), np.sum(y_test)],#                base=[-500,-600],

                    marker_color='crimson',

                    name='Malignant'))

fig.add_trace(go.Bar(x=split, 

                     y=[len(y_train)- np.sum(y_train), len(y_test) - np.sum(y_test)],

                    base=0,

                    marker_color='lightgreen',

                    name='Benign'                ))

fig.update_layout(width = 800, height = 400)

fig.update_layout(title = 'Count of Samples in Train and Test Split', title_x = 0.5, xaxis_title = "Category", yaxis_title = 'Sample Count')

fig.show()
clf = LogisticRegression(max_iter = 200, regularization= None)

clf.fit(X_train, y_train, verbose = False)
import plotly.graph_objects as go

import numpy as np



y = clf.loss_log 



fig = go.Figure(data=go.Scatter(x= np.arange(start =1, stop = len(y)), 

                                y=y,

                                mode = 'lines+markers'))



fig.update_layout(title = "Error Plot over Iterations", title_x = 0.5,

                  xaxis_title = 'Iteration',

                  yaxis_title = 'Log Loss',

                  width = 800,

                  height = 500)

fig.show()
y_pred = clf.predict(X_test)

y_pred



match = (y_pred == y_test)*1.0

match.values





accuracy = (np.sum(match)*100/ len(match))

print(f'Accuracy {accuracy}%')
import seaborn as sns

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)