import random



class myPerceptron:



    bias = 0

    learning_factor = 0

    inputs = []

    labels = []

    weights = []



    def __init__(self, bias, learning_factor):

        self.bias = bias

        self.learning_factor = learning_factor



    # helper methods ---------------------------------

    

    # given the deltas to be applied to the weights, weights += deltas

    def refresh_weights(self, deltas):

        for i in range(len(self.weights)):

            self.weights[i] += deltas[i]

    

    # applies the threshold function to the weighted sums given

    def classification(self, predictions):

        classification = []

        for p in predictions:

            classification.append(self.threshold_function(p))

        return classification

    

    # weighted sum (since we say that neurons fire, my perceptron fires too)

    def fire(self, input):

        y = self.bias

        for i in range(len(input)):

            y += input[i]*self.weights[i]

        return y



    def threshold_function(self, value):

        if value > 0:

            return 1

        else: return 0



    # / helper methods ---------------------------------

        

    def fit(self, X_train, y_train):

        deltas = [0]*len(X_train[0]) # deltas represents the change that need to be made to the weights

        

        # if weights are not set, randomly generate them

        if len(self.weights) == 0:

            self.weights = [random.uniform(-.0001, .0001) for i in range(len(X_train[0]))]

        

        # calculating deltas as delta = learning_factor * (y_train - perceptron_prediction) * input

        for i in range(0, len(X_train)):

            for j in range(0, len(X_train[0])):

                deltas[j] = (self.learning_factor * (y_train[i]-self.threshold_function(self.fire(X_train[i])))*X_train[i][j])

        

        # sum deltas to weights

        self.refresh_weights(deltas)   

       

    def predict(self, x_test):

        y = []

        for i in x_test:

            y.append(self.fire(i))

        return self.classification(y)

import numpy as np

import pandas as pd



data = pd.read_csv('/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv', parse_dates=['Date'])



# taking only a few years

data = data[data.Date > np.datetime64('2008-12-31')]

data = data[data.Date < np.datetime64('2012-01')]



# taking data only for sydney

data = data[data.Location == 'Sydney']



# keeping just a few features for semplicity

features = ['Date', 'Rainfall', 'Humidity3pm', 'RainTomorrow']



data = data[features]



data = data.dropna(axis=1, how='all')

data = data.dropna(axis=0, how='any')



# date to day of the year

data['Date'] = data['Date'].apply(lambda x: pd.Timestamp(x).dayofyear)



# classes label to 0-1

data['RainTomorrow'] = data['RainTomorrow'].apply(lambda x: 0 if x=='No' else 1)



# trying out MinMaxScaler

from sklearn.preprocessing import MinMaxScaler

standardized_columns = ['Date', 'Rainfall', 'Humidity3pm']

scaler = MinMaxScaler()

data[standardized_columns] = scaler.fit_transform(data[standardized_columns])
import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('seaborn-whitegrid')

plt.rcParams['figure.figsize'] = [14, 10]



plt.scatter(data.Humidity3pm, data.Rainfall, color=data['RainTomorrow'].apply(lambda x: 'Red' if x==0 else 'Blue'))
from sklearn.model_selection import train_test_split



X = data[list(set(data.columns)-set(['RainTomorrow']))]

y = data.RainTomorrow



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)

X_train = X_train.to_numpy()

X_test = X_test.to_numpy()

y_train = y_train.to_numpy()

y_test = y_test.to_numpy()
from sklearn.linear_model import Perceptron

clf = Perceptron(tol=1e-3, random_state=2252332)

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)



summ=0

for i in range(0, len(predictions)):

    if predictions[i] != y_test[i]:

        summ += 1



print('real error {}%'.format(summ/len(y_test)*100))
perc = myPerceptron(0, .01)

perc.fit(X_train, y_train)

predictions = perc.predict(X_test)



summ=0

for i in range(0, len(predictions)):

    if predictions[i] != y_test[i]:

        summ += 1



print('real error {}%'.format(summ/len(y_test)*100))