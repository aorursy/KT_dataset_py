# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv(os.path.join('/kaggle/input/titanic/', 'train.csv'))

print(data.head())
def data_preprocessing(data):

    def remove_unwanted(data):

        """

        This function removes unnecessary columns as the passenger ID, Name, Ticket number and Cabin number are unnecessary features to predict if he/she survived or not.

        """

        new_data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

        return new_data

    def replace_NaN(data):

        """

        This function replaces empty cells in the table by mean of the column.

        """

        for col in data.columns:

            if data[col].isnull().any():

                data[col] = data[col].fillna(data[col].mean())

        return data

    def make_categorical(data):

        for col in data.columns:

            data[col] = data[col].astype("category").cat.codes

        return data

    def normalize(data):

        new_data = (data - data.min()) / (data.max() - data.min())

        return new_data

    new_data = remove_unwanted(data)

    new_data = make_categorical(new_data)

    new_data = replace_NaN(new_data)

    new_data = normalize(new_data)

    return new_data
data = data_preprocessing(data)

print(data.head())
import matplotlib.pyplot as plt



plt.matshow(data.corr())

plt.xticks(range(data.shape[1]), data.columns, rotation=90)

plt.yticks(range(data.shape[1]), data.columns)

cb = plt.colorbar()

plt.title('Correlation Matrix')

plt.show()
from sklearn.linear_model import LogisticRegression as sk_LogisticRegression

from sklearn.metrics import accuracy_score



class LogisticRegression:

    def __init__(self, learning_rate, random_state, shape_of_data):

        super().__init__()

        self.learning_rate = learning_rate

        self.shape_of_data = shape_of_data

        np.random.seed(random_state)

        # self.weights = np.random.rand(shape_of_data + 1)

        self.weights = np.ones(shape_of_data+1)

        # print("Weights", self.weights)



        self.sk_logReg = sk_LogisticRegression(random_state=random_state)



    def predict_float(self, X_test):

        """

        Predict the value

        """

        assert (len(X_test) == self.shape_of_data or len(X_test) == self.shape_of_data + 1), str("Number of features does not match expected number of features ")

        if len(X_test) == self.shape_of_data:

            X_test = np.append(X_test, [1], axis=0)



        return 1/(1+np.exp(-np.dot(X_test, self.weights)))



    def predict(self, X_test):

        """

        Predict the class

        """

        itrs = 1

        if X_test.ndim == 1:

            pre = self.predict_float(X_test)

            if pre >= 0.5:

                return 1

            else:

                return 0

        if X_test.ndim == 2:

            X_test = X_test.to_numpy()

            itrs = X_test.shape[0]



            preds = []

            for i in range(itrs):

                preds.append(self.predict(X_test[i]))

            return preds



    def fit(self, X_train, Y, epochs = 100):

        """

        Fit function to fit the data with model

        """

        X_train = X_train.to_numpy()

        Y = Y.to_numpy()

        assert (X_train.shape[1] == self.shape_of_data), str(

            "Number of features does not match expected number of features : " + 

            "Features got : " + str(X_train.shape[1]) + 

            " Features expected : " + str(self.shape_of_data))



        assert (len(X_train) == len(Y)), str(

            "X_train and Y do not have same shape : " + 

            "X_train : " + str(len(X_train)) + " Y : " + str(len(Y)))

        



        X_train = np.append(X_train, np.ones((len(X_train), 1)), axis=1)

        for _ in range(epochs):

            for r in range(len(X_train)):

                Y_pred = self.predict_float(X_train[r])

                cost = Y[r] - Y_pred

                for t in range(len(self.weights)):

                    self.weights[t] = self.weights[t] + self.learning_rate*cost*Y_pred*(1-Y_pred)*X_train[r][t]

            



    def score(self, X_test, Y_test):

        """

        Score function for our model

        """

        # X_test = X_test.to_numpy()

        # Y_test = Y_test.to_numpy()

        assert (len(X_test) == len(Y_test)), str(

            "X_test and Y_test are not of same size. " + 

            "X_test : " + str(len(X_test)) + 

            " Y_test : " + str(len(Y_test)))



        # preds = []

        # for x in X_test:

        #     preds.append(self.predict(x))

        preds = self.predict(X_test)

        return accuracy_score(preds, Y_test)



    def fit_sklearn(self, X_train, Y):

        """

        Fit function to fit sklearn model

        """

        X_train = X_train.to_numpy()

        Y = Y.to_numpy()

        Y = np.transpose(Y)

        Y = Y[0]

        assert (len(X_train) == len(Y)), str(

            "Size of training features and training target do not match. " +

            "X_train : " + str(len(X_train)) +

            " Y : " + str(len(Y)))

        self.sk_logReg.fit(X_train, Y)



    def predict_sklearn(self, X_test):

        """

        Predict class for sklearn model

        """

        return self.sk_logReg.predict(X_test)



    def score_sklearn(self, X_test, Y_test):

        """

        Score function for sklearn model

        """

        return self.sk_logReg.score(X_test, Y_test)

""" 

Reading the data files

"""

print("Reading the data files...")

print('training data, ')

XY_train_raw = pd.read_csv(os.path.join('/kaggle/input/titanic/', 'train.csv'))

print(XY_train_raw.head(), '\n')



print('testing data, ')

X_test_raw = pd.read_csv(os.path.join('/kaggle/input/titanic/', 'test.csv'))

print(X_test_raw.head(), '\n')



print('output')

Y_test_raw = pd.read_csv(os.path.join('/kaggle/input/titanic/', 'gender_submission.csv'))

print(Y_test_raw.head(), '\n')

print("Done.\n")
"""

Preprocessing the data

"""

print("Preprocessing the data...")

print('XY_train')

XY_train = data_preprocessing(XY_train_raw)

print(XY_train.head(), '\n')



print('X_test')

X_test = data_preprocessing(X_test_raw)

print(X_test.head(), '\n')



print('Y_test')

Y_test = Y_test_raw['Survived']

PassengerID_test = list(Y_test_raw['PassengerId'])

print(Y_test.head(), '\n')

print("Done.\n")
"""

Cut the dataset into features and target

"""

print("Cutting the dataset into features and target...")

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

target = ['Survived']

X_train = XY_train[features]

Y_train = XY_train[target]

print("Done.\n")
"""

Initialize Model

"""

print("Initializing the model...")

model = LogisticRegression(learning_rate=0.01, random_state=0, shape_of_data=len(features))

print("Done.\n")
"""

Fit dataset in model

"""

print("Fitting dataset in model...")

model.fit(X_train = X_train, Y = Y_train, epochs = 100)

model.fit_sklearn(X_train = X_train, Y = Y_train)

print("Done.\n")
"""

Showing Training Score

"""

train_score = model.score(X_test = X_train, Y_test = Y_train)

sklearn_train_score = model.score_sklearn(X_test = X_train, Y_test = Y_train)



print("Training Accuracy for my model : ", train_score)

print("Training Accuracy for sklearn model : ", sklearn_train_score)

print('\n')
"""

Show score 

"""

score = model.score(X_test = X_test, Y_test = Y_test)

sk_score = model.score_sklearn(X_test = X_test, Y_test = Y_test)



print("Test Accuracy for my model : ", score)

print("Test Accuracy for sklearn model : ", sk_score)

print('\n')



predictions_mymodel = model.predict(X_test = X_test)

predictions_sklearn = model.predict_sklearn(X_test=X_test)



# print(PassengerID_test)

Predictions = {"PassengerId" : PassengerID_test, "Survived" : predictions_mymodel}

Predictions_df = pd.DataFrame(Predictions)
print(Predictions_df)
Predictions_df.to_csv('submission.csv', index=False)