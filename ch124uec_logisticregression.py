# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import os
from sklearn.feature_extraction.text import CountVectorizer


def get_data_from_json(path):
    """Reads json files, and returns arrays of cuisines and ingredients."""

    train_data = os.path.join(path, "train.json")
    test_data = os.path.join(path, "test.json")

    recipes = pd.read_json(train_data).set_index('id')
    ingredients = recipes.ingredients.str.join(' ')

    cv = CountVectorizer()
    cv.fit(ingredients)
    num_unique_ingredients = len(cv.get_feature_names())
    num_recipes = len(recipes)
    train_x = pd.DataFrame(cv.transform(ingredients).todense())
    train_y = recipes.cuisine
    num_cuisines = recipes.cuisine.nunique()
    print("{0} unique ingredients used in {1} recipes from {2} different cuisines around the world.".format(num_unique_ingredients, num_recipes, num_cuisines))

    recipes_test = pd.read_json(test_data).set_index('id')
    test_x = pd.DataFrame(cv.transform(recipes_test.ingredients.str.join(' ')).todense())
    index = recipes_test.index
    return train_x, train_y, test_x, index


def write_predictions_to_file(index, test_y, path):
    """Writes the predictions on test data to a .csv file in suitable format for submission."""
    print(type(test_y))
    print(test_y.shape)
    submission_df = pd.Series(test_y, index=index, name='cuisine')
    submission_df.to_csv(path=os.path.join(path, "submissions4.csv"), header=True)

from sklearn.linear_model.logistic import LogisticRegression

class Model(object):
    """Define base class for prediction models"""
    def __init__(self, name):
        self.name = name

    def train(self, train_x, train_y):
        """Train the prediction model"""

    def test(self, test_x):
        """Function to predict and generate file for submission."""


class LogisticClassifier(Model):
    """Multi-label logistic classifier class."""

    def __init__(self, epochs):
        super(LogisticClassifier, self).__init__("logistic regression")
        self.max_epochs = epochs
        self.lr = LogisticRegression(max_iter=epochs)

    def train(self, train_x, train_y):
        print("Training {} model.......".format(self.name))
        self.lr.fit(train_x, train_y)
        print("Training complete!!")

    def test(self, test_x):
        test_y = self.lr.predict(test_x)
        print("Successfully generated predictions for test data.")
        return test_y
class LogisticClassifier2(Model):

    def __init__(self, epochs):
        super(LogisticClassifier2, self).__init__("self-written Logistic Regression")
        self.max_epochs = epochs
        self.labels = None
        self.W = None
        self.b = None

    def train(self, train_x, train_y, alpha=0.05):
        num_classes = train_y.nunique()
        self.labels = train_y.unique()
        train_x = train_x.as_matrix()
        # train_x = train_x[:1000, :]
        # train_y = train_y[:1000]
        m, n = train_x.shape
        print (m, n)
        print ("Number of training examples: {}, number of features: {}".format(m, n))
        self.W = np.random.rand(num_classes, n)
        self.b = np.random.rand(num_classes, 1)

        y = np.zeros((num_classes, m), dtype='int8')
        for i, label in enumerate(self.labels):
            y[i, train_y == label] = 1

        # print "y.shape: " + str(y.shape)
        # now train binary logistic regression for all classes together
        for j in range(self.max_epochs):
            print("Epoch #{}".format(j+1))

            # forward pass
            z = np.matmul(self.W, train_x.T) + self.b
            a = 1 / (1 + np.exp(-z))

            # backward pass gradient descent
            dz = y-a
            # print "dz.shape: " + str(dz.shape)
            dw = -1.0*np.matmul(dz, train_x)/m
            # print dw.shape
            db = -1.0*np.mean(dz, axis=1, keepdims=True)
            # print db.shape

            self.W -= alpha * dw
            self.b -= alpha * db

    def test(self, test_x):
        test_x = test_x.as_matrix()
        predictions = np.matmul(self.W, test_x.T) + self.b
        predictions = np.argmax(predictions, axis=0)
        test_y = self.labels[predictions]
        return test_y
data_dir = os.path.join(os.pardir, "input")

# generate array from json files
train_x, train_y, test_x, index = get_data_from_json(data_dir)

model = LogisticClassifier2(epochs=150)
model.train(train_x, train_y)
test_y = model.test(test_x)
write_predictions_to_file(index, test_y, os.getcwd())