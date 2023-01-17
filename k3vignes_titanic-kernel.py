# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

%matplotlib inline

# Any results you write to the current directory are saved as output.
class NeuralNetwork:

    # layers: list of numbers, each number is number of neurons in that hidden layer.
    #         Thus, length of list is number of hidden layers
    def __init__(self, n_inputs, n_outputs, h_layers, inputs, expected_outputs):
        self.n_inputs = n_inputs
        self.h_layers = h_layers
        self.n_outputs = n_outputs
        self.x = inputs
        self.y_ = expected_outputs
        self.input_layer = tf.placeholder(tf.float32, [None, self.n_inputs], name="InputLayer")
        self.expected_output_layer = tf.placeholder(tf.float32, [None, self.n_outputs],
                                                    name="ExpectedOutputLayer")
        self.weights = []
        self.bias = []
        self.cost_history = []
        self.learning_rate = 0.01
        self.output_layer = None
        self.session = None
        self.training_step = None
        self.cost_function = None

    def set_init_vals(self, is_weights_rand):
        if is_weights_rand:
            return tf.truncated_normal
        else:
            return tf.zeros

    def initialize_weights_bias(self, is_weights_rand=True):
        init_vals = self.set_init_vals(is_weights_rand)
        for i in range(0, len(self.h_layers) + 1):
            if i == 0:
                curr_weights = tf.Variable(init_vals([self.n_inputs, self.h_layers[0]]), name="Weights_Input_to_h1")
                curr_bias = tf.Variable(tf.truncated_normal([self.h_layers[0]]), name="Bias_h1")
            elif i < len(self.h_layers):
                curr_weights = tf.Variable(init_vals([self.h_layers[i - 1], self.h_layers[i]]),
                                           name='Weights_h{0}_to_h{1}'.format(i-1, i))
                curr_bias = tf.Variable(tf.truncated_normal([self.h_layers[i]]), name='Bias_h{0}'.format(i))
            else:
                curr_weights = tf.Variable(init_vals([self.h_layers[-1], self.n_outputs]),
                                           name="Weights_h{0}_to_output".format(i))
                curr_bias = tf.Variable(tf.truncated_normal([self.n_outputs]),
                                        name="Bias_outputLayer")

            self.weights.append(curr_weights)
            self.bias.append(curr_bias)

        init = tf.global_variables_initializer()
        self.session.run(init)

    def forward_propagation(self):
        curr_layer = tf.add(tf.matmul(self.input_layer, self.weights[0]), self.bias[0])
        curr_layer = tf.nn.sigmoid(curr_layer)

        for i in range(1, len(self.weights)):
            curr_layer = tf.add(tf.matmul(curr_layer, self.weights[i]), self.bias[i])
            curr_layer = tf.nn.sigmoid(curr_layer)

        self.output_layer = curr_layer

    def train(self, epochs=1000, learning_rate=0.01):
        self.learning_rate = learning_rate
        for i in range(epochs):
            self.train_step()

        # print('output: ', self.session.run(self.output_layer, feed_dict={self.input_layer: self.x}))
        # print('expected: ', self.y_)

    def train_step(self):
        curr_epoch = len(self.cost_history)
        self.session.run(self.training_step,
                         feed_dict={self.input_layer: self.x,
                                    self.expected_output_layer: self.y_})
        cost = self.session.run(self.cost_function, feed_dict={self.input_layer: self.x, self.expected_output_layer: self.y_})
        if curr_epoch % 10000 == 0:
            print('epoch: ', curr_epoch, ' cost: ', cost)
        self.cost_history.append(cost)

    def test(self, test_inputs, test_outputs=None):
        outputs = self.session.run(self.output_layer, feed_dict={self.input_layer: test_inputs})
        cost = None
        if test_outputs is not None: 
            cost = self.session.run(self.cost_function, feed_dict={self.input_layer: test_inputs,
                                                               self.expected_output_layer: test_outputs})
        return outputs, cost

    def plot(self):
        plt.figure()
        plt.plot(self.cost_history)
        plt.xlabel('Epoch')
        plt.ylabel('Cost')
        # plt.ylim(ymin=0)
        plt.title('Learning Rate')
        plt.show()

    def open_session(self):
        self.session = tf.Session()

    def close_session(self):
        self.session.close()

    def make_model(self, is_weights_rand=True):
        self.initialize_weights_bias(is_weights_rand)
        self.forward_propagation()
        self.cost_function = tf.reduce_mean((self.output_layer - self.expected_output_layer) ** 2)
        self.training_step = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost_function)


    # To open tensor graph
    # cd to project dir then run
    # run python -m tensorboard.main --logdir="./"
    def make_tensor_board(self):
        tf.summary.FileWriter('./tensorgraph', self.session.graph)
# Prep training data
# Load and select feature cols
df = pd.read_csv('../input/train.csv', index_col=0)
x = df.loc[:, ['Pclass', 'Sex', 'Age', 'Embarked']]
y = df.loc[:, ['Survived']]

# Replace null values with most frequent value
x.loc[x['Sex'].isnull(), 'Sex'] = x['Sex'].value_counts().idxmax()
x.loc[x['Embarked'].isnull(), 'Embarked'] = x['Embarked'].value_counts().idxmax()
x.loc[x['Age'].isnull(), 'Age'] = x['Age'].value_counts().idxmax()

# Convert strings to int-categories
label_encoder = LabelEncoder()
sex_int_encoded = label_encoder.fit_transform(x.loc[:, 'Sex'].values)

# Convert int-categories to binary-categories
sex_int_encoded = sex_int_encoded.reshape(len(sex_int_encoded), 1)
sex_one_hot_encoded = OneHotEncoder(sparse=False).fit_transform(sex_int_encoded)

# Inverting back to our orginal label
# inverted = label_encoder.inverse_transform([np.argmax(sex_one_hot_encoded[0, :])])
x['Female'] = np.NaN
x['Male'] = np.NaN
x.loc[:, ['Female', 'Male']] = sex_one_hot_encoded

x.drop(['Sex', 'Embarked'], inplace=True, axis=1)

doa = y.values
doa = doa.reshape(len(doa), 1)
doa_one_hot_encoded = OneHotEncoder(sparse=False).fit_transform(doa)
y['Dead'] = np.NaN
y['Survived'] = np.NaN
y[['Dead', 'Survived']] = doa_one_hot_encoded
# Load and select feature cols
test_df = pd.read_csv('../input/test.csv', index_col=0)
test_x = test_df.loc[:, ['Pclass', 'Sex', 'Age', 'Embarked']]
test_y = pd.read_csv('../input/gender_submission.csv', index_col=0)

# Replace null values with most frequent value
test_x.loc[test_x['Sex'].isnull(), 'Sex'] = test_x['Sex'].value_counts().idxmax()
test_x.loc[test_x['Embarked'].isnull(), 'Embarked'] = test_x['Embarked'].value_counts().idxmax()
test_x.loc[test_x['Age'].isnull(), 'Age'] = test_x['Age'].value_counts().idxmax()

# Convert strings to int-categories
label_encoder = LabelEncoder()
sex_int_encoded = label_encoder.fit_transform(test_x.loc[:, 'Sex'].values)

# Convert int-categories to binary-categories
sex_int_encoded = sex_int_encoded.reshape(len(sex_int_encoded), 1)
sex_one_hot_encoded = OneHotEncoder(sparse=False).fit_transform(sex_int_encoded)

# Inverting back to our orginal label
# inverted = label_encoder.inverse_transform([np.argmax(sex_one_hot_encoded[0, :])])
test_x['Female'] = np.NaN
test_x['Male'] = np.NaN
test_x.loc[:, ['Female', 'Male']] = sex_one_hot_encoded

test_x.drop(['Sex', 'Embarked'], inplace=True, axis=1)

doa = test_y.values
doa = doa.reshape(len(doa), 1)
doa_one_hot_encoded = OneHotEncoder(sparse=False).fit_transform(doa)
test_y['Dead'] = np.NaN
test_y['Survived'] = np.NaN
test_y[['Dead', 'Survived']] = doa_one_hot_encoded
h_layers = [10, 10]
inputs = x.values
outputs = y.values
nn = NeuralNetwork(n_inputs=4,
                   n_outputs=2,
                   h_layers=h_layers,
                   inputs=inputs,
                   expected_outputs=outputs)
nn.open_session()
nn.make_model(is_weights_rand=True)
nn.make_tensor_board()
s = time.time()
nn.train(epochs=999, learning_rate=0.01)
outputs, cost = nn.test(test_inputs=test_x.values, test_outputs=test_y.values)
print("my bvalues")
test_y['pred_survived'] = np.NaN
test_y['pred_dead'] = np.NaN
test_y[['pred_survived', 'pred_dead']] = outputs
test_y['pred_survived'] = [np.rint(x) for x in test_y['pred_survived']]
test_y['pred_dead'] = [np.rint(x) for x in test_y['pred_dead']]
test_y['cost'] = ((test_y['pred_survived'] - test_y['Survived']) ** 2).mean()
print(test_y)
print('cost: ', cost)
e = time.time() - s
print("Training took ", e, "seconds")
nn.close_session()
nn.plot()
