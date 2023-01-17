# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_df =pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

train_df.info()

test_df.info()
train_df.describe()
train_df.head()
train_df.describe(include=['O'])
selected_feature = ['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

parameters = {}

parameters['selected_feature'] = selected_feature
def cleanup_data(train_df, test_df):

    age_mean = pd.concat([train_df['Age'], test_df['Age']], ignore_index=True).mean()

    fare_mean = pd.concat([train_df['Fare'], test_df['Fare']], ignore_index=True).mean()

    

    train = train_df[['Survived'] + selected_feature].copy()

    

    train['Sex'] = train['Sex'].map({'male': 1, 'female': 0}).astype(int)

    train['Age'] = train['Age'].fillna(age_mean)

    train = train.dropna()

    train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    

    test = test_df[selected_feature].copy()

    test['Sex'] = test['Sex'].map({'male': 1, 'female': 0}).astype(int)

    test['Age'] = test['Age'].fillna(age_mean)

    test['Fare'] = test['Fare'].fillna(fare_mean)

    test['Embarked'] = test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    

    return train, test



train, test = cleanup_data(train_df, test_df)
train.describe()
test.describe()
def feature_scaling(parmeters):

    def get_mean(data_list):

        return pd.concat(data_list, ignore_index=True).mean()

    

    def get_std(data_list):

        return pd.concat(data_list, ignore_index=True).std()



    def get_min(data_list):

        return pd.concat(data_list, ignore_index=True).min()



    def get_max(data_list):

        return pd.concat(data_list, ignore_index=True).max()



    for feature in parameters['selected_feature']:

        if parameters['feature_scaling'] == 'rescaling':

            data_list = [train[feature], test[feature]]

            min_ = get_min(data_list)

            max_ = get_max(data_list)

            train[feature] = (train[feature] - min_) / (max_ - min_)

            test[feature] = (test[feature] - min_) / (max_ - min_)

        elif parameters['feature_scaling'] == 'mean_normalization':

            data_list = [train[feature], test[feature]]

            mean = get_mean(data_list)

            min_ = get_min(data_list)

            max_ = get_max(data_list)

            train[feature] = (train[feature] - mean) / (max_ - min_)

            test[feature] = (test[feature] - mean) / (max_ - min_)

        else:

            data_list = [train[feature], test[feature]]

            mean = get_mean(data_list)

            std = get_std(data_list)

            train[feature] = (train[feature] - mean) / std

            test[feature] = (test[feature] - mean) / std
parameters['feature_scaling'] = 'standardization'

feature_scaling(parameters)
train.describe()
test.describe()
m = int(train.values.shape[0] * 0.7)

train_X = train[selected_feature].values[:m, :]

train_Y = train['Survived'].values.reshape(-1, 1)[:m, :]

valid_X = train[selected_feature].values[m:, :]

valid_Y = train['Survived'].values.reshape(-1, 1)[m:, :]

test_X = test[selected_feature].values

print(train_X.shape, train_Y.shape)

print(valid_X.shape, valid_Y.shape)

print(test_X.shape)
import math

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):

    """

    Creates a list of random minibatches from (X, Y)

    

    Arguments:

    X -- input data, of shape (input size, number of examples)

    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)

    mini_batch_size -- size of the mini-batches, integer

    

    Returns:

    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)

    """

    

    np.random.seed(seed)            # To make your "random" minibatches the same as ours

    m = X.shape[0]                  # number of training examples

    mini_batches = []

        

    # Step 1: Shuffle (X, Y)

    permutation = list(np.random.permutation(m))

    shuffled_X = X[permutation, :]

    shuffled_Y = Y[permutation, :].reshape((m,1))



    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.

    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning

    for k in range(0, num_complete_minibatches):

        ### START CODE HERE ### (approx. 2 lines)

        mini_batch_X = shuffled_X[k * mini_batch_size : (k + 1) * mini_batch_size, :]

        mini_batch_Y = shuffled_Y[k * mini_batch_size : (k + 1) * mini_batch_size, :]

        ### END CODE HERE ###

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)

    

    # Handling the end case (last mini-batch < mini_batch_size)

    if m % mini_batch_size != 0:

        ### START CODE HERE ### (approx. 2 lines)

        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size:, :]

        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size:, :]

        ### END CODE HERE ###

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)

    

    return mini_batches
import tensorflow as tf

def make_model(paramters):

    num_feature = len(parameters['selected_feature'])

    X = tf.placeholder(tf.float32, [None, num_feature])

    Y = tf.placeholder(tf.float32, [None, 1])



    layers_dim = paramters['layers_dim']

    fc = tf.contrib.layers.stack(X, tf.contrib.layers.fully_connected, layers_dim)

    hypothesis = tf.contrib.layers.fully_connected(fc, 1, activation_fn=None)

    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=hypothesis, labels=Y)

    cost = tf.reduce_mean(loss)

    

    learning_rate = parameters['learning_rate']

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    

    prediction = tf.round(tf.sigmoid(hypothesis))

    correct_prediction = tf.equal(prediction, Y)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    

    model = {'X': X, 'Y': Y, 'hypothesis': hypothesis, 'cost': cost,

             'train_op': train_op, 'prediction': prediction, 'accuracy': accuracy}

    

    return model
def train(parameters, model):

    num_epochs = parameters['num_epochs']

    minibatch_size = parameters['minibatch_size']

    train_size = train_X.shape[0]

    saver = tf.train.Saver()

    epoch_list = []

    cost_list = []

    with tf.Session() as sess:

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        sess.run(init_op)

        for epoch in range(num_epochs):

            epoch_cost = 0.

            num_minibatches = int(train_size / minibatch_size)

            minibatches = random_mini_batches(train_X, train_Y, minibatch_size)

            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch

                feed_dict = {model['X'] : minibatch_X, model['Y'] : minibatch_Y}

                _ ,minibatch_cost = sess.run([model['train_op'], model['cost']], feed_dict= feed_dict)

                epoch_cost += minibatch_cost / num_minibatches

            if parameters['print'] and (epoch % parameters['print_freq'] == 0):

                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))

            if parameters['save_cost'] and (epoch % parameters['save_cost_freq'] == 0):

                epoch_list.append(epoch)

                cost_list.append(epoch_cost)

        saver.save(sess, parameters['model_name'])

    return {'epoch_list': epoch_list, 'cost_list' : cost_list}
# set model parameters

parameters['layers_dim'] = [14]

parameters['learning_rate'] = 0.01

# set train parameters (hyper parameter)

parameters['num_epochs'] = 2000

parameters['minibatch_size'] = 16

# set option parameters

parameters['model_name'] = 'titanic'

parameters['print'] = True

parameters['print_freq'] = 100

parameters['save_cost'] = True

parameters['save_cost_freq'] = 10



for k, v in parameters.items():

    print(k, '=', v)
with tf.Graph().as_default():

    model = make_model(parameters)

    plot_data = train(parameters, model)
import matplotlib.pyplot as plt

%matplotlib inline

print

if parameters['save_cost']:

    plt.plot(plot_data['epoch_list'], plot_data['cost_list'])
def evaluate(parameters, model):

    saver = tf.train.Saver()

    with tf.Session() as sess:

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        sess.run(init_op)

        saver.restore(sess, parameters['model_name'])

        print ("Train Accuracy:", model['accuracy'].eval({model['X']: train_X, model['Y']: train_Y}))

        print ("Valid Accuracy:", model['accuracy'].eval({model['X']: valid_X, model['Y']: valid_Y}))
with tf.Graph().as_default():

    model = make_model(parameters)

    evaluate(parameters, model)
def predict(parameters, model):

    saver = tf.train.Saver()

    with tf.Session() as sess:

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        sess.run(init_op)

        saver.restore(sess, parameters['model_name']) 

        return model['prediction'].eval({model['X']: test_X})
answer = pd.DataFrame(test_df['PassengerId'], columns=['PassengerId'])

with tf.Graph().as_default():

    model = make_model(parameters)

    test_Y = predict(parameters, model)

    answer['Survived'] = test_Y.astype(int)

answer.to_csv('answer.csv', index=False)
from sklearn import linear_model

regr = linear_model.LinearRegression()

regr.fit(train_X, train_Y)

match = np.sum(test_Y == np.round(regr.predict(test_X)))

print('match ratio with linear_model of scikit-learn: ', match / test_Y.shape[0])