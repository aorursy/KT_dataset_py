# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
%matplotlib inline
# use this to control random state when we do things like shuffling
SEED = 0
np.random.seed(SEED)
%pwd
# make sure output directory exists (where we store tensorflow files, output CSV etc), if not, create it.
# (This folder will not be tracked by git.)
# import os
# os.makedirs("./titanic",exist_ok=True)
# parse CSV into a Pandas DataFrame
df =pd.read_csv('../input/train.csv')
# Take a peek at first 5 rows
df.head()
# Take a peek at last 5 rows
df.tail()
# What columns do we have?
# What are the inferred datatypes by Pandas?
# How complete are our data?
df.info()
df.shape
# Before: we have 891 rows and 12 columns
df.shape
assert df.shape[0] == 891
assert df.shape[1] == 12
# 1 - We ignore the column Cabin (too many missing data)
df2 = df.drop(['Cabin'], axis=1)

# Now we have 1 less column
df2.shape
assert df2.shape[0] == 891
assert df2.shape[1] == 11
df2.loc[pd.isnull(df["Embarked"])]
df3 = df2.copy()
df3.loc[df3["PassengerId"] == 62, "Embarked"] = "S"
df3.loc[df3["PassengerId"] == 830, "Embarked"] = "S"
df3.loc[df3["PassengerId"].isin([62, 830]), :]
# Just to double check we now have 100% non-null Embarked values
df3.loc[pd.notnull(df3["Embarked"])].shape
assert df3.shape[0] == 891
assert df3.shape[1] == 11
# describe numerical columns
df3.describe()
# Let's store this median value away.
median_age = df["Age"].median()
median_age
df3.loc[pd.isnull(df3["Age"])].head()
# make a copy just in case as we are replacing values in original dataframe
df4 = df3.copy()
# Create a `FakeAge` flag (1=True/0=False) to indicate the Age column is fake due to NaN
df4["FakeAge"] = df4["Age"].apply(lambda x: True if np.isnan(x) else False)
# Do some spot check - our FakeAge flag should be correct.

# Passengers with non-null Age values should get a FakeAge of False
assert df4.loc[df4["PassengerId"] == 1]["FakeAge"].bool() is False
assert df4.loc[df4["PassengerId"] == 2]["FakeAge"].bool() is False
assert df4.loc[df4["PassengerId"] == 3]["FakeAge"].bool() is False
assert df4.loc[df4["PassengerId"] == 4]["FakeAge"].bool() is False

# Passengers with null Age values should get a FakeAge of True
assert df4.loc[df4["PassengerId"] == 6]["FakeAge"].bool() is True
assert df4.loc[df4["PassengerId"] == 18]["FakeAge"].bool() is True
assert df4.loc[df4["PassengerId"] == 20]["FakeAge"].bool() is True
assert df4.loc[df4["PassengerId"] == 27]["FakeAge"].bool() is True
assert df4.loc[df4["PassengerId"] == 29]["FakeAge"].bool() is True
# fill NaN age with our `median_age`
df4["Age"] = df4["Age"].fillna(median_age)
# we can confirm none of our rows contain missing `Age`.
df4.loc[pd.isnull(df4["Age"])].head()
sample_missing_age_passengers = df3.loc[pd.isnull(df3["Age"])].loc[:, "PassengerId"].head()
sample_missing_age_passengers
df4.loc[df4["PassengerId"].isin(sample_missing_age_passengers), :]
# Ensure none of our age values are missing
assert df4["Age"].isnull().values.any() == False
assert df4["Age"].notnull().values.all() == True
assert df4.isnull().values.any() == False
assert df4.notnull().values.all() == True
df4.info()
df4.shape
assert df4.shape[0] == 891
assert df4.shape[1] == 12
df5 = df4.copy()
df5 = df5.set_index("PassengerId")
df5.head()
df5.info()
# Tidy up
del df, df2, df3, df4
df5.head()
# show string categorical column overview
# df5.describe(include="O")
pclass_group = df5.groupby('Pclass')["Pclass"]
pclass_group.count()
pclass_group.count().plot(kind="bar")
sex_group = df5.groupby('Sex')['Sex']
sex_group.count()
sex_group.count().plot(kind="bar")
embarked_group = df5.groupby('Embarked')['Embarked']
embarked_group.count()
embarked_group.count().plot(kind='bar')
# embarked_group.count().sort_values(ascending=False).plot(kind="bar")
# Fare histogram overall - note the artifially high age of 28 is probably from our
# data cleaning step (fill NaN with missing value of 28)
df5["Age"].plot(kind="hist", bins=20)
p = pd.DataFrame(
  {'0 Real Age': df5.groupby('FakeAge').get_group(False)["Age"],
   '1 Fake Age': df5.groupby('FakeAge').get_group(True)["Age"]})
p.plot.hist(bins=20, stacked=True)
sibsp_group = df5.groupby('SibSp')['SibSp']
sibsp_group.count()
sibsp_group.count().plot(kind="bar")
parch_group = df5.groupby('Parch')['Parch']
parch_group.count()
parch_group.count().plot(kind="bar")
# Fare histogram overall
df5["Fare"].hist(bins=20)
# Fare histogram breakdown by class
# https://stackoverflow.com/questions/41622054/stacked-histogram-of-grouped-values-in-pandas
pd.DataFrame({'1st Class': df5.groupby('Pclass').get_group(1)["Fare"],
              '2nd Class': df5.groupby('Pclass').get_group(2)["Fare"],
              '3rd Class': df5.groupby('Pclass').get_group(3)["Fare"]}
            ).plot.hist(bins=20, stacked=True)
# to view histograms by class separately
df5["Fare"].hist(by=df5['Pclass'])
# https://stackoverflow.com/questions/6871201/plot-two-histograms-at-the-same-time-with-matplotlib
# Overlay histograms by class
plt.clf()
plt.hist(df5.loc[df5["Pclass"]==1, "Fare"], bins=20, color="red", alpha=0.5, label='1st class')
plt.hist(df5.loc[df5["Pclass"]==2, "Fare"], bins=20, color="blue", alpha=0.5, label='2nd class')
plt.hist(df5.loc[df5["Pclass"]==3, "Fare"], bins=20, color="green", alpha=0.5, label='3rd class')
plt.legend(loc='upper right')
plt.show()
# https://stackoverflow.com/questions/6871201/plot-two-histograms-at-the-same-time-with-matplotlib
# overlay histogram by class, but show side-by-side for clarity.
plt.clf()
plt.hist([
  df5.loc[df5["Pclass"]==1, "Fare"].values,
  df5.loc[df5["Pclass"]==2, "Fare"].values,
  df5.loc[df5["Pclass"]==3, "Fare"].values
], color=['red','blue', 'green'], alpha=0.8, bins=20, label=["1st", "2nd", "3rd"])
plt.legend(loc='upper right')
plt.show()
df6 = df5.copy()
# required for ease of doing pivot tables later on
df6["Dummy"] = "yo"
# transform Sex into Binary IsMale
df6["IsMale"] = df6['Sex'].map({'male': 1, 'female': 0}).astype(int)
df6[["Sex", "IsMale"]].head()
# transform Embarked into Binary EmbarkedC, EmbarkedQ, and EmbarkedS
df_embarked = pd.pivot_table(df6[["Dummy", "Embarked"]], index="PassengerId", columns="Embarked", aggfunc="count")\
    .fillna(0.).astype(int).rename(columns={"C": "EmbarkedC", "Q": "EmbarkedQ", "S": "EmbarkedS"})
df_embarked.columns = df_embarked.columns.droplevel()
df_embarked.columns = ["EmbarkedC", "EmbarkedQ", "EmbarkedS"]
df_embarked.head()
# transform Pclass into Binary Is1stClass, Is2ndClass, and Is3rdClass
df_pclass = pd.pivot_table(df6[["Dummy", "Pclass"]], index="PassengerId", columns="Pclass", aggfunc="count")\
    .fillna(0.).astype(int).rename(columns={1: "Is1stClass", 2: "Is2ndClass", 3: "Is3rdClass"})
df_pclass.columns = df_pclass.columns.droplevel()
df_pclass.columns = ["Is1stClass", "Is2ndClass", "Is3rdClass"]
df_pclass.head()
# Stitch these transformed categorical columns back to the main dataframe
df7 = pd.concat([df6, df_embarked, df_pclass], axis="columns")
df7.head()
# Select only the columns we want - for tidiness
params = {}
params['selected_features'] = ['IsMale', "EmbarkedC", "EmbarkedQ", "EmbarkedS", "Is1stClass",
                              "Is2ndClass", "Is3rdClass", 'SibSp', 'Parch', 'Age', 'Fare']
params['label_col'] = "Survived"
params['labeled_cols'] = params['selected_features'] + [params['label_col']]
df_labeled = df7.loc[:, params["labeled_cols"]]
df_labeled.head()
df_labeled.info()
# percentage of labeled data we wish to split out for validation
val_percent = 10
num_samples = df_labeled.shape[0]
val_size = int(num_samples * val_percent / 100)
train_size = num_samples - val_size

print("Training Size: {}".format(train_size))
print("Validation Size: {}".format(val_size))
df_labeled_shuffled = df_labeled.sample(frac=1, random_state=0)
df_labeled_shuffled.shape
# check that it's been shuffled
df_labeled_shuffled.head()
# Obtain our training features and ground truth labels
train_X = df_labeled_shuffled[params["selected_features"]].values[:train_size, :]
train_Y = df_labeled_shuffled[params['label_col']].values.reshape(-1, 1)[:train_size, :]

# Obtain our validation features and ground truth labels
valid_X = df_labeled_shuffled[params["selected_features"]].values[train_size:, :]
valid_Y = df_labeled_shuffled[params['label_col']].values.reshape(-1, 1)[train_size:, :]

print(train_X.shape, train_Y.shape)
print(valid_X.shape, valid_Y.shape)
from sklearn import preprocessing

train_X_scaled_std = preprocessing.StandardScaler().fit(train_X)
train_X_scaled = train_X_scaled_std.transform(train_X)
train_X_scaled.shape
# Take a peek at first 5 training samples - not that all the values are more or less within the same range.
train_X_scaled[:5]
# After standardization we should expect a mean value of 0, and standard deviation of 1.
print("input feature mean after standardization: {}".format(train_X_scaled.mean()))
print("input feature std after standardization: {}".format(train_X_scaled.std()))
# Do the same for the validation set
valid_X_scaled_std = preprocessing.StandardScaler().fit(valid_X)
valid_X_scaled = valid_X_scaled_std.transform(valid_X)

# After standardization we should expect a mean value of 0, and standard deviation of 1.
print("input feature mean after standardization: {}".format(valid_X_scaled.mean()))
print("input feature std after standardization: {}".format(valid_X_scaled.std()))
import math
def random_mini_batches(X, Y, mini_batch_size = 64, seed = SEED):
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
def make_model(parameters):
    num_feature = len(parameters['selected_features'])
    X = tf.placeholder(tf.float32, [None, num_feature])
    Y = tf.placeholder(tf.float32, [None, 1])

    layers_dim = parameters['layers_dim']
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
params['layers_dim'] = [11]
params['learning_rate'] = 0.01
# set train parameters (hyper parameter)
params['num_epochs'] = 2000
params['minibatch_size'] = 16
# set option parameters
# save files to the folder ./titanic, within it the files are labeled "titanic*"
params['model_name'] = './titanic/titanic'
params['print'] = True
params['print_freq'] = 100
params['save_cost'] = True
params['save_cost_freq'] = 10

for k, v in params.items():
    print(k, '=', v)
with tf.Graph().as_default():
    model = make_model(params)
    plot_data = train(params, model)
import matplotlib.pyplot as plt
if params['save_cost']:
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
    model = make_model(params)
    evaluate(params, model)
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(train_X, train_Y)
match = np.sum(valid_Y == np.round(regr.predict(valid_X)))
print('match ratio with linear_model of scikit-learn: ', match / valid_Y.shape[0])
# parse CSV into a Pandas DataFrame
df_test =pd.read_csv('../input/test.csv')
df_test.head()
df_test.info()
df_test.shape
# 1 - We ignore the column Cabin (too many missing data)
df_test_2 = df_test.drop(['Cabin'], axis=1)
# 2 - fill missing age with median age
median_age_test = df_test_2["Age"].median()
median_age_test
# Create a `FakeAge` flag (1=True/0=False) to indicate the Age column is fake due to NaN
df_test_3 = df_test_2.copy()
df_test_3["FakeAge"] = df_test_3["Age"].apply(lambda x: True if np.isnan(x) else False)
# fill NaN age with our `median_age`
df_test_4 = df_test_3.copy()
df_test_4["Age"] = df_test_4["Age"].fillna(median_age_test)
# we can confirm none of our rows contain missing `Age`.
df_test_4.loc[pd.isnull(df_test_4["Age"])].head()
# Ensure none of our age values are missing
assert df_test_4["Age"].isnull().values.any() == False
assert df_test_4["Age"].notnull().values.all() == True
# Who has missing Fare? (Passenger 152, a 3rd class)
df_test_4.loc[pd.isnull(df_test_4["Fare"])].head()
median_fare_3rd_class_test = df_test_4.loc[df_test_4["Pclass"] == 3, "Fare"].median()
median_fare_3rd_class_test
# fill the missing Fare with the median of Fare of the pessenger of the same class
df_test_5 = df_test_4.copy()
df_test_5["Fare"] = df_test_5["Fare"].fillna(median_fare_3rd_class_test)
# we can confirm none of our rows contain missing `Fare`.
df_test_5.loc[pd.isnull(df_test_5["Fare"])].head()
assert df_test_5.isnull().values.any() == False
assert df_test_5.notnull().values.all() == True
df_test_5.info()
df_test_5.shape
assert df_test_5.shape[0] == 418
assert df_test_5.shape[1] == 11
df_test_6 = df_test_5.copy()
df_test_6 = df_test_6.set_index("PassengerId")
df_test_6.head()
# required for ease of doing pivot tables later on
df_test_6["Dummy"] = "yo"
# transform Sex into Binary IsMale
df_test_6["IsMale"] = df_test_6['Sex'].map({'male': 1, 'female': 0}).astype(int)
df_test_6[["Sex", "IsMale"]].head()
# transform Embarked into Binary EmbarkedC, EmbarkedQ, and EmbarkedS
df_embarked_test = pd.pivot_table(df_test_6[["Dummy", "Embarked"]], index="PassengerId", columns="Embarked", aggfunc="count")\
    .fillna(0.).astype(int).rename(columns={"C": "EmbarkedC", "Q": "EmbarkedQ", "S": "EmbarkedS"})
df_embarked_test.columns = df_embarked_test.columns.droplevel()
df_embarked_test.columns = ["EmbarkedC", "EmbarkedQ", "EmbarkedS"]
df_embarked_test.head()
# transform Pclass into Binary Is1stClass, Is2ndClass, and Is3rdClass
df_pclass_test = pd.pivot_table(df_test_6[["Dummy", "Pclass"]], index="PassengerId", columns="Pclass", aggfunc="count")\
    .fillna(0.).astype(int).rename(columns={1: "Is1stClass", 2: "Is2ndClass", 3: "Is3rdClass"})
df_pclass_test.columns = df_pclass_test.columns.droplevel()
df_pclass_test.columns = ["Is1stClass", "Is2ndClass", "Is3rdClass"]
df_pclass_test.head()
# Stitch these transformed categorical columns back to the main dataframe
df_test_7 = pd.concat([df_test_6, df_embarked_test, df_pclass_test], axis="columns")
df_test_7.head()
df_test_7.shape
# Obtain our training features and ground truth labels
test_X = df_test_7[params["selected_features"]].values

print(test_X.shape)
from sklearn import preprocessing

test_X_scaled_std = preprocessing.StandardScaler().fit(test_X)
test_X_scaled = test_X_scaled_std.transform(test_X)
test_X_scaled.shape
# Take a peek at first 5 test samples - not that all the values are more or less within the same range.
test_X_scaled[:5]
# After standardization we should expect a mean value of 0, and standard deviation of 1.
print("test input feature mean after standardization: {}".format(test_X_scaled.mean()))
print("test input feature std after standardization: {}".format(test_X_scaled.std()))
def predict(parameters, model):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        saver.restore(sess, params['model_name'])
        return model['prediction'].eval({model['X']: test_X})
answer = pd.DataFrame(df_test['PassengerId'], columns=['PassengerId'])
with tf.Graph().as_default():
    model = make_model(params)
    test_Y = predict(params, model)
    answer['Survived'] = test_Y.astype(int)
# This is the CSV that we will submit to Kaggle
# Uncomment the following line to export to CSV

# answer.to_csv('./titanic/titanic_test_predictions.csv', index=False)
answer.head(20)
test_survived_group = answer.groupby('Survived')['Survived']
test_survived_group.count().plot(kind="bar")
