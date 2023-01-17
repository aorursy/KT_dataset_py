import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import ShuffleSplit

import numpy as np

import tensorflow as tf

import tflearn
train_df = pd.read_csv("../input/train.csv")

train_df.drop('ID', axis=1, inplace=True)

test_df = pd.read_csv("../input/test.csv")

test_df.drop('ID', axis=1, inplace=True)

print("Train shape : ", train_df.shape)

print("Test shape : ", test_df.shape)
dtype_df = train_df.dtypes.reset_index()

dtype_df.columns = ['name','type']

dtype_df.groupby('type').aggregate('count').reset_index()
def get_dummy_values(dummy_fields, base_df):

    # This will create dummy values for all categorical features

    # and remove the original features from the dataset

    for each in dummy_fields:

        dummies = pd.get_dummies(base_df[each], prefix=each, drop_first=False)

        base_df = pd.concat([base_df, dummies], axis=1)

    data = base_df.drop(dummy_fields, axis=1)

    return data



def get_scaled_values(quant_features, data):

    # Store scalings in a dictionary so we can convert back later

    scaled_features = {}

    for each in quant_features:

        mean, std = data[each].mean(), data[each].std()

        scaled_features[each] = [mean, std]

        data.loc[:, each] = (data[each] - mean)/std

    return scaled_features, data
# ID all the categoical and numerical features

cat_vars = dtype_df.name[dtype_df.type=='object'].tolist()

print(cat_vars)

num_vars = dtype_df.name[dtype_df.type=='float64'].tolist()

print(num_vars)
# We want to create dummy variables for all categorical features

data = get_dummy_values(cat_vars, train_df)

# We only need to scale the target variable. All other numerical features 

# are binary categories so no need to scale them

scaled_features, data = get_scaled_values(num_vars, data)



# just need to get the categorical variables for the test data

test_data = get_dummy_values(cat_vars, test_df)



print(data.shape, test_data.shape)

data.head()
# Create train and test sets. In this case we'll set the test size to 0 as we want to use all the data for training

ss = ShuffleSplit(n_splits=1, test_size=0.0)

target_fields = ['y']

features, targets = data.drop(target_fields, axis=1), data[target_fields]

for train_index, test_index in ss.split(features, targets):

    train_x, test_x = features.values[train_index], features.values[test_index]

    train_y, test_y = targets.values[train_index], targets.values[test_index]
print("Train shapes (x, y):", train_x.shape, train_y.shape)

print("Test shapes (x, y):", test_x.shape, test_y.shape)
# Define the neural network

def build_model():

    # This resets all parameters and variables, leave this here

    tf.reset_default_graph()

    

    # Inputs

    net = tflearn.input_data([None, train_x.shape[1]])



    # Hidden layer(s)

    net = tflearn.fully_connected(net, 100, activation='ReLU')

    net = tflearn.fully_connected(net, 100, activation='ReLU')

    net = tflearn.fully_connected(net, 100, activation='ReLU')

    

    # Output layer and training model

    net = tflearn.fully_connected(net, 1, activation='linear')

    

    # The regression layer is used in TFLearn to apply a regression (linear or logistic) to the provided input. 

    # It requires to specify a TensorFlow gradient descent optimizer 'optimizer' that will minimize the provided 

    # loss function 'loss' (which calculate the errors). A metric can also be provided, to evaluate the model performance.

    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001, loss='mean_square', metric='R2')

    

    model = tflearn.DNN(net, tensorboard_verbose=3)

    return model
model = build_model()
# Training

model.fit(train_x, train_y, validation_set=0.1, show_metric=True, batch_size=None, n_epoch=100)
# There are differences between the trainging and test sets provided

dlist = data.columns.tolist()

tlist = test_data.columns.tolist()

# This tells us which features are in training and not in test

buffer_list_one = list(set(dlist)-set(tlist))

# Add these to test with 0 value

buffer_list_one.remove('y')

for each in buffer_list_one:

    test_data[each] = 0
# This tells us which features are in test and not in training

buffer_list_two = list(set(tlist)-set(dlist))

# drop these columns from the test data set

test_data.drop(buffer_list_two, axis=1, inplace=True)
test_data.shape
mean, std = scaled_features['y']

predictions = np.array(model.predict(test_data))*std + mean
predictions
submission_data = pd.read_csv('../input/test.csv')

submission_data['y'] = np.absolute(predictions)

submission_data.to_csv('submission.csv',columns=['ID','y'],header=['ID','y'],index=False)