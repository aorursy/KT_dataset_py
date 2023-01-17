%matplotlib inline



import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
# Open the file for reading...

data = pd.read_csv('../input/Iris.csv', index_col='Id')

data.head()
# Normalise data ~ May not need doing as all measurements in CM

data['SepalLengthCm'] /= data['SepalLengthCm'].max()

data['SepalWidthCm'] /= data['SepalWidthCm'].max()

data['PetalLengthCm'] /= data['PetalLengthCm'].max()

data['PetalWidthCm'] /= data['PetalWidthCm'].max()
data.head()
# Split data for test and train

train, test = train_test_split(data, test_size=0.25)
train = train.drop( 'Species', axis=1)

test_target = test['Species']

test = test.drop( 'Species', axis=1)
# Find distance between two inputs

def euclidean_distance( input1, input2):

    if len(input1) != len(input2):

        raise ValueError('euclidean_distance: input dimensions do not match.')

    else:

        dist = 0

        for col in range(len(input1)):

            dist += (input2[col] - input1[col])**2

        return np.sqrt(dist)
# Find nearest neighbour for a single input row

def find_nn( train, test_row):

    distances = []

    for index, row in train.iterrows():

        distances.append((index, euclidean_distance( row, test_row)))

    distances.sort(key=lambda tup: tup[1])

    return distances[0][0] # Returns index of NN
def find_prediction( index, full_data):

    return full_data.get_value(index, 'Species')
def predict_nn_value( train, test, full):

    predictions = []

    for index, test_row in test.iterrows():

        idx = find_nn( train, test_row)

        predictions.append(find_prediction( idx, full))

    return predictions
predictions = predict_nn_value( train, test, data)

print('The accuracy score of the model is ', accuracy_score(test_target, predictions) )
predictions[:10]