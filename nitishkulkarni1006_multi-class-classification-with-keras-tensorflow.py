import numpy as np

import pandas as pd

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

#from keras.utils import np_utils

#from sklearn.preprocessing import LabelEncoder
# Random seed for reproducibility

seed = 10

np.random.seed(seed)

# Import data

df = pd.read_csv('../input/Sensorless_drive_diagnosis.csv', sep = ' ', header = None)

# Print first 10 samples

print(df.head(10))
# Check missing values

print(df.isna().sum())
# Remove missing values IF AVAILABLE and print first 10 samples

# df = df.dropna()

# print(df.head(10))

# print(df.shape)

# Divide data into features X and target (Classes) Y

X = df.loc[:,0:47]

Y = df.loc[:,48]

print(X.shape)

print(Y.shape)
# Statistical summary of the variables

print(X.describe())
# Check for class imbalance

print(df.groupby(Y).size())
# Normalize features within range 0 (minimum) and 1 (maximum)

scaler = MinMaxScaler(feature_range=(0, 1))

X = scaler.fit_transform(X)

X = pd.DataFrame(X)
# Convert target Y to one hot encoded Y for Neural Network

Y = pd.get_dummies(Y)

# If target is in string form, use following code:

# First encode target values as integers from string

# Then perform one hot encoding

# encoder = LabelEncoder()

# encoder.fit(Y)

# Y = encoder.transform(Y)

# Y = np_utils.to_categorical(Y)
# For Keras, convert dataframe to array values (Inbuilt requirement of Keras)

X = X.values

Y = Y.values
# First define baseline model. Then use it in Keras Classifier for the training

def baseline_model():

    # Create model here

    model = Sequential()

    model.add(Dense(15, input_dim = 48, activation = 'relu')) # Rectified Linear Unit Activation Function

    model.add(Dense(15, activation = 'relu'))

    model.add(Dense(11, activation = 'softmax')) # Softmax for multi-class classification

    # Compile model here

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    return model
# Create Keras Classifier and use predefined baseline model

estimator = KerasClassifier(build_fn = baseline_model, epochs = 100, batch_size = 10, verbose = 0)

# Try different values for epoch and batch size
# KFold Cross Validation

kfold = KFold(n_splits = 5, shuffle = True, random_state = seed)

# Try different values of splits e.g., 10
# Object to describe the result

results = cross_val_score(estimator, X, Y, cv = kfold)

# Result

print("Result: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))