# Import libraries for data wrangling, preprocessing and visualization

import numpy 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
# Importing libraries for building the neural network

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score
# Read data file

data = pd.read_csv("../input/data.csv", header=0)

seed = 5

numpy.random.seed(seed)
# Take a look at the data

print(data.head(2))
# Take a look at the types of data

data.info()
# Column Unnamed : 32 holds only null values, so it is of no use to us. We simply drop that column.

data.drop("Unnamed: 32",axis=1,inplace=True)

data.drop("id", axis=1, inplace=True)
# Check whether the column has been dropped

data.columns
# Select the columns to use for prediction in the neural network

prediction_var = ['radius_mean', 'texture_mean', 'perimeter_mean',

       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',

       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',

       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',

       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',

       'fractal_dimension_se', 'radius_worst', 'texture_worst',

       'perimeter_worst', 'area_worst', 'smoothness_worst',

       'compactness_worst', 'concavity_worst', 'concave points_worst',

       'symmetry_worst', 'fractal_dimension_worst']

X = data[prediction_var].values

Y = data.diagnosis.values
# Diagnosis values are strings. Changing them into numerical values using LabelEncoder.

encoder = LabelEncoder()

encoder.fit(Y)

encoded_Y = encoder.transform(Y)
# Baseline model for the neural network. We choose a hidden layer of 10 neurons. The lesser number of neurons helps to eliminate the redundancies in the data and select the more important features.

def create_baseline():

    # create model

    model = Sequential()

    model.add(Dense(10, input_dim=30, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    # Compile model. We use the the logarithmic loss function, and the Adam gradient optimizer.

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
# Evaluate model using standardized dataset. 

estimators = []

estimators.append(('standardize', StandardScaler()))

estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)))

pipeline = Pipeline(estimators)

kfold = StratifiedKFold(n_splits=10, shuffle=True)

results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)

print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))