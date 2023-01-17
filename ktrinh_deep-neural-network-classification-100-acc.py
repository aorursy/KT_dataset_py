'''
Project:    Mushroom Classification - Data Visualizer
Purpose:    Explore the Mushroom data set prior to ML

@author:    Kevin Trinh
'''

import numpy as np
from numpy.core.defchararray import add
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def histCompare(edf, pdf, feature):
    '''Plot a dual histogram of edible and poisonous mushrooms for a 
    certain feature.
    @param edf --> (pandas dataframe) a dataframe of edible mushrooms
    @param pdf --> (pandas dataframe) a dataframe of poisonous mushrooms
    @param feature --> (string) the name of the feature to be compared
    '''
    e_array = list(edf[feature])
    p_array = list(pdf[feature])
    plt.hist([e_array, p_array], color=['b', 'g'], alpha=0.5)
    plt.xlabel(feature)
    plt.title('Histogram (n = 8124)')
    plt.legend(['Edible', 'Poisonous'])
    plt.show()

# read in .csv data as pandas dataframe
mushroom_df = pd.read_csv('../input/mushroom-classification/mushrooms.csv', encoding='utf-8')

# separate dataframe by class
edible_df = mushroom_df.loc[mushroom_df['class'] == 'e']
poisonous_df = mushroom_df.loc[mushroom_df['class'] == 'p']

# obtain list of features
features = list(mushroom_df)

# generate comparative histograms for each feature
for feat in features:
    histCompare(edible_df, poisonous_df, feat)
"""
Project:    Mushroom Classification -- Feature Engineering
Purpose:    - Encode categorical data
            - omit redundant and highly skewed features

@author:    Kevin Trinh
"""


import numpy as np
from numpy.core.defchararray import add
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def encodeDummy(df, feature):
    '''Encode a given feature into dummy variables, omitting the first
    alphabetically-sorted category. Remove the original feature.
    
    @param df --> (pandas dataframe) dataframe to be modified
    @param feature --> (str) name of feature
    @return df --> (pandas dataframe) modified dataframe
    '''
    labels = np.unique(df[feature])
    labels = add(feature, labels)
    le = LabelEncoder()
    dummy_labels = le.fit_transform(df[feature])
    df[feature] = dummy_labels
    dummy_features = pd.get_dummies(df[feature], drop_first=True)
    df[labels[1:]] = dummy_features
    return df.drop(feature, axis=1)
    

def encodeBinary(df, feature, positive):
    '''Encode a given feature into a binary variable with 'positive' as 1 and
    all other values as 0.
    
    @param df --> (pandas dataframe) dataframe to be modified
    @param feature --> (str) name of feature
    @param positive --> (str) category to be a positive binary
    @return df --> (pandas dataframe) modified dataframe
    '''
    positive_arr = df[feature] == positive
    df.loc[positive_arr, feature] = 1
    df.loc[~positive_arr, feature] = 0
    return df

def encodeOmit(df, feature):
    '''Omit feature from dataframe.
    
    @param df --> (pandas dataframe) dataframe to be modified
    @param feature --> (str) name of feature
    @return df --> (pandas dataframe) modified dataframe
    '''
    return df.drop(feature, axis=1)


# read in .csv data as pandas dataframe
mushroom_df = pd.read_csv('mushrooms.csv', encoding='utf-8')

# select features to encode or omit
my_dummies = ['cap-shape', 'cap-surface', 'cap-color', 'odor', 'gill-color',
              'stalk-root', 'stalk-surface-above-ring', 
              'stalk-color-above-ring', 'ring-type', 'spore-print-color', 
              'population', 'habitat']

my_binaries = [('class', 'e'), ('bruises', 't'), ('gill-attachment', 'f'),
               ('gill-spacing', 'c'), ('gill-size', 'b'), ('stalk-shape', 't'), 
               ('ring-number', 'o')]

my_omissions = ['stalk-surface-below-ring', 'stalk-color-below-ring',
                'veil-type', 'veil-color']


# encode dataframe
for feat in my_dummies:
    mushroom_df = encodeDummy(mushroom_df, feat)
for feat, pos in my_binaries:
    mushroom_df = encodeBinary(mushroom_df, feat, pos)
for feat in my_omissions:
    mushroom_df = encodeOmit(mushroom_df, feat)



mushroom_df.to_csv('mushrooms_encoded.csv')
"""
Project:    Mushroom Classification -- Neural Network
Purpose:    Construct a neural network to predict mushroom edibility

            Note: Run mushroomEncoder.py before running mushroomClassifier.py

@author:    Kevin Trinh
"""


import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.metrics import binary_crossentropy
import matplotlib.pyplot as plt


# read in and shuffle encoded data
mushroom_df = pd.read_csv('../input/mushrooms-encoded/mushrooms_encoded.csv', encoding='utf-8')
mushroom_df = mushroom_df.drop(mushroom_df.columns[0], axis=1) # omit index column
mushroom_df = mushroom_df.sample(frac=1)

# partition into training (60%), validation (20%), and test set (20%)
samples = mushroom_df.shape[0]
train_count = round(samples * 0.6)
val_count = round(samples * 0.2)
test_count = samples - train_count - val_count

train_df = mushroom_df.iloc[:train_count]
validation_df = mushroom_df.iloc[train_count:train_count + val_count]
test_df = mushroom_df.iloc[-test_count:]

X_train = train_df.drop(['class'], axis=1)
X_validation = validation_df.drop(['class'], axis=1)
X_test = test_df.drop(['class'], axis=1)

y_train = train_df['class']
y_validation = validation_df['class']
y_test = test_df['class']


### Build neural network architecture ###
num_features = mushroom_df.shape[1] - 1

model = Sequential()
model.add(Dense(16, input_dim=num_features, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(12, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid', name='output'))
  
model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['binary_accuracy'])

# train NN
my_epochs = 50
history = model.fit(X_train, y_train, epochs=my_epochs, batch_size=20,
                    validation_data=(X_validation, y_validation))

# plot model loss while training
epochs_arr = np.arange(1, my_epochs + 1, 1)
my_history = history.history
line1 = plt.plot(epochs_arr, my_history['loss'], 'r-', label='training loss')
line2 = plt.plot(epochs_arr, my_history['val_loss'], 'b-', label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model loss')
plt.legend()
plt.show()

# plot model accuracy while training
line1 = plt.plot(epochs_arr, my_history['binary_accuracy'], 'r-', label='training accuracy')
line2 = plt.plot(epochs_arr, my_history['val_binary_accuracy'], 'b-', label='validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model accuracy')
plt.legend()
plt.show()


# evaluate the keras model against the test set (DO ONCE)
_, accuracy = model.evaluate(X_test, y_test)
print('Test Accuracy: %.2f' % (accuracy*100))