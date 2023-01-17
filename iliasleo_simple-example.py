# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf # tensorflow

import glob 

import re # regular expression

# Input data files are available in the "../input/" directory.
'''

  This will load the data into memory (without any processing)

  Each sample will be returned in a dictionary that contains:

    - the sample_id

    - the label (if its a train sample)

    - a pandas dataframe with all the accelerometer/gyro data for this sample

    

  For a list of samples (dictionaries) will be returned seperatelly for train/test set

    

'''







def load_data():

    train_data = [] # we will store the train data here

    test_data = [] # we will store the test data here

    

    # read training labels

    labels = pd.read_csv("../input/smartphone_data/train_labels.csv")

    label_dictionary = labels.set_index("id")["label"].to_dict() # get the labels as a dictionary by sample ID

    

    # read training data. Iterate files in the train folder

    for filename in glob.glob("../input/smartphone_data/train/train_*.csv"): # iterate through all the files

        # extract the sample id from the file name

        m = re.search(r"\S+\_(\d+).csv", filename)

        if m is not None:

            # if the filename is as expected then store the sample

            sample_id = int(m.group(1)) # get the group from the filename (regular expression)

            label = label_dictionary[sample_id] # get the training label from the dictionary created above

            sample_data = pd.read_csv(filename) # finally read the accelerometer data

            # store the data in a dictionary for further processing later

            sample = {"sample_id": sample_id, "label": label, "sample_data" : sample_data}

            train_data.append(sample)

      

    # similarly, load data for test set (notice that no labels exist here)

    for filename in glob.glob("../input/smartphone_data/test/test_*.csv"): # iterate through all the files

        # extract the sample id from the file name

        m = re.search(r"\S+\_(\d+).csv", filename)

        if m is not None:

            # if the filename is as expected then store the sample

            sample_id = int(m.group(1)) # get the group from the filename (regular expression)

            label = None

            sample_data = pd.read_csv(filename) # finally read the accelerometer data

            # store the data in a dictionary for further processing later

            sample = {"sample_id": sample_id, "label": label, "sample_data" : sample_data}

            test_data.append(sample)

    return train_data, test_data

        



# load data

train_data, test_data = load_data()
# lets print an example

sample = train_data[20]

print (sample['label'])

print (sample['sample_id'])

display(sample['sample_data'].head(3))

sample['sample_data'].plot(figsize=(10,5))
'''

  This function will be used to process the data and return your model features

  Feel free to modify this, but in theory you should return:

  - For training: 

     - X a n-dimentional numpy array with the training input (or pandas array if only 2 Dimentions are used)

     - Y (if available) the  labels for these samples. Notice that the order of samples should be the same as train_X. For the test set this will be NaN

     - original_sample_ids this is to keep track of which sample corresponds to which ID in the original filenames. You will need this to align the results with the submission IDs



  Feature construction:

      In this example a simple mean and standard deviation is returned for the acceleration values per sample. 

      Therefore, the whole time-series is collapsed into a few numbers. 

      As a results train_X is a 2-D array that fontains one line per sample with 6 fatures (columns)

  

  WHAT TO MODIFY:

      This is one of the functions that you will need to modify. 

      For RNNs/CNNs you might want to return the whole time-series as numpy arrays. In this case you would need

      to return 3D numpy arrays in the format (samples, time, features)

      You can also choose to return a second X-array if you plan to have a multi-input model

'''



labels = ["dws","ups", "wlk", "jog", "std", "sit"]





def process_sample_data(data, labels):

    X = []

    y = []

    original_sample_ids = []

    

    for d in data:

        original_sample_id = d['sample_id']

        label     = d['label']

        sample_data = d['sample_data']

        

        # Extracting features. In this example we collapse the time series into mean and std for the accelerometer values

        features = sample_data[['userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z']].agg(['mean','std']).unstack().tolist()

        X.append(features)

        y.append(label)

        original_sample_ids.append(original_sample_id)

        

    # convert to X to numpy array (you can also directly store your features in numpy)

    X = np.array(X)

    

    # one-hot encode Y (expected by softmax classification)

    if labels is None:

        y = None

    else:

        y = pd.get_dummies(y)[labels] # get dummies performs one-hot encode. We also use the "labels" list to make sure the order is as expected

        y = np.array(y)

    

    return X, y, original_sample_ids

        

  



train_X, train_y, train_sample_ids =  process_sample_data(train_data, labels)

test_X, _ , test_sample_ids =  process_sample_data(test_data, None)
print (train_X.shape)  # in this example we have 255 sample to train with 6 features each

print (train_y.shape)  # we have 6 possible labels for each sample ("dws","ups", "wlk", "jog", "std", "sit")



print (test_X.shape)   # we have 105 test samples. We don't have the labels for those. our model has to guess them from the input data
def create_keras_model( num_classes):

    model = tf.keras.Sequential()

    # Adds a densely-connected layer with 64 units to the model:

    model.add(tf.keras.layers.Dense(64, activation='relu'))

    # Add one more layer

    model.add(tf.keras.layers.Dense(64, activation='relu'))

    # Add a softmax layer with num_classes output units:

    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    

    model.compile(optimizer=tf.train.AdamOptimizer(),

              loss='categorical_crossentropy',

              metrics=['accuracy'])

    

    return model

# simple model training. 

# you might want to avoid overfitting by monitoring validation loss and implement early stopping, etc

def train_model(model, X, y):

    model.fit(X, y, epochs=200, batch_size=32)



    

    

def predict(model, X):

    y_pred = model.predict(X, batch_size=32)

    return y_pred
# running it

model = create_keras_model(6)

train_model(model, train_X, train_y)
y_pred = predict(model, test_X)



# convert predictions to the kaggle format

y_pred_numerical = np.argmax(y_pred, axis = 1) # one-hot to numerical

y_pred_cat = [labels[x] for x in y_pred_numerical] # numerical to string label



# generate the table with the correct IDs for kaggle.

# we get the correct sample ID from the stored array (test_sample_ids)

submission_results = pd.DataFrame({'id':test_sample_ids, 'label':y_pred_cat})

submission_results.to_csv("submission.csv", index=False)
