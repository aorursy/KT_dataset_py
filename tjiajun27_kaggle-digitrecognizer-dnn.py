# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"
import matplotlib.pyplot as plt
%matplotlib inline

# Read training data from csv file.
train_data_path = '/kaggle/input/digit-recognizer/train.csv'
train_data = pd.read_csv(train_data_path)
train_data.head()
train_labels = train_data[['label']]
train_labels.head()
index = dict()

for label in train_labels.label.unique():
    index[label] = train_labels.loc[train_labels['label'] == label].index
# Inspect the number of data in each class
count_of_each_label = train_labels.label.value_counts(sort=False)
print(count_of_each_label)
count_of_each_label.plot(kind="bar", rot=0);
# create train and val set, 80-20 ratio for each class
from sklearn.model_selection import train_test_split

train_index = []
val_index = []
for label in train_labels.label.unique():
    train, val = train_test_split(index[label], train_size=0.8, shuffle=True, random_state=42)
    train_index = [*train_index, *train]
    val_index = [*val_index, *val]
    
train_X = train_data.iloc[train_index].drop(columns=['label'])
train_y = train_data.iloc[train_index][['label']]
val_X = train_data.iloc[val_index].drop(columns=['label'])
val_y = train_data.iloc[val_index][['label']]
# Normalize
train_X = train_X/255
val_X = val_X/255
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import datetime

def create_model(input_shape, layer, num_hidden_node, hiddenX_rate, dropout):
    
    inputs = tf.keras.Input(shape=input_shape)
    model = tf.keras.layers.Dense(num_hidden_node, activation=tf.nn.relu)(inputs)
    if(dropout):
        model = tf.keras.layers.Dropout(0.2)(model)
    if(layer>0):
        for x in range (layer):
            model = tf.keras.layers.Dense(int(num_hidden_node*hiddenX_rate[x]), activation=tf.nn.relu)(model)
            if(dropout):
                model = tf.keras.layers.Dropout(0.2)(model)
    output = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(model)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy'],
    )
    return model

model =dict()
num_models = 3
output_shape =10
input_shape = ((train_X).shape[1])
hidden_node = (input_shape+output_shape) *(2/3)
hiddenX_rate = [.75, .5, .25]

for index in range(num_models):
    model[index] = create_model(input_shape, index, int(hidden_node), hiddenX_rate, dropout=False)
    model[index].summary()
history = [0]* len(model)

for index in range(len(model)):
    history[index] = model[index].fit(train_X, train_y, validation_data= (val_X, val_y), batch_size=64, epochs=30)
# summarize history for accuracy
for x in range (len(history)):
    print("model {}: Epochs={}, Train accuracy={}, Validation accuracy={}".format(x,30,max(history[x].history['accuracy']),
                                                                                     max(history[x].history['val_accuracy']) ))
    plt.plot(history[x].history['val_accuracy'])
    
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['model {}'.format(y) for y in range(len(history))], loc='lower right')
plt.show()
# summarize history for loss
for x in range (len(history)):
    plt.plot(history[x].history['loss'])
    
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['model {}'.format(y) for y in range(len(history))], loc='upper right')
plt.show()
# Read in the test features.
test_data_path = '/kaggle/input/digit-recognizer/test.csv'
test_data = pd.read_csv(test_data_path)
test_data.head()
# Predict the all the test features.
test_probs = model[0].predict(test_data)
test_labels = test_probs.argmax(axis=-1)
# Generate Submission File.
submission_df = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
submission_df.Label = test_labels
submission_df.to_csv("output.csv", index=False)