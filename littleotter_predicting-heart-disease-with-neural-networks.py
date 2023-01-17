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
import pandas as pd

import numpy as np

import math

import matplotlib.pyplot as plt



# Make numpy values easier to read.

np.set_printoptions(precision=3, suppress=True)



from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras import layers

from tensorflow.keras.layers.experimental import preprocessing
heart_disease = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

heart_disease.head()
heart_disease.dtypes
def convert_datatypes(df):

    """Converts datatypes of the UCI heart disease dataset

    

    Arg:

        df: UCI heart disease dataframe

    

    Return:

        returns the dataframe with the converted datatypes

    """

    df['age'] = df['age'].astype('float')

    df['trestbps'] = df['trestbps'].astype('float')

    df['chol'] = df['chol'].astype('float')

    df['thalach'] = df['thalach'].astype('float')

    df['oldpeak'] = df['oldpeak'].astype('float')

    

    return df
# Convert datatypes

heart_disease = convert_datatypes(heart_disease)



# Extract features and labels

heart_disease_features = heart_disease.copy()

heart_disease_labels = heart_disease_features.pop('target')



# Split train and test data

X_train, X_test, y_train, y_test = train_test_split(heart_disease_features, heart_disease_labels, test_size=0.2, random_state=42)
# Initialize a dictionary to build a set of symbolic keras.Input objects matching the names and data-types of the CSV columns.

inputs = {}



for name, column in X_train.items():

    dtype = column.dtype

    if dtype == object:

        dtype = tf.string

    elif dtype == 'int64':

        dtype = tf.int64

    else:

        dtype = tf.float32

        

    inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

inputs
# Concatenate the numeric inputs together and run them through a normalization layer

numeric_inputs = {name:input for name,input in inputs.items() if input.dtype==tf.float32}



x = layers.Concatenate()(list(numeric_inputs.values()))

norm = preprocessing.Normalization()

norm.adapt(np.array(X_train[numeric_inputs.keys()]))

all_numeric_inputs = norm(x)
# Collect all the symbolic preprocessing results

preprocessed_inputs = [all_numeric_inputs]
# For the string inputs use the preprocessing.StringLookup function to map from strings to integer indices in a vocabulary. 

# Use preprocessing.CategoricalEncoding to convert the indexes into float32 data appropriate for the model.

for name, input in inputs.items():

    

    if input.dtype == tf.float32:

        continue

        

    elif input.dtype == tf.int64:

        # One hot encode integers

        one_hot = preprocessing.CategoryEncoding(max_tokens=len(X_train[name].unique())) 

        x = one_hot(input)

        preprocessed_inputs.append(x)

        

    else:

        # One hot encode strings

        lookup = preprocessing.StringLookup(vocabulary=np.unique(X_train[name]))

        one_hot = preprocessing.CategoryEncoding(max_tokens=lookup.vocab_size())



        x = lookup(input)

        x = one_hot(x)

        preprocessed_inputs.append(x)
# Concatenate all the preprocessed inputs together, and build a model that handles the preprocessing

preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)



heart_disease_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)



tf.keras.utils.plot_model(model = heart_disease_preprocessing , rankdir="LR", dpi=72, show_shapes=True)
def heart_disease_model(preprocessing_head, inputs):

    body = tf.keras.Sequential([

        layers.Dense(64, activation="relu"),

        layers.BatchNormalization(),

        layers.Dropout(.2),

        layers.Dense(64, activation="relu"),

        layers.BatchNormalization(),

        layers.Dropout(.2),

        layers.Dense(1),

    ])



    preprocessed_inputs = preprocessing_head(inputs)

    result = body(preprocessed_inputs)

    model = tf.keras.Model(inputs, result)



    model.compile(optimizer='adam',

              loss=tf.losses.BinaryCrossentropy(from_logits=True),

              metrics=['accuracy'])

    

    return model



heart_disease_model = heart_disease_model(heart_disease_preprocessing, inputs)
epochs = 50



# Feature dictionary

X_train_features_dict = {name: np.array(value) for name, value in X_train.items()}



# Train Model

history = heart_disease_model.fit(x=X_train_features_dict,

                               y=y_train,

                               epochs=epochs,

                               shuffle=True,

                               validation_split = 0.2,

                               callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5)])



# Save Model

print("Saving model...")

heart_disease_model.save('/kaggle/working/heart_disease_model')

loaded_model = tf.keras.models.load_model('/kaggle/working/heart_disease_model')
frame = pd.DataFrame(history.history)

epochs = np.arange(len(frame))



fig = plt.figure(figsize=(12,5))



# Accuracy plot

ax = fig.add_subplot(122)

ax.plot(epochs, frame['accuracy'], label="Train") 

ax.plot(epochs, frame['val_accuracy'], label="Validation") 

ax.set_xlabel("Epochs")

ax.set_ylabel("Mean Absolute Error") 

ax.set_title("Accuracy vs Epochs") 

ax.legend()



# Loss plot

ax = fig.add_subplot(121)

ax.plot(epochs, frame['loss'], label="Train") 

ax.plot(epochs, frame['val_loss'], label="Validation") 

ax.set_xlabel("Epochs")

ax.set_ylabel("Loss")

ax.set_title("Loss vs Epochs")

ax.legend()





plt.show()
input_dict = {name: tf.convert_to_tensor(value) for name, value in X_train.items()}

predictions = loaded_model.predict(input_dict)

prob = tf.nn.sigmoid(predictions)

predction_labels = list(map(lambda x: 1 if x > .5 else 0, np.array(prob).flatten()))
X_test_features_dict = {name: np.array(value) for name, value in X_test.items()}
test_loss, test_accuracy = loaded_model.evaluate(X_test_features_dict, y_test)

print(f"""

Accuracy: {test_accuracy}

Loss: {test_loss}

""")