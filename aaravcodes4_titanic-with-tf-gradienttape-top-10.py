# Import Some Libraries Needed

# If you are actually running the notebook uncomment the last

# 'mkdir' line to make the directory

# then comment it back after running



import numpy as np # Vital for Math

print('Numpy Import Success')

import pandas as pd  # Data Reading

print('Pandas Import Success')

import tensorflow as tf # Main Model Library

print('Tensorflow Import Success')

from keras.models import Sequential # Keras

from keras.layers import Dense # Keras

from keras import optimizers # Keras

from keras import losses # Keras

print('Keras Import Success')

from tqdm import tqdm # Progress Bar

print('Tqdm Import Success')

import matplotlib.pyplot as plt # Data Visualization

print('Matplotlib Import Success')

import os # Creating file to save model

os.mkdir("/kaggle/working/models")

# Load the data from training

df = pd.read_csv('../input/titanic/train.csv')

df.columns
# Passenger Id and Name would have no effect

# drop them

X = df.drop(['PassengerId','Name','Ticket','Survived'], axis=1)

y = df['Survived']

print(X)
# A Bunch of preprocessing functions

# from this helpful notebook: https://www.kaggle.com/jameskhoo/deep-learning-with-keras-and-tensorflow



def simplify_ages(df):

    #df['Age'] = df['Age'].fillna(-0.5)

    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)

    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

    categories = pd.cut(df['Age'], bins, labels=group_names)

    df['Age'] = categories.cat.codes 

    return df



def simplify_cabins(df):

    df['Cabin'] = df['Cabin'].fillna('N')

    df['Cabin'] = df['Cabin'].apply(lambda x: x[0])

    df['Cabin'] =  pd.Categorical(df['Cabin'])

    df['Cabin'] = df['Cabin'].cat.codes 

    return df



def simplify_fares(df):

    df['Fare'] = df.Fare.fillna(-0.5)

    bins = (-1, 0, 8, 15, 31, 1000)

    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']

    categories = pd.cut(df['Fare'], bins, labels=group_names)

    df['Fare'] = categories.cat.codes 

    return df



def simplify_sex(df):

    df['Sex'] = pd.Categorical(df['Sex'])

    df['Sex'] = df['Sex'].cat.codes 

    return df



def simplify_embarked(df):

    df['Embarked'] = pd.Categorical(df['Embarked'])

    df['Embarked'] = df['Embarked'].cat.codes + 1

    return df



def transform_features(df):

    df = simplify_ages(df)

    df = simplify_cabins(df)

    df = simplify_fares(df)

    df = simplify_sex(df)

    df = simplify_embarked(df)

    return df



X = transform_features(X).to_numpy()

y = y.to_numpy()
# Defining and building the model

def build_keras_model():

    model = Sequential([

        Dense(32, activation='relu'),

        Dense(32, activation='relu'),

        Dense(16, activation='relu'),

        Dense(1, activation='sigmoid')

    ])

    return model

model = build_keras_model()
# Initializing

model(tf.constant(X, dtype=tf.float32))
# Getting a Batch of Data

def get_batch(x, y, batch_size):

    idx = np.random.choice(len(x), batch_size)

    input_batch = [x[i] for i in idx]

    label_batch = [y[i] for i in idx]

    return input_batch, label_batch

# Making 1D array to 2D matrix

new_y = np.reshape(y, (-1, 1))
# Binary Crossentropy Function

def compute_loss(labels, logits):

  loss = tf.keras.losses.binary_crossentropy(labels, logits, from_logits=True)

  return loss
# IMPORTANT:

# parameters for training

# changing them may change the model a lot



# Learning Rate

learning_rate = 1e-2

# Iterations

epochs = 50000

# Batch Size

batch_size = 16



# Defining our optimizer

# Adam and Adagrad are also suitable

optimizer = tf.keras.optimizers.RMSprop(learning_rate)



# Defining a checkpoint to save model

checkpoint_dir = '/kaggle/working/models/model.ckpt'



# Train Step Function

@tf.function

def train_step(x, y): 

  # Use tf.GradientTape()

  with tf.GradientTape() as tape:

    # Prediciting labels

    y_hat = model(x) 

    # Computing Loss

    loss = compute_loss(y, y_hat) 

  # Backprogating through landscape

  grads = tape.gradient(loss, model.weights) # TODO

  # Descending to local minimum

  optimizer.apply_gradients(zip(grads, model.weights))

  return loss



# History for finding minimum

history = []



# Clearing Bar

if hasattr(tqdm, '_instances'): tqdm._instances.clear()

    



for iter in tqdm(range(epochs)):

  # Get a Batch

  x_batch, y = get_batch(X, new_y, batch_size)

  y_batch = tf.constant(np.reshape(y, (-1, 1)), dtype=tf.float32)

  x_batch = tf.constant(x_batch, dtype=tf.float32)

  loss = train_step(x_batch, y_batch)

    

  # Update the progress bar

  history.append(loss.numpy().mean())

  num_iterations = [i for i in range(len(history))]

  # Update the model with the changed weights!

  if iter % 100 == 0:     

    model.save_weights(checkpoint_dir)

    

# Save the trained model and the weights

plt.plot(num_iterations, history)

plt.ylabel('Loss')

plt.xlabel('Epochs')

model.save_weights(checkpoint_dir)
print(f'Local Minimum: {min(history)}')

print(f'Final Loss: {history[len(history) -1]}')
# Submitting Predictions

testing_set = pd.read_csv('../input/titanic/test.csv')

x_test = testing_set.drop(['PassengerId','Name','Ticket'], axis=1)

x_test = transform_features(x_test)



predictions = model.predict_classes(x_test)

ids = testing_set['PassengerId'].copy()

new_output = ids.to_frame()

new_output["Survived"]=predictions

new_output.to_csv("another_submission3.csv",index=False)

print(new_output)