# Basic import of libraries to support our analysis

import numpy as np 

import pandas as pd



# Visualization Libs

import matplotlib.pyplot as plt

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import tensorflow as tf
# Check which version of TF is hosted 

print(tf.__version__)
train=pd.read_csv("../input/Kannada-MNIST/train.csv")

test=pd.read_csv("../input/Kannada-MNIST/test.csv")



print("Train Shape: {}".format(train.shape))

print("Test Shape: {}".format(test.shape))
train.head()
test.head()
print("Train Nulls: {}".format(train.isna().any().sum()))

print("Test Nulls: {}".format(test.isna().any().sum()))
# Training set

fig, axs = plt.subplots(2, 5, figsize=(16,6))

for i, ax  in zip(range(0,10), axs.flat):

    ax.imshow(train[train['label'] == i].drop(columns=['label']).iloc[0].values.reshape(28, 28), cmap='gray')
fig, axs = plt.subplots(2, 5, figsize=(16,6))

for i, ax  in zip(range(0,10), axs.flat):

    ax.imshow(test.drop(columns=['id']).iloc[i].values.reshape(28, 28), cmap='gray')
# Plot the distribution of each label

plt.figure(figsize=(24, 7))

plt.hist(train['label'], color='c', rwidth=0.5, align='mid')

plt.xlabel('Digits')

plt.ylabel('Frequency')

plt.title('Distribution of Labels')

plt.show()
# Downcasting all the values to save memory

y = train['label'].astype('int8')



# Downcast to float16 for every column except label

X = train.drop(columns=['label']).astype('float16').values



# Reshape the arrays so they are easier to visualize and input to NN

X = X.reshape(len(train), 28,28, 1)
from sklearn.model_selection import train_test_split



X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2)
from tensorflow.keras.preprocessing.image import ImageDataGenerator



# Create our image generator

train_image_generator = ImageDataGenerator(

    rescale=1./255

)



validation_image_generator = ImageDataGenerator(

    rescale=1./255

)



# Create instance of image generator attach to the dataset

train_image_gen = train_image_generator.flow(

    x=X_train, 

    y=y_train,

)



validation_image_gen = validation_image_generator.flow(

    x=X_validation, 

    y=y_validation,

)
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3,3), activation='relu',padding='same', input_shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu',padding='same'),

    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Dropout(0.1),

    

    tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu',padding='same'),

    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Dropout(0.1),

    

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Dense(10, activation='softmax')

])
tf.keras.backend.clear_session()  # For easy reset of notebook state.



model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



history = model.fit(

    train_image_gen,

    epochs=5,

    validation_data=validation_image_gen

)
# Plot the loss and accuracy of the validation and training set

fig, axs = plt.subplots(1,2, figsize=(20,5))

axs[0].plot(history.history['loss'], label='Train')

axs[0].plot(history.history['val_loss'], label='Validation')

axs[0].set_title("Loss of Validation and Train")

axs[0].legend()



axs[1].plot(history.history['accuracy'], label='Train')

axs[1].plot(history.history['val_accuracy'], label='Validation')

axs[1].set_title("Accuracy of Validation and Train")

axs[1].legend()



fig.show()
# Lets predict the values on the validation set

validation_predict = model.predict(X_validation)



# Create a dataframe to store the label and the confidence in their predictions

low_predictions = pd.DataFrame()

low_predictions['label'] = np.argmax(validation_predict, axis=1)

low_predictions['Confidence'] = np.max(validation_predict, axis=1)



low_index = low_predictions.sort_values(by=['Confidence'])[:10].index

low_labels = low_predictions.sort_values(by=['Confidence'])['label']



fig, axs = plt.subplots(2, 5, figsize=(24,9.5))

for i, low_label, ax  in zip(low_index,low_labels, axs.flat):

    image = X_validation[i].astype('float32').reshape(28, 28)

    ax.imshow(image, cmap='gray')

    ax.set_xlabel("True: {}".format(y_validation.iloc[i]))

    ax.set_title("Guessed:{} ".format(low_label))
# Create our image generator

train_image_generator = ImageDataGenerator(

    rescale=1./255,

    rotation_range=25,

    width_shift_range=.1,

    height_shift_range=.1,

    zoom_range=0.1

)



validation_image_generator = ImageDataGenerator(

    rescale=1./255,

    rotation_range=25,

    width_shift_range=.1,

    height_shift_range=.1,

    zoom_range=0.1

)



# Create the dataset

train_image_gen = train_image_generator.flow(

    x=X_train, 

    y=y_train,

)



validation_image_gen = validation_image_generator.flow(

    x=X_validation, 

    y=y_validation,

)



lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(

    initial_learning_rate=0.001,

    decay_steps=250,

    decay_rate=1,

    staircase=False

)



def get_optimizer():

    return tf.keras.optimizers.Adam(lr_schedule)



tf.keras.backend.clear_session()  # For easy reset of notebook state.



model.compile(optimizer=get_optimizer(),

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



history = model.fit(

    train_image_gen,

    epochs=100,

    validation_data=validation_image_gen

)
# Plot the loss and accuracy of validation and train

fig, axs = plt.subplots(1,2, figsize=(20,5))

axs[0].plot(history.history['loss'], label='Train')

axs[0].plot(history.history['val_loss'], label='Validation')

axs[0].set_title("Loss of Validation and Train")

axs[0].legend()



axs[1].plot(history.history['accuracy'], label='Train')

axs[1].plot(history.history['val_accuracy'], label='Validation')

axs[1].set_title("Accuracy of Validation and Train")

axs[1].legend()



fig.show()
X_test = test.drop(columns=['id']).astype('float16').values

X_test = X_test / 255.0

X_test = X_test.reshape(len(X_test), 28, 28, 1)



label_pred = model.predict_classes(X_test, verbose=0)



submission = pd.DataFrame()

submission['label'] = label_pred

submission['id'] = submission.index

submission.to_csv('../working/submission.csv', index=False)