import pandas as pd # Data processing, CSV file I/O 

import numpy as np # Linear Algebra

import matplotlib.pyplot as plt # Plotting graphs



import tensorflow as tf



# Since I have a GPU & I've GPU enabled, I am going to use the GPU version of keras 

# (NOTE: Ignore if you do not have GPU enabled)

from keras import backend as K

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

K.tensorflow_backend._get_available_gpus()



print(f'Tensorflow {tf.__version__}')
# Reading the training and testing datasets

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
# Let's explore the datasets by plotting a few images (from their pixel values)

# The number of rows & cols to plot

rows, cols = 4, 4



# Creating a matplotlib figure that holds 16 subplots (for 16 digit images)

plt.figure(figsize=(8,7))

plt.suptitle('Training Data')



# Plotting the first 16 datapoints from the training data

for i in range(4):

    for j in range(4):

        index = (i * cols) + j # Row major ordering

        plt.subplot(rows, cols, index + 1)

        plt.xticks([])

        plt.yticks([])

        

        label = train_data['label'].values[index]

        image = train_data.drop('label', axis=1).iloc[index].values.reshape(28, 28) # Reshaping the 1D array into a 2D array of pixel values

        

        plt.title(label)

        

        # Using a binary color map

        plt.imshow(image, cmap='binary')
# A function to scale the values and return the data as numpy arrays

def preprocess_data(df):

    # Training Data

    if 'label' in df.columns: 

        df_x = df.drop('label', axis=1) / 255.0

        df_y = df['label']

        

        df_x = df_x.values.reshape(df.shape[0], 28, 28, 1)        

        return (df_x, df_y)

    

    # Testing Data

    else: 

        df_x = df.div(255.0)

        

        df_x = df_x.values.reshape(df.shape[0], 28, 28, 1)

        return df_x



# Applying the preprocess_data function to both the train & test data

train_X, train_y = preprocess_data(train_data)

test_X = preprocess_data(test_data)



print(f'Training data shape: X-{train_X.shape} & y-{train_y.shape}')

print(f'Testing data shape: X-{test_X.shape}')
# Initializing some constants for the Neural Network Architecture

INPUT_SHAPE = train_X.shape[1:] # (28, 28, 1)



# The number of training examples per batch of training

BATCH_SIZE = 128



# The number of epochs or iterations of the training loop

EPOCHS = 10
# The various layers used for this Neural Network Model.

# Dense - A layer that is fully connected (densely-connected.)

# Conv2D - A 2-dimensional convolutional layer.

# Dropout - A layer that helps prevent overfitting.

# Flatten - A layer that flattens the input.

# MaxPooling2D - A layer that performs Max Pooling of the input.

# BatchNormalization - A layer that normalizes the values of each batch.



# Building the model

model = tf.keras.models.Sequential()



model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same', input_shape=INPUT_SHAPE))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same'))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.BatchNormalization())



model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2))) 

model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.BatchNormalization())



model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'))

model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2))) 

model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.BatchNormalization())



model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(512, activation='relu'))

model.add(tf.keras.layers.Dense(10, activation='softmax'))



model.summary()
# 'adam' is the most used optimizer

# The loss function used is SCC because it's a multi-class classification problem with integer value classes

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])



# Fitting the model and using 10% of the data for validation

history = model.fit(x=train_X, y=train_y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)
# Creating the predictions file to submit

submission_df = pd.read_csv('../input/sample_submission.csv')

predictions = model.predict(test_X)

submission_vals = []



for i in range(len(predictions)):

    submission_vals.append(np.argmax(predictions[i]))



submission_df['Label'] = submission_vals



# Saving the submission file

submission_df.to_csv('submission.csv', index=False)