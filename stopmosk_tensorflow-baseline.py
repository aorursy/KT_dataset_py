import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.model_selection import KFold



import tensorflow as tf



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D

from tensorflow.keras.utils import to_categorical

from tensorflow.keras import regularizers



#tf.debugging.set_log_device_placement(True)



print(tf.config.experimental.list_physical_devices('CPU'))

print(tf.config.experimental.list_physical_devices('GPU'))

print(tf.__version__)
train_data = pd.read_csv('../input/bird-or-aircraft-dafe-open/train_x.csv', index_col=0, header=None)

train_labels = pd.read_csv('../input/bird-or-aircraft-dafe-open/train_y.csv', index_col=0)

test_data = pd.read_csv('../input/bird-or-aircraft-dafe-open/test_x.csv', index_col=0, header=None)
train_data.shape, train_labels.shape, test_data.shape
train_data
test_data
train_labels
# Check classes balance



train_labels['target'].value_counts()
# Convert to numpy arrays



train_data = train_data.to_numpy()

test_data = test_data.to_numpy()



train_labels = train_labels.to_numpy()
def get_sample_image(sample):

    image = sample.reshape(32, 32, 3)

    return image
# Sample image



plt.figure(figsize=(2, 2))

plt.imshow(get_sample_image(train_data[555]))

plt.show()
# Sample images 



plt.figure(figsize=(8, 8))

for i in range(25):

    plt.subplot(5, 5, i+1)

    plt.xticks([])

    plt.yticks([])

    plt.imshow(get_sample_image(train_data[i]))

    plt.xlabel('plane' if train_labels[i] else 'bird')

plt.show()
# Normalize and reshape to 32 x 32 x 3 (RGB)



train_data = train_data / 255

train_data = train_data.reshape(train_data.shape[0], 32, 32, 3)
# Convolutional NN



def build_model():

    

    model = Sequential([

        Conv2D(16, 3, padding='same', activation='relu', input_shape=(32, 32, 3)),

        MaxPooling2D(),

        Conv2D(32, 3, padding='same', activation='relu'),

        MaxPooling2D(),

        Conv2D(64, 3, padding='same', activation='relu'),

        MaxPooling2D(),

        Flatten(),

        Dropout(0.5),

        Dense(512, activation='relu', kernel_regularizer=regularizers.l1_l2(0.0001)),

        Dense(1, activation='sigmoid')

    ])

    

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
# Training with KFold validation



kf = KFold(n_splits=5, shuffle=True, random_state=42)



epochs_num = 150

all_loss = []

all_accuracy = []



i = 0

for train_index, val_index in kf.split(train_data):

    i += 1

    print('Processing fold #', i)

    

    X_train = train_data[train_index] 

    y_train = train_labels[train_index]



    X_val = train_data[val_index]

    y_val = train_labels[val_index]

    

    model = build_model()

    history = model.fit(X_train, y_train,

                        epochs=epochs_num, 

                        batch_size=128,

                        validation_data=(X_val, y_val),

                        verbose=0)

    

    loss_history = history.history['val_loss']

    all_loss.append(loss_history)



    accuracy_history = history.history['val_accuracy']

    all_accuracy.append(accuracy_history)
# Exponential moving average function



def smooth_curve(points, factor=0.9):

    smoothed_points = []

    for point in points:

        if smoothed_points:

            previous = smoothed_points[-1]

            smoothed_points.append(previous * factor + point * (1 - factor))

        else:

            smoothed_points.append(point)

    return smoothed_points
# Plots



n_from = 0

n_to = epochs_num

    

plt.figure(figsize=(20, 10))



# Accuracy plots

plt.subplot(2, 4, 1)

for i in range(5):

    sns.lineplot(np.arange(n_to - n_from), all_accuracy[i][n_from:n_to]);



plt.subplot(2, 4, 2)

for i in range(5):

    sns.lineplot(np.arange(n_to - n_from), smooth_curve(all_accuracy[i][n_from:n_to], factor=0.7));



plt.subplot(2, 4, 3)

all_accuracy_np = np.array(all_accuracy)

sns.lineplot(np.arange(n_to - n_from), all_accuracy_np.mean(axis=0)[n_from:n_to]);



plt.subplot(2, 4, 4)

sns.lineplot(np.arange(n_to - n_from), smooth_curve(all_accuracy_np.mean(axis=0)[n_from:n_to], factor=0.7));



# Loss plots

plt.subplot(2, 4, 5)

for i in range(5):

    sns.lineplot(np.arange(n_to - n_from), all_loss[i][n_from:n_to]);



plt.subplot(2, 4, 6)

for i in range(5):

    sns.lineplot(np.arange(n_to - n_from), smooth_curve(all_loss[i][n_from:n_to], factor=0.7));



plt.subplot(2, 4, 7)

all_loss_np = np.array(all_loss)

sns.lineplot(np.arange(n_to - n_from), all_loss_np.mean(axis=0)[n_from:n_to]);



plt.subplot(2, 4, 8)

sns.lineplot(np.arange(n_to - n_from), smooth_curve(all_loss_np.mean(axis=0)[n_from:n_to], factor=0.7));



plt.show()
# Final train



epochs_num = 75



X_train = train_data

y_train = train_labels



model = build_model()

model.fit(X_train, y_train, epochs=epochs_num, batch_size=128, verbose=0);
# Normalize and reshape to 32 x 32 x 3 (RGB)



X_test = test_data / 255

X_test = X_test.reshape(test_data.shape[0], 32, 32, 3)
# Make Predictions



y_pred = model.predict(X_test)



y_pred[:10]
# Show some predictions



plt.figure(figsize=(8, 8))

for i in range(25):

    plt.subplot(5, 5, i+1)

    plt.xticks([])

    plt.yticks([])

    plt.imshow(get_sample_image(test_data[i]))

    plt.xlabel('plane' if y_pred[i] >= 0.5 else 'bird')

plt.show()
# Make submission



submission = pd.DataFrame({'id': range(test_data.shape[0]),

                           'target': (y_pred >= 0.5).astype('int').flatten()

                          })
submission.tail(3)
submission.to_csv('submission.csv', index=False)