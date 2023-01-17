import datetime, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, Model, Input, Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Activation, AveragePooling2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import to_categorical
import tensorflow_datasets as tfds

plt.style.use('dark_background')
%matplotlib inline

print(tf.__version__)
def plot(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend(loc=0)
    plt.figure()
    plt.show()

def plot_image(i, predictions_array, true_label, img):
    # true_label, img = np.argmax(true_label[i]), img[i]
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    color = 'white' if true_label == predicted_label else 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                    100*np.max(predictions_array),
                                    class_names[true_label]),
                                    color=color)

def plot_value_array(i, predictions_array, true_label):
    # true_label = np.argmax(true_label[i])
    true_label = true_label[i]
    plt.grid(False)
    plt.yticks(range(10), class_names, rotation=0)
    plt.xticks([])
    thisplot = plt.barh(range(10), predictions_array, color="#777777")
    plt.xlim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('white')
# Mapping Classes
class_names = {0 : 'T-shirt/top',
            1 : 'Trouser',
            2 : 'Pullover',
            3 : 'Dress',
            4 : 'Coat',
            5 : 'Sandal',
            6 : 'Shirt',
            7 : 'Sneaker',
            8 : 'Bag',
            9 : 'Ankle boot'}
training = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')
testing = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')
training.describe()
training_X, training_Y = training.iloc[:,1:], training.iloc[:,0]
testing_X, testing_Y = testing.iloc[:,1:], testing.iloc[:,0]
training_X, testing_X = training_X.values.reshape((-1, 28, 28, 1)), testing_X.values.reshape((-1, 28, 28, 1))

training_X, testing_X = training_X/255.0, testing_X/255.0

training_X.shape, testing_X.shape
plt.figure()
plt.imshow(training_X[0, :, :, 0])
plt.colorbar()
plt.grid(False)
plt.show()
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(training_X[i, :, :, 0], cmap=plt.cm.BuGn)
    plt.colorbar()
    plt.xlabel(class_names[np.argmax(training_Y[i])])
plt.show()
# initializer = tf.keras.initializers.GlorotNormal()
# model = Sequential([
#                 Input(shape=(28, 28, 1), name="input"),
#                 Conv2D(32, kernel_size=(3,3), padding='same', kernel_initializer=initializer),
#                 BatchNormalization(),
#                 Activation("swish"),
#                 MaxPooling2D(pool_size=(2, 2), strides=1, padding='same'),
#                 Conv2D(32, kernel_size=(3,3), kernel_initializer=initializer),
#                 BatchNormalization(),
#                 Activation("swish"),
#                 MaxPooling2D(),
#                 Conv2D(64, kernel_size=(3,3), kernel_initializer=initializer),
#                 BatchNormalization(),
#                 Activation("swish"),
#                 MaxPooling2D(),
#                 Flatten(name='flatten'),
#                 Dense(128, kernel_initializer=initializer),
#                 BatchNormalization(),
#                 Activation("swish"),
#                 Dropout(0.50),
#                 Dense(10, activation="softmax", name="output")], name='sequential')

# model.compile(optimizer=RMSprop(learning_rate=1e-4), loss="sparse_categorical_crossentropy", metrics=['accuracy'])
# model.summary()

model = Sequential([
                Input(shape=(28, 28, 1), name="input"),
                Conv2D(32, kernel_size=(3,3), padding='same'),
                BatchNormalization(),
                Activation("relu"),
                Conv2D(32, kernel_size=(3,3), padding='same'),
                BatchNormalization(),
                Activation("relu"),
                MaxPooling2D((2,2), name='pooling_1'),
                Conv2D(64, kernel_size=(3,3)),
                BatchNormalization(),
                Activation("relu"),
                MaxPooling2D((2,2), name='pooling_2'),
                Flatten(name='flatten'),
                Dense(512),
                BatchNormalization(),
                Activation("relu"),
                Dropout(0.50),
                Dense(256),
                BatchNormalization(),
                Activation("relu"),
                Dropout(0.50),
                Dense(10, activation="softmax", name="output")
])

model.compile(optimizer=Adam(learning_rate=1e-4), loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model.summary()
%%time
logdir = os.path.join("/logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
modelCheckpointCallback = tf.keras.callbacks.ModelCheckpoint("fashion_mnist.h5", save_best_only=True)
earlyStoppingCallback = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1)
tensorboardCallback = tf.keras.callbacks.TensorBoard(logdir)

history = model.fit(training_X, training_Y, epochs=25, batch_size=128, shuffle=True, validation_split=0.2, callbacks = [modelCheckpointCallback, earlyStoppingCallback, tensorboardCallback])
plot(history)
model.evaluate(testing_X,  testing_Y, verbose=2)
predictions = model.predict(testing_X)
topK = tf.keras.metrics.sparse_top_k_categorical_accuracy(testing_Y, predictions, k=1)

unique, counts = np.unique(topK.numpy(), return_counts=True)
wrongs=np.where(topK.numpy() == 0)[0]
print(dict(zip(unique, counts)))
i = 13
plt.figure(figsize=(8,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], testing_Y, testing_X[:,:,:,0])
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  testing_Y)
plt.tight_layout()
plt.show()
num_rows = 5
num_cols = 5
num_images = num_rows*num_cols
plt.figure(figsize=(3*2*num_cols, 2*num_rows))
for i, item in zip(range(num_images), wrongs):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], testing_Y, testing_X[:,:,:,0])
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], testing_Y)
plt.tight_layout(pad=2.0)
plt.show()
# Let's define a new Model that will take an image as input, and will output
# intermediate representations for all layers in the previous model after
# the first.
successive_outputs = [layer.output for layer in model.layers[1:]]

#visualization_model = Model(img_input, successive_outputs)
visualization_model = Model(inputs = model.input, outputs = successive_outputs)
# Let's run our image through our network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(testing_X)

# These are the names of the layers, so can have them as part of our plot
layer_names = [layer.name for layer in model.layers]

# -----------------------------------------------------------------------
# Now let's display our representations
# -----------------------------------------------------------------------
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  
    if len(feature_map.shape) == 4:
        #-------------------------------------------
        # Just do this for the conv / maxpool layers, not the fully-connected layers
        #-------------------------------------------
        n_features = feature_map.shape[-1]  # number of features in the feature map
        size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)
        
        # We will tile our images in this matrix
        display_grid = np.zeros((size, size * n_features))
        
        #-------------------------------------------------
        # Postprocess the feature to be visually palatable
        #-------------------------------------------------
        for i in range(n_features):
            x  = feature_map[wrongs[1], :, :, i]
            x -= x.mean()
            x /= x.std ()
            x *=  64
            x += 128
            x  = np.clip(x, 0, 255).astype('uint8')
            display_grid[:, i * size : (i + 1) * size] = x # Tile each filter into a horizontal grid

        #-----------------
        # Display the grid
        #-----------------

        scale =  20. / n_features
        plt.figure(figsize=(scale * n_features, scale))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis' ) 