%matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import math
# from tf.keras.models import Sequential  # This does not work!
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
train_dataset=pd.read_csv('../input/train.csv')
train_features=train_dataset.iloc[:,1:]
print(train_features)
train_label=train_dataset['label']

test_dataset=pd.read_csv('../input/test.csv')
#test_dataset.head()
print("Size of:")
print("- Training-set:\t\t{}".format(len(train_dataset['label'])))
print("- Test-set:\t\t{}".format(len(test_dataset['pixel0'])))
img_size = 28

img_size_flat = img_size * img_size

img_shape = (img_size, img_size)

img_shape_full = (img_size, img_size, 1)

num_channels = 1

num_classes = 10
def plot_images(images, cls_true, typ):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        xlabel = "{0}: {1}".format(typ,cls_true[i] )


        ax.set_xlabel(xlabel)
        
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
# Get the first 9 images from the test-set.
images = train_features.iloc[0:9].values

plot_images(images=images, cls_true=train_label.iloc[0:9].values, typ="true")
# Start construction of the Keras Sequential model.
model = Sequential()

# Add an input layer which is similar to a feed_dict in TensorFlow.
# Note that the input-shape must be a tuple containing the image-size.
model.add(InputLayer(input_shape=(img_size_flat,)))

# The input is a flattened array with 784 elements,
# but the convolutional layers expect images with shape (28, 28, 1)
model.add(Reshape(img_shape_full))

# First convolutional layer with ReLU-activation and max-pooling.
model.add(Conv2D(kernel_size=5, strides=1, filters=3, padding='same',
                 activation='relu', name='layer_conv1'))
model.add(MaxPooling2D(pool_size=2, strides=2))

# Second convolutional layer with ReLU-activation and max-pooling.
model.add(Conv2D(kernel_size=5, strides=1, filters=5, padding='same',
                 activation='relu', name='layer_conv2'))
model.add(MaxPooling2D(pool_size=2, strides=2))

# Flatten the 4-rank output of the convolutional layers
# to 2-rank that can be input to a fully-connected / dense layer.
model.add(Flatten())

# First fully-connected / dense layer with ReLU-activation.
model.add(Dense(50, activation='relu'))

# Last fully-connected / dense layer with softmax-activation
# for use in classification.
model.add(Dense(num_classes, activation='softmax'))
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.optimizers import SGD

#optimizer = Adam(lr=1e-3)
optimizer = SGD(lr=1e-3)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# convert labels into one hot vectors
train_one_hot=tf.keras.backend.one_hot(
    train_label.values,
    num_classes
)

print(tf.Session().run(train_one_hot))
train_one_hot = tf.Session().run(train_one_hot)
# Train the model over training data
model.fit(x=train_features.values,
          y=train_one_hot,
          epochs=3, batch_size=25)
y_pred = model.predict(x=test_dataset.values)
cls_pred = np.argmax(y_pred,axis=1)
print(len(cls_pred),"\n",len(test_dataset.values))
plot_images(images=test_dataset.values[0:9] , cls_true=cls_pred[0:9], typ='pred')
write_df=pd.DataFrame()
write_df['Label']=cls_pred
write_df.index=[i for i in range(1,len(cls_pred)+1)]
print(write_df.index)
write_df['ImageId']=write_df.index
write_df.to_csv('prediction.csv',index=False)