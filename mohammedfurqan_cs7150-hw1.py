import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import regularizers
from datetime import datetime
from numpy import expand_dims
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

val_images = train_images[-10000:]
val_labels = train_labels[-10000:]
train_images = train_images[:-10000]
train_labels = train_labels[:-10000]

# Normalize pixel values to be between 0 and 1
train_images, val_images, test_images = train_images / 255.0, val_images / 255.0, test_images / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
def create_model(bn=False, learning_rate=0.001):
    inputs = keras.Input(shape=(32, 32, 3), name='img')
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    block_1_output = layers.MaxPooling2D(3)(x)
    
    # ResNet Block 1
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_1_output)
    if bn:
        x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    block_2_output = layers.add([x, block_1_output])
    
    # ResNet Block 2
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_2_output)
    if bn:
        x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    block_3_output = layers.add([x, block_2_output])
    
    # ResNet Block 3
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_3_output)
    if bn:
        x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    block_4_output = layers.add([x, block_3_output])
    
    # ResNet Block 4
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_4_output)
    if bn:
        x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    block_5_output = layers.add([x, block_4_output])
    
    x = layers.Conv2D(64, 3, activation='relu')(block_5_output)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10)(x)
    
    model = keras.Model(inputs, outputs, name='toy_resnet')
    
    # Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model

wbn_model_01 = create_model(0.001)
history1 = wbn_model_01.fit(train_images, train_labels, epochs=50, batch_size=1000, validation_data=(val_images, val_labels))

wbn_model_05 = create_model(0.005)
history2 = wbn_model_05.fit(train_images, train_labels, epochs=50, batch_size=1000, validation_data=(val_images, val_labels))

wbn_model_10 = create_model(0.01)
history3 = wbn_model_10.fit(train_images, train_labels, epochs=50, batch_size=1000, validation_data=(val_images, val_labels))

wbn_model_30 = create_model(0.03)
history4 = wbn_model_30.fit(train_images, train_labels, epochs=50, batch_size=1000, validation_data=(val_images, val_labels))

bn_model_01 = create_model(True, 0.001)
history5 = bn_model_01.fit(train_images, train_labels, epochs=50, batch_size=1000, validation_data=(val_images, val_labels))

bn_model_05 = create_model(True, 0.005)
history6 = bn_model_05.fit(train_images, train_labels, epochs=50, batch_size=1000, validation_data=(val_images, val_labels))

bn_model_10 = create_model(True, 0.01)
history7 = bn_model_10.fit(train_images, train_labels, epochs=50, batch_size=1000, validation_data=(val_images, val_labels))

bn_model_30 = create_model(True, 0.03)
history8 = bn_model_30.fit(train_images, train_labels, epochs=50, batch_size=1000, validation_data=(val_images, val_labels))
# # # Save the entire model as a SavedModel.
# # !mkdir -p saved_model
# loc = './saved_model/bn_model_' + datetime.today().strftime('%Y-%m-%d-%H:%M')
# # model.save(loc) 

# # Save weights
# model.save(loc)
# # # Create a new model instance
# # saved_model = create_model()

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # Load the previously saved model
# saved_model = models.load_model('/kaggle/input/saved_model/base_model_2020-05-27-01:51')
plt.figure(figsize=(15,15))

plt.plot(history1.history['accuracy'], label='w/ BN, lr=0.001')
plt.plot(history2.history['accuracy'], label='w/ BN, lr=0.005')
plt.plot(history3.history['accuracy'], label='w/ BN, lr=0.01')
plt.plot(history4.history['accuracy'], label='w/ BN, lr=0.03')

plt.plot(history5.history['accuracy'], label='w/o BN, lr=0.001')
plt.plot(history6.history['accuracy'], label='w/o BN, lr=0.005')
plt.plot(history7.history['accuracy'], label='w/o BN, lr=0.01')
plt.plot(history8.history['accuracy'], label='w/o BN, lr=0.03')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

# test_loss, test_acc = wbn_model_01.evaluate(test_images,  test_labels, verbose=2)
bn_test_loss, bn_test_acc = model.evaluate(test_images,  test_labels, verbose=2)
for i in range(len(wbn_model_01.layers)):
	layer = wbn_model_01.layers[i]
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
	# summarize output shape
	print(i, layer.name, layer.output.shape)

saved_model = wbn_model_01
# Sub models to study layers
model_2 = keras.Model(inputs=saved_model.inputs, outputs=saved_model.layers[2].output)
model_8 = keras.Model(inputs=saved_model.inputs, outputs=saved_model.layers[8].output)
model_16 = keras.Model(inputs=saved_model.inputs, outputs=saved_model.layers[16].output)
test_img = test_images[2] # Boat
plt.imshow(test_img, cmap=plt.cm.binary) 

test_img = expand_dims(test_img, axis=0)
feature_maps_2 = model_2.predict(test_img)
feature_maps_8 = model_8.predict(test_img)
feature_maps_16 = model_16.predict(test_img)

fmap_layers = [feature_maps_2, feature_maps_8, feature_maps_16]

# plot the output from each block
square = 8
for layer in fmap_layers:
    for fmap in layer:
        # plot all 64 maps in an 8x8 squares
        ix = 1
        plt.figure(figsize=(15,15))
        for _ in range(square):
            for _ in range(square):
                # specify subplot and turn of axis
                ax = plt.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                plt.imshow(fmap[:, :, ix-1], cmap='gray')
                ix += 1
        # show the figure
        plt.show()
test_img = test_images[4] # Frog
plt.imshow(test_img, cmap=plt.cm.binary) 

test_img = expand_dims(test_img, axis=0)
feature_maps_2 = model_2.predict(test_img)
feature_maps_8 = model_8.predict(test_img)
feature_maps_16 = model_16.predict(test_img)

fmap_layers = [feature_maps_2, feature_maps_8, feature_maps_16]

# plot the output from each block
square = 8
for layer in fmap_layers:
    for fmap in layer:
        # plot all 64 maps in an 8x8 squares
        ix = 1
        plt.figure(figsize=(15,15))
        for _ in range(square):
            for _ in range(square):
                # specify subplot and turn of axis
                ax = plt.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                plt.imshow(fmap[:, :, ix-1], cmap='gray')
                ix += 1
        # show the figure
        plt.show()
test_img = test_images[6] # Car
plt.imshow(test_img, cmap=plt.cm.binary) 

test_img = expand_dims(test_img, axis=0)
feature_maps_2 = model_2.predict(test_img)
feature_maps_8 = model_8.predict(test_img)
feature_maps_16 = model_16.predict(test_img)

fmap_layers = [feature_maps_2, feature_maps_8, feature_maps_16]

# plot the output from each block
square = 8
for layer in fmap_layers:
    for fmap in layer:
        # plot all 64 maps in an 8x8 squares
        ix = 1
        plt.figure(figsize=(15,15))
        for _ in range(square):
            for _ in range(square):
                # specify subplot and turn of axis
                ax = plt.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                plt.imshow(fmap[:, :, ix-1], cmap='gray')
                ix += 1
        # show the figure
        plt.show()