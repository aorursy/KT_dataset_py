import os
from os import listdir, path
from zipfile import ZipFile
import random
import numpy as np
import keras
from keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.io import imread
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
if not path.exists('../input/cat/'):
    print('Extracting cat image files...')
    zf = ZipFile('../input/cat.zip')
    zf.extractall('../input/')
if not path.exists('../input/dog/'):
    print('Extracting dog image files...')
    zf = ZipFile('../input/dog.zip')
    zf.extractall('../input/')
def show(image):
    plt.imshow(np.squeeze(image.astype("uint8")), cmap="gray")
    plt.title("image shape: "+ str(image.shape), fontsize=14)
    plt.axis('off');
    
def show_multiple(images, figsize):
    fig, ax = plt.subplots(ncols=len(images), figsize=figsize)
    for col, image in zip(ax, images):
        col.imshow(np.squeeze(image.astype("uint8")), cmap="gray")
        col.set_title("image shape: "+ str(image.shape), fontsize=14)
    plt.tight_layout()
    plt.axis('off');
sample_image = imread("../input/panda.jpg")
show(sample_image)
def conv_2d(x, k, strides, padding, conv_type):
    if conv_type == 'depthwise':
        return tf.nn.depthwise_conv2d(
            x, k, strides=strides, padding=padding
        )
    elif conv_type == 'standard':
        return tf.nn.conv2d(
            x, k, strides=strides, padding=padding
        )   
def visualize_kernel(kernel):
    # move the channel dimension to the first one
    # this way, it is easier to see the spacial organization of the kernel with print
    print(np.transpose(kernel, (2, 0, 1)))
kernel_data = np.ones(shape=(5, 5, 3)).astype(np.float32)
kernel_data /= kernel_data.sum(0).sum(0)
visualize_kernel(kernel_data)
image = tf.placeholder(tf.float32, shape=(None, None, None, 3)) # [batch, height, width, channels]
kernel = tf.placeholder(tf.float32, shape=(5, 5, 3, 1)) # [filter_height, filter_width, in_channels, out_channels]

output_image = conv_2d(image, kernel, strides=(1, 1, 1, 1), padding='SAME', conv_type='depthwise')

with tf.Session() as sess:
    image_batch_expanded = np.expand_dims(sample_image, axis=0)
    kernel_data_expanded = np.expand_dims(kernel_data, axis=-1)
    print('Kernel shape: %s' % str(kernel_data_expanded.shape))
    feed_dict = {image: image_batch_expanded, kernel: kernel_data_expanded}
    feature_map = sess.run(output_image, feed_dict=feed_dict)
    show(feature_map)
kernel_data = np.zeros(shape=(3, 3, 3)).astype(np.float32)
kernel_data[1, 1, :] = 1
visualize_kernel(kernel_data)
image = tf.placeholder(tf.float32, shape=(None, None, None, 3))
kernel = tf.placeholder(tf.float32, shape=(3, 3, 3, 1))

output_same_padding = conv_2d(image, kernel, strides=(1, 1, 1, 1), 
                              padding='SAME', conv_type='depthwise')
output_valid_padding = conv_2d(image, kernel, strides=(1, 1, 1, 1), 
                               padding='VALID', conv_type='depthwise')
output_larger_strides = conv_2d(image, kernel, strides=(1, 10, 10, 1), 
                                padding='SAME', conv_type='depthwise')

with tf.Session() as sess:
    image_batch_expanded = np.expand_dims(sample_image, axis=0)
    kernel_data_expanded = np.expand_dims(kernel_data, axis=-1)
    feed_dict = {image: image_batch_expanded, kernel: kernel_data_expanded}
    feature_map_same_padding, feature_map_valid_padding, feature_map_larger_strides = sess.run(
            [output_same_padding, output_valid_padding, output_larger_strides], 
            feed_dict=feed_dict
    )
    show_multiple([
        feature_map_same_padding, 
        feature_map_valid_padding, 
        feature_map_larger_strides
    ], figsize=(16, 12))
grey_sample_image = np.expand_dims(sample_image.sum(axis=2) / 3., axis=-1)
show(grey_sample_image)
# Implement a 3x3 edge detection kernel
line_detection_kernel = np.asarray(
    [
        # TODO:
        [1., 2., 1.],
        [0., 1, 0.],
        [-1., -2., -1.]
    ]
)

kernel_data = np.expand_dims(line_detection_kernel, axis=-1)
visualize_kernel(kernel_data)
image = tf.placeholder(tf.float32, shape=(None, None, None, 1))
kernel = tf.placeholder(tf.float32, shape=(3, 3, 1, 1))

output_line_detection = conv_2d(image, kernel, strides=(1, 1, 1, 1), 
                                padding='SAME', conv_type='standard')

with tf.Session() as sess:
    image_batch = np.expand_dims(grey_sample_image, axis=0)
    kernel_data = np.expand_dims(kernel_data, axis=-1)
    feed_dict = {image: image_batch, kernel: kernel_data}
    feature_map = sess.run(output_line_detection, feed_dict=feed_dict)
    show(feature_map)
image = tf.placeholder(tf.float32, [None, None, None, 3])
# TODO:
output_max_pool = tf.nn.max_pool(
    value=image_batch,
    ksize=[1,5,5,1],
    strides=[1,2,2,1],
    padding="VALID"
)
output_avg_pool = tf.nn.avg_pool(
    value=image_batch,
    ksize=[1,2,2,1],
    strides=[1,1,1,1],
    padding="SAME"
)

with tf.Session() as sess:
    feed_dict={image:[sample_image], kernel: kernel_data}
    # TODO:
    feature_map_max_pool, feature_map_avg_pool = sess.run(output_max_pool, feed_dict=feed_dict), sess.run(output_avg_pool, feed_dict=feed_dict)
    # TODO:
    show_multiple([feature_map_max_pool, feature_map_avg_pool], figsize=(8, 6))
def get_splitted_data_with_size(image_size, sample_size, test_ratio, classes, seed):
    X, Y = [], []
    for label, animal in enumerate(classes):
        files = listdir(path.join('../input/', animal))
        random.shuffle(files)
        files = files[:(sample_size // len(classes))]
        for i, file in enumerate(files):
            img = load_img(path.join('../input/', animal, file), 
                           target_size=image_size)
            #print(i, file)
            X.append(img_to_array(img))
            Y.append(label)
    return train_test_split(np.asarray(X), np.asarray(Y), test_size=test_ratio, random_state=seed)
def plot_model_history(model_history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    for ax, metric, name in zip(axs, ['acc', 'loss'], ['Accuracy', 'Loss']):
        ax.plot(
            range(1, len(model_history.history[metric]) + 1), 
            model_history.history[metric]
        )
        ax.plot(
            range(1, len(model_history.history['val_' + metric]) + 1), 
            model_history.history['val_' + metric]
        )
        ax.set_title('Model ' + name)
        ax.set_ylabel(name)
        ax.set_xlabel('Epoch')
        ax.legend(['train', 'val'], loc='best')
    plt.show()
def scale_data(X_tr, X_val, return_scaler=False):
    shape_tr, shape_val = X_tr.shape, X_val.shape
    X_tr_flat = np.ravel(X_tr).reshape(-1, 1)
    X_val_flat = np.ravel(X_val).reshape(-1, 1)
    min_max_scaler = MinMaxScaler()
    X_tr_scaled = min_max_scaler.fit_transform(X_tr_flat).reshape(shape_tr)
    X_val_scaled = min_max_scaler.transform(X_val_flat).reshape(shape_val)
    if not return_scaler:
        return X_tr_scaled, X_val_scaled
    else:
        return X_tr_scaled, X_val_scaled, min_max_scaler
    
def apply_scaling(X, scaler):
    shape_X = X.shape
    X_flat = np.ravel(X).reshape(-1, 1)
    X_scaled = scaler.transform(X_flat).reshape(shape_X)
    return X_scaled
image_size = (32, 32, 3)
sample_size = 10000

classes = ['cat/cat', 'dog/dog']
X_tr, X_val, Y_tr, Y_val = get_splitted_data_with_size(
    image_size=image_size, sample_size=sample_size, test_ratio=0.25, classes=classes, seed=42
)
X_tr.shape, X_val.shape, Y_tr.shape, Y_val.shape
i = np.random.choice(len(X_tr))
show(X_tr[i])
print('True label: {0}'.format(classes[Y_tr[i]]))
X_tr_scaled, X_val_scaled, scaler = scale_data(X_tr, X_val, return_scaler=True)
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam

"""
Return a compiled Keras model
"""
def design_and_compile_model():
    model = Sequential()
    # TODO:
    model.add(Conv2D(filters=64,kernel_size=3, padding='Valid', input_shape=(32, 32, 3)))
    model.add(Activation(activation = "relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense( 1, kernel_initializer= "normal", activation= "sigmoid"))
    
    # TODO:

    # Compiling the model adds a loss function, optimiser and metrics to track during training
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=['accuracy']
    )
    
    # TODO:
    return model
design_and_compile_model().summary() if design_and_compile_model() else None
batch_size = 128
num_epochs = 25  # The number of epochs (full passes through the data) to train for

model = design_and_compile_model()

# The fit function allows you to fit the compiled model to some training data
if model:
    model_history = model.fit(
        x=X_tr_scaled, 
        y=Y_tr, 
        batch_size=batch_size, 
        epochs=num_epochs,
        verbose=1,
        validation_data=(X_val_scaled, Y_val)
    )
    print('Training complete')
else:
    model_history = None
plot_model_history(model_history) if model_history else None
from keras.preprocessing.image import ImageDataGenerator
batch_size = 128
# Instantiate a ImageDataGenerator object with the right parameters and then fit it on your training dataset
# TODO:
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20, #40
    #width_shift_range=0.2,
    #height_shift_range=0.2,  
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20, #40
    #width_shift_range=0.2,
    #height_shift_range=0.2,  
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
##### Finaly

print(X_tr.shape , Y_tr.shape)
train_generator = train_datagen.flow(
    np.array(X_tr), 
    Y_tr, 
    batch_size=batch_size)
validation_generator = val_datagen.flow(
    np.array(X_val), 
    Y_val, 
    batch_size=batch_size)
setattr(tf.contrib.rnn.GRUCell, '__deepcopy__', lambda self, _: self)
setattr(tf.contrib.rnn.BasicLSTMCell, '__deepcopy__', lambda self, _: self)
setattr(tf.contrib.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)

model = design_and_compile_model()
# Fit your model with model.fit_generator() and feed it with data_generator.flow()
# TODO:
        
#model.fit_generator(generator=train_datagen.flow(X_tr_scaled,Y_tr))
#model.fit_generator(train_datagen.flow(X_tr_scaled, Y_tr, batch_size=batch_size),
#                    steps_per_epoch=len(X_tr_scaled) / 64, 
#                    validation_data=test_datagen.flow(X_val_scaled, Y_val, batch_size=batch_size),
#                    validation_steps=800 // batch_size,
#                    epochs=num_epochs)

###############
model_history = model.fit_generator(
    train_generator,
    steps_per_epoch = len(X_tr) / (batch_size/2), # // 64
    epochs = num_epochs,
    validation_data = validation_generator,
    validation_steps = len(X_val) / (batch_size/2)) # // 64

plot_model_history(model_history) if model_history else None
cat_sample_path = "../input/cat/cat/cat_1.jpg"
dog_sample_path = "../input/dog/dog/dog_1.jpg"
resnet_input_size = (224, 224)
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.models import Model
from skimage.transform import resize

model_ResNet50 = ResNet50(include_top=True, weights='imagenet')
model_ResNet50.summary()
img = imread(cat_sample_path)
img_resized = resize(img, resnet_input_size, mode='reflect', preserve_range=True)
show(img_resized)

# Use preprocess_input() to apply the same preprocessing as ResNet, 
# get the prediction from the loaded model, and then decode the predictions

# TODO:
x_img = np.expand_dims(img_resized, axis=0)
x_img = preprocess_input(x_img)

decoded_predictions = decode_predictions(model_ResNet50.predict(x_img), top=3)[0]

if decoded_predictions:
    for _, name, score in decoded_predictions:
        print(name, score)
        
print(len(model_ResNet50.layers))
# Create a truncated Model using ResNet50.input and the before last layer

# TODO:
model_ResNet50.layers.pop()
#model_ResNet50.summary()
output = model_ResNet50.get_layer('avg_pool').output
new_model = Model(model_ResNet50.input, output)
feat_extractor_model = new_model
def preprocess_resnet(x, size):
    x = resize(x, size, mode='reflect', preserve_range=True)
    x = np.expand_dims(x, axis=0)
    if x.ndim == 3:
        x = np.expand_dims(x, axis=0)
    return preprocess_input(x)
cat_img = imread(cat_sample_path)
cat_img_processed = preprocess_resnet(cat_img, resnet_input_size)
if feat_extractor_model:
    cat_representation = feat_extractor_model.predict(cat_img_processed)
    print("Cat deep representation shape: (%d, %d)" % cat_representation.shape)
    for activation in np.ravel(cat_representation):
        print(activation)
if feat_extractor_model:
    plt.hist(np.where(cat_representation == 0)[1])
    plt.title("cat zeros positions")
    plt.show()

    dog_img = imread(dog_sample_path)
    dog_img_processed = preprocess_resnet(dog_img, resnet_input_size)
    dog_representation = feat_extractor_model.predict(dog_img_processed)

    plt.hist(np.where(dog_representation == 0)[1])
    plt.title("dog zeros positions")
    plt.show()
classes = ['cat/cat', 'dog/dog']
X_tr, X_val, Y_tr, Y_val = get_splitted_data_with_size(
    image_size=(224, 224, 3), sample_size=2000, test_ratio=0.25, classes=classes, seed=42
)
if feat_extractor_model:
    X_extracted_tr = feat_extractor_model.predict(preprocess_input(X_tr), verbose=1)
    X_extracted_val = feat_extractor_model.predict(preprocess_input(X_val), verbose=1)
    print('Done extracting resnet50 features..')
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

# TODO:
transfer_model = Sequential()
transfer_model.add(Dense(128,activation='relu',input_shape=(X_extracted_tr.shape[1],)))
transfer_model.add(Dropout(0.25))
transfer_model.add(Dense(1,activation='sigmoid'))

# Compiling the model adds a loss function, optimiser and metrics to track during training
transfer_model.compile(
    optimizer="adam",
    loss="binary_crossentropy", #"binary_crossentropy"
    metrics=["accuracy"]
)

model_history = transfer_model.fit(X_extracted_tr, Y_tr, epochs=40, validation_data=(X_extracted_val, Y_val))
plot_model_history(model_history) if model_history else None
