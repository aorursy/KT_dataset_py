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
import numpy as np
import os
import glob
import tensorflow as tf
import argparse

# Display
from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# The local path to our target image
img_path = '../input/covid19-xray-images-using-cnn/images/train/corona/01E392EE-69F9-4E33-BFCE-E5C968654078.jpeg'

display(Image(img_path))
class default_parameters:
    exp_id = 'elbow_efficientnet_lr_1e-4'
    learning_rate = 1e-4
    epochs = 50
    rotation_range = 0
    width_shift_range = 0.0
    height_shift_range = 0.0
    brightness_range = None
    zoom_range = 0.0
    horizontal_flip = False
    vertical_flip = False
    validation_split = 0.2
    
args = default_parameters

try:
    os.mkdir(args.exp_id)
except:
    pass
exp_path = os.path.join('..',args.exp_id)
train_dir = '../input/covid19-xray-images-using-cnn/images/train'

base_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=args.rotation_range,
                                                                           width_shift_range=args.width_shift_range,
                                                                           height_shift_range=args.height_shift_range,
                                                                           brightness_range=args.brightness_range,
                                                                           zoom_range=args.zoom_range,
                                                                           horizontal_flip=args.horizontal_flip,
                                                                           vertical_flip=args.vertical_flip,
                                                                           rescale=1./255, validation_split = args.validation_split)



train_data_gen = base_image_generator.flow_from_directory(batch_size=20,
                                                        directory=train_dir,
                                                        shuffle=True,
                                                        class_mode='binary',
                                                        subset='training')



val_data_gen = base_image_generator.flow_from_directory(batch_size=10,
                                                        directory=train_dir,
                                                        shuffle=True,
                                                        class_mode='binary',
                                                        subset='validation')
images, labels = next(train_data_gen)
print(images.shape)
print(labels.shape)
model_builder = tf.keras.applications.xception.Xception
img_size = (256, 256)
preprocess_input = tf.keras.applications.xception.preprocess_input
decode_predictions = tf.keras.applications.xception.decode_predictions

last_conv_layer_name = "block14_sepconv2_act"
classifier_layer_names = [
    "avg_pool",
    "predictions",
]

Xception_model = model_builder(weights="imagenet",include_top=True)

model = tf.keras.Sequential()
model.add(Xception_model)
model.add(tf.keras.layers.Dense(units = 128, activation='relu'))
model.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))
model.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))

model.compile(optimizer=tf.keras.optimizers.Adam(lr=args.learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
Xception_model.summary()
model.summary()
# Callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint(exp_path+'/weights.{epoch}-{val_accuracy:.4f}.hdf5', monitor='val_accuracy', verbose=1,save_best_only=True)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', verbose = 1, patience=5)
callbacks_list = [checkpoint, earlystopping]    

history = model.fit(
    train_data_gen,
    epochs = args.epochs,
    steps_per_epoch = len(train_data_gen),
    #steps_per_epoch = 1,
    validation_data = val_data_gen,
    verbose = 1,
    validation_steps = len(val_data_gen),
    callbacks=callbacks_list
)                                
test_dir = '../input/covid19-xray-images-using-cnn/images/test'
test_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split = 0.0)



test_data_gen = test_image_generator.flow_from_directory(batch_size=10,
                                                        directory=test_dir,
                                                        shuffle=True,
                                                        class_mode='binary',
                                                        subset='training')
images, labels = next(test_data_gen)
print(images.shape)
print(labels.shape)
model.evaluate(test_data_gen)
from tensorflow.keras.preprocessing import image
def prepare_image(img_path):    
    # Read the image and resize it
    img = image.load_img(img_path, target_size=img_size)
    # Convert it to a Numpy array with target shape.
    x = image.img_to_array(img)
    # Reshape
    x = x.reshape((1,) + x.shape)
    x /= 255.
    result = model.predict([x])[0][0]
    if result > 0.5:
        status = "normal"
    else:
        status = "corona"
        result = 1 - result
            
    return status,result
normal_image_path = '../input/covid19-xray-images-using-cnn/images/test/normal/IM-0117-0001.jpeg'
Image(filename=img_path)
prepare_image(normal_image_path)
corona_image_path = '../input/covid19-xray-images-using-cnn/images/test/corona/SARS-10.1148rg.242035193-g04mr34g09b-Fig9b-day19.jpeg'
Image(filename=img_path)
prepare_image(corona_image_path)
from tensorflow import keras

def get_img_array(img_path):
    # Read the image and resize it
    img = image.load_img(img_path, target_size=img_size)
    # Convert it to a Numpy array with target shape.
    x = image.img_to_array(img)
    # Reshape
    x = x.reshape((1,) + x.shape)
    x /= 255.
    return x


def make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, classifier_layer_names):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap
img_path = '../input/covid19-xray-images-using-cnn/images/test/normal/IM-0125-0001.jpeg'
img_array = get_img_array(img_path)

# Generate class activation heatmap
heatmap = make_gradcam_heatmap(
    img_array, Xception_model, last_conv_layer_name, classifier_layer_names
)


# Display heatmap
plt.matshow(heatmap)
plt.show()
# We load the original image
img = keras.preprocessing.image.load_img(img_path)
img = keras.preprocessing.image.img_to_array(img)

# We rescale heatmap to a range 0-255
heatmap = np.uint8(255 * heatmap)

# We use jet colormap to colorize heatmap
jet = cm.get_cmap("jet")

# We use RGB values of the colormap
jet_colors = jet(np.arange(256))[:, :3]
jet_heatmap = jet_colors[heatmap]

# We create an image with RGB colorized heatmap
jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

# Superimpose the heatmap on original image
superimposed_img = jet_heatmap * 0.4 + img
superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

# Save the superimposed image
save_path = "normal_cam.jpg"
superimposed_img.save(save_path)

# Display Grad CAM
display(Image(save_path))
img_path = '../input/covid19-xray-images-using-cnn/images/test/corona/streptococcus-pneumoniae-pneumonia-temporal-evolution-1-day1.jpg'
img_array = get_img_array(img_path)

# Generate class activation heatmap
heatmap = make_gradcam_heatmap(
    img_array, Xception_model, last_conv_layer_name, classifier_layer_names
)


# Display heatmap
plt.matshow(heatmap)
plt.show()
# We load the original image
img = keras.preprocessing.image.load_img(img_path)
img = keras.preprocessing.image.img_to_array(img)

# We rescale heatmap to a range 0-255
heatmap = np.uint8(255 * heatmap)

# We use jet colormap to colorize heatmap
jet = cm.get_cmap("jet")

# We use RGB values of the colormap
jet_colors = jet(np.arange(256))[:, :3]
jet_heatmap = jet_colors[heatmap]

# We create an image with RGB colorized heatmap
jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

# Superimpose the heatmap on original image
superimposed_img = jet_heatmap * 0.4 + img
superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

# Save the superimposed image
save_path = "normal_cam.jpg"
superimposed_img.save(save_path)

# Display Grad CAM
display(Image(save_path))