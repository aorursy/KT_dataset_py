import numpy as np

import pandas as pd

import os



PATH = "/kaggle/input/applications-of-deep-learning-wustl-fall-2020/final-kaggle-data/"

PATH_TRAIN = os.path.join(PATH, "train.csv")

PATH_TEST = os.path.join(PATH, "test.csv")
# What version of Python do you have?

import sys



import tensorflow.keras

import pandas as pd

import sklearn as sk

import tensorflow as tf



print(f"Tensor Flow Version: {tf.__version__}")

print(f"Keras Version: {tensorflow.keras.__version__}")

print()

print(f"Python {sys.version}")

print(f"Pandas {pd.__version__}")

print(f"Scikit-Learn {sk.__version__}")

print("GPU is", "available" if tf.test.is_gpu_available() \

      else "NOT AVAILABLE")
df_train = pd.read_csv(PATH_TRAIN)

df_test = pd.read_csv(PATH_TEST)



df_train = df_train[df_train.id != 1300]



df_train['filename'] = df_train["id"].astype(str)+".png"

df_train['stable'] = df_train['stable'].astype(str)



df_test['filename'] = df_test["id"].astype(str)+".png"
import matplotlib.pyplot as plt



df_train.stable.value_counts().plot(kind='bar')

plt.title('Labels counts')

plt.xlabel('Stable')

plt.ylabel('Count')

plt.show()
TRAIN_PCT = 0.9

TRAIN_CUT = int(len(df_train) * TRAIN_PCT)



df_train_cut = df_train[0:TRAIN_CUT]

df_validate_cut = df_train[TRAIN_CUT:]



print(f"Training size: {len(df_train_cut)}")

print(f"Validate size: {len(df_validate_cut)}")
import tensorflow as tf

import keras_preprocessing

from keras_preprocessing import image

from keras_preprocessing.image import ImageDataGenerator



WIDTH = 150

HEIGHT = 150



training_datagen = ImageDataGenerator(

  rescale = 1./255,

  #horizontal_flip=True,

  #vertical_flip=True,

  fill_mode='nearest')



train_generator = training_datagen.flow_from_dataframe(

        dataframe=df_train_cut,

        directory=PATH,

        x_col="filename",

        y_col="stable",

        target_size=(HEIGHT, WIDTH),

        batch_size=32,

        class_mode='categorical')



validation_datagen = ImageDataGenerator(rescale = 1./255)



val_generator = validation_datagen.flow_from_dataframe(

        dataframe=df_validate_cut,

        directory=PATH,

        x_col="filename",

        y_col="stable",

        target_size=(HEIGHT, WIDTH),

        batch_size=32,

        class_mode='categorical')
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.applications import MobileNet

from tensorflow.keras.applications.mobilenet import preprocess_input

from tensorflow.keras.layers import Dense,GlobalAveragePooling2D

from tensorflow.keras.models import Model



base_model=MobileNet(weights='imagenet',include_top=False) 



x=base_model.output

x=GlobalAveragePooling2D()(x)

#x=Dense(1024,activation='relu')(x) 

#x=Dense(1024,activation='relu')(x) 

preds=Dense(2,activation='softmax')(x)



model=Model(inputs=base_model.input,outputs=preds)



for layer in model.layers[:20]:

    layer.trainable=False

for layer in model.layers[20:]:

    layer.trainable=True



model.summary()
validation_steps = len(df_validate_cut)

model.compile(loss = 'categorical_crossentropy', optimizer='adam')

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto',

        restore_best_weights=True)



history = model.fit(train_generator, epochs=25, steps_per_epoch=250, 

                    validation_data = val_generator, 

                    verbose = 1, validation_steps=20)
import numpy as np

import tensorflow as tf

from tensorflow import keras



# Display

from IPython.display import Image

import matplotlib.pyplot as plt

import matplotlib.cm as cm



def get_img_array(img_path, size):

    # `img` is a PIL image of size 299x299

    img = keras.preprocessing.image.load_img(img_path, target_size=size)

    # `array` is a float32 Numpy array of shape (299, 299, 3)

    array = keras.preprocessing.image.img_to_array(img)

    # We add a dimension to transform our array into a "batch"

    # of size (1, 299, 299, 3)

    array = np.expand_dims(array, axis=0)

    return array





def make_gradcam_heatmap(

    img_array, model, last_conv_layer_name, classifier_layer_names

):

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
# Activation for last CONV layer.

last_conv_layer_name = "conv_pw_13_relu"





classifier_layer_names = [

    "global_average_pooling2d",

    "dense",

]
def process_grad_cam(filename):

    img_path = os.path.join(PATH, filename)

    img_array = get_img_array(img_path, size=(HEIGHT, WIDTH))

    img_array /= 255.0

    preds = model.predict(img_array)

    print(f"Prediction: {preds}")

    #img = Image(img_path)

    #display(img)

    heatmap = make_gradcam_heatmap(

        img_array, model, last_conv_layer_name, classifier_layer_names

    )

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



    # Display Grad CAM

    display(superimposed_img)
# Unstable

process_grad_cam("8.png")
# Stable

process_grad_cam("1.png")
submit_datagen = ImageDataGenerator(rescale = 1./255)



submit_generator = submit_datagen.flow_from_dataframe(

        dataframe=df_test,

        directory=PATH,

        x_col="filename",

        batch_size = 1,

        shuffle = False,

        target_size=(HEIGHT, WIDTH),

        class_mode=None)



submit_generator.reset()

pred = model.predict(submit_generator,steps=len(df_test))

df_submit = pd.DataFrame({"id":df_test['id'],'stable':pred[:,1].flatten()})

df_submit.to_csv("/kaggle/working/submit.csv",index = False)
df_submit