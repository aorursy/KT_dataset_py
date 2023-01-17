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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19
base_img_path=keras.utils.get_file("paris.jpg", "https://i.imgur.com/F28w3Ac.jpg")
style_reference_image_path = keras.utils.get_file(
    "starry_night.jpg", "https://i.imgur.com/9ooB60I.jpg"
)
result_prefix = "paris_generated"
from IPython.display import Image,display
display(Image(base_img_path))
display(Image(style_reference_image_path))
def preprocess_img(img_path):
    img=keras.preprocessing.image.load_img(img_path,target_size=(img_nrows, img_ncols))
    img=keras.preprocessing.image.img_to_array(img)
    img=np.expand_dims(img,axis=0)
    img=vgg19.preprocess_input(img)
    return img
def deprocess_image(x):
    x=x.reshape((img_nrows, img_ncols,3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x
def gram_style(x):
    x=tf.transpose(x,(2,0,1))
    features=tf.reshape(x,(tf.shape(x)[0],-1))
    gram=tf.matmul(features,tf.transpose(features))
    return gram
def style_loss(style,combination):
    S=gram_style(style)
    C=gram_style(combination)
    channels=3
    size=img_nrows * img_ncols
    return tf.reduce_sum(tf.square(S-C))/(4*(channels**2)*(size**2))
    
def content_loss(base,combination):
    return tf.reduce_sum(tf.square(combination-base))

def total_variation_loss(x):
    a = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, 1:, : img_ncols - 1, :]
    )
    b = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, : img_nrows - 1, 1:, :]
    )
    return tf.reduce_sum(tf.pow(a + b, 1.25))
model=vgg19.VGG19(weights="imagenet",include_top=False) 
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers]) 
feature_dict = keras.Model(inputs=model.inputs,outputs=outputs_dict)
# List of layers to use for the style loss.
style_layer_names = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]
# The layer to use for the content loss.
content_layer_name = "block5_conv2"
def compute_loss(combination_image,base_image,style_reference_image):
    input_tensor=tf.concat([base_image,style_reference_image,combination_image],axis=0)
    features=feature_dict(input_tensor)
    
    loss=tf.zeros(shape=())
    layer_features=features[content_layer_name]
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss = loss + content_weight * content_loss(
        base_image_features, combination_features
    )
    
    for layer_name in style_layer_names:
        layer_features=features[layer_name]
    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(style_layer_names)) * sl

    # Add total variation loss
    loss += total_variation_weight * total_variation_loss(combination_image)
    return loss
@tf.function
def compute_loss_and_grads(combination_image, base_image, style_reference_image):
    with tf.GradientTape() as tape:
        loss=compute_loss(combination_image,base_image,style_reference_image)
    grads=tape.gradient(loss,combination_image)
    return loss ,grads

total_variation_weight = 1e-6
style_weight = 1e-6
content_weight = 2.5e-8

# Dimensions of the generated picture.
width, height = keras.preprocessing.image.load_img(base_img_path).size
img_nrows = 400
img_ncols = int(width * img_nrows / height)
optimizer = keras.optimizers.SGD(keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96
    ))
base_image=preprocess_img(base_img_path)
style_reference_image=preprocess_img(style_reference_image_path)
combination_image=tf.Variable(preprocess_img(base_img_path))
iterations = 4000
for i in range(1, iterations + 1):
    loss, grads = compute_loss_and_grads(
        combination_image, base_image, style_reference_image
    )
    optimizer.apply_gradients([(grads, combination_image)])
if i % 100 == 0:
        print("Iteration %d: loss=%.2f" % (i, loss))
        img = deprocess_image(combination_image.numpy())
        fname = result_prefix + "_at_iteration_%d.png" % i
        keras.preprocessing.image.save_img(fname, img)
display(Image(result_prefix + "_at_iteration_4000.png"))
