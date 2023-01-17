import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

%matplotlib inline

from random import shuffle

import glob

from PIL import Image

import time

import functools

from kaggle_datasets import KaggleDatasets



import tensorflow as tf

from tensorflow.keras.preprocessing.image import img_to_array

from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam
content1 = '../input/contentimg/shannon-kunkle-dM8INmkyDas-unsplash.jpg'

content2 = '../input/contentimg/bailey-zindel-NRQV-hBF10M-unsplash.jpg'

content3 = '../input/contentimg/sebastian-boring-8zD7rs8UpxU-unsplash.jpg'



wave = '../input/style-images/kanagawa wave.jpg'

nature = '../input/style-images/nature oil.jpg'

wheat = '../input/style-images/wheat field vincent.jpg'
plt.figure(figsize = (15,10))

style = Image.open(wave)

plt.imshow(style)
plt.figure(figsize = (15,10))

style = Image.open(nature)

plt.imshow(style)
plt.figure(figsize = (15,10))

style = Image.open(wheat)

plt.imshow(style)
plt.figure(figsize = (15,10))

cont = Image.open(content1)

plt.imshow(cont)
plt.figure(figsize = (15,10))

cont = Image.open(content2)

plt.imshow(cont)
plt.figure(figsize = (15,10))

cont = Image.open(content3)

plt.imshow(cont)
def load_img(path_to_img):

    max_dim = 1080

    img = Image.open(path_to_img)

    long = max(img.size)

    scale = max_dim/long

    img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)

  

    img = img_to_array(img)

  

  # We need to broadcast the image array such that it has a batch dimension 

    img = np.expand_dims(img, axis=0)

    return img
def preprocess_img(img_path):

    img = load_img(img_path)

    img = preprocess_input(img)

    return img
def deprocess_img(process_img):

    x = process_img.copy()

    if len(x.shape) == 4:

        x = np.squeeze(x, axis = 0)

    assert len(x.shape) == 3   #Expected input shape [1,H,W,C] or [H,W,C]

    if len(x.shape) !=3:

        raise ValueError('Invalid input')

    

    #We will add the individual channel mean.

    x[:, :, 0] += 103.939

    x[:, :, 1] += 116.779

    x[:, :, 2] += 123.68

    

    x = x[..., ::-1]             #Converting BGR to RGB

    

    x = np.clip(x, 0, 255).astype('uint8')    #clipping pixel values

    

    return x
content_layers = ['block5_conv2']

style_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']
def get_model():

    vgg = VGG19(include_top = False, weights = 'imagenet')

    vgg.trainable = False

    

    content_output = [vgg.get_layer(layer).output for layer in content_layers]

    style_output = [vgg.get_layer(layer).output for layer in style_layers]

    output = content_output + style_output

    return Model(inputs = vgg.inputs, outputs = output)
def content_loss(content, base_img):

    return tf.reduce_mean(tf.square(content - base_img))
def gram_matrix(input_tensor):

    channels = int(input_tensor.shape[-1])

    vector = tf.reshape(input_tensor, [-1, channels])

    n = tf.shape(vector)[0]

    gram = tf.matmul(vector, vector, transpose_a = True)

    return gram / tf.cast(n, tf.float32)



def style_loss(style, base_img):

    gram_style = gram_matrix(style)

    gram_gen = gram_matrix(base_img)

    return tf.reduce_mean(tf.square(gram_style - gram_gen))
def get_feature_representation(model, content_path, style_path):

    content_img = preprocess_img(content_path)

    style_img = preprocess_img(style_path)

    

    content_output = model(content_img)

    style_output = model(style_img)

    

    content_features = [content_layer[0] for content_layer in content_output[:len(style_layers)]]

    style_features = [style_layer[0] for style_layer in style_output[len(style_layers):]]

    

    return content_features, style_features
def compute_loss(model, content_features, style_features, base_img, loss_weights):

    

    """This function will compute the loss total loss.

  

    Arguments:

    model: The model that will give us access to the intermediate layers

    base_img: Our initial base image. This image is what we are updating with 

      our optimization process. We apply the gradients wrt the loss we are 

      calculating to this image.

    style_features: Precomputed gram matrices corresponding to the 

      defined style layers of interest.

    content_features: Precomputed outputs from defined content layers of 

      interest.

    loss_weight: The weights of each contribution of each loss function. 

      (style weight, content weight, and total variation weight)  

    Returns:

    returns the total loss, content loss, style loss

    """

    

    content_weight, style_weight = loss_weights    #Also known as alpha and beta

    

    output = model(base_img)

    content_base_features = output[:len(style_layers)]    #feature output of base_img w.r.t. content image

    style_base_features = output[len(style_layers):]    #feature output of base_img w.r.t. style image

    

    content_score, style_score = 0, 0

    

    weights_per_content_layer = 1.0 / float(len(content_layers))       #getting weights from content layer 

    #content_feature is from content image and content_base_feature are from base_img or generated noise(image)

    for content_feature, content_base_feature in zip(content_features, content_base_features):

        content_score += weights_per_content_layer * content_loss(content_feature, content_base_feature[0])

        

    weights_per_style_layer = 1.0 / float(len(style_layers))     #getting equally distributed weights from individual layer

    for style_feature, style_base_feature in zip(style_features, style_base_features):

        style_score += weights_per_style_layer * style_loss(style_feature, style_base_feature[0])

        

    content_score *= content_weight

    style_score *= style_weight

    

    total_loss = content_score + style_score

    return total_loss, content_score, style_score
def compute_grad(args):

    with tf.GradientTape() as grad:

        loss = compute_loss(**args)

    

    gradients = grad.gradient(loss[0], args['base_img'])

    return gradients, loss
def style_transfer(content_path, style_path, epochs, content_weight, style_weight):

    model = get_model()

    for layer in model.layers:

        layer.trainable = False

        

    content_features, style_features = get_feature_representation(model, content_path, style_path)

    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

    

    base_img = preprocess_img(content_path)

    base_img = tf.Variable(base_img, dtype = tf.float32)

    

    #optimizer = Adam(lr = 5, beta_1 = 0.99, epsilon = 1e-1)

    optimizer = tf.keras.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)

    best_loss, best_img = float('inf'), None    #https://stackoverflow.com/questions/34264710/what-is-the-point-of-floatinf-in-python

    loss_weights = (content_weight, style_weight)   #alpha and beta

    

    args = {'model':model,'content_features':content_features,'style_features':style_features,'base_img':base_img,'loss_weights':loss_weights}

    

    channel_normalized_means = np.array([103.939, 116.779, 123.68])

    min_val = -channel_normalized_means

    max_val = 255 - channel_normalized_means

    

    iter_count = 1

    

    plt.figure(figsize=(15, 15))

    num_rows = (epochs / 100) // 5

    

    start_time = time.time()

    global_start = time.time()

    

    images = []

    for i in range(epochs):

        gradients, loss = compute_grad(args)

        total_loss, content_score, style_score = loss

        optimizer.apply_gradients([(gradients, base_img)])

        clip = tf.clip_by_value(base_img, min_val, max_val)     #https://stackoverflow.com/questions/44796793/difference-between-tf-clip-by-value-and-tf-clip-by-global-norm-for-rnns-and-how/44798131

        base_img.assign(clip)

        end_time = time.time() 



        

        if total_loss < best_loss:

            best_loss = total_loss

            best_img = deprocess_img(base_img.numpy())

  

    print('Total time: {:.4f}s'.format(time.time() - global_start))

      

    return best_img, best_loss             
best, best_loss = style_transfer(content1, nature, 1000, 1e2, 2e3)

Image.fromarray(best)
best, best_loss = style_transfer(content2, wheat, 1000, 1e2, 2e3)

Image.fromarray(best)
best, best_loss = style_transfer(content3, wave, 1000, 1e2, 1e3)

Image.fromarray(best)