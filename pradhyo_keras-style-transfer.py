# https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py
from __future__ import print_function
from keras.preprocessing.image import load_img, save_img, img_to_array
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import tensorflow as tf
from tensorflow.losses import *

from keras.applications import vgg19
from keras import backend as K
from glob import glob
import urllib.request
import os

directory = "images"
out_dir = "output"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)   
    
img_names = ["afremo_rain", "maldives", "miami", "mosaic", "nyc", "nyc2", "sf", "udnie", "wave"]   
img_urls = {
    f"{img_name}.jpg": f"https://raw.githubusercontent.com/Pradhyo/machine-learning-practice-notebooks/master/style-transfer/images/{img_name}.jpg" 
    for img_name in img_names
}    
        
# function to download images to directory    
def get_images(directory, img_urls):
    """Download images to directory"""
    if not os.path.exists(directory):
        os.makedirs(directory)    
    for name, url in img_urls.items():
        urllib.request.urlretrieve(url, directory + "/" + name)

get_images(directory, img_urls)
from os import listdir

# just making sure the images were downloaded successfully
print(listdir(directory))
from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

# function to display list of images
def display_images(image_paths):
    plt.figure(figsize=(20,20))
    columns = 3
    for i, image in enumerate(image_paths):
        plt.subplot(len(image_paths) / columns + 1, columns, i + 1)
        plt.imshow(mpimg.imread(image))    
        
display_images(list(map(lambda img: directory + "/" + img, img_urls.keys())))
    
# util function to open, resize and format pictures into appropriate tensors
def preprocess_image(image_path, img_nrows, img_ncols):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img
# util function to convert a tensor into a valid image
def deprocess_image(x, img_nrows, img_ncols):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x
# compute the neural style loss
# first we need to define 4 util functions

# the gram matrix of an image tensor (feature-wise outer product)
def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram
# the "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image
def style_loss(style, combination):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    return absolute_difference(C, S) ** 2

# an auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image
def content_loss(base, combination):
    _content_loss = K.sum(K.square(combination - base))
    return _content_loss
# the 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent
def total_variation_loss(x):
    img_nrows = x.get_shape()[1]
    img_ncols = x.get_shape()[2]
    assert K.ndim(x) == 4
    if K.image_data_format() == 'channels_first':
        a = K.square(
            x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
        b = K.square(
            x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
    else:
        a = K.square(
            x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        b = K.square(
            x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    _total_variation_loss = K.sum(K.pow(a + b, 1.25))
    return(_total_variation_loss)
# this Evaluator class makes it possible
# to compute loss and gradients in one pass
# while retrieving them via two separate functions,
# "loss" and "grads". This is done because scipy.optimize
# requires separate functions for loss and gradients,
# but computing them separately would be inefficient.
class Evaluator(object):

    def __init__(self, f_outputs, img_nrows, img_ncols):
        self.loss_value = None
        self.grads_values = None
        self.f_outputs = f_outputs
        self.img_nrows = img_nrows
        self.img_ncols = img_ncols
        
    def eval_loss_and_grads(self, x):
        if K.image_data_format() == 'channels_first':
            x = x.reshape((1, 3, self.img_nrows, self.img_ncols))
        else:
            x = x.reshape((1, self.img_nrows, self.img_ncols, 3))
        outs = self.f_outputs([x])
        loss_value = outs[0]
        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
        return loss_value, grad_values        

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = self.eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values
def loss1(base_image, style_reference_image, combination_image):
    content_weight = 1.0
    style_weight = 1.0
    total_variation_weight = 1.0
    
    # combine the 3 images into a single Keras tensor
    input_tensor = K.concatenate([base_image,
                                  style_reference_image,
                                  combination_image], axis=0)
    
    # build the VGG19 network with our 3 images as input
    # the model will be loaded with pre-trained ImageNet weights
    model = vgg19.VGG19(input_tensor=input_tensor,
                        weights='imagenet', include_top=False)
    print('Model loaded.')
    
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])


    # combine these loss functions into a single scalar
    loss = K.variable(0.0)
    layer_features = outputs_dict['block5_conv2']
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss += content_weight * content_loss(base_image_features,
                                          combination_features)

    feature_layers = [f'block{i}_conv{j}' for i in range(4, 6) for j in range(1, 5)]
    for layer_name in feature_layers:
        layer_features = outputs_dict[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(feature_layers)) * sl
    loss += total_variation_weight * total_variation_loss(combination_image)
    return loss

def style_transfer(base_image_path, 
                   style_reference_image_path, 
                   result_prefix, 
                   loss_fn,
                   iterations=10):

    # Remove existing files with prefix
    for existing_result_file in glob(f"{result_prefix}*"):
        os.remove(existing_result_file)    
    
    # dimensions of the generated picture.
    width, height = load_img(base_image_path).size
    img_nrows = 400
    img_ncols = int(width * img_nrows / height)

    # get tensor representations of our images
    base_image = K.variable(preprocess_image(base_image_path, img_nrows, img_ncols))
    style_reference_image = K.variable(preprocess_image(style_reference_image_path, img_nrows, img_ncols))

    # this will contain our generated image
    if K.image_data_format() == 'channels_first':
        combination_image = K.placeholder((1, 3, img_nrows, img_ncols))
    else:
        combination_image = K.placeholder((1, img_nrows, img_ncols, 3))


    loss = loss_fn(base_image, style_reference_image, combination_image)
    
    # get the gradients of the generated image wrt the loss
    grads = K.gradients(loss, combination_image)

    outputs = [loss]
    if isinstance(grads, (list, tuple)):
        outputs += grads
    else:
        outputs.append(grads)

    f_outputs = K.function([combination_image], outputs)

    evaluator = Evaluator(f_outputs, img_nrows, img_ncols)

    # run scipy-based optimization (L-BFGS) over the pixels of the generated image
    # so as to minimize the neural style loss
    x = preprocess_image(base_image_path, img_nrows, img_ncols)

    curr_loss = 0.0
    for i in range(iterations):
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                         fprime=evaluator.grads, maxfun=20)
        print('Iteration ' + str(i) + ' loss value:', min_val)
        # save current generated image
        img = deprocess_image(x.copy(), img_nrows, img_ncols)
        # save every 10 images and the last one    
        if (i % 10 == 0) or (i == iterations - 1):
            fname = result_prefix + '_at_iteration_%d.png' % i
            save_img(fname, img)

        # Stop if loss doesn't reduce by more than 10%    
        if curr_loss and (((curr_loss - min_val)/curr_loss) <= 0.03):
            fname = result_prefix + '_at_iteration_%d.png' % i
            save_img(fname, img)            
            break   
        
        curr_loss = min_val            
# Running an image with a style
iterations = 5
input_name = "nyc"
style = "wave"

result_prefix = "output/" + input_name + "_" + style
print("Input: " + input_name + "; Style: " + style)
style_transfer("images/" + input_name + ".jpg", 
               "images/" + style + ".jpg", 
               result_prefix, 
               loss1,
               iterations)

display_images(sorted(glob(f"{result_prefix}*")))
def loss2(base_image, style_reference_image, combination_image):
    content_weight = 1.0
    style_weight = 3.0
    total_variation_weight = 1.0
    
    # combine the 3 images into a single Keras tensor
    input_tensor = K.concatenate([base_image,
                                  style_reference_image,
                                  combination_image], axis=0)
    
    # build the VGG19 network with our 3 images as input
    # the model will be loaded with pre-trained ImageNet weights
    model = vgg19.VGG19(input_tensor=input_tensor,
                        weights='imagenet', include_top=False)
    print('Model loaded.')
    
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])


    # combine these loss functions into a single scalar
    loss = K.variable(0.0)
    layer_features = outputs_dict['block5_conv2']
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss += content_weight * content_loss(base_image_features,
                                          combination_features)

    # https://www.kaggle.com/pradhyo/keras-style-transfer-different-losses
    feature_layers = [f'block{i}_conv{j}' for i in range(4, 6) for j in range(1, 5)]
    
    for layer_name in feature_layers:
        layer_features = outputs_dict[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(feature_layers)) * sl
    loss += total_variation_weight * total_variation_loss(combination_image)  
    return loss

# Running an image with a style
iterations = 10
input_name = "nyc"
style = "wave"

result_prefix = "output/" + input_name + "_" + style
print("Input: " + input_name + "; Style: " + style)
style_transfer("images/" + input_name + ".jpg", 
               "images/" + style + ".jpg", 
               result_prefix, 
               loss2,
               iterations)

display_images(sorted(glob(f"{result_prefix}*")))
# Running multiple images with multiple styles
iterations = 50

input_names = ['sf', 'nyc2', 'miami', 'maldives', 'nyc']
styles = ['mosaic', 'udnie', 'wave', 'afremo_rain']
for input_name in input_names:
    for style in styles:
        result_prefix = "output/" + input_name + "_" + style
        print("Input: " + input_name + "; Style: " + style)
        style_transfer(f"images/{input_name}.jpg", 
                       f"images/{style}.jpg", 
                       result_prefix, 
                       loss2,
                       iterations)

display_images(sorted(glob(f"{result_prefix}*")))