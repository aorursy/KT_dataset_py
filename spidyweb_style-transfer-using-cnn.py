# Importing the libraries

from __future__ import print_function
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg19
from keras import backend as K
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse
import cv2
import matplotlib.pyplot as plt

%matplotlib inline
# loading the content image and style image

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        

content_image_path = ("../input/rabbit-and-a-style-for-style-transfer/rabbit.jpg")
style_reference_image_path = ("../input/rabbit-and-a-style-for-style-transfer/style.jpg")

# prefix for saving the generated image 
result_prefix = 'gen'

# Number of iterations 
iterations = 10

# these are the weights of the different loss components
content_weight = 0.025
style_weight = 1.0

from IPython.display import Image
Image("../input/rabbit-and-a-style-for-style-transfer/rabbit.jpg")
# defining dimensions of the generated image
width, height = load_img(content_image_path).size

gen_height = 400

# Resizing width according to the height
gen_width = int( (width/ height) * gen_height)

print("Ratio of width is to height of content image is ", (width/height))
print("Ratio of width is to height of generated image is ", (gen_width/gen_height))
# Reading the content image
fig=plt.figure(figsize=(8, 8))
content_img = cv2.imread(content_image_path)
plt.imshow(cv2.cvtColor(content_img, cv2.COLOR_BGR2RGB))
# Loading the style image
fig=plt.figure(figsize=(8, 8))
style_img = cv2.imread(style_reference_image_path)
plt.imshow(cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB))
# preprocessing the image
# resizing the image to the size of generated image
# first we are expanding the dimension so as to include the batch size so it can be given to the network
# Then we are preprocessing the standard way of vgg19 as we are using its trained model
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(gen_height, gen_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img
def deprocess_image(x):
    x = x.reshape((gen_height, gen_width, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x
# Get tensor representations of our images and store them in tensorflow variable
# These will go as input to the model
content_image = K.variable(preprocess_image(content_image_path))
style_reference_image = K.variable(preprocess_image(style_reference_image_path))

# placeholder will contain our generated image
generated_image = K.placeholder((1, gen_height, gen_width, 3))

# combine the 3 images into a single Keras tensor along the axis of batch size(axis = 0)
# batch size = 3 (content_image,style_reference_image, generated_image)
input_tensor = K.concatenate([content_image,
                              style_reference_image,
                              generated_image], axis=0)
# build the VGG19 network with our 3 images as input
# the model will be loaded with pre-trained ImageNet weights
model = vgg19.VGG19(input_tensor=input_tensor,
                    weights='imagenet', include_top=False)
print('Model loaded.')
# Getting the model layers
print("Summary of the model", model.summary())
# get the symbolic outputs of each "key" layer (we gave them unique names).
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
print(" The output layers are ", outputs_dict)
# Selecting the layer  for content loss
layer_features = outputs_dict['block5_conv2']

# Along the dimension of batch size, the first is content image, second is style image and third the generated image
# get the content image feature 
content_image_features = layer_features[0, :, :, :]
print("Shape of content_image_featues is ", content_image_features.shape )

# get the generated image feature
generated_features = layer_features[2, :, :, :]
print("Shape of generated_features is ", generated_features.shape )
# compute the neural style loss
# the gram matrix of an image tensor (feature-wise outer product)
# In the output of size (25, 47, 512) there are 512 feature vectors, each of size 25 x 47

def gram_matrix(x):
    # Since the features are along the 3rd axis, we permute them to bring them to the first axis.
    # The tensor will become a tensor of size (512, 25, 47)
    # Then we flatten the tensor. It becomes (512, 1175)
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    
    # We take dot product with itself - (512 x 1175  x  1175 x 512) - which results in (512 x 512) matrix
    gram = K.dot(features, K.transpose(features))
    return gram
def style_loss(style, generation):
    
    '''
    Input: 
    style      -  the style feature vector that we get using the VGG19
    generation -  the generated feature vector that we get using the VGG19
    
    Returns: style loss
    '''
    
    assert K.ndim(style) == 3
    assert K.ndim(generation) == 3
    
    S = gram_matrix(style)
    C = gram_matrix(generation)
    
    channels = 3
    size = gen_width * gen_height
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

# an auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image

def content_loss(content, generation):
    
    '''
    Input: 
    content    -  the content feature vector that we get using the VGG19
    generation -  the generated feature vector that we get using the VGG19
    
    Returns: content loss
    '''
    return K.sum(K.square(generation - content))
# make variable to store total loss and initialise it with '0.'
loss = K.variable(0.)

# calculating the content loss by multiplying content loss with its weight
lossy = content_weight * content_loss(content_image_features,
                                      generated_features)
loss = loss + lossy

# defining the layers for which we want to calculate style loss
feature_layers = ['block1_conv1', 'block2_conv1',
                  'block3_conv1', 'block4_conv1',
                  'block5_conv1']

# calculating the style loss for all the feature layers
for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    
    # style feature is the second layer along the batch dimension and generated features is along the third 
    style_reference_features = layer_features[1, :, :, :]
    generated_features = layer_features[2, :, :, :]
    
    # calculating the style loss for each feature layer 
    sl = style_loss(style_reference_features, generated_features)
    
    # add the style loss to previous loss to calculate total loss
    loss += (style_weight / len(feature_layers)) * sl
# get the gradients of the generated image w.r.t. the loss
grads = K.gradients(loss, generated_image)

# get the loss and gradient in the output variable
outputs = [loss]

# add gradient to the output
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)
    
# K.function(inputs, outputs, updates=None), like how we make model, where
# inputs: List of placeholder tensors, here it is generated_image which we have defined above
# outputs: List of output tensors, here it is loss with which we want to update the inputs
# returns outputs(loss here) after evaluating inputs(generated image) on model
# https://keras.io/getting-started/faq/

f_outputs = K.function([generated_image], outputs)
def eval_loss_and_grads(x):
    
    '''
    x : x is the generated image on which we do iteration. So, here we feed generated image to the model which
        we have defined using K.function(input, output). It will return output(outs) which is combination of loss and grad.
        We segregate the loss value and grad value.
    
    Returns: loss_value, grad_values
    
    '''
    
    x = x.reshape((1, gen_height, gen_width, 3))
    
    # here we feed generated input to the model to calulate the output
    outs = f_outputs([x])
    
    # Rembember the first value of outs is loss value and the second value is gradient
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values
class Evaluator(object):
    
    '''
    Input: Input to the fucntion is generated image. Here to pass it to eval_loss_and_grads function
           to calculate the loss and grad on it. 
    
    Return: loss function will return loss value 
            grad fucntion will return grad value
    '''

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    # Return loss value     
    def loss(self, x):
        assert self.loss_value is None
        # call the eval_loss_and_grads function, using input as generated image
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        # Return loss value
        return self.loss_value

    # Return grad values
    def grads(self, x):
        assert self.loss_value is not None
        # Make copy of grad value
        grad_values = np.copy(self.grad_values)
        # set the loss value and grad values none for next iteration
        self.loss_value = None
        self.grad_values = None
        # Return grad value for current iteration
        return grad_values
# Initialize the class
evaluator = Evaluator()
# run scipy-based optimization (L-BFGS) over the pixels of the generated image
# so as to minimize the neural style loss

# Initialize the generated image with content image (x) and pass to  fmin_l_bfgs_b()
x = preprocess_image(content_image_path)
iterations = 10
for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    
    # optimizing the neural style loss
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    
    # save current generated image
    img = deprocess_image(x.copy())
    fname = result_prefix + '_at_iteration_%d.png' % i
    cv2.imwrite(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i, end_time - start_time))
# Generated imaage at first iteration
fig=plt.figure(figsize=(8, 8))
gen_image_final = cv2.imread('gen_at_iteration_1.png')    
plt.imshow(cv2.cvtColor(gen_image_final, cv2.COLOR_BGR2RGB))
# Generated imaage at fifth iteration
fig=plt.figure(figsize=(8, 8))
gen_image_final = cv2.imread('gen_at_iteration_5.png')    
plt.imshow(cv2.cvtColor(gen_image_final, cv2.COLOR_BGR2RGB))
# Generated imaage at 9th iteration
fig=plt.figure(figsize=(8, 8))
gen_image_final = cv2.imread('gen_at_iteration_9.png')    
plt.imshow(cv2.cvtColor(gen_image_final, cv2.COLOR_BGR2RGB))
