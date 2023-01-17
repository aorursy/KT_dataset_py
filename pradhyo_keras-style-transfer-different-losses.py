# https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py
from __future__ import print_function
from keras.preprocessing.image import load_img, save_img, img_to_array
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import tensorflow as tf
from scipy import ndimage
from glob import glob

from keras.applications import vgg19
from keras import backend as K
import urllib.request
import os
import ssl

# To avoid ssl: certificate_verify_failed error
ssl._create_default_https_context = ssl._create_unverified_context

directory = "images"
out_dir = "output"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)   
    
# img_names = ["afremo_rain", "maldives", "miami", "mosaic", "nyc", "nyc2", "sf", "udnie", "wave"] 
img_names = ["nyc", "wave"]
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
width, height = 400, 400
img_nrows = height
img_ncols = int(width * img_nrows / height)
print(img_ncols)

opt_img = np.random.uniform(0, 1, size=(img_nrows, img_ncols, 3)).astype(np.float32)
plt.imshow(opt_img);
opt_img_smooth = ndimage.filters.median_filter(opt_img, [8,8,1])
plt.imshow(opt_img_smooth)
save_img(f"images/uniform.jpg", opt_img_smooth)
display_images([f"images/uniform.jpg"])
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
def style_transfer(base_image_path, 
                   style_reference_image_path, 
                   result_prefix, 
                   loss_fn,
                   style_loss_fn,
                   feature_layers,
                   iterations=10,
                   save_every=10):
    
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


    loss = (loss_fn(base_image, style_reference_image, combination_image, style_loss_fn, feature_layers))
    
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
        # save every 'save_every' images and the last one    
        if (i % save_every == 0) or (i == iterations - 1):
            fname = result_prefix + '_at_iteration_%d.png' % i
            save_img(fname, img)

        # Stop if loss doesn't reduce by more than 10%    
        if curr_loss and (((curr_loss - min_val)/curr_loss) <= 0.1):
            fname = result_prefix + '_at_iteration_%d.png' % i
            save_img(fname, img)            
            break   
        
        curr_loss = min_val
# the gram matrix of an image tensor (feature-wise outer product)
def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss_gram(style, combination):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = 400 * 400 # constant of the order of the image size
    _style_loss = K.sum(K.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))
    return(_style_loss) 
from tensorflow.keras.metrics import *

# the "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image
def style_loss_kld(style, combination):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    
    if K.image_data_format() == 'channels_first':
        style_features = K.flatten(style)
        combination_features = K.flatten(combination)
    else:
        style_features = K.flatten(K.permute_dimensions(style, (2, 0, 1)))    
        combination_features = K.flatten(K.permute_dimensions(combination, (2, 0, 1)))    
    
    _loss = kullback_leibler_divergence(style_features, combination_features)
    return  _loss
def total_loss(base_image, style_reference_image, combination_image, style_loss_fn, feature_layers):
    style_weight = 1000.0 # to get to the order of 
    
    # combine the 3 images into a single Keras tensor
    input_tensor = K.concatenate([base_image,
                                  style_reference_image,
                                  combination_image], axis=0)
    
    # build the VGG19 network with our 3 images as input
    # the model will be loaded with pre-trained ImageNet weights
    model = vgg19.VGG19(input_tensor=input_tensor,
                        weights='imagenet', include_top=False)
    
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    # combine these loss functions into a single scalar
    loss = K.variable(0.0)
    layer_features = outputs_dict['block5_conv2']
    
    for layer_name in feature_layers:
        layer_features = outputs_dict[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss_fn(style_reference_features, combination_features)
        loss += (style_weight / len(feature_layers)) * sl

    return loss

iterations = 100
input_name = "uniform"
style = "wave"
loss_fn = total_loss
style_loss_fns = [style_loss_gram]
save_every = 100

# 'block1_conv1', 'block1_conv2', 'block1_pool', 
# 'block2_conv1', 'block2_conv2', 'block2_pool', 
# 'block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_conv4', 'block3_pool', 
# 'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_conv4', 'block4_pool', 
# 'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_conv4', 'block5_pool'
feature_layers_list = [['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'],
                       [f'block{i}_conv{j}' for i in range(1, 6) for j in range(1, 3)] + [f'block{i}_conv{j}' for i in range(3, 6) for j in range(3, 5)],
                       [f'block{i}_conv{j}' for i in range(2, 6) for j in range(1, 3)] + [f'block{i}_conv{j}' for i in range(3, 6) for j in range(3, 5)],
                       [f'block{i}_conv{j}' for i in range(3, 6) for j in range(1, 3)] + [f'block{i}_conv{j}' for i in range(3, 6) for j in range(3, 5)],
                       [f'block{i}_conv{j}' for i in range(4, 6) for j in range(1, 3)] + [f'block{i}_conv{j}' for i in range(4, 6) for j in range(3, 5)],                      
                       [f'block{i}_conv{j}' for i in range(3, 6) for j in range(1, 5)],                      
                       [f'block{i}_conv{j}' for i in range(4, 6) for j in range(1, 5)],                      
                       [f'block{i}_conv{j}' for i in range(5, 6) for j in range(1, 5)],                                             
                      ]

for style_loss_fn in style_loss_fns:
    with open("feature_layers.txt", "w") as f:
        f.write("Features:")
        f.write("\n")
    for fl_index, feature_layers in enumerate(feature_layers_list):
        result_prefix = f"output/{input_name}_{style}_{style_loss_fn.__name__}_fl{fl_index}"
        print(f"Input: {input_name}; Style: {style}; Style Loss: {style_loss_fn.__name__}")
        with open("feature_layers.txt", "a") as f:
            f.write(",".join(feature_layers))
            f.write("\n")        
        style_transfer("images/" + input_name + ".jpg", 
                       "images/" + style + ".jpg", 
                       result_prefix, 
                       loss_fn,
                       style_loss_fn,
                       feature_layers,
                       iterations,
                       save_every)

        display_images(sorted(glob(f"{result_prefix}*")))
[0, 2] + [1, 2]
