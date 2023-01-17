import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

from matplotlib.image import imread

from PIL import Image

import IPython.display



import tensorflow as tf

import keras.preprocessing.image as process_im

from keras.models import Model

from tensorflow.python.keras import models 

from tensorflow.python.keras import losses

from tensorflow.python.keras import layers

from tensorflow.python.keras import backend as K



import functools

# Set paths to content and style pictures



content_paths = {'rainbow': '/kaggle/input/pictures/DSC04907_IG.jpg',

                'rainbow2': '/kaggle/input/pictures/DSC05029_IG.jpg',

                'tower': '/kaggle/input/pictures/DSC06205-Edit_IG.jpg',

                'mp': '/kaggle/input/pictures/LRM_EXPORT_20190925_190945.jpg'}

style_paths = {'vg': '/kaggle/input/pictures/vangogh.jpeg',

              'umbrella': '/kaggle/input/pictures/rainprincess.jpg',

              'face': '/kaggle/input/pictures/face.png',

              'scream': '/kaggle/input/pictures/the-scream.jpg'}
# Plot sample images



def plot_samples(path1, path2):  

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))

    image = imread(path1)

    ax[0].imshow(image)

    ax[0].axis('off')

    ax[0].set_title("Content Image", fontsize=20)

    image = imread(path2)

    ax[1].imshow(image)

    ax[1].axis('off')

    ax[1].set_title("Style Image", fontsize=20)

    

%matplotlib inline 

plot_samples(content_paths['rainbow'], style_paths['face'])
# load image as array

def load_file(image_path):

    image =  Image.open(image_path)

    max_dim = 512

    factor = max_dim/max(image.size)

    image = image.resize((round(image.size[0]*factor),round(image.size[1]*factor)),

                         Image.ANTIALIAS)

    im_array = process_im.img_to_array(image)

    #adding extra axis to the array as to generate a batch of single image 

    im_array = np.expand_dims(im_array,axis=0) 

    return im_array



# preprocess image to vgg19 input requirements

def img_preprocess(img_path):

    image = load_file(img_path)

    img = tf.keras.applications.vgg19.preprocess_input(image)

    return img



# inverse preprocess step

def deprocess_img(processed_img):

  x = processed_img.copy()

  if len(x.shape) == 4:

    x = np.squeeze(x, 0)

  # Input dimension must be [1, height, width, channel] or [height, width, channel]

  assert len(x.shape) == 3 

  

  

  # perform the inverse of the preprocessing step

  x[:, :, 0] += 103.939

  x[:, :, 1] += 116.779

  x[:, :, 2] += 123.68

  x = x[:, :, ::-1] # converting BGR to RGB channel



  x = np.clip(x, 0, 255).astype('uint8')

  return x
# Test image loading functions



img = img_preprocess(content_paths['rainbow'])



fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))

ax[0].imshow(img[0])

ax[0].axis('off')

ax[0].set_title("Processed Image", fontsize=20)



img_deprocess = deprocess_img(img)

ax[1].imshow(img_deprocess)

ax[1].axis('off')

ax[1].set_title("Deprocessed Image", fontsize=20)



plt.show()
# set the layers to get content and style 

content_layers = ['block5_conv2']

style_layers = ['block1_conv1',

                'block2_conv1',

                'block3_conv1', 

                'block4_conv1', 

                'block5_conv1']

number_content=len(content_layers)

number_style =len(style_layers)
def get_model():

    # get vgg model

    vgg=tf.keras.applications.vgg19.VGG19(include_top=False,weights='imagenet')

    vgg.trainable=False

    # get the content and style layers

    content_output=[vgg.get_layer(layer).output for layer in content_layers]

    style_output=[vgg.get_layer(layer).output for layer in style_layers]

    model_output= style_output+content_output

    return models.Model(vgg.input,model_output) # set model input and output



model = get_model()

model.summary()
model.output
def get_content_loss(generated_content_feature,content_features):

    # equation 2

    loss = tf.reduce_mean(tf.square(generated_content_feature-content_features))

    return loss
def gram_matrix(tensor):

    # refer to picture above for better visualization

    channels=int(tensor.shape[-1]) # get number of channels

    vector=tf.reshape(tensor,[-1,channels]) # unroll into 2d matrix (h*w,channels)

    n=tf.shape(vector)[0]

    gram_matrix=tf.matmul(vector,vector,transpose_a=True) # compute gram matrix

    return gram_matrix/tf.cast(n,tf.float32)



def get_style_loss(generated_style_features,style_gram_matrix):

    # get gram matrix of generated activations

    generated_gram_matrix = gram_matrix(generated_style_features) 

    # frobenius norm of gram matrices -- equation 3

    loss = tf.reduce_mean(tf.square(style_gram_matrix-generated_gram_matrix)) 

    return loss
# extract out features/activations of content and style images 

# from model output

def get_features(model,content_path,style_path):

    # preprocess images

    content_img=img_preprocess(content_path)

    style_image=img_preprocess(style_path)

    

    # output of model for each image

    content_output=model(content_img)

    style_output=model(style_image)

    

    # extract out the features/activations, content is last layer

    content_feature = [layer[0] for layer in content_output[number_style:]]

    # style has few layers

    style_feature = [layer[0] for layer in style_output[:number_style]]

    return content_feature,style_feature
def compute_loss(model, loss_weights,image, style_gram_matrix, content_features):

    #style weight and content weight are user given parameters

    #that define what percentage of content and/or style will be preserved 

    # in the generated image

    style_weight,content_weight = loss_weights 

    

    output=model(image)

    content_loss=0

    style_loss=0

    

    # extract content,style activations of generated image

    generated_style_features = output[:number_style]

    generated_content_feature = output[number_style:]

    

    # compute style loss

    weight_per_layer = 1.0/float(number_style) # our λ weighting is equal here

    for a,b in zip(style_gram_matrix,generated_style_features):

        style_loss+=weight_per_layer*get_style_loss(b[0],a) # equation 4

        

    # compute content loss

    weight_per_layer =1.0/ float(number_content)

    for a,b in zip(generated_content_feature,content_features):

        content_loss+=weight_per_layer*get_content_loss(a[0],b)

    

    # apply α and β in equation 1

    style_loss *= style_weight

    content_loss *= content_weight

    # get total loss -- equation 1

    total_loss = content_loss + style_loss

    

    

    return total_loss,style_loss,content_loss
def compute_grads(dictionary):

    with tf.GradientTape() as tape:

        all_loss=compute_loss(**dictionary)

        

    total_loss=all_loss[0]

    # return gradients and loss

    return tape.gradient(total_loss,dictionary['image']),all_loss
def run_style_transfer(content_path,style_path,epochs=500, content_weight=1e3, style_weight=1e-3, show_img=False):

    

    model=get_model()

    

    for layer in model.layers:

        layer.trainable = False

    

    # get activations and style matrix

    content_feature,style_feature = get_features(model,content_path,style_path)

    # get gram matrix of style image

    style_gram_matrix=[gram_matrix(feature) for feature in style_feature]

    

    # initialize generated image

    generated = img_preprocess(content_path)

    generated = tf.Variable(generated,dtype=tf.float32)

    

    optimizer = tf.keras.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)

    

    best_loss,best_img=float('inf'),None

    

    # relative weightings of content and style image

    loss_weights = (style_weight, content_weight)

    dictionary={'model':model,

              'loss_weights':loss_weights,

              'image':generated,

              'style_gram_matrix':style_gram_matrix,

              'content_features':content_feature}

    

    # for clipping image

    norm_means = np.array([103.939, 116.779, 123.68])

    min_vals = -norm_means

    max_vals = 255 - norm_means   

  

    imgs = []

    for i in range(epochs):

        # run gradient descent and update generated image

        grad,all_loss = compute_grads(dictionary)

        total_loss,style_loss,content_loss = all_loss

        optimizer.apply_gradients([(grad,generated)])

        clipped=tf.clip_by_value(generated,min_vals,max_vals)

        generated.assign(clipped)

        

        if total_loss<best_loss:

            best_loss = total_loss

            best_img = deprocess_img(generated.numpy())

            

        # for visualization 

        # print out every 5 epochs

        

        if show_img:

            if i%5==0:

                plot_img = generated.numpy()

                plot_img = deprocess_img(plot_img)

                imgs.append(plot_img)

                IPython.display.clear_output(wait=True)

                IPython.display.display_png(Image.fromarray(plot_img))

                print('Epoch: {}'.format(i))        

                print('Total loss: {:.4e}, ' 

                  'style loss: {:.4e}, '

                  'content loss: {:.4e}, '.format(total_loss, style_loss, content_loss))

    

    IPython.display.clear_output(wait=True)

    

    

    return best_img,best_loss,imgs
best, best_loss,image = run_style_transfer(content_paths['rainbow'], 

                                           style_paths['face'], epochs=1000,

                                           style_weight=1e-4,

                                           show_img=True)
def plot_images(generated, content_path, style_path):

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,5))

    ax[0].imshow(generated)

    ax[0].axis('off')

    ax[0].set_title("Generated Image", fontsize=20)

    

    image = imread(content_path)

    ax[1].imshow(image)

    ax[1].axis('off')

    ax[1].set_title("Content Image", fontsize=20)

    

    image = imread(style_path)

    ax[2].imshow(image)

    ax[2].axis('off')

    ax[2].set_title("Style Image", fontsize=20)
plot_images(best, content_paths['rainbow'], style_paths['face'])
best, best_loss,image = run_style_transfer(content_paths['mp'], 

                                           style_paths['vg'], epochs=1000,

                                           style_weight=1e-3)

plot_images(best, content_paths['mp'], style_paths['vg'])