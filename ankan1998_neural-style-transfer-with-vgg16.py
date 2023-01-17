# Referred from Research Paper Controlling perceptual factors in neural style transfer - Gatys et al

# Andrew Ng's neural style transfer



import numpy as np

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.applications import VGG16

"""

# Google-colab

from kaggle import files

content=files.upload()

for c in content.keys():

  # Upload Content image

  content_image_path='/content/' + c"""

  
content_image_path="../input/taj-pic/Taj Mahal.jpeg"
"""

# Google Colab

from google.colab import files

style=files.upload()

for s in style.keys():

  # Upload Style image

  style_art_image_path='/content/' + s"""

  
style_art_image_path="../input/oil-painting/oil_painting.jpg"
# Combined image dimension

# Size of content image is taken into account for columns

w, h = keras.preprocessing.image.load_img(content_image_path).size

rows = 600

cols = int(w * rows / h)

from tensorflow.keras.applications import vgg16

def preprocess_image(image_path):

    # Util function to open, resize and format pictures into appropriate tensors

    img = keras.preprocessing.image.load_img(

        image_path, target_size=(rows, cols)

    )

    img = keras.preprocessing.image.img_to_array(img)

    # Expanding dimension to add 1 to dimension to match the dimensions

    img = np.expand_dims(img, axis=0)

    img = vgg16.preprocess_input(img)

    return tf.convert_to_tensor(img)
# Source of three type of image

content_image = preprocess_image(content_image_path)

style_art_image = preprocess_image(style_art_image_path)

# Intializing tf.Variable constructor with random tensor, properties of tensor from content image

generated_combined_image = tf.Variable(preprocess_image(content_image_path))

#######
# VGG16 model loaded 

# weights="imagenet" loads pre-trained ImageNet weights

# include_top=False to not include the 3 fully-connected layers at the top of the network

model = VGG16(weights="imagenet", include_top=False)

#use model.summary() to check the layers in vgg16

#It would helpful to extract desired activation layer



#layer_name_output_dict contains key value pair of key being layer_name and its output

layer_name_output_dict = dict([(layer.name, layer.output) for layer in model.layers])



# Set up a model that returns the activation values for every layer in VGG16 (as a dict).

#This vgg16_features_engine will take a input tensor and outputs 

vgg16_features_engine = keras.Model(inputs=model.inputs, outputs=layer_name_output_dict)
model.summary()
#vgg16_features_engine.save('/tmp/mvgg16.h5')
#from tensorflow.keras.models import load_model

#xvgg16=load_model('/tmp/mvgg16.h5')
# CALCULATE TOTAL LOSS= alpha x CONTENT_COST_FUNCTION + beta x STYLE_COST_FUNCTION

# Layer for the content loss

# Content_loss_layer is chosen from higher layer to get the context of image

content_layer_name = "block5_conv3"



# Layers for the style loss.

# Style_loss_layer is chosen from every layer get the texture of the style image

style_layer_names = [

    "block1_conv1",

    "block2_conv1",

    "block3_conv2",

    "block4_conv2",

    "block5_conv1",

]



# Weights of the different loss components

total_loss_coeff = 1.05e-6

beta = 1.1e-6

alpha = 2.3e-7



def total_loss(generated_combined_image, content_image, style_art_image):

    #we take tensors

    extracted_tensors = tf.concat([content_image, style_art_image, generated_combined_image], axis=0)

    

    features = vgg16_features_engine(extracted_tensors)

    

    # Loss initialized with zeros

    loss = tf.zeros(shape=())

    # As in extracted_tensors 0=content_image;1=style_art_image;2=generated_combined_image

    # Add content loss

    layer_features = features[content_layer_name]

    # activation of layer "block5_conv3" of content_image 

    content_image_features = layer_features[0, :, :, :]

    combination_features = layer_features[2, :, :, :]

    # Calculate element-wise sum of squared difference multiplied with alpha and added to loss

    loss = loss + alpha * tf.reduce_sum(tf.square(combination_features - content_image_features))

    

    

    def style_matrix(feature_matrix_X):

    # style is defined as correlation between activation across different channel

    # style_matrix is 2D whereas the image is 3D. 

    # The first line permute transpose to bring channel dimension in row dimension ==> (2,0,1)

    # Flattened it along spatial dimension i.e we multiply its height and width together that reshape does

    # style_matrix is also called gram-matrix, it is a autocorrelation i.e X.X'(transpose)/N

    # Finally it takes X(c,hw)==> X(c,c)

      feature_matrix_X = tf.transpose(feature_matrix_X, (2, 0, 1))

      features = tf.reshape(feature_matrix_X, (tf.shape(feature_matrix_X)[0], -1))

      return tf.matmul(features, tf.transpose(features))

       



    # Add style loss

    for layer_name in style_layer_names:

        layer_features = features[layer_name]

        #activation of layer from block_1 to block_5 of style image

        style_art_features = layer_features[1, :, :, :]

        # activation of layer of generated_combined_image

        combination_features = layer_features[2, :, :, :]

        # To get correlation between channels of style image and combination image

        S = style_matrix(style_art_features)

        G = style_matrix(combination_features)

        channels = 3

        size = rows * cols

        # Calculate element-wise sum of squared difference multiplied with beta and added to loss

        # And dividing by a constant

        style_loss = tf.reduce_sum(tf.square(S - G)) / (4.0 * (channels ** 2) * (size ** 2))

        

        loss =loss+ (beta / len(style_layer_names)) * style_loss



    

    a = tf.square(generated_combined_image[:, : rows - 1, : cols - 1, :] - generated_combined_image[:, 1:, : cols - 1, :])

    b = tf.square(generated_combined_image[:, : rows - 1, : cols - 1, :] - generated_combined_image[:, : rows - 1, 1:, :])  

    

    # Adding total loss

    loss =loss+ total_loss_coeff * tf.reduce_sum(tf.pow(a + b, 1.25))

    return loss



# Custom loss function with custom differentiation in tensorflow to get gradient 

def total_loss_and_gradient(generated_combined_image, content_image, style_art_image):

  # Gradient Tape performs differentiation 

    with tf.GradientTape() as t:

        loss = total_loss(generated_combined_image, content_image, style_art_image)

    # To simply put does gradient descent on loss function

    grads = t.gradient(loss, generated_combined_image)

    return loss, grads

import datetime

# Inherits from  LearningRateSchedule

# A useful method for hyper-paramter optimisation to determine learning Rate Combined with callbacks

# SGD= Stochastic gradient descent

learning_rate_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(

    initial_learning_rate=80.0,

    decay_steps=100,

    decay_rate=0.9)

# initial_learning_rate * decay_rate ^ (step / decay_steps)

# Tried with different optimizers like Adam,RMSprop, l-bfgs but it worked best

optimizer = keras.optimizers.SGD(learning_rate=learning_rate_scheduler,momentum=0.9)

# Backtrace from here

# Iterating and reducing the loss

#more iteration the better but 3000-4000 is optimal

t0=datetime.datetime.now()

iterations = 3000

for i in range(1, iterations + 1):

    loss, gradient = total_loss_and_gradient(generated_combined_image, content_image, style_art_image)

    optimizer.apply_gradients([(gradient, generated_combined_image)])

    if i%50==0:

      print("*"*25,"PAINTING","*"*25)

      print("time elapsed",datetime.datetime.now()-t0)

      print("loss=%.1f" %(loss))



#deprocess image

numpy_img_matrix=generated_combined_image.numpy()

numpy_img_matrix = numpy_img_matrix.reshape((rows, cols, 3))

# Remove zero-center by mean pixel

# Reversing effect of Keras pre-processing

# Just a boiler-plate(dont worry about the number)

numpy_img_matrix[:, :, 0] += 103.939

numpy_img_matrix[:, :, 1] += 116.779

numpy_img_matrix[:, :, 2] += 123.68

numpy_img_matrix = numpy_img_matrix[:, :, ::-1]

img = np.clip(numpy_img_matrix, 0, 255).astype("uint8")

print("Final--> loss=%.1f" %(loss))

print("Completed in-->",datetime.datetime.now()-t0)

keras.preprocessing.image.save_img("Combined_Image.jpg", img)

"""

# Download Final image

from google.colab import files

files.download("/content/"+"Combined_Image.jpg")"""