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
import os

import sys

import scipy.io

import imageio

import scipy.misc

from scipy import io

import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow

from PIL import Image

import numpy as np

import tensorflow as tf

import pprint                  #to format printing of the vgg model

%matplotlib inline
class CONFIG:                  #Class is like an object constructor

    IMAGE_WIDTH = 1280

    IMAGE_HEIGHT = 720

    COLOR_CHANNELS = 3

    NOISE_RATIO = 0.5

    MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)) 
def load_vgg_model(path):

    vgg = io.loadmat(path)

    vgg_layers = vgg['layers']

    def _weights(layer, expected_layer_name):

        wb = vgg_layers[0][layer][0][0][2]

        W = wb[0][0]

        b = wb[0][1]

        layer_name = vgg_layers[0][layer][0][0][0][0]

        assert layer_name == expected_layer_name

        return W, b

        return W, b

    def _relu(conv2d_layer):

        return tf.nn.relu(conv2d_layer)

    def _conv2d(prev_layer, layer, layer_name):    

        W, b = _weights(layer, layer_name)

        W = tf.constant(W)

        b = tf.constant(np.reshape(b, (b.size)))

        return tf.nn.conv2d(prev_layer, filters=W, strides=[1, 1, 1, 1], padding='SAME') + b

    def _conv2d_relu(prev_layer, layer, layer_name):

        return _relu(_conv2d(prev_layer, layer, layer_name))

    

    def _avgpool(prev_layer):

        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    

    # Constructs the graph model.

    graph = {}

    graph['input']   = tf.Variable(np.zeros((1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.COLOR_CHANNELS)), dtype = 'float32')

    graph['conv1_1']  = _conv2d_relu(graph['input'], 0, 'conv1_1')

    graph['conv1_2']  = _conv2d_relu(graph['conv1_1'], 2, 'conv1_2')

    graph['avgpool1'] = _avgpool(graph['conv1_2'])

    graph['conv2_1']  = _conv2d_relu(graph['avgpool1'], 5, 'conv2_1')

    graph['conv2_2']  = _conv2d_relu(graph['conv2_1'], 7, 'conv2_2')

    graph['avgpool2'] = _avgpool(graph['conv2_2'])

    graph['conv3_1']  = _conv2d_relu(graph['avgpool2'], 10, 'conv3_1')

    graph['conv3_2']  = _conv2d_relu(graph['conv3_1'], 12, 'conv3_2')

    graph['conv3_3']  = _conv2d_relu(graph['conv3_2'], 14, 'conv3_3')

    graph['conv3_4']  = _conv2d_relu(graph['conv3_3'], 16, 'conv3_4')

    graph['avgpool3'] = _avgpool(graph['conv3_4'])

    graph['conv4_1']  = _conv2d_relu(graph['avgpool3'], 19, 'conv4_1')

    graph['conv4_2']  = _conv2d_relu(graph['conv4_1'], 21, 'conv4_2')

    graph['conv4_3']  = _conv2d_relu(graph['conv4_2'], 23, 'conv4_3')

    graph['conv4_4']  = _conv2d_relu(graph['conv4_3'], 25, 'conv4_4')

    graph['avgpool4'] = _avgpool(graph['conv4_4'])

    graph['conv5_1']  = _conv2d_relu(graph['avgpool4'], 28, 'conv5_1')

    graph['conv5_2']  = _conv2d_relu(graph['conv5_1'], 30, 'conv5_2')

    graph['conv5_3']  = _conv2d_relu(graph['conv5_2'], 32, 'conv5_3')

    graph['conv5_4']  = _conv2d_relu(graph['conv5_3'], 34, 'conv5_4')

    graph['avgpool5'] = _avgpool(graph['conv5_4'])

    

    return graph
datapath = "../input/imagenetvggverydeep19mat/imagenet-vgg-verydeep-19.mat"

pp = pprint.PrettyPrinter(indent=4)

model = load_vgg_model(datapath)

pp.pprint(model)
import imageio

content_image = imageio.imread("https://www.economist.com/img/b/1280/720/90/sites/default/files/20200905_BKP503.jpg")

plt.figure(figsize=(10,10))

plt.imshow(content_image);

content_image.shape
# Compute Content Cost



def compute_content_cost(a_C,a_G):

    #Retrieve dimensions from a_G

    m,n_H,n_W,n_C = a_G.get_shape().as_list()

    

    #Unrolled a_C and a_G

    a_C_unrolled = tf.reshape(a_C,shape=[m,-1,n_C])

    a_G_unrolled = tf.reshape(a_G,shape=[m,-1,n_C])

    

    #Compute the Content Cost

    j_content = (1/(4*n_H*n_W*n_C))*(tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled,a_G_unrolled))))



    return j_content
style_image = imageio.imread("https://www.thevintagenews.com/wp-content/uploads/2020/04/van_gogh_painting-1280x720.jpg")

plt.figure(figsize=(10,10))

plt.imshow(style_image)

style_image.shape

# Style Matrix is also known as Gram Matrix, Gij which is the dot product of np.dot(vi,vj). Gij compares how similar vi to vj if it is similar then dot product to be very high.

# In Neural Style Transfer (NST), you can compute the Style matrix by multiplying the "unrolled" filter matrix with its transpose.

# The result is a matrix of dimension (n_C,n_C) where n_C is the number of filters (channels).

# The diagonal elements  G(gram)ii  measure how "active" a filter  i  is.

# For example, suppose filter i is detecting vertical textures in the image. Then  G(gram)i  measures how common vertical textures are in the image as a whole.

# If  G(gram)ii is large, this means that the image has a lot of vertical texture.
### Computing Cost for a single layer

def compute_style_cost_layer(a_S,a_G):

    #Retrieve dimensions from a_G

    m,n_H,n_W,n_C = a_G.get_shape().as_list()

    

    #unroll a_S and a_G

    a_S = tf.transpose(tf.reshape(a_S,shape=[n_H*n_W,n_C]))

    a_G = tf.transpose(tf.reshape(a_G,shape=[n_H*n_W,n_C]))

    

    #Computing gram matrics

    S = tf.matmul(a_S,tf.transpose(a_S))

    G = tf.matmul(a_G,tf.transpose(a_G))

    

    #Compute the Style Cost

    j_style_layer = (1/(4*(n_C**2)*((n_H*n_W)**2)))*(tf.reduce_sum(tf.reduce_sum(tf.square(tf.subtract(S,G)))))

    

    return j_style_layer
# We have captured the style from only one layer.

# We'll get better results if we "merge" style costs from several different layers.

# Each layer will be given weights (λ[l]) that reflect how much each layer will contribute to the style.
style_layers = [('conv1_2',0.3),

               ('conv3_2',0.3),

               ('conv3_3',0.3),

               ('conv4_2',0.3),

               ('conv5_1',0.3)]
#Compute Style Cost

def compute_style_cost(model, style_layers):

    

    #initialize overall cost

    j_style=0

    

    for layer_name,coeff in style_layers:

        #select the output tensor of the currently selected layer

        out = model[layer_name]

        

        #set a_S to be the activaltion of the currently selected layer by running the session on out

        a_S = sess.run(out)

        

        #set a_G to be the activaltion from the same layer

        a_G = model[layer_name]

        

        #compute style cost for the current layer

        j_style_layer = compute_style_cost_layer(a_S,a_G)

        

        #Add coeff to j_style layer to compute the cost from overall layers

        j_style += coeff*j_style_layer

        

    return j_style

    
#compute total cost

#alpha and beta are the hypermeters that control the weights between content and style. 

def total_cost(j_content, j_style, alpha=10, beta=40):

    

    j = (alpha*j_content) + (beta*j_style)

    

    return j 
tf.compat.v1.disable_eager_execution()

tf.compat.v1.reset_default_graph()

sess = tf.compat.v1.InteractiveSession()
#  Reshape and normalize the input image (content or style)

def reshape_and_normalize_image(image):

    image = np.reshape(image, ((1,) + image.shape))

    image = image - CONFIG.MEANS

    return image
content_image = imageio.imread("https://www.economist.com/img/b/1280/720/90/sites/default/files/20200905_BKP503.jpg")

content_image = reshape_and_normalize_image(content_image)

style_image = imageio.imread("https://www.thevintagenews.com/wp-content/uploads/2020/04/van_gogh_painting-1280x720.jpg")

style_image = reshape_and_normalize_image(style_image)
style_image.shape
# Now we initialize generated image as a noisy image by adding random noise to the content image



def generate_noisy_image(content_image, noise= CONFIG.NOISE_RATIO):

    

    noisy_image  = np.random.uniform(-20,20,(1,CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.COLOR_CHANNELS)).astype('float32')

    input_image = (noisy_image * noise) + (content_image * (1-noise))

    

    return input_image
generated_image = generate_noisy_image(content_image)

plt.imshow(generated_image[0])
model = load_vgg_model(datapath)
##CONTENT COST

#Assign content image to the input of vgg model

sess.run(model['input'].assign(content_image))



# Select the output tensor of layer conv4_2

out = model['conv4_2']



# Set a_C to be the hidden layer activation from the layer we have selected

a_C = sess.run(out)



# Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2'] 

# and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that

# when we run the session, this will be the activations drawn from the appropriate layer, with G as input.

a_G = out



# Compute the content cost

j_content = compute_content_cost(a_C, a_G)
##STYLE COST

sess.run(model['input'].assign(style_image))

j_style = compute_style_cost(model, style_layers)
#TOTAL COST

j = total_cost(j_content,j_style,alpha=10,beta=40)
#OPTIMIZER

#Using Adam Optimizer to minimize the cost

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.3)

train_set = optimizer.minimize(j)
def save_image(path, image):  

    # Un-normalize the image so that it looks good

    image = image + CONFIG.MEANS

    

    # Clip and Save the image

    image = np.clip(image[0], 0, 255).astype('uint8')

    imageio.imsave(path, image)
def model_nn(sess, input_image, num_iterations = 150):

    sess.run(tf.compat.v1.global_variables_initializer())

    # Run the noisy input image

    generated_image = sess.run(model["input"].assign(input_image))

    for i in range(num_iterations):

        # Run the session on the train_step to minimize the total cost

        sess.run(train_set)

        generated_image = sess.run(model["input"])

        # Print every 20 iteration.

        if i%20 == 0:

            J, Jc, Js = sess.run([j, j_content, j_style])

            print("Iteration " + str(i) + " :")

            print("total cost = " + str(J))

            print("content cost = " + str(Jc))

            print("style cost = " + str(Js))

            save_image("" + str(i) + ".png", generated_image)

    

    # save last generated image

    save_image('generated_image.jpg', generated_image)

    

    return generated_image
model_nn(sess,generated_image)