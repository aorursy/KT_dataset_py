# Importing the libraries:



import numpy as np

from PIL import Image



# Keras libraries

from keras.applications.vgg16 import VGG16

from keras.applications.vgg16 import preprocess_input,decode_predictions

from keras import backend

from keras.models import Model

from scipy.optimize import fmin_l_bfgs_b

from scipy.misc import imsave
# Loading the content image:



content_image = Image.open('../input/images2/content.jpg')

content_image = content_image.resize((512, 512))

content_image
# Loading the styling image: 



style_image = Image.open('../input/images2/style.jpg')

style_image = style_image.resize((512, 512))

style_image
# Convertng the content image into an array:



content_array = np.asarray(content_image, dtype = 'float32')

content_array = np.expand_dims(content_array, axis = 0)

print("The shape of the content image array is:", content_array.shape)
# Converting the styling array into an array:



style_array = np.asarray(style_image, dtype = 'float32')

style_array = np.expand_dims(style_array, axis=0)

print("The shape of the styling image array is:", style_array.shape)
# Preprocessing the content array:



content_array[:, :, :, 0] -= 103.939

content_array[:, :, :, 1] -= 116.779

content_array[:, :, :, 2] -= 123.68

content_array = content_array[:, :, :, ::-1]

content_array.shape
# Preprocessing the styling array:



style_array[:, :, :, 0] -= 103.939

style_array[:, :, :, 1] -= 116.779

style_array[:, :, :, 2] -= 123.68

style_array = style_array[:, :, :, ::-1]

style_array.shape
# Creating a placeholder for the three image variable:



# Defining the height and width of the images:

height = 512

width = 512



# Creating a placeholder for the content image:

content_image = backend.variable(content_array)



# Creating a placeholder for the styling image:

style_image = backend.variable(style_array)



# Creating a placeholder for the combined image:

combination_image = backend.placeholder((1,height,width,3))



# Concatenating the content and style image to feed it to the VGG network:



input_tensor=backend.concatenate([content_image,style_image,combination_image],axis=0)
# Loading the model:



model=VGG16(input_tensor = input_tensor, weights = 'imagenet', include_top = False)
# Defining alpha and beta for the total loss:



alpha = 0.05

beta = 5.0

total_variation_weight = 1.0
# Creating the dictionary containing all the layers in he model:



layers = dict([(layer.name, layer.output) for layer in model.layers])
# Defining the loss variable:



loss = backend.variable(0.)
# Defining the content loss:



def content_loss(content, combination):

    return backend.sum(backend.square(content-combination))
# Extracting the layer features for the content image:



layer_features = layers['block2_conv2']

content_image_features = layer_features[0,:,:,:]

combination_features = layer_features[2,:,:,:]



# Calculating the content loss:

loss += alpha * content_loss(content_image_features,combination_features)
# Defining a function to form the gram matrix:



def gram_matrix(x):

    features  =backend.batch_flatten(backend.permute_dimensions(x,(2,0,1)))

    gram = backend.dot(features, backend.transpose(features))

    return gram
# Defining the style loss:



def style_loss(style,combination):

    S = gram_matrix(style)

    C = gram_matrix(combination)

    channels = 3

    size = height * width

    st = backend.sum(backend.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

    return st
# Creating a dictionary to store the layers which will be used to form the styling image:



feature_layers = ['block1_conv2', 'block2_conv2',

                  'block3_conv3', 'block4_conv3',

                  'block5_conv3']
# Extracting the layer features and the styling loss:



for layer_name in feature_layers:

    layer_features = layers[layer_name]

    style_features = layer_features[1,:,:,:]

    combination_features = layer_features[2,:,:,:]

    sl = style_loss(style_features,combination_features)

    loss += (beta/len(feature_layers))*sl
# Defining the total loss:



def total_variation_loss(x):

    a=backend.square(x[:,:height-1,:width-1,:]-x[:,1:,:width-1,:])

    b = backend.square(x[:, :height-1, :width-1, :] - x[:, :height-1, 1:, :])

    return backend.sum(backend.pow(a + b, 1.25))

loss += total_variation_weight * total_variation_loss(combination_image)
# Calculating the gradients:



grads = backend.gradients(loss, combination_image)
# Forming the output array and applying the function:



outputs=[loss]

if isinstance(grads, (list, tuple)):

    outputs += grads

else:

    outputs.append(grads)

f_outputs = backend.function([combination_image], outputs)
# Defining a function to get the loss and gradient values:



def eval_loss_and_grads(x):

    x = x.reshape((1, height, width, 3))

    outs = f_outputs([x])

    loss_value = outs[0]

    grad_values = outs[1].flatten().astype('float64')

    return loss_value, grad_values
# Defining the Evaluator class:



class Evaluator(object):

    def __init__(self):

        self.loss_value = None

        self.grads_values = None

    

    def loss(self, x):

        assert self.loss_value is None

        loss_value, grad_values = eval_loss_and_grads(x)

        self.loss_value = loss_value

        self.grad_values = grad_values

        return self.loss_value



    def grads(self, x):

        assert self.loss_value is not None

        grad_values = np.copy(self.grad_values)

        self.loss_value = None

        self.grad_values = None

        return grad_values
# Creating an instance of the class Evaluator:



evaluator = Evaluator()
# Defining the array x and number of iterations:



x = np.random.uniform(0,255,(1,height,width,3))-128.0



iterations = 10
# Generating the image:



import time

for i in range(iterations):

    print('Start of iteration', i + 1)

    start_time = time.time()

    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),

                           fprime = evaluator.grads, maxfun = 20)

    print(min_val)

    end_time = time.time()

    print('Iteration %d completed in %ds' % (i + 1, end_time - start_time))
# Reshaping the generated image in a representable form:



x = x.reshape((height, width, 3))

x = x[:, :, ::-1]

x[:, :, 0] += 103.939

x[:, :, 1] += 116.779

x[:, :, 2] += 123.68

x = np.clip(x, 0, 255).astype('uint8')
# Printing the generated Image:



Image.fromarray(x)