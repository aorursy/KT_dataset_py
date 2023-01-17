import tensorflow as tf

from tensorflow.keras.models import Model

import cv2

import numpy as np

import matplotlib.pyplot as plt

from skimage import io

import keras



%matplotlib inline
keras_model = tf.keras.applications.vgg19.VGG19(weights = 'imagenet', include_top = False)

preprocess_input = tf.keras.applications.vgg19.preprocess_input

keras_model.summary()
def load_img(url, target_size = None):

    image = io.imread(url)

    image = image / 255.0

    image = tf.convert_to_tensor(image, dtype=tf.float32)

    image = tf.expand_dims(image, axis = 0)

    return image



def show_image(image, name = ''):

    image = tf.reshape(image, shape = (image.shape[1], image.shape[2], image.shape[3]))

    image = image.numpy()

    plt.imshow(image)

    plt.xlabel(name)
content_file = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'

style_file = 'https://st2.depositphotos.com/3224051/6529/i/950/depositphotos_65299155-stock-photo-abstract-artificial-computer-generated-iterative.jpg'

content_image = load_img(content_file)

style_image = load_img(style_file)
images = { 'content_image' : content_image, 'style_image' : style_image }

_ = plt.figure(figsize=(25, 7))

for ix, name in enumerate(images.keys()):

    _ = plt.subplot(1, 2, ix + 1)

    show_image(images[name], name)

    _ = plt.xticks([])

    _ = plt.yticks([])
content_layers = {

    'block1_conv1' : 1

} 



style_layers = {

    'block1_conv1' : 1,

                'block2_conv1' : 1,

                'block3_conv1' : 1, 

                'block4_conv1' : 1, 

                'block5_conv1' : 1

}
def gram_matrix(a_s):

    _, n_h, n_w, n_c = a_s.shape

    a_s = tf.reshape(a_s, (n_h * n_w, n_c))

    a_s = tf.matmul(tf.transpose(a_s), a_s)

    a_s = a_s / (n_h * n_w)

    return a_s
def get_model(model_layers):

    inp = keras_model.input

    output = [keras_model.get_layer(name).output for name in model_layers]

    model = Model([keras_model.input], output)

    model.trainable = False  

    return model
extractor = get_model(list(content_layers.keys()) + list(style_layers.keys()))

extractor.summary()
def get_style_content_activation(image):

    preprocess_image = preprocess_input(image * 255)

    output = extractor(preprocess_image)

    content_output = output[:len(content_layers)]

    style_output = output[len(content_layers):]

    

    content_output = {layer : value for layer , value in zip(content_layers.keys(), content_output)}

    style_output = {layer : gram_matrix(value) for layer , value in zip(style_layers.keys(), style_output)}

    

    return {'content' : content_output, 'style' : style_output}
content_targets = get_style_content_activation(content_image)['content']

style_targets = get_style_content_activation(style_image)['style']
style_weight = 1e-2

content_weight = 1e4
def get_style_content_loss(image):

    output = get_style_content_activation(image)

    content_output = output['content']

    style_output = output['style']



    content_loss = tf.add_n([content_layers[name] * tf.reduce_mean((content_output[name] - content_targets[name])**2) 

                            for name in content_output.keys()])

    style_loss = tf.add_n([style_layers[name] * tf.reduce_mean((style_output[name] - style_targets[name])**2) 

                        for name in style_output.keys()])



    content_loss = content_loss * content_weight / len(content_layers)

    style_loss = style_loss * style_weight / len(style_layers)

    loss = content_loss + style_loss

    return loss
tf_generated_image = tf.Variable(content_image)

opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
def clip_0_1(image):

    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
def train_step(tf_generated_image):

    with tf.GradientTape() as tape:

        loss = get_style_content_loss(tf_generated_image)

        #print('{}. loss: {}'.format(ix,loss))

    grad = tape.gradient(loss, tf_generated_image)

    opt.apply_gradients([(grad, tf_generated_image)])

    tf_generated_image.assign(clip_0_1(tf_generated_image))
for ix in range(10):

    train_step(tf_generated_image)

    

images = { 'content_image' : content_image, 'style_image' : style_image, 'generated_image' : tf_generated_image }

_ = plt.figure(figsize=(25, 7))

for ix, name in enumerate(images.keys()):

    _ = plt.subplot(1, len(images), ix + 1)

    show_image(images[name], name)

    _ = plt.xticks([])

    _ = plt.yticks([])
epochs = 100

steps_per_epoch = 10

plt.figure(figsize = (25, 140))

for epoch in range(epochs):

    for _ in range(steps_per_epoch):

        train_step(tf_generated_image)

    plt.subplot(20, 5, epoch + 1)

    show_image(tf_generated_image, str(epoch))

    _ = plt.xticks([])

    _ = plt.yticks([])    
images = { 'content_image' : content_image, 'style_image' : style_image, 'generated_image' : tf_generated_image}

_ = plt.figure(figsize=(25, 7))

for ix, name in enumerate(images.keys()):

    _ = plt.subplot(1, len(images), ix + 1)

    show_image(images[name], name)

    _ = plt.xticks([])

    _ = plt.yticks([])