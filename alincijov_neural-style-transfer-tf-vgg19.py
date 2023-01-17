from PIL import Image

Image.open(r"../input/best-artworks-of-all-time/images/images/Leonardo_da_Vinci/Leonardo_da_Vinci_112.jpg") 
import pandas as pd

import numpy as np

import cv2

import matplotlib.pyplot as plt

import IPython.display as display

import PIL.Image
import tensorflow as tf
content_pic_path = '../input/best-artworks-of-all-time/images/images/Vincent_van_Gogh/Vincent_van_Gogh_1.jpg'

style_pic_path = '../input/best-artworks-of-all-time/images/images/Jackson_Pollock/Jackson_Pollock_14.jpg'
dim = (224, 224)

content_image = cv2.resize(cv2.imread(content_pic_path), dim, interpolation=cv2.INTER_AREA)

style_image = cv2.resize(cv2.imread(style_pic_path), dim, interpolation=cv2.INTER_AREA)
f, axarr = plt.subplots(1,2, figsize=(20,20),gridspec_kw = {'wspace':0, 'hspace':0})

plt.axis("off")



axarr[0].set_title('Content Image')

axarr[0].set_axis_off()

axarr[0].imshow(cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB))



axarr[1].set_title('Style Image')

axarr[1].set_axis_off()

axarr[1].imshow(cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB))
content_image = tf.convert_to_tensor(content_image, dtype=tf.float32) / 255.0

style_image = tf.convert_to_tensor(style_image, dtype=tf.float32) / 255.0



content_image = tf.expand_dims(content_image, axis=0)

style_image = tf.expand_dims(style_image, axis=0)
content_layers = ['block5_conv2'] 



style_layers = ['block1_conv1',

                'block2_conv1',

                'block3_conv1', 

                'block4_conv1', 

                'block5_conv1']



num_content_layers = len(content_layers)

num_style_layers = len(style_layers)

def vgg_layers(layer_names):

    """ Creates a vgg model that returns a list of intermediate output values."""

    # Load our model. Load pretrained VGG, trained on imagenet data

    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

    vgg.trainable = False



    outputs = [vgg.get_layer(name).output for name in layer_names]



    model = tf.keras.Model([vgg.input], outputs)

    return model
style_extractor = vgg_layers(style_layers)

style_outputs = style_extractor(content_image*255)
#Look at the statistics of each layer's output

for name, output in zip(style_layers, style_outputs):

    print(name)

    print("  shape: ", output.numpy().shape)

    print("  min: ", output.numpy().min())

    print("  max: ", output.numpy().max())

    print("  mean: ", output.numpy().mean())

    print()
def gram_matrix(input_tensor):

    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)

    input_shape = tf.shape(input_tensor)

    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)

    return result/(num_locations)
class StyleContentModel(tf.keras.models.Model):

    def __init__(self, style_layers, content_layers):

        super(StyleContentModel, self).__init__()

        self.vgg =  vgg_layers(style_layers + content_layers)

        self.style_layers = style_layers

        self.content_layers = content_layers

        self.num_style_layers = len(style_layers)

        self.vgg.trainable = False



    def call(self, inputs):

        "Expects float input in [0,1]"

        inputs = inputs*255.0

        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)

        outputs = self.vgg(preprocessed_input)

        style_outputs, content_outputs = (outputs[:self.num_style_layers], 

                                          outputs[self.num_style_layers:])



        style_outputs = [gram_matrix(style_output)

                         for style_output in style_outputs]



        content_dict = {content_name:value 

                        for content_name, value 

                        in zip(self.content_layers, content_outputs)}



        style_dict = {style_name:value

                      for style_name, value

                      in zip(self.style_layers, style_outputs)}



        return {'content':content_dict, 'style':style_dict}

extractor = StyleContentModel(style_layers, content_layers)



results = extractor(content_image)



print('Styles:')

for name, output in sorted(results['style'].items()):

    print("  ", name)

    print("    shape: ", output.numpy().shape)

    print("    min: ", output.numpy().min())

    print("    max: ", output.numpy().max())

    print("    mean: ", output.numpy().mean())

    print()



print("Contents:")

for name, output in sorted(results['content'].items()):

    print("  ", name)

    print("    shape: ", output.numpy().shape)

    print("    min: ", output.numpy().min())

    print("    max: ", output.numpy().max())

    print("    mean: ", output.numpy().mean())
style_targets = extractor(style_image)['style']

content_targets = extractor(content_image)['content']
image = tf.Variable(content_image)
def clip_0_1(image):

    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
style_weight=1e-2

content_weight=1e4
def style_content_loss(outputs):

    style_outputs = outputs['style']

    content_outputs = outputs['content']

    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 

                           for name in style_outputs.keys()])

    style_loss *= style_weight / num_style_layers



    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 

                             for name in content_outputs.keys()])

    content_loss *= content_weight / num_content_layers

    loss = style_loss + content_loss

    return loss
@tf.function()

def train_step(image):

    with tf.GradientTape() as tape:

        outputs = extractor(image)

        loss = style_content_loss(outputs)



    grad = tape.gradient(loss, image)

    opt.apply_gradients([(grad, image)])

    image.assign(clip_0_1(image))

def tensor_to_image(tensor):

    tensor = tensor*255

    tensor = np.array(tensor, dtype=np.uint8)

    if np.ndim(tensor)>3:

        assert tensor.shape[0] == 1

        tensor = tensor[0]

    return PIL.Image.fromarray(tensor).resize((356, 356))
import time

start = time.time()



epochs = 10

steps_per_epoch = 100



step = 0

for n in range(epochs):

    for m in range(steps_per_epoch):

        step += 1

        train_step(image)

        print(".", end='')

    display.clear_output(wait=True)

    display.display(tensor_to_image(image))

    print("Train step: {}".format(step))



end = time.time()

print("Total time: {:.1f}".format(end-start))