## importing modules

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as mpl

import matplotlib.image as mpimg

import PIL.Image

from tqdm import tqdm

mpl.rcParams['figure.figsize'] = (12,12)

mpl.rcParams['axes.grid'] = False
def load_img(path_to_img):

    max_dim = 512

    # read file

    img = tf.io.read_file(path_to_img)

    # decode with three channels RGB

    img = tf.image.decode_image(img, channels=3)

    # convert the datatype of tensor to float32

    img = tf.image.convert_image_dtype(img, tf.float32)

    # get the size of the image as float32

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)

    # get the longer dimension

    long_dim = max(shape)

    # define a scaling factor

    scale = max_dim / long_dim

    # scale the image by that factor and save the new shape as int32

    # this keeps the aspect ratio

    new_shape = tf.cast(shape * scale, tf.int32)

    # resize the image to the new shape

    img = tf.image.resize(img, new_shape)

    # make it a batch

    img = img[tf.newaxis, :]

    return img
# let's load the images

content_image = load_img("../input/photos/photo_2020-07-29_21-52-26.jpg")

style_image = load_img("../input/dataset4/style.png")



plt.subplot(1, 2, 1)

plt.imshow(content_image[0])



plt.subplot(1, 2, 2)

plt.imshow(style_image[0])
vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")

# The top is the last layer in the network, this is used for classification so we don't need it
# the feature maps are extracted from the convolutional layers of the model

# let's find out how they're named

vgg.summary()
# you can change these if you want, playing around is never bad ;)

content_layers = ["block5_conv4"]

style_layers = [

    "block1_conv1",

    "block2_conv2",

    "block3_conv2",

    "block4_conv3",

    "block5_conv3",

]

no_content = len(style_layers)

no_style = len(content_layers)
outputs = [vgg.get_layer(name).output for name in style_layers + content_layers]

extractor = tf.keras.Model(inputs=vgg.inputs, outputs=outputs)

extractor.trainable = False
# let's take it for a spin

outputs = extractor(content_image)

len(outputs)
def gram_matrix(input_tensor):

    # the b dimension is just the batch, don't worry about it

    result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)

    input_shape = tf.shape(input_tensor)

    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)

    return result / num_locations

    
# the first layer in the outputs is block1_conv1

# this one has output shape of (None, None, None, 64)

# its gram matrix should have shape (None, 64, 64) with none being the batches (just 1)

g = gram_matrix(outputs[0])

g.shape
class StyleContentExtractor(tf.keras.models.Model):

    def __init__(self, style_layers, content_layers):

        super(StyleContentExtractor, self).__init__()

        outputs = [vgg.get_layer(name).output for name in style_layers + content_layers]

        self.model = tf.keras.Model(inputs=vgg.inputs, outputs=outputs)

        self.model.trainable = False

        

        self.style_layers = style_layers

        self.content_layers = content_layers

        self.no_style_layers = len(style_layers)

        

    def call(self, inputs):

        inputs = inputs*255.

        inputs = tf.keras.applications.vgg19.preprocess_input(inputs)

        outputs = self.model(inputs)

        

        content_outputs = outputs[self.no_style_layers:]

        style_outputs = [gram_matrix(out) for out in outputs[:self.no_style_layers]]

        

        content_dict = {name: val for name, val in zip(self.content_layers, content_outputs)}

        style_dict = {name: val for name, val in zip(self.style_layers, style_outputs)}

        

        return {"style": style_dict, "content": content_dict}

        

        

extractor = StyleContentExtractor(style_layers, content_layers)
content_target = extractor(content_image)["content"]

style_target = extractor(style_image)["style"]

alpha = 1e4

beta = 1e-2
def content_style_loss(outputs):

    style_outputs = outputs['style']

    content_outputs = outputs['content']

    

    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_target[name])**2) 

                           for name in style_outputs.keys()])

    

    style_loss *= beta / no_style

    

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_target[name])**2) 

                           for name in content_outputs.keys()])

    

    content_loss *= alpha / no_content

    

    return content_loss + style_loss

    
# let'd define the variable for which we want to optimize the loss ( O )

# this is the generated image

# initially, it has the exact contents of the content image

image = tf.Variable(content_image)
opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)



# we need to make sure the tensor values remain between 0 and 1

# let's clip its values

def clip_0_1(image):

    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)





@tf.function()

def step(img):

    # this context manager is used to calculate the gradient

    with tf.GradientTape() as tape:

        outputs = extractor(img)

        loss = content_style_loss(outputs)



    grad = tape.gradient(loss, img)

    opt.apply_gradients([(grad, img)])

    img.assign(clip_0_1(img))
# What we'll get is a tensor having our information.

# let's write a function that turns it into an image

def tensor_to_image(tensor):

    # tensor images are normalized so denormalize them

    tensor = tensor * 255

    tensor = np.array(tensor, dtype=np.uint8)

    

    if np.ndim(tensor)>3:

        assert tensor.shape[0] == 1

        tensor = tensor[0]

    return PIL.Image.fromarray(tensor)
# let's run a few steps and see what we've got

for _ in tqdm(range(40)):

    step(image)
out = tensor_to_image(image)

plt.figure(figsize=(6, 6))

plt.imshow(out)
diff = 1

def high_pass_x_y(image):

    x_var = image[:,:,diff:,:] - image[:,:,:-diff,:] # fast variations on the x axis

    y_var = image[:,diff:,:,:] - image[:,:-diff,:,:] # fast variations on the y axis



    return x_var, y_var

x_deltas, y_deltas = high_pass_x_y(content_image)



plt.figure(figsize=(14,10))

plt.subplot(2,2,1)

plt.title("Horizontal Deltas: Original")

plt.imshow(clip_0_1(2*y_deltas[0]+0.5))



plt.subplot(2,2,2)

plt.title("Vertical Deltas: Original")

plt.imshow(clip_0_1(2*x_deltas[0]+0.5))



x_deltas, y_deltas = high_pass_x_y(image)



plt.subplot(2,2,3)

plt.title("Horizontal Deltas: Styled")

plt.imshow(clip_0_1(2*y_deltas[0]+0.5))



plt.subplot(2,2,4)

plt.title("Vertical Deltas: Styled")

plt.imshow(clip_0_1(2*x_deltas[0]+0.5))
noise_weight = 1e3

def content_style_noise_loss(outputs):

    style_outputs = outputs['style']

    content_outputs = outputs['content']

    

    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_target[name])**2) 

                           for name in style_outputs.keys()])

    

    style_loss *= beta / no_style

    

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_target[name])**2) 

                           for name in content_outputs.keys()])

    

    content_loss *= alpha / no_content

    

    x_deltas, y_deltas = high_pass_x_y(image)

    noise_loss = tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))

    

    noise_loss *= noise_weight

    

    return content_loss + style_loss + noise_loss
# let's do it again

@tf.function()

def step(img):

    # this context manager is used to calculate the gradient

    with tf.GradientTape() as tape:

        outputs = extractor(img)

        loss = content_style_noise_loss(outputs) # added noise



    grad = tape.gradient(loss, img)

    opt.apply_gradients([(grad, img)])

    img.assign(clip_0_1(img))
image = tf.Variable(content_image)

for _ in tqdm(range(40)):

    step(image)
# ready? even if not, here we go

out_filtered = tensor_to_image(image)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)

plt.title("With noise")

plt.imshow(out)



plt.subplot(1, 2, 2)

plt.title("Without noise")

plt.imshow(out_filtered)
plt.figure(figsize=(16, 4))



plt.subplot(1, 4, 1)

plt.imshow(content_image[0])



plt.subplot(1, 4, 2)

plt.imshow(style_image[0])



plt.subplot(1, 4, 3)

plt.imshow(out)



plt.subplot(1, 4, 4)

plt.imshow(out_filtered)
