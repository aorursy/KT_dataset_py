import time
import IPython.display as display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [12, 9]
import cv2
import tensorflow as tf
import tensorflow.keras.backend as K
import sklearn
def read_img(path):
    img = cv2.imread(path)
    img = cv2.resize(img,image_size,interpolation=cv2.INTER_LANCZOS4)
    return img

def read_img_step2(img):
    img = img.astype(np.float32)
    img = img/255.
    img = np.expand_dims(img, axis=0)
    return img

def show_img(img_list):
    title = ['content image','style_image']
    pos = 1
    for i in img_list:
        i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        plt.subplot(1, len(img_list), pos)
        plt.title(title[pos-1])
        plt.imshow(i)
        pos += 1
'''Define gram matrix which is used to extract the style features from the style image'''

def gram_matrix(inputs):
    shape = K.shape(inputs)
    F = K.reshape(inputs, (shape[1] * shape[2], shape[0] * shape[3]))
    num_locations = tf.cast(shape[1]*shape[2], tf.float32)
    return K.dot(K.transpose(F), F)

def gram_matrix_tf(inputs):
    result = tf.linalg.einsum('bijc,bijd->bcd', inputs, inputs)
    input_shape = tf.shape(inputs)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/num_locations

def architecture(architecture,content=None,styles=None,weights=None):
    architecture = architecture
    if architecture == 'vgg':
        if styles is None:
            content_layers = ['block5_conv2']
            style_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']
            style_layers_weights = [1,1,1,1,1]
        else:
            content_layers,style_layers,style_layers_weights = content,styles,weights
    if architecture == 'resnet':
        if styles is None:
            content_layers = ['conv4_block2_1_conv']
            style_layers = ['conv1_conv','conv2_block1_1_conv','conv3_block1_1_conv','conv4_block1_1_conv','conv5_block1_1_conv']
            style_layers_weights = [1,1,1,1,1]
        else:
            content_layers,style_layers,style_layers_weights = content,styles,weights
    return architecture, content_layers, style_layers, style_layers_weights

'''
Here we download a pre-trained VGG19 model trained with Imagenet dataset, to make use of its trained weights. 
These can help us to extract the style features (i.e. the feature maps) from the style image. 
It would be a painful process to train a model by your own from scratch and use it here.
'''

def vgg_layers(layer_names):
    if architecture == 'vgg':
        vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet',pooling='avg',input_shape=image_size+(3,))
    if architecture == 'resnet':
        vgg = tf.keras.applications.ResNet152V2(include_top=False, weights='imagenet',pooling='avg',input_shape=image_size+(3,))
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

class StyleContentModel(tf.keras.models.Model):

    def __init__(self, style_layers, content_layers):
        super().__init__()
        self.vgg =  vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.num_content_layers = len(content_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        inputs = inputs[:,:,:,::-1]
        inputs = inputs*255
        inputs = tf.keras.applications.vgg16.preprocess_input(inputs)
             
        outputs = self.vgg(inputs)        
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])
        style_outputs = [gram_matrix_tf(style_output) for style_output in style_outputs]
        content_dict = {content_name:value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name:value for style_name, value in zip(self.style_layers, style_outputs)}
        return {'content':content_dict, 'style':style_dict}
def total_variation_loss(x, kind='isotropic'):
    h, w = x.shape[1], x.shape[2]
    if kind == 'anisotropic':
        # take the absolute value between this image, and the image one pixel
        # down, and one pixel to the right. take the absolute value as
        # specified by anisotropic loss
        a = K.abs(x[:, :h-1, :w-1, :] - x[:, 1:, :w-1, :])
        b = K.abs(x[:, :h-1, :w-1, :] - x[:, :h-1, 1:, :])
        # add up all the differences
        return K.sum(a + b)
    elif kind == 'isotropic':
        # take the absolute value between this image, and the image one pixel
        # down, and one pixel to the right. take the square root as specified
        # by isotropic loss
        a = K.square(x[:, :h-1, :w-1, :] - x[:, 1:, :w-1, :])
        b = K.square(x[:, :h-1, :w-1, :] - x[:, :h-1, 1:, :])
        # take the vector square root of all the pixel differences, then sum
        # them all up
        return K.sum(K.pow(a + b, 2))
    
def style_content_loss(outputs):
    style_loss,content_loss = 0.0,0.0
    style_outputs,content_outputs = outputs['style'],outputs['content']
    weights_sum = sum(style_layers_weights)
    for count,name in enumerate(style_outputs.keys()):
        style_loss += tf.reduce_mean(((style_outputs[name]-style_targets[name])**2)) * style_layers_weights[count]
    content_loss = 0.5*tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) for name in content_outputs.keys()])
    loss = style_loss*style_weight/weights_sum + content_loss*content_weight/num_content_layers
    return loss/(image_size[0]*image_size[1])
        
def style_content_loss_tf(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) for name in style_outputs.keys()])
    style_loss *= style_weight/num_style_layers
    content_loss = 0.5*tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) for name in content_outputs.keys()])
    content_loss *= content_weight/num_content_layers
    loss = style_loss + content_loss
    return loss/(image_size[0]*image_size[1])
    
def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

@tf.function
def train_step(img,smoothing_factor):
    with tf.GradientTape() as tape:
        loss1 = smoothing_factor*total_variation_loss(img,kind='isotropic')
        loss2 = style_content_loss(extractor(img))
        loss = loss1 + loss2
    grad = tape.gradient(loss, img)
    opt.apply_gradients([(grad, img)])
    img.assign(clip_0_1(img))
    return loss1, loss2

def train(img,epochs=10,steps_per_epoch=100,smoothing_factor=1e-3):
    step = 0
    start = time.time()
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            loss1,loss2 = train_step(img,smoothing_factor)
            print(".", end='')
        print("\nTrain step: {}".format(step),' Loss:',loss1,loss2)
    end = time.time()
    print("Total time: {:.1f}".format(end-start))
    return img
def proc_tensor_to_image(tensor):
    IMAGENET_MEANS = [103.939, 116.779, 123.68]
    tensor = tensor[0]
    tf.math.add(tensor[:,:,0],IMAGENET_MEANS[0])
    tf.math.add(tensor[:,:,1],IMAGENET_MEANS[1])
    tf.math.add(tensor[:,:,2],IMAGENET_MEANS[2])
    tensor = tensor[:,:,::-1]
    plt.imshow(tensor)
    return tensor

def tensor_to_image(tensor):
    tensor = np.array(tensor*255, dtype=np.uint8)
    if np.ndim(tensor)==4:
        tensor = tensor[0]
    tensor_show = cv2.cvtColor(tensor, cv2.COLOR_BGR2RGB)
    plt.imshow(tensor_show)
    return tensor
image_size = (960,720)
'''
Setting up models and layers that used for content/style reconstruction.
Use these layers for content and style reconstruction, which will be used to calculate the total loss. 
For content images, in higher layers of the network, detailed pixel information is lost while the high-level content of the image is preserved.
'''

architecture,content_layers,style_layers,style_layers_weights = architecture('vgg',
                content=['block4_conv2'],
                styles=['block1_conv1','block1_conv2','block2_conv1','block2_conv2','block3_conv1','block3_conv2','block4_conv1','block4_conv2','block5_conv1','block5_conv2'],
                weights=[2,1,2,1,2,1,3,2,3,2])

num_content_layers,num_style_layers = len(content_layers),len(style_layers)
extractor = StyleContentModel(style_layers, content_layers)

opt = tf.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.InverseTimeDecay(0.05, decay_steps=500, decay_rate=1))
style_weight=1000
content_weight=1
'''Read images, divide it by 255 and reshape it to n=4'''

content_img = read_img('../input/my-photos/2.jpg')
style_img = read_img('../input/paintings-for-artistic-style-transfer/Vincent Van Gogh - The Starry Night.jpg')
show_img([content_img,style_img])
start_content_img, start_style_img = read_img_step2(content_img), read_img_step2(style_img)
style_targets = extractor.call(start_style_img)['style']
content_targets = extractor.call(start_content_img)['content']
'''Initialisation of the gradient descent'''

random = np.random.rand(1,image_size[1],image_size[0],3)
black = np.zeros((1,image_size[1],image_size[0],3))
white = np.ones((1,image_size[1],image_size[0],3))
# start_content_img, start_style_img
image = tf.Variable(start_content_img,dtype=tf.float32)
image = train(image,10,100)
result1 = tensor_to_image(image)
cv2.imwrite('result1.jpg', result1)
content_img = read_img('../input/my-photos/2.jpg')
style_img = read_img('../input/paintings-for-artistic-style-transfer/Vincent Van Gogh - The Olive Trees.jpg')
show_img([content_img,style_img])
start_content_img, start_style_img = read_img_step2(content_img), read_img_step2(style_img)
style_targets = extractor.call(start_style_img)['style']
content_targets = extractor.call(start_content_img)['content']
image2 = tf.Variable(start_content_img,dtype=tf.float32)
@tf.function
def train_step(img,smoothing_factor):
    with tf.GradientTape() as tape:
        loss1 = smoothing_factor*total_variation_loss(img,kind='isotropic')
        loss2 = style_content_loss(extractor(img))
        loss = loss1 + loss2
    grad = tape.gradient(loss, img)
    opt.apply_gradients([(grad, img)])
    img.assign(clip_0_1(img))
    return loss1, loss2
image2 = train(image2,10,100)
result2 = tensor_to_image(image2)
cv2.imwrite('result2.jpg', result2)
content_img = read_img('../input/my-photos/2.jpg')
style_img = read_img('../input/paintings-for-artistic-style-transfer/Edvard Munch - The Scream.jpg')
show_img([content_img,style_img])
start_content_img, start_style_img = read_img_step2(content_img), read_img_step2(style_img)
style_targets = extractor.call(start_style_img)['style']
content_targets = extractor.call(start_content_img)['content']
image3 = tf.Variable(start_content_img,dtype=tf.float32)
@tf.function
def train_step(img,smoothing_factor):
    with tf.GradientTape() as tape:
        loss1 = smoothing_factor*total_variation_loss(img,kind='isotropic')
        loss2 = style_content_loss(extractor(img))
        loss = loss1 + loss2
    grad = tape.gradient(loss, img)
    opt.apply_gradients([(grad, img)])
    img.assign(clip_0_1(img))
    return loss1, loss2
image3 = train(image3,10,100)
result3 = tensor_to_image(image3)
cv2.imwrite('result3.jpg', result3)