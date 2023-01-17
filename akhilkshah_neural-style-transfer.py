import os
import numpy as np
import tensorflow as tf

from skimage import io
import cv2
import matplotlib.pyplot as plt 
vgg19 =  tf.keras.applications.vgg19.VGG19(input_shape=(224,224,3),include_top=False,weights='imagenet')
vgg19.trainable = False
MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
content = io.imread('https://dynaimage.cdn.cnn.com/cnn/q_auto,w_412,c_fill,g_auto,h_412,ar_1:1/http%3A%2F%2Fcdn.cnn.com%2Fcnnnext%2Fdam%2Fassets%2F190906133333-isleworth-mona-lisa-crop.jpg')
style = io.imread('https://images.fineartamerica.com/images/artworkimages/mediumlarge/1/energized--abstract-art-by-fidostudio-tom-fedro--fidostudio.jpg')

content = cv2.resize(content,(224,224))
content = np.reshape(content,(1,224,224,3))

style = cv2.resize(style,(224,224))
style = np.reshape(style,(1,224,224,3))

content = content.astype('float32') - MEANS
style = style.astype('float32') - MEANS
layer_name = 'block4_conv2'
style_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']
def content_loss(a_c,a_g):
    n_H,n_W,n_C = a_c.shape[1],a_c.shape[2],a_c.shape[3]
    return tf.reduce_mean(tf.square(tf.subtract(a_c,a_g)))

def style_loss(a_s,a_g):
    
    n_H,n_W,n_C = a_s.shape[1],a_s.shape[2],a_s.shape[3]
    
    a_s = tf.transpose(tf.reshape(a_s, [1, a_s.shape[1]*a_s.shape[2], a_s.shape[3]]), perm=[0,2,1])
    gram_s = tf.matmul(a_s,a_s,transpose_a=False,transpose_b=True)
    
    a_g = tf.transpose(tf.reshape(a_g, [1, a_g.shape[1]*a_g.shape[2], a_g.shape[3]]), perm=[0,2,1])
    gram_g = tf.matmul(a_g,a_g,transpose_a=False,transpose_b=True)
    
    return tf.reduce_mean(tf.square(tf.subtract(gram_s,gram_g)))/(4*(n_C**2)*((n_H*n_W)))

def total_loss(content_loss,style_loss):
    return 200*content_loss + style_loss
#generator = tf.Variable(tf.random.uniform([1,224,224,3]))
generator = tf.Variable(np.random.uniform(-20,20, (1,224, 224, 3)).astype('float32'))
#generator = tf.Variable(content)

@tf.function
def compute_loss():
    
    # Content Image
    c_layer_model = tf.keras.Model(inputs=vgg19.input,
                                     outputs=vgg19.get_layer(layer_name).output)

    a_g = c_layer_model(generator)
    a_c = c_layer_model(content)
    
    c_loss = content_loss(a_c,a_g)

    # Style Image
    s_loss = 0
    i = 0
    for layer in style_layers:

        s_layer_model = tf.keras.Model(inputs=vgg19.input,
                                     outputs=vgg19.get_layer(layer).output)

        a_g = s_layer_model(generator)
        a_s = s_layer_model(style)

        s_loss += 0.2*style_loss(a_s,a_g)
    
    # Total Loss
    t_loss = total_loss(c_loss,s_loss)
    
    return t_loss


opt = tf.keras.optimizers.Adam(2)

for epoch in range(200):
    
    with tf.GradientTape() as t:
        
        loss = compute_loss()
        grad = t.gradient(loss,generator)
        opt.apply_gradients([(grad,generator)])
        
        if epoch%50 == 0:
            print("Loss at epoch ", epoch, " - ", loss.numpy())
plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')

plt.subplot(1,3,1)
img = tf.squeeze(content + MEANS)
img = tf.cast(img, tf.uint8)
plt.imshow(img)

plt.subplot(1,3,2)
img = tf.squeeze(style + MEANS)
img = tf.cast(img, tf.uint8)
plt.imshow(img)

plt.subplot(1,3,3)
img = tf.squeeze(generator + MEANS)
img = tf.cast(img, tf.uint8)
plt.imshow(img)

plt.show()
