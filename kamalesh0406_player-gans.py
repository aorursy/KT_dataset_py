# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
pictures = os.listdir("../input/Pictures")


def normalize(array):
    #Convert this to a (0,1) range
    array = (array - array.min())/(255 - array.min())
    #Now convert it into a range of (-1,1)
    array = 2*array - 1
    
    return array
dataset = []


for picture in pictures[:320]:
    im = Image.open('../input/Pictures/'+picture)
    pix_val = np.array(im.getdata())
    pix_val = pix_val.reshape(64,64,-1)[:,:,:3]
    normalized = normalize(pix_val)
    dataset.append(normalized)
dataset = np.array(dataset)
print(dataset.shape)

def batch_generator(dataset, index, batch_size):
    data = dataset[(index*batch_size):((index+1)*batch_size)]
    return data
def model_inputs(real_dim, z_dim):
    input_real = tf.placeholder(tf.float32, shape = (None, *real_dim), name = "real_input")
    input_z = tf.placeholder(tf.float32, shape = (None, z_dim), name = "z_input")
    
    return (input_real , input_z)
def generator(input_z, alpha = 0.2, reuse = False, training = True):
    
    with tf.variable_scope("generator", reuse=reuse):
        
        input_x = tf.layers.dense(input_z, 16*16*1024)
        input_x = tf.reshape(input_x, (-1, 16, 16, 1024))
        input_x = tf.maximum(input_x, alpha*input_x)
        
        first_layer = tf.layers.conv2d_transpose(input_x, 512, 5, strides = 2, padding = 'same')
        first_layer_normal = tf.layers.batch_normalization(first_layer, training = training)
        first_layer_output = tf.maximum(first_layer_normal, alpha*first_layer_normal)
        
        second_layer = tf.layers.conv2d_transpose(first_layer_output, 256, 5, strides = 2, padding = 'same')
        second_layer_normal = tf.layers.batch_normalization(second_layer, training = training )
        second_layer_output = tf.maximum(second_layer_normal, alpha*second_layer_normal)
        
        logits = tf.layers.conv2d_transpose(second_layer_output, 3, 5, strides = 2, padding = 'same')
        out = tf.tanh(logits)
        return(out)  
        
def discriminator(input_x, alpha = 0.2, reuse = True):
    
    with tf.variable_scope("discriminator", reuse=reuse):
        
        first_layer = tf.layers.conv2d(input_x, 128, 3, strides = 2, padding = 'same')
        first_layer_output = tf.maximum(first_layer, alpha*first_layer)
        
        second_layer = tf.layers.conv2d(first_layer_output, 256, 3, strides = 2, padding = 'same')
        second_layer_normal = tf.layers.batch_normalization(second_layer, training = True)
        second_layer_output = tf.maximum(second_layer_normal, alpha * second_layer_normal)
        
        final_layer = tf.layers.conv2d(second_layer_output, 512, 3, strides=2, padding='same')
        final_layer_normal = tf.layers.batch_normalization(final_layer, training = True)
        final_layer_output = tf.maximum(final_layer_normal, alpha*final_layer_normal)
        
        final_layer_output = tf.reshape(final_layer_output, (-1,8*8*512))
        logits = tf.layers.dense(final_layer_output, 1)
        out = tf.sigmoid(logits)
        
        return (logits, out)
def model_loss(input_real, fake_input, alpha):
    
    g_out = generator(fake_input, alpha)
    d_real_logits, d_real_out = discriminator(input_real, alpha, reuse = False)
    d_fake_logits, d_fake_out = discriminator(g_out, alpha)
    
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_real_logits, labels = tf.ones_like(d_real_out)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_fake_logits, labels = tf.zeros_like(d_fake_out)))
    
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_fake_logits, labels = tf.ones_like(d_fake_out)))
    d_loss = d_loss_real + d_loss_fake
    
    return (g_loss, d_loss)
def model_optimizer(g_loss, d_loss, learning_rate, beta1):
    
    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if var.name.startswith('generator')]
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        g_opt = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = beta1).minimize(g_loss, var_list = g_vars)
        d_opt = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = beta1).minimize(d_loss, var_list = d_vars)
    
    return g_opt, d_opt

class Generator:
    def __init__(self, real_dim, z_dim, learning_rate, alpha, beta1):
        tf.reset_default_graph()
        self.input_real , self.input_z = model_inputs(real_dim, z_dim)
        self.g_loss, self.d_loss = model_loss(self.input_real, self.input_z, alpha)
        self.g_opt , self.d_opt = model_optimizer(self.g_loss, self.d_loss, learning_rate, beta1)
        
def view_samples(epoch, samples, nrows, ncols, figsize=(8,8)):
    fig, axes = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols, 
                             sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples):
        ax.axis('off')
        img = ((img - img.min())*255 / (img.max() - img.min())).astype(np.uint8)
        ax.set_adjustable('box-forced')
        im = ax.imshow(img, aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)
    return fig, axes
def train(net, data, batch_size, epochs):
    sample_input = np.random.uniform(-1, 1,size=(36, 86))
    loss = []
    samples = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for n_epoch in range(1, epochs+1):
            for ii in range(int(len(data)/batch_size)):
                dataset = batch_generator(data, ii, batch_size)
                random_input = np.random.uniform(-1, 1,size=(batch_size, 86))
                
                _ = sess.run(net.d_opt, feed_dict = {net.input_real:dataset, net.input_z:random_input})
                _ = sess.run(net.g_opt, feed_dict = {net.input_real:dataset, net.input_z:random_input})
                

                if n_epoch%1 == 0:
                    train_loss_g = net.g_loss.eval({net.input_z:random_input})
                    train_loss_d = net.d_loss.eval({net.input_real:dataset, net.input_z: random_input})

                    loss.append(train_loss_g)
                    print("Epoch {}/{} Discriminator Loss {:.4f} Generator Loss {:.4f}".format(
                            n_epoch, epochs, train_loss_d, train_loss_g))
                if n_epoch%10 ==0 :
                    gener_samples = sess.run(generator(net.input_z, reuse = True, training = False),feed_dict= {net.input_z:sample_input})
                    fig, axes = view_samples(-1, gener_samples, 6, 6)
                    plt.show()
                    samples.append(gener_samples)
    return samples    
real_dim = (64, 64, 3)
z_dim = 86
batch_size = 64
learning_rate = 0.0002
epochs = 100
alpha = 0.2
beta1 =0.5