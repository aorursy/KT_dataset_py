%matplotlib inline
import os
import sys
import h5py
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math

from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.layers import InputLayer, Input, BatchNormalization, LeakyReLU
from tensorflow.python.keras.layers import Reshape, MaxPooling2D, Activation
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Conv2DTranspose
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam, SGD
from tensorflow.python.keras.models import load_model
#laod and process real images
with h5py.File(os.path.join('../input','real_gaze.h5'),'r') as t_file:
    assert 'image' in t_file, "Images are missing"
    print('Real Images found:',len(t_file['image']))
    for _, (ikey, ival) in zip(range(1), t_file['image'].items()):
        print('image',ikey,'shape:',ival.shape)
        img_height, img_width = ival.shape
        img_channels = 1
    real_image_stack = np.stack([np.expand_dims(a,-1) for a in t_file['image'].values()],0)

#shape images to (32,64)
real_image_stack = np.pad(real_image_stack[:,2:34,:,:], pad_width=((0, 0), (0, 0), (4, 5), (0,0)), mode='constant', constant_values=0)
#show example image
plt.imshow(real_image_stack[1223,:,:,0], cmap='viridis')
plt.show()
#normalization helper functions: we want to normalize the images before we use them as inputs. We want to use the inverse of this normalization
#when we read the outputs of our generator as an image
def norm(x):
    return (x-129)/73.8

def anti_norm(x):
    return (x*73.8)+129

#helper function for displaying test images
def display_images(images):
    for i in range(images.shape[0]):
        plt.subplot(1, num_test, i + 1)
        plt.imshow(images[i], cmap='viridis')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
tf.reset_default_graph()
#hyperparamters
height = 32
width = 64
img_size_flat = 32*64
img_shape = (height, width)
img_shape_full = (height, width, 1)
num_channels = 1

#generator input size
n_z = 64
num_test = 6
#test inputs for generator
z_test = np.random.normal(0,1.0,size=[num_test,n_z])

#denerator/discriminator learning rates
g_learning_rate = 0.00005
d_learning_rate = 0.02
#generator network
g_inputs = Input(shape=(n_z,))
g_net = g_inputs
g_net = Reshape((1,1,n_z))(g_net)
g_net = Conv2DTranspose(kernel_size=(2,4), strides=(2,4), filters=256, padding='same',
             kernel_initializer=RandomNormal(stddev=0.02), name='g_conv1')(g_net)
g_net = BatchNormalization(axis=3)(g_net)
g_net = Activation('relu')(g_net)
g_net = Conv2DTranspose(kernel_size=5, strides=2, filters=128, padding='same',
             kernel_initializer=RandomNormal(stddev=0.02), name='g_conv2')(g_net)
g_net = BatchNormalization(axis=3)(g_net)
g_net = Activation('relu')(g_net)
g_net = Conv2DTranspose(kernel_size=5, strides=2, filters=64, padding='same',
             kernel_initializer=RandomNormal(stddev=0.02), name='g_conv3')(g_net)
g_net = BatchNormalization(axis=3)(g_net)
g_net = Activation('relu')(g_net)
g_net = Conv2DTranspose(kernel_size=5, strides=2, filters=32, padding='same',
             kernel_initializer=RandomNormal(stddev=0.02), name='g_conv4')(g_net)
g_net = BatchNormalization(axis=3)(g_net)
g_net = Activation('relu')(g_net)
g_net = Conv2DTranspose(kernel_size=5, strides=2, filters=1, padding='same',
             activation='tanh', kernel_initializer=RandomNormal(stddev=0.02), name='g_conv5')(g_net)
g_outputs = g_net

g_model = Model(inputs=g_inputs, outputs=g_outputs,name='generator')
print('Generator:')
g_model.summary()

g_model.compile(optimizer=Adam(lr=g_learning_rate, beta_1=0.5),
               loss='binary_crossentropy')

z_batch = np.random.normal(-1.0,1.0,size=[1,n_z])
g_model.predict(z_batch)
#discriminator network
d_inputs = Input(shape=img_shape_full)
d_net = d_inputs
d_net = Conv2D(kernel_size=5, strides=(2,4), filters=16, padding='same',
               kernel_initializer=RandomNormal(stddev=0.02), name='d_conv1')(d_net)
d_net = BatchNormalization(axis=3)(d_net)
d_net = LeakyReLU(alpha=0.2)(d_net)
d_net = Conv2D(kernel_size=5, strides=2, filters=32, padding='same',
             kernel_initializer=RandomNormal(stddev=0.02), name='d_conv2')(d_net)
d_net = BatchNormalization(axis=3)(d_net)
d_net = LeakyReLU(alpha=0.2)(d_net)
d_net = Conv2D(kernel_size=5, strides=2, filters=64, padding='same',
            kernel_initializer=RandomNormal(stddev=0.02), name='d_conv3')(d_net)
d_net = BatchNormalization(axis=3)(d_net)
d_net = LeakyReLU(alpha=0.2)(d_net)
d_net = Conv2D(kernel_size=3, strides=2, filters=128, padding='same',
            kernel_initializer=RandomNormal(stddev=0.02), name='d_conv4')(d_net)
d_net = BatchNormalization(axis=3)(d_net)
d_net = LeakyReLU(alpha=0.2)(d_net)
d_net = Conv2D(kernel_size=3, strides=2, filters=256, padding='same',
            kernel_initializer=RandomNormal(stddev=0.02), name='d_conv5')(d_net)
d_net = BatchNormalization(axis=3)(d_net)
d_net = LeakyReLU(alpha=0.2)(d_net)
d_net = Conv2D(kernel_size=1, strides=1, filters=1, padding='same',
            activation='sigmoid', kernel_initializer=RandomNormal(stddev=0.02), name='d_conv6')(d_net)
d_net = Flatten()(d_net)
d_outputs = d_net

d_model = Model(inputs=d_inputs, outputs=d_outputs,name='discriminator')
print('Discriminator:')
d_model.summary()

d_model.compile(optimizer=Adam(lr=d_learning_rate, beta_1=0.5),
               loss='binary_crossentropy')

z_batch = np.random.normal(-1.0,1.0,size=[2,32,64,1])
d_model.predict(z_batch)
#gan network
d_model.trainable = False
z_inputs = Input(shape=(n_z,))
x_inputs = g_model(z_inputs)
gan_outputs = d_model(x_inputs)

gan_model = Model(inputs=z_inputs, outputs=gan_outputs,name='gan')
print('GAN:')
gan_model.summary()

gan_model.compile(optimizer=Adam(lr=g_learning_rate, beta_1=0.5),
               loss='binary_crossentropy')

z_batch = np.random.uniform(-1.0,1.0,size=[1,n_z])
gan_model.predict(z_batch)
#model save directories
save_dir = 'checkpoints/'
checkpoint_dir = save_dir+'check/'
final_model_dir = save_dir+'final/'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(final_model_dir):
    os.makedirs(final_model_dir)

d_final_model_path = final_model_dir+'d_final_model.keras'
d_save_epoch_path = d_final_model_path+'.epoch'
d_save_loss_path = d_final_model_path+'.loss'
d_checkpoint_path = checkpoint_dir+'d_model.keras'
d_checkpoint_epoch_path = d_checkpoint_path+'.epoch'
d_checkpoint_loss_path = d_checkpoint_path+'.loss'

g_final_model_path = final_model_dir+'g_final_model.keras'
g_save_epoch_path = g_final_model_path+'.epoch'
g_save_loss_path = g_final_model_path+'.loss'
g_checkpoint_path = checkpoint_dir+'g_model.keras'
g_checkpoint_epoch_path = g_checkpoint_path+'.epoch'
g_checkpoint_loss_path = g_checkpoint_path+'.loss'

#model save helper fucntion
def progress_saver(mode, epoch, g_best_loss, d_best_loss):
    if mode == 'checkpoint':
        g_model.save(g_checkpoint_path)
        d_model.save(d_checkpoint_path)
        with open(g_checkpoint_epoch_path, "wb") as f:
            f.write(b"%d" % (epoch + 1))
        with open(d_checkpoint_epoch_path, "wb") as f:
            f.write(b"%d" % (epoch + 1))
        
        with open(g_checkpoint_loss_path, "wb") as f:
            f.write(b"%d" % g_best_loss)
        with open(d_checkpoint_loss_path, "wb") as f:
            f.write(b"%d" % d_best_loss)
    if mode == 'final':
        g_model.save(g_final_model_path)
        d_model.save(d_final_model_path)
        with open(g_save_epoch_path, "wb") as f:
            f.write(b"%d" % (epoch + 1))
        with open(d_save_epoch_path, "wb") as f:
            f.write(b"%d" % (epoch + 1))
        with open(g_save_loss_path, "wb") as f:
            f.write(b"%d" % g_best_loss)
        with open(d_save_loss_path, "wb") as f:
            f.write(b"%d" % d_best_loss)

#model restore helper function
def progress_restore(mode):
    if mode == 'checkpoint':
        best_loss = None
        g_model = load_model(g_checkpoint_path)
        d_model = load_model(d_checkpoint_path)
        with open(g_checkpoint_epoch_path, "rb") as f:
            start_epoch = int(f.read())
        with open(d_checkpoint_epoch_path, "rb") as f:
            start_epoch = int(f.read())
        with open(g_checkpoint_loss_path, "rb") as f:
            g_best_loss = int(f.read())
        with open(d_checkpoint_loss_path, "rb") as f:
            d_best_loss = int(f.read())
        print('Chekpoint')
    if mode == 'final':
        g_model = load_model(g_final_model_path)
        d_model = load_model(d_final_model_path)
        with open(g_save_epoch_path, "rb") as f:
            start_epoch = int(f.read())
        with open(d_save_epoch_path, "rb") as f:
            start_epoch = int(f.read())
        with open(g_save_loss_path, "rb") as f:
            g_best_loss = int(f.read())
        with open(d_save_loss_path, "rb") as f:
            d_best_loss = int(f.read())
        print("Final")
    print("Training was interrupted. Continuing at epoch", start_epoch)
    return start_epoch, g_best_loss, d_best_loss

#object for saving test images after each training epoch
saved_images = [[] for _ in range(num_test)]
n_epochs = 6000
batch_size = 300
n_batches = int(len(real_image_stack) / batch_size)
g_best_loss = 1000000000
d_best_loss = 1000000000
epochs_without_progress = 0
max_epochs_without_progress = 100

n_epochs_save = 25
n_epochs_print = 5

#load model if exists
if os.path.isfile(g_checkpoint_epoch_path):
    start_epoch, g_best_loss, d_best_loss  = progress_restore('checkpoint')
else:
    try:
        start_epoch, g_best_loss, d_best_loss = progress_restore('final')
    except:
        start_epoch = 0
#traing epochs
for epoch in range(start_epoch,start_epoch+n_epochs+1):
    print(epoch)
    epoch_d_loss = 0.0
    epoch_g_loss = 0.0
    #train batches
    for batch in range(n_batches):
        #train discrimantor
        x_batch = real_image_stack[batch*batch_size:(batch+1)*batch_size,:]
        x_batch = norm(x_batch.astype('float32'))
        z_batch = np.random.normal(0,1.0,size=[batch_size,n_z])
        g_batch = g_model.predict(z_batch)
        x_in = np.concatenate([x_batch, g_batch])
        y_out = np.ones(batch_size*2)
        y_out[:batch_size]=0.9
        y_out[batch_size:]=0.1
        d_model.trainable = True
        x_batch_d_loss = d_model.train_on_batch(x_batch, y_out[:batch_size])
        g_batch_d_loss = d_model.train_on_batch(g_batch, y_out[batch_size:])
        batch_d_loss = (x_batch_d_loss+g_batch_d_loss)/2
        
        #train generator
        for _ in range(5):
            z_batch = np.random.normal(0,1.0,size=[batch_size,n_z])
            x_in = z_batch
            y_out = np.ones(batch_size)
            d_model.trainable = False
            batch_g_loss = gan_model.train_on_batch(x_in, y_out)
        
        epoch_d_loss += batch_d_loss
        epoch_g_loss += batch_g_loss
    
    #save test images
    x_pred = g_model.predict(z_test)
    for g in range(6):
        saved_images[g].append(anti_norm(x_pred.reshape(-1,height,width)[g]).astype('uint8'))
    
    #print progress
    if epoch%n_epochs_print == 0:
        average_d_loss = epoch_d_loss / n_batches
        average_g_loss = epoch_g_loss / n_batches
        print('epoch: {0:04d}   d_loss = {1:0.6f}  g_loss = {2:0.6f}'.format(epoch,average_d_loss,average_g_loss))
        x_pred = g_model.predict(z_test)
        display_images(anti_norm(x_pred.reshape(-1,height,width)).astype('uint8'))
        
        #update checkpoint model
        progress_saver('checkpoint', epoch, g_best_loss, d_best_loss)
        
        #update final model
        if batch_g_loss + batch_d_loss < g_best_loss + d_best_loss:
            g_best_loss = batch_g_loss
            d_best_loss = batch_d_loss
            progress_saver('final', epoch, g_best_loss, d_best_loss)
            print('Final Model Updated')
        else:
            epochs_without_progress += n_epochs_print
        #early stopping
        if epochs_without_progress > max_epochs_without_progress:
            print("Early stopping")
            break
if not os.path.exists('figs/'):
    os.makedirs('figs/')

for i in range(len(saved_images[0])):
    print(i)
    plt.imshow(saved_images[0][i], cmap='viridis')
    plt.grid(b=False)
    plt.axis('off')
    plt.savefig('figs/eye'+str(i)+'.png', transparent = True, bbox_inches = 'tight', pad_inches = 0)
