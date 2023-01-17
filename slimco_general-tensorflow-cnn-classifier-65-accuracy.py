import tensorflow as tf

import matplotlib.pyplot as plt 

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split as tts

import os

import math

import cv2

import sklearn.preprocessing as pp

from tqdm import tqdm_notebook as tqdm

from sklearn.utils import shuffle
#Function for loading the image data

def loadData(path):

    "parse the root path with the labled child folders folder"

    imgs = []

    labels = []

    folders = os.listdir(path)

    

    for folder in folders:

        os.chdir(os.path.join(path,folder))

        files = os.listdir()

        for file in files:

            img = cv2.imread(file)

            if img.shape != (150,150,3):

                continue

            else:

                imgs.append(img)

                labels.append(folder)



    

    

    lbl_encoded = pp.LabelEncoder()

    lbl_vls = lbl_encoded.fit_transform(labels).reshape(-1,1)

    one_hot = pp.OneHotEncoder(sparse=False)

    labels = one_hot.fit_transform(lbl_vls)

    return imgs, labels

        

    

    

    
X_train, Y_train = loadData('/kaggle/input/seg_train/seg_train')

X_test, Y_test = loadData('/kaggle/input/seg_test/seg_test')
print(len(X_test))

print(len(Y_test))

print(len(X_train))

print(len(Y_train))

X_train = np.array(X_train)

Y_train = np.array(Y_train)

X_test = np.array(X_test)

Y_test = np.array(Y_test)

fig,ax =plt.subplots(5,5,figsize=(18,16))

for i in range(5):

    for j in range(5):

        index=np.random.randint(X_train.shape[0])

        ax[i][j].imshow(X_train[index])

        ax[i][j].axis('off')

        ax[i][j].set_title(Y_train[index])
class CNNConstructor():

    

    def __init__(self):

        self.active=True

        

    def initWeights(self,shape,name='Weights'):

        return tf.Variable(tf.truncated_normal(shape=shape,stddev=0.05,name=name))

    

    def initBias(self,num,name='Bias'):

        return tf.Variable(tf.constant(0.7,shape=[num]),name=name)

    

    def convLayer(self,input,kernel_size, num_kernels=64,

                 num_channels=3,conv_stride=1,pool_stride=2,padding='SAME',

                 pooling=True,activate=True, name='Conv_layer'):

        

        

        conv_volume = [ kernel_size,kernel_size,num_channels,num_kernels]

        weights = self.initWeights(shape = conv_volume)

        bias = self.initBias(num = num_kernels)

        

        conv_layer = tf.nn.conv2d(input, filter=weights, strides= [1,conv_stride,conv_stride,1],

                                 padding=padding)

        conv_layer = tf.add(conv_layer, bias)

        

        if pooling:

            conv_layer = tf.nn.max_pool(conv_layer,

                                       ksize=[1,2,2,1],

                                       strides=[1,pool_stride,pool_stride,1],

                                       padding=padding)

        if activate:

            conv_layer = tf.nn.relu(conv_layer)

            

        return conv_layer, weights

    

    def flatten_conv_layer(self, conv_layer, name='Flatten_layer'):

        conv_shapes = conv_layer.get_shape()

        n_features = conv_shapes[1:4].num_elements()



        return tf.reshape(conv_layer, shape=[-1, n_features]), n_features

    

    def dens_layer(self,input,n_inpt,n_out,activation=True,name='Dens'):

        

        weights = self.initWeights(shape=[n_inpt,n_out])

        bias = self.initBias(num=n_out)

        

        dens = tf.matmul(input, weights)

        dens = tf.add(dens, bias)

        

        if activation:

            dens = tf.nn.relu(dens)

        return dens

        
class batch():

    def __init__(self,imgs,labels):

        self.X = imgs

        self.Y = labels

        self.batch=0



    def shuffle(self,state=101,new=True):

        self.X = shuffle(self.X,random_state=state)

        self.Y = shuffle(self.Y,random_state=state)

        if new:

            self.batch = 0

        

        

    def next_batch(self,batch_size=100):

        

        if self.X.shape[0]<batch_size*self.batch:

            X,Y = self.X[self.batch*batch_size:],self.Y[self.batch*batch_size:]

            self.batch=0

            return X,Y

        else:

            X,Y = self.X[self.batch*batch_size:(self.batch+1)*batch_size],self.Y[self.batch*batch_size:(self.batch+1)*batch_size]

            self.batch += 1 

            return X,Y

        

        
def plot_conv_weights(weights, input_channel=0):

    

    # Retrieve the values of the weight-variables

    w_values = session.run(weights)

    

    # Number of filters used in the conv. layer.

    n_kernels = w_values.shape[3]

    n_grids = math.ceil(np.sqrt(n_kernels))

    print('Total number of kernels: {} \n\t  Plotting grid: {}x{}'.format(n_kernels,

                                                                          n_grids,

                                                                          n_grids))

    # Plot.

    fig, axes = plt.subplots(n_grids, n_grids, figsize=(10, 10))

    axes = axes.flatten()

    

    for i, ax in enumerate(axes):

        if i < n_kernels:

            ax.imshow(w_values[:, :, input_channel, i],

                      vmin=np.min(w_values),

                      vmax=np.max(w_values),

                      interpolation='nearest',

                      cmap='viridis')

        

        ax.set_xticks([])

        ax.set_yticks([])

    plt.show()

    

    return fig



def plot_conv_layer(layer, image):

    

    # Feed an image to the leyer of interest.

    feed_dict = {X: [image]}

    conv_values = session.run(layer, feed_dict=feed_dict)

    

    

    # Number of filters used in the conv. layer.

    n_kernels = conv_values.shape[3]

    n_grids = math.ceil(np.sqrt(n_kernels))

    print('Total number of kernels: {} \n\t  Plotting grid: {}x{}'.format(n_kernels,

                                                                          n_grids,

                                                                          n_grids))

    # Plot.

    fig, axes = plt.subplots(n_grids, n_grids, figsize=(10, 10))

    axes = axes.flatten()

    for i, ax in enumerate(axes):

        if i < n_kernels:

            ax.imshow(conv_values[0, :, :, i],

                      interpolation='nearest',

                      cmap='binary')

        

        ax.set_xticks([])

        ax.set_yticks([])

    plt.show()

    

    return fig
n_classes = 6

n_channels = 3

n_pixels = 150



kernel_size = 3

n_kernels_l1 = 128

n_kernels_l2 = 128

n_kernels_l3 = 64

n_kernels_l4 = 32

n_neurons_fc1 = 1024

n_neurons_fc2 = 512

tf.reset_default_graph()



X = tf.placeholder(tf.float32, shape= (None, n_pixels,n_pixels,n_channels), name = 'InputImages')

Y = tf.placeholder(tf.float32, shape = (None, n_classes), name = 'Labels')
CNN = CNNConstructor()



C1,W1 = CNN.convLayer(input=X,kernel_size=kernel_size,num_kernels=n_kernels_l1, pooling=False)



C2,W2 = CNN.convLayer(input = C1,kernel_size=kernel_size,num_channels=n_kernels_l1, num_kernels=n_kernels_l2, pooling=True)



C3,W3 = CNN.convLayer(input= C2,kernel_size=kernel_size,num_channels=n_kernels_l2, num_kernels=n_kernels_l3, pooling = False)



C4,W4 = CNN.convLayer(input= C3,kernel_size=kernel_size,num_channels=n_kernels_l3, num_kernels=n_kernels_l4, pooling = True)



flat, n_features = CNN.flatten_conv_layer(C3)



fc = CNN.dens_layer(input=flat, n_inpt=n_features, n_out=n_neurons_fc1, activation=True, name='Dense_1')



dpo = tf.nn.dropout(fc,keep_prob=0.5)



fc2 = CNN.dens_layer(input=dpo, n_inpt=n_neurons_fc1, n_out=n_neurons_fc2, activation=True, name='Dense_2')



out_layer = CNN.dens_layer(input=fc2, n_inpt=n_neurons_fc2, n_out=n_classes, activation=False, name='Output')


# this adds a name to a node in the tensorboar

with tf.name_scope('Cross_Entropy'):  

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=out_layer,

                                                               labels=Y)

cost_function = tf.reduce_mean(cross_entropy)

with tf.name_scope('Loss_function'):  # this adds a name to a node in the tensorboar

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost_function)



# Evaluation

Y_pred = tf.argmax(out_layer, axis=1)

Y_true = tf.argmax(Y, axis=1)



with tf.name_scope('Accuracy'):

    correct_prediction = tf.equal(Y_pred, Y_true)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
batch_size = 100

epochs = 25

path_logs = '/kaggle/working'



#saver = tf.train.Saver()

session = tf.Session()

session.run(tf.global_variables_initializer())

accuaracy_values = []

valid_accuracy_values = []



#merged = tf.summary.merge_all()

#writer=tf.summary.FileWriter(path_logs) # creates file with tensorboard data stored

#writer.add_graph(session.graph)         # writes data to the file,

                                        # for variables need to use merged = tf.summary.merge_all()

# to initialize the tensorboard use 'tensorboard --logdir=path' in cmd





data=batch(X_train,Y_train)

valid_data = batch(X_test,Y_test)





for i in tqdm(range(epochs)):

    acc_epoch = []

    valid_acc_epoch = []

    data.shuffle(state=np.random.randint(0,500))

    valid_data.shuffle(state=np.random.randint(0,500))

    

    for j in tqdm(range(len(Y_train)//batch_size)):

        

        batch_x, batch_y = data.next_batch(batch_size)

        

        feed_train = {X: batch_x,

                      Y: batch_y}



        session.run(optimizer, feed_dict=feed_train)



        if j % 10 == 0:

            acc_j = session.run(accuracy, feed_dict=feed_train)

            acc_epoch.append(acc_j)

            valid_x,valid_y = valid_data.next_batch(batch_size)

            feed_valid = {X:valid_x,Y:valid_y}

            v_acc_j=session.run(accuracy, feed_dict=feed_valid)

            valid_acc_epoch.append(v_acc_j)

            #msg = "Batch: {0:>6}, Training Accuracy: {1:>6.1%}"

            #print(msg.format(j + 1, acc_j))

            #summary, acc = session.run([merged, accuracy], feed_dict=feed_train)

            #test_writer.add_summary(summary, j)

        

    acc = np.mean(acc_epoch)

    accuaracy_values.append(acc)

    val_acc = np.mean(valid_acc_epoch)

    valid_accuracy_values.append(val_acc)

    print('Epoch {} train accuracy :{}'.format(i+1,acc))

    print('Epoch {} validation accuracy :{}'.format(i+1,val_acc))

    #if val_acc>=valid_accuracy_values[-1]:

        #save_path = saver.save(session, "{}/model.ckpt".format(path_logs))

        #print('model saved to {}'.format(save_path))

        
plt.plot(accuaracy_values)

plt.plot(valid_accuracy_values)

plt.show()
saver = tf.train.Saver()



with tf.Session() as session:

    saver.restore(session,'/kaggle/working/model.ckpt')

    test_data = batch(imgs=X_test,labels=Y_test)

    test_data.shuffle()

    x_data_test, y_data_test = test_data.next_batch(100)

    plot_conv_layer(C1,image=x_data_test[50,:,:,:])

    plot_conv_layer(C2,image=x_data_test[50,:,:,:])

    plot_conv_layer(C3,image=x_data_test[50,:,:,:])

    plot_conv_layer(C4,image=x_data_test[50,:,:,:])  