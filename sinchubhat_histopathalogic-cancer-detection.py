! git clone https://github.com/tueimage/SE2CNN.git
! pip install tensorflow-gpu==1.13.1
# Import tensorflow and numpy
import tensorflow as tf
import numpy as np
import math as m
import time
import glob

# For validation
from sklearn.metrics import confusion_matrix
import itertools

# For plotting
from PIL import Image
from matplotlib import pyplot as plt

# Add the library to the system path
import os,sys
se2cnn_source =  os.path.join(os.getcwd(),'/kaggle/working/SE2CNN/')
if se2cnn_source not in sys.path:
    sys.path.append(se2cnn_source)

# Import the library
import se2cnn.layers
# help(se2cnn.layers.z2_se2n)
# help(se2cnn.layers.se2n_se2n)
# help(se2cnn.layers.spatial_max_pool)
# Xavier's/He-Rang-Zhen-Sun initialization for layers that are followed ReLU
def weight_initializer(n_in, n_out):
    return tf.random_normal_initializer(mean=0.0, stddev=m.sqrt(2.0 / (n_in))
    )
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
def size_of(tensor) :
    # Multiply elements one by one
    result = 1
    for x in tensor.get_shape().as_list():
         result = result * x 
    return result
CANCERtraindata = glob.glob('/kaggle/input/histopathologiccancerwithcsv/SplitData/train80/*.jpeg')
train_data = np.array([np.array(Image.open(fname)) for fname in CANCERtraindata])

# validation data
CANCERtestdata = glob.glob('/kaggle/input/histopathologiccancerwithcsv/SplitData/validation20/*.jpeg')
eval_data = np.array([np.array(Image.open(fname)) for fname in CANCERtestdata])
''' To read CSV data into a record array in NumPy you can use NumPy modules genfromtxt() function, In this function’s argument, you need to set the delimiter to a comma. '''
# genfromtxt - function to create arrays from tabular data

from numpy import genfromtxt
train_labels_2 = genfromtxt('/kaggle/input/histopathologiccancerwithcsv/SplitData/Label80.csv', delimiter=',')
train_labels = train_labels_2[:,1]
eval_labels_2 = genfromtxt('/kaggle/input/histopathologiccancerwithcsv/SplitData/Label20.csv', delimiter=',')
eval_labels = eval_labels_2[:,1]
print(" Length of train_data ")
print(len(train_data))
print(" Length of eval_data ")
print(len(eval_data))

print(" Length of train_labels ")
print(len(train_labels))
print(" Length of eval_labels ")
print(len(eval_labels))

#print(' Train data ')
#print(train_data)
#print(' Test data ')
#print(eval_data)

#print(' Train labels ')
#print(train_labels)
#print(' Test labels ')
#print(eval_labels)

# Reshape to 2D multi-channel images
train_data_2D = train_data.reshape([len(train_data),32,32,1]) # [batch_size, Nx, Ny, Nc]
eval_data_2D = eval_data.reshape([len(eval_data),32,32,1])
# To know the information about the structure of train_data_2D
# type
#print(' Type ',type(train_data_2D))
# Number of dimensions
#print('Number of dimensions',train_data_2D.ndim)
# shape
#print('shape',train_data_2D.shape)
# size
#print('size',train_data_2D.size)
# type of elements in it
#print(' type of elements in it',train_data_2D.dtype)

#print(' Train_data_2D ')
#print(train_data_2D)
#print(' eval_data_2D ')
#print(eval_data_2D)
# Plot the first sample 

plt.plot()
plt.title(' Label : %d ' % train_labels[0])
plt.imshow(train_data_2D[0,:,:,0], interpolation='nearest')
plt.show()
train_data_2D = np.pad(train_data_2D,((0,0),(2,2),(2,2),(0,0)),'constant', constant_values=((0,0),(0,0),(0,0),(0,0)))
eval_data_2D = np.pad(eval_data_2D,((0,0),(2,2),(2,2),(0,0)),'constant', constant_values=((0,0),(0,0),(0,0),(0,0)))
#print(' Train_data_2D ')
#print(train_data_2D)
#print(' eval_data_2D ')
#print(eval_data_2D)
graph = tf.Graph()
graph.as_default()
#tf.reset_default_graph()
tf.compat.v1.reset_default_graph()
Ntheta = 12 # Kernel size in angular direction
Nxy=5       # Kernel size in spatial direction
Nc = 4      # Number of channels in the initial layer
#inputs_ph = tf.placeholder( dtype = tf.float32, shape = [None,36,36,1] )
#labels_ph = tf.placeholder( dtype = tf.int32, shape = [None,] )
inputs_ph = tf.compat.v1.placeholder( dtype = tf.float32, shape = [None,36,36,1] )
labels_ph = tf.compat.v1.placeholder( dtype = tf.int32, shape = [None,] )
#print(' inputs_ph ')
#print(inputs_ph)
#print(' labels_ph ')
#print(labels_ph)
tensor_in = inputs_ph
Nc_in = 1
kernels={}
with tf.variable_scope("Layer_{}".format(1)) as _scope:
    ## Settings
    Nc_out = Nc

    ## Perform lifting convolution
    # The kernels used in the lifting layer
    kernels_raw = tf.get_variable(
                        'kernel', 
                        [Nxy,Nxy,Nc_in,Nc_out],
                        initializer=weight_initializer(Nxy*Nxy*Nc_in,Nc_out))
    tf.add_to_collection('raw_kernels', kernels_raw)
    bias = tf.get_variable( # Same bias for all orientations
                        "bias",
                        [1, 1, 1, 1, Nc_out], 
                        initializer=tf.constant_initializer(value=0.01))
    # Lifting layer
    tensor_out, kernels_formatted = se2cnn.layers.z2_se2n(
                            input_tensor = tensor_in,
                            kernel = kernels_raw,
                            orientations_nb = Ntheta)
    # Add bias
    tensor_out = tensor_out + bias
    
    ## Perform (spatial) max-pooling
    tensor_out = se2cnn.layers.spatial_max_pool( input_tensor=tensor_out, nbOrientations=Ntheta)
    
    ## Apply ReLU
    tensor_out = tf.nn.relu(tensor_out)

    ## Prepare for the next layer
    tensor_in = tensor_out
    Nc_in = Nc_out
    
    ## Save kernels for inspection
    kernels[_scope.name] = kernels_formatted

tensor_in.get_shape()
with tf.variable_scope("Layer_{}".format(2)) as _scope:
    ## Settings
    Nc_out = 2*Nc

    ## Perform group convolution
    # The kernels used in the group convolution layer
    kernels_raw = tf.get_variable(
                        'kernel', 
                        [Nxy,Nxy,Ntheta,Nc_in,Nc_out],
                        initializer=weight_initializer(Nxy*Nxy*Ntheta*Nc_in,Nc_out))
    tf.add_to_collection('raw_kernels', kernels_raw)
    bias = tf.get_variable( # Same bias for all orientations
                        "bias",
                        [1, 1, 1, 1, Nc_out], 
                        initializer=tf.constant_initializer(value=0.01))
    # The group convolution layer
    tensor_out, kernels_formatted = se2cnn.layers.se2n_se2n(
                            input_tensor = tensor_in,
                            kernel = kernels_raw)
    tensor_out = tensor_out + bias
    
    ## Perform max-pooling
    tensor_out = se2cnn.layers.spatial_max_pool( input_tensor=tensor_out, nbOrientations=Ntheta)
    
    ## Apply ReLU
    tensor_out = tf.nn.relu(tensor_out)

    ## Prepare for the next layer
    tensor_in = tensor_out
    Nc_in = Nc_out
    
    ## Save kernels for inspection
    kernels[_scope.name] = kernels_formatted
tensor_in.get_shape()
# Concatenate the orientation and channel dimension
tensor_in = tf.concat([tensor_in[:,:,:,i,:] for i in range(Ntheta)],3)
Nc_in = tensor_in.get_shape().as_list()[-1]

# 2D convolution layer
with tf.variable_scope("Layer_{}".format(3)) as _scope:
    ## Settings
    Nc_out = 4*Nc

    ## Perform group convolution
    # The kernels used in the group convolution layer
    kernels_raw = tf.get_variable(
                        'kernel', 
                        [Nxy,Nxy,Nc_in,Nc_out],
                        initializer=weight_initializer(Nxy*Nxy*Nc_in,Nc_out))
    tf.add_to_collection('raw_kernels', kernels_raw)
    bias = tf.get_variable( # Same bias for all orientations
                        "bias",
                        [1, 1, 1, Nc_out], 
                        initializer=tf.constant_initializer(value=0.01))
    # Convolution layer
    tensor_out = tf.nn.conv2d(
                        input = tensor_in,
                        filter=kernels_raw,
                        strides=[1, 1, 1, 1],
                        padding="VALID")
    tensor_out = tensor_out + bias
    
    ## Apply ReLU
    tensor_out = tf.nn.relu(tensor_out)

    ## Prepare for the next layer
    tensor_in = tensor_out
    Nc_in = Nc_out
    
    ## Save kernels for inspection
    kernels[_scope.name] = kernels_raw
tensor_in.get_shape()
# 2D convolution layer
with tf.variable_scope("Layer_{}".format(4)) as _scope:
    ## Settings
    Nc_out = 128

    ## Perform group convolution
    # The kernels used in the group convolution layer
    kernels_raw = tf.get_variable(
                        'kernel', 
                        [1,1,Nc_in,Nc_out],
                        initializer=weight_initializer(1*1*Nc_in,Nc_out))
    tf.add_to_collection('raw_kernels', kernels_raw)
    bias = tf.get_variable( # Same bias for all orientations
                        "bias",
                        [1, 1, 1, Nc_out], 
                        initializer=tf.constant_initializer(value=0.01))
    # Convolution layer
    tensor_out = tf.nn.conv2d(
                        input = tensor_in,
                        filter=kernels_raw,
                        strides=[1, 1, 1, 1],
                        padding="VALID")
    tensor_out = tensor_out + bias
    
    ## Apply ReLU
    tensor_out = tf.nn.relu(tensor_out)

    ## Prepare for the next layer
    tensor_in = tensor_out
    Nc_in = Nc_out
    
    ## Save kernels for inspection
    kernels[_scope.name] = kernels_raw

tensor_in.get_shape()
with tf.variable_scope("Layer_{}".format(5)) as _scope:
    ## Settings
    Nc_out = 2

    ## Perform group convolution
    # The kernels used in the group convolution layer
    kernels_raw = tf.get_variable(
                        'kernel', 
                        [1,1,Nc_in,Nc_out],
                        initializer=weight_initializer(1*1*Nc_in,Nc_out))
    tf.add_to_collection('raw_kernels', kernels_raw)
    bias = tf.get_variable( # Same bias for all orientations
                        "bias",
                        [1, 1, 1, Nc_out], 
                        initializer=tf.constant_initializer(value=0.01))

    
    ## Convolution layer
    tensor_out = tf.nn.conv2d(
                        input = tensor_in,
                        filter=kernels_raw,
                        strides=[1, 1, 1, 1],
                        padding="VALID")
    tensor_out = tensor_out + bias
    
    ## The output logits
    logits = tensor_out[:,0,0,:]
    predictions = tf.argmax(input=logits, axis=1)
    probabilities = tf.nn.softmax(logits)
    
    ## Save the kernels for later inspection
    kernels[_scope.name] = kernels_raw

logits.get_shape()
# Cross-entropy loss
loss = tf.losses.sparse_softmax_cross_entropy(labels=labels_ph, logits=logits)
#-- Define the l2 loss 
weightDecay=5e-4
# Get the raw kernels
variables_wd = tf.get_collection('raw_kernels')
print('-----')
print('RAW kernel shapes:')
for v in variables_wd: print( "[{}]: {}, total nr of weights = {}".format(v.name, v.get_shape(), size_of(v)))
print('-----')
loss_l2 = weightDecay*sum([tf.nn.l2_loss(ker) for ker in variables_wd])
# Configure the Training Op (for TRAIN mode)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

train_op = optimizer.minimize(
    loss=loss + loss_l2,
    global_step=tf.train.get_global_step())
#-- Start the (GPU) session
initializer = tf.global_variables_initializer()
session = tf.Session(graph=tf.get_default_graph()) #-- Session created
session.run(initializer)
batch_size=100
n_epochs=10
for epoch_nr in range(n_epochs):
    loss_average = 0
    data = train_data_2D
    labels = train_labels
    # KBatch settings
    NItPerEpoch = m.floor(len(data)/batch_size) #number of iterations per epoch
    samples=np.random.permutation(len(data))
    # Loop over dataset
    tStart = time.time()
    for iteration in range(NItPerEpoch):
        feed_dict = {
                inputs_ph: np.array(data[samples[iteration*batch_size:(iteration+1)*batch_size]]),
                labels_ph: np.array(labels[samples[iteration*batch_size:(iteration+1)*batch_size]])
                }
        operators_output = session.run([ loss , train_op ], feed_dict)
        loss_average += operators_output[0]/NItPerEpoch
    tElapsed = time.time() - tStart
    print('Epoch ' , epoch_nr , ' finished... Average loss = ' , round(loss_average,4) , ', time = ',round(tElapsed,4))
batch_size = 5
labels_pred = []
for i in range(round(len(eval_data_2D)/batch_size)):
    [ labels_pred_batch ] = session.run([ predictions ], { inputs_ph: eval_data_2D[i*batch_size:(i+1)*batch_size] })
    labels_pred = labels_pred + list(labels_pred_batch)
labels_pred = np.array(labels_pred)
print(' Compare the first 10 results with the ground truth ')
print(' The first 10 predicted results ')
print(labels_pred[0:10])
print(' The first 10 ground truth (actual results) ')
print(eval_labels[0:10])

print(' The accuracy (average number of successes) ')
print(((labels_pred - eval_labels)**2==0).astype(float).mean())
print(' Total number of errors ')
print(((labels_pred - eval_labels)**2>0).astype(float).sum())
print(' Error rate ')
print(100*((labels_pred - eval_labels)**2>0).astype(float).mean())
cm = confusion_matrix(eval_labels, labels_pred)
plot_confusion_matrix(cm, range(1))
