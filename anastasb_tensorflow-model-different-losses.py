import tensorflow
import sys
import numpy as np
import keras
from keras.utils import Sequence
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
import imgaug as ia
from imgaug import augmenters as iaa
import cv2
from matplotlib import image

BASE_DIR = "../input"
def train_data():
    ''' Gets paths to images for training data and labels '''
    x_train_path  = os.path.join(BASE_DIR,'train')
    y_train_path  = os.path.join(BASE_DIR,'train.csv')
    data          = pd.read_csv(y_train_path)
    paths         = []
    labels        = []
    for example_id, protein_ids in zip(data['Id'], data['Target'].str.split(' ')):
        y = np.zeros(28)
        y[np.array(protein_ids,dtype=np.int32)]=1
        paths.append(os.path.join(x_train_path, example_id))
        labels.append(y)
    return np.array(paths), np.array(labels)

def test_data():
    ''' Gets paths to images for test data '''
    x_test_path = os.path.join(BASE_DIR,'test')
    data_test   = pd.read_csv(os.path.join(BASE_DIR,'sample_submission.csv'))
    paths       = []
    y           = []
    for example_id in data_test.Id:
        paths.append(os.path.join(x_test_path,example_id))
        y.append(np.zeros(28))
    return np.array(paths), np.array(y)
paths, labels = train_data()

label_counts = np.sum(labels,axis=0)
protein_id   = np.arange(len(label_counts))

localization_names = {
    0:  "Nucleoplasm",  
    1:  "Nuclear membrane",   
    2:  "Nucleoli",   
    3:  "Nucleoli fibrillar center",   
    4:  "Nuclear speckles",
    5:  "Nuclear bodies",   
    6:  "Endoplasmic reticulum",   
    7:  "Golgi apparatus",   
    8:  "Peroxisomes",   
    9:  "Endosomes",   
    10:  "Lysosomes",   
    11:  "Intermediate filaments",   
    12:  "Actin filaments",   
    13:  "Focal adhesion sites",   
    14:  "Microtubules",   
    15:  "Microtubule ends",   
    16:  "Cytokinetic bridge",   
    17:  "Mitotic spindle",   
    18:  "Microtubule organizing center",   
    19:  "Centrosome",   
    20:  "Lipid droplets",   
    21:  "Plasma membrane",   
    22:  "Cell junctions",   
    23:  "Mitochondria",   
    24:  "Aggresome",   
    25:  "Cytosol",   
    26:  "Cytoplasmic bodies",   
    27:  "Rods & rings"
}

# proteins 8,9,10 and 28 are severely underrepresented in the 
plt.figure(1, figsize=(8,5))
plt.bar(protein_id,label_counts/1000,color='c')
plt.xticks(protein_id,[localization_names[int(el)] for el in protein_id],rotation='vertical')
plt.xlabel('Localizations')
plt.ylabel('Frequency in K')
plt.title('Frequency of protein localizations')
plt.box(on=None) 
plt.show()


sorted_by_counts = sorted(zip(protein_id,label_counts),key = lambda item: item[1], reverse=False)
frequency_counts = pd.DataFrame(data=np.array(sorted_by_counts,dtype=np.int32), columns=['ProteinID','Counts'])
print(frequency_counts)
# How many localizations there are per example
labels_per_image = np.sum(labels,axis=1)
from collections import Counter
cnt = sorted(Counter(labels_per_image).items(),key=lambda item: item[0])
lpi    = [int(e[0]) for e in cnt]
counts = [int(e[1]) for e in cnt] 


plt.figure()
plt.bar(lpi,counts,color='g')
plt.xticks(lpi, [str(e) for e in lpi])
plt.xlabel('Labels per image')
plt.ylabel('Count')
plt.box(on=None) 
plt.title('Number of labels per data point')
# We want to make split so that train/val/test sets will look the same way, due to 
# significant data imbalances random split may result lead to rarely occuring classes 
# being absent in train or test or validation. In particular we are interested in making 
# sure that examples that contain protein with ID=27 are included.

from sklearn.model_selection import train_test_split

paths,labels = train_data()
paths_train, paths_val, labels_train, labels_val = train_test_split(paths,labels,test_size=0.2,stratify=labels[:,27])
#paths_val, paths_test, labels_val, labels_test = train_test_split(paths_val_test,labels_val_test,test_size=0.5,stratify=labels_val_test[:,27])

# As you can see from the plot train/val/test have the same distribution
plt.figure(figsize=(12,6))
for series,name,clr in zip([labels_train,labels_val],['train','val'],['ro','bo']):
     plt.plot(np.arange(series.shape[1]),series.sum(axis=0)/series.sum(),clr)
plt.xlabel('Protein ID')
plt.ylabel('Proportion of label')
plt.title('Comparison of train/val/test')
plt.show()

# train/val/test contain Protein with ID = 27
print(labels_train[:,27].sum())
print(labels_val[:,27].sum())
#print(labels_test[:,27].sum())
    
# Vizualize second image 
path = paths_train[1]
path_completions = ["_green.png",'_yellow.png','_blue.png','_red.png']
color_maps       = ['Greens','Oranges','Blues','Reds']
plt.figure(figsize=(6,6))
plt.subplots_adjust(left=0.05, right=0.95)
for i, (cmpl,cmp) in enumerate(zip(path_completions,color_maps)):
    A = image.imread(path+cmpl)
    plt.subplot(2,2,i+1)
    plt.imshow(A,cmap=cmp)
    plt.title(cmp)
    plt.box(on=None) 
plt.show()   
# loading single image

image_yellow = image.imread(paths_train[0]+'_yellow.png')
image_green  = image.imread(paths_train[0]+'_green.png')
image_blue   = image.imread(paths_train[0]+'_blue.png')
image_red    = image.imread(paths_train[0]+'_red.png')

image_general = np.concatenate([np.expand_dims(channel,axis=-1) for channel in (image_yellow,image_green,image_blue,image_red)],axis=-1)

def image_loader(image_path, width=512, height=512, channels_to_include=['green']):
    ''' Image loader '''
    image_matrix = np.zeros([width,height,len(channels_to_include)],dtype=np.float32)
    channel_index = 0
    for i,channel_color in enumerate(channels_to_include):
        channel_extension = "_"+channel_color+'.png'
        image_matrix[:,:,i] = image.imread(image_path+channel_extension)
    return image_matrix
from skimage.transform import resize
from itertools import islice

def train_generator_single_epoch(paths,labels,batch_size=128,row_resize_ratio=4,col_resize_ratio=4,
                                width=512,height=512,channels_to_include=['green','blue','red','yellow']):
    ''' Generator for training data that will automatically resize image to a given proportions '''
    iter_idx = 0
    while iter_idx*batch_size < labels.shape[0]:
        batch_paths  = paths[iter_idx*batch_size:(iter_idx+1)*batch_size]
        batch_images = [image_loader(path,width,height,channels_to_include) for path in batch_paths]
        batch_labels = labels[iter_idx*batch_size:(iter_idx+1)*batch_size,:]
        if row_resize_ratio==1 and col_resize_ratio==1:
            iter_idx += 1
            yield batch_images,batch_labels
        else:
            iter_idx += 1
            resize_shape = (int(width/row_resize_ratio),int(height/col_resize_ratio),len(channels_to_include))
            batched_resized_images = [resize(img,resize_shape) for img in batch_images]
            yield np.array(batched_resized_images),batch_labels

# vizual comparison of resized and original image
train_iterator_original   = train_generator_single_epoch(paths_train,labels_train,batch_size=1,row_resize_ratio=1,col_resize_ratio=1)
train_iterator_resized    = train_generator_single_epoch(paths_train,labels_train,batch_size=1,row_resize_ratio=4,col_resize_ratio=4)
image_original,_          = next(train_iterator_original)
image_resized,_           = next(train_iterator_resized)
plt.figure(figsize=(6,6))
plt.subplots_adjust(left=0.05, right=0.95)
plt.subplot(1,2,1)
plt.imshow(np.squeeze(image_resized),cmap='Greens')
plt.title('Resized Image')


plt.subplot(1,2,2)
plt.imshow(np.squeeze(image_original),cmap='Greens')
plt.title('Original Image')
plt.show()

# Comments: it is quite obvious from these images that resizing to the size (128,128) from (512,512) resulted in
# more blurry image and probably significant loss of information.
def eval_predict_proba_generator(paths,predict_fn,batch_size=128,row_resize_ratio=4,col_resize_ratio=4,
                                 width=512,height=512,channels_to_include=['green']):
    iter_idx    = 0
    accumulator = []
    resize_shape = (int(width/row_resize_ratio),int(height/col_resize_ratio),len(channels_to_include))
    while iter_idx*batch_size < paths.shape[0]:
        batch_paths  = paths[iter_idx*batch_size:(iter_idx+1)*batch_size]
        batch_images = [image_loader(path,width,height,channels_to_include) for path in batch_paths]
        if row_resize_ratio==1 and col_resize_ratio==1:
            iter_idx += 1
            predicted_probs = predict_fn(np.concatenate([np.expand_dims(im,axis=0) for im in batch_images],axis=0))
            accumulator.append(np.concatenate(predicted_probs,axis=1))
        else:
            iter_idx += 1
            batched_resized_images = [resize(img,resize_shape) for img in batch_images]
            predicted_probs = predict_fn(np.concatenate([np.expand_dims(im,axis=0) for im in batched_resized_images],axis=0))
            accumulator.append(np.concatenate(predicted_probs,axis=1))
    return np.concatenate(accumulator,axis=0)
    

def compute_task_specific_gamma(y_true):
    '''Heuristic to compute values of gamma for different tasks'''
    props = np.sum(y_true,0) / y_true.shape[0]
    gammas = [0]*28
    for i in range(y_true.shape[1]):
        if props[i] > 0.2:
            gammas[i] = 2
        elif props[i] >= 1e-2 and props[i] <= 0.2:
            gammas[i] = 4
        else:
            gammas[i] = 6
    return gammas
        
# Since our labels are not mutually exclusive using softmax is not an option. As our first baseline we are going to 
# experiment with multi-task learning with hard parameter sharing, where in shared layers we are going to use convolutional 
# layers while task specific layers will be dense ones. Each task will have weighted binary cross-entropy loss function, 
# we will try to use weights to account for severe imbalance in the data.
# Credit: coursera/convolutional_neural_networks/week_1/home_assignment
from sklearn.metrics import f1_score
from functools import partial
import tensorflow as tf

def create_placeholders(height=128, width=128, channels=4, n_targets=28):
    ''' Creates the placeholders for the tensorflow session. '''
    X = tf.placeholder(shape=[None,height,width,channels], dtype=tf.float32, name='X')
    Y = tf.placeholder(shape=[None,n_targets], dtype=tf.float32, name='Y')    
    is_train = tf.placeholder(tf.bool, name="is_train")
    return X, Y, is_train


#-------------------------------------------- BASELINE ARCHITECTURE ------------------------------------------------

def initialize_parameters(channels=4):
    ''' Initializes weight parameters to build a neural network with tensorflow'''
    # We attempt to do architecture that closely resembles VGG19 by Karen Simonyan
    # 1) 2 convolutional layers 
    W1 = tf.get_variable(name='W1',shape=[4,4,channels,24],initializer=tf.contrib.layers.xavier_initializer_conv2d())
    W2 = tf.get_variable(name='W2',shape=[4,4,24,32],initializer=tf.contrib.layers.xavier_initializer_conv2d())
    # 2) Another 2 convolutional layers
    W3 = tf.get_variable(name='W3',shape=[4,4,32,32],initializer=tf.contrib.layers.xavier_initializer_conv2d())
    W4 = tf.get_variable(name='W4',shape=[4,4,32,32],initializer=tf.contrib.layers.xavier_initializer_conv2d())
    # 3) Another 2 convolutional layers
    W5 = tf.get_variable(name='W5',shape=[4,4,32,64],initializer=tf.contrib.layers.xavier_initializer_conv2d())
    W6 = tf.get_variable(name='W6',shape=[4,4,64,64],initializer=tf.contrib.layers.xavier_initializer_conv2d())
    return {"W1": W1,"W2": W2,"W3":W3,"W4":W4,'W5':W5,'W6':W6}

def forward_propagation_shared_layers(X, parameters, is_train):
    """
    Implements forward propagation for shared layers, took inspiration from VGG16 model
    by Karen Simonyan.
    """    
    # Retrieve the parameters from the dictionary "parameters" 
    W1  = parameters['W1']
    W2  = parameters['W2']
    W3  = parameters['W3']
    W4  = parameters['W4']
    W5  = parameters['W5']
    W6  = parameters['W6']
    # BLOCK 1: CONV2D -> RELU -> CONV2D -> RELU -> MaxPool
    layer1_Z1    = tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding='SAME')
    layer1_A1    = tf.nn.relu(layer1_Z1)
    layer1_Z2    = tf.nn.conv2d(layer1_A1,W2,strides=[1,1,1,1],padding='SAME')
    layer1_A2    = tf.nn.relu(layer1_Z2)
    layer1_MP    = tf.nn.max_pool(layer1_A2,ksize=[1,4,4,1],strides=[1,4,4,1],padding='SAME')
    # BLOCK 2: CONV2D -> RELU -> CONV2D -> RELU -> MaxPool
    layer2_Z1    = tf.nn.conv2d(layer1_MP,W3,strides=[1,1,1,1],padding='SAME')
    layer2_A1    = tf.nn.relu(layer2_Z1)
    layer2_Z2    = tf.nn.conv2d(layer2_A1,W4,strides=[1,1,1,1],padding='SAME')
    layer2_A2    = tf.nn.relu(layer2_Z2)
    layer2_MP    = tf.nn.max_pool(layer2_A2,ksize=[1,4,4,1],strides=[1,4,4,1],padding='SAME')
    # BLOCK 3: CONV2D -> RELU -> CONV2D -> RELU -> MaxPool
    layer3_Z1    = tf.nn.conv2d(layer2_MP,W5,strides=[1,1,1,1],padding='SAME')
    layer3_A1    = tf.nn.relu(layer3_Z1)
    layer3_Z2    = tf.nn.conv2d(layer3_A1,W6,strides=[1,1,1,1],padding='SAME')
    layer3_A2    = tf.nn.relu(layer3_Z2)
    layer3_MP    = tf.nn.max_pool(layer3_A2,ksize=[1,4,4,1],strides=[1,4,4,1],padding='SAME')
    # FLATTEN
    layer4_FLAT  = tf.contrib.layers.flatten(layer3_MP)
    # Fully connected layers
    layer5_FC1   = tf.contrib.layers.fully_connected(layer4_FLAT, 108, activation_fn=tf.nn.relu)
    layer6_FC2   = tf.contrib.layers.fully_connected(layer5_FC1, 56, activation_fn=tf.nn.relu)
    return layer6_FC2
    
def forward_propagation_task_specific_layers(shared_last_layer):
    ''' Forward propagation for task specific layers '''
    task_specific_layers = [0]*28
    for i in range(28):
        task_specific_layers[i] = tf.contrib.layers.fully_connected(shared_last_layer, 1, activation_fn=None)
    return task_specific_layers

#-------------------------------------------- ARCHITECTURE VERSION 1 ------------------------------------------------

def initialize_parameters_v1(channels=4):
    ''' Initializes weight parameters to build a neural network with tensorflow'''
    # We attempt to do architecture that closely resembles VGG19 by Karen Simonyan
    # 1) 2 convolutional layers 
    W1 = tf.get_variable(name='W1',shape=[3,3,channels,8],initializer=tf.contrib.layers.xavier_initializer_conv2d())
    W2 = tf.get_variable(name='W2',shape=[3,3,8,8],initializer=tf.contrib.layers.xavier_initializer_conv2d())
    # 2) Another 2 convolutional layers
    W3 = tf.get_variable(name='W3',shape=[3,3,8,16],initializer=tf.contrib.layers.xavier_initializer_conv2d())
    W4 = tf.get_variable(name='W4',shape=[3,3,16,16],initializer=tf.contrib.layers.xavier_initializer_conv2d())
    # 3) Another 2 convolutional layers
    W5 = tf.get_variable(name='W5',shape=[3,3,16,32],initializer=tf.contrib.layers.xavier_initializer_conv2d())
    W6 = tf.get_variable(name='W6',shape=[3,3,32,32],initializer=tf.contrib.layers.xavier_initializer_conv2d())
    # 4) Another 2 convolutional layers
    W7 = tf.get_variable(name='W7',shape=[3,3,32,64],initializer=tf.contrib.layers.xavier_initializer_conv2d())
    W8 = tf.get_variable(name='W8',shape=[3,3,64,128],initializer=tf.contrib.layers.xavier_initializer_conv2d())
    return {"W1": W1,"W2": W2,"W3":W3,"W4":W4,'W5':W5,'W6':W6,'W7':W7,'W8':W8}


def forward_propagation_shared_layers_v1(X, parameters, is_train):
    """
    Implements forward propagation for shared layers, took inspiration from VGG16 model
    by Karen Simonyan.
    """    
    # Retrieve the parameters from the dictionary "parameters" 
    W1  = parameters['W1']
    W2  = parameters['W2']
    W3  = parameters['W3']
    W4  = parameters['W4']
    W5  = parameters['W5']
    W6  = parameters['W6']
    W7  = parameters['W7']
    # BLOCK 1: CONV2D -> RELU -> CONV2D -> BN -> RELU -> MaxPool
    layer1_Z1    = tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding='SAME')
    layer1_A1    = tf.nn.relu(layer1_Z1)
    layer1_Z2    = tf.nn.conv2d(layer1_A1,W2,strides=[1,1,1,1],padding='SAME')
    layer1_BN    = tf.layers.batch_normalization(layer1_Z2,training=is_train)
    layer1_A2    = tf.nn.relu(layer1_BN)
    layer1_MP    = tf.nn.max_pool(layer1_A2,ksize=[1,2,2,1],strides=[1,4,4,1],padding='VALID')
    # BLOCK 2: CONV2D -> RELU -> CONV2D -> BN -> RELU -> MaxPool
    layer2_Z1    = tf.nn.conv2d(layer1_MP,W3,strides=[1,1,1,1],padding='SAME')
    layer2_A1    = tf.nn.relu(layer2_Z1)
    layer2_Z2    = tf.nn.conv2d(layer2_A1,W4,strides=[1,1,1,1],padding='SAME')
    layer2_BN    = tf.layers.batch_normalization(layer2_Z2,training=is_train)
    layer2_A2    = tf.nn.relu(layer2_BN)
    layer2_MP    = tf.nn.max_pool(layer2_A2,ksize=[1,2,2,1],strides=[1,4,4,1],padding='VALID')
    # BLOCK 3: CONV2D -> RELU -> CONV2D -> BN-> RELU -> MaxPool
    layer3_Z1    = tf.nn.conv2d(layer2_MP,W5,strides=[1,1,1,1],padding='SAME')
    layer3_A1    = tf.nn.relu(layer3_Z1)
    layer3_Z2    = tf.nn.conv2d(layer3_A1,W6,strides=[1,1,1,1],padding='SAME')
    layer3_BN    = tf.layers.batch_normalization(layer3_Z2,training=is_train)
    layer3_A2    = tf.nn.relu(layer3_BN)
    layer3_MP    = tf.nn.max_pool(layer3_A2,ksize=[1,2,2,1],strides=[1,4,4,1],padding='VALID')
    # BLOCK 4: CONV2D -> RELU -> CONV2D -> BN-> RELU -> MaxPool
    layer4_Z1    = tf.nn.conv2d(layer3_MP,W7,strides=[1,1,1,1],padding='SAME')
    layer4_BN    = tf.layers.batch_normalization(layer4_Z1,training=is_train)
    layer4_A2    = tf.nn.relu(layer4_BN)
    layer4_MP    = tf.nn.max_pool(layer4_A2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    # FLATTEN
    layer4_FLAT  = tf.contrib.layers.flatten(layer3_MP)
    # Fully connected layers
    layer5_FC1   = tf.contrib.layers.fully_connected(layer4_FLAT, 108, activation_fn=tf.nn.relu)
    layer6_FC2   = tf.contrib.layers.fully_connected(layer5_FC1, 28, activation_fn=None)
    return layer6_FC2

#--------------------------------------- Loss Fuctions ----------------------------------------------

def multitask_heads(mlt,weights,labels):
    ''' Headers for each task, computes predicted probabilities and task specific loss functions'''
    predict_proba = [0]*28
    loss = 0
    for i,w in enumerate(weights):
        predict_proba[i] = tf.nn.sigmoid(mlt[i])
        task_label       = tf.expand_dims(tf.gather(labels,indices=i,axis=1),axis=1)
        loss += tf.reduce_mean(tf.pow(tf.constant(w,dtype=tf.float32),labels[:,i])*tf.nn.sigmoid_cross_entropy_with_logits(logits=mlt[i], labels=task_label))
    return predict_proba, loss
                                                                
def multitask_heads_focal(mlt,weights,labels,gamma=2):
    ''' Focal Loss '''
    predict_proba = [0]*28
    loss          = 0
    for i in range(28):
        predict_proba[i] = tf.nn.sigmoid(mlt[i])
        task_label       = tf.expand_dims(tf.gather(labels,indices=i,axis=1),axis=1)
        focal_loss_multiplier  = tf.where(tf.equal(task_label,tf.constant(1.,dtype=tf.float32)),tf.pow(1-predict_proba[i],gamma),tf.pow(predict_proba[i],gamma))
        loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=mlt[i], labels=task_label)*focal_loss_multiplier)
    return predict_proba, loss   

def multitask_heads_weighted_focal(mlt,weights,labels,gamma=2):
    ''' Weighted Focal Loss '''
    predict_proba = [0]*28
    loss          = 0
    for i,w in enumerate(weights):
        predict_proba[i] = tf.nn.sigmoid(mlt[i])
        task_label       = tf.expand_dims(tf.gather(labels,indices=i,axis=1),axis=1)
        focal_loss_multiplier  = tf.where(tf.equal(task_label,tf.constant(1.,dtype=tf.float32)),tf.pow(1-predict_proba[i],gamma),tf.pow(predict_proba[i],gamma))
        weighting_multiplier   = tf.pow(tf.constant(w,dtype=tf.float32),labels[:,i])
        loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=mlt[i], labels=task_label)*focal_loss_multiplier*weighting_multiplier)
    return predict_proba, loss  

def multitask_heads_weighted_focal_different_gammas(mlt,weights,labels,gammas):
    ''' Focal loss with different gammas for different tasks'''
    predict_proba = [0]*28
    loss          = 0
    for i,gamma in enumerate(gammas):
        predict_proba[i] = tf.nn.sigmoid(mlt[i])
        task_label       = tf.expand_dims(tf.gather(labels,indices=i,axis=1),axis=1)
        focal_loss_multiplier = tf.where(tf.equal(task_label,tf.constant(1.,dtype=tf.float32)),tf.pow(1-predict_proba[i],gamma),tf.pow(predict_proba[i],gamma))
        loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=mlt[i], labels=task_label)*focal_loss_multiplier)
    return predict_proba, loss
        
def multitask_heads_weighted_hinge(mlt,weights,labels):
    ''' Weighted Hinge Loss'''
    predict_proba = [0]*28
    labels_transformed = (labels + 1)/2
    loss               = 0
    for i,w in enumerate(weights):
        predict_proba[i] = tf.nn.sigmoid(mlt[i])
        task_label = tf.expand_dims(tf.gather(labels_transformed,indices=i,axis=1),axis=1)
        weighting  = tf.pow(tf.constant(w,dtype=tf.float32),task_label)
        loss      += tf.reduce_mean(weighting*tf.maximum(0., 1 - task_label*mlt[i]))
    return predict_proba, loss


def multitask_heads_weighted_hinge_squared(mlt,weights,labels):
    ''' Squared Weighted Hinge Loss '''
    predict_proba = [0]*28
    labels_transformed = (labels+1)/2
    loss          = 0
    for i,w in enumerate(weights):
        predict_proba[i] = tf.nn.sigmoid(mlt[i])
        task_label       = tf.expand_dims(tf.gather(labels_transformed,indices=i,axis=1),axis=1)
        weighting        = tf.pow(tf.constant(w,dtype=tf.float32),task_label)
        loss             = tf.reduce_mean(weighting*tf.square(tf.maximum(0.,1.-task_label*mlt[i])))
    return predict_proba, loss


def _predict_probabilities(x_batch,X,sess,predict_proba,is_train):
    ''' Helper function that is later wrapped in finctools.partial, for batch evaluation'''
    return sess.run(predict_proba,feed_dict={X:x_batch,is_train:False})

def balanced_weights_per_class(targets):
    ''' Computes balancing weights for imbalanced classes'''
    pos_weights = np.sum(targets,axis=0)
    neg_weights = targets.shape[0] - pos_weights
    return neg_weights/pos_weights
from functools import partial
import imgaug as ia
from imgaug import augmenters as iaa
tf.reset_default_graph()

BATCH       = 27 # does not include augmentation
RESIZE_ROWS = 3.2
RESIZE_COLS = 3.2
W, H        = 512, 512
RESIZED_W   = int(W/RESIZE_ROWS)
RESIZED_H   = int(H/RESIZE_COLS)
CHANNELS    = ['green','red','blue','yellow']
LOSSES      = ['WBCE','Focal','WeightedFocal','WeightedHinge','WeightedHingeSquared']
LOSS_TO_USE = 'Focal'
# Baseline architecture has more than 7 million weights, taking into account number of examples we have in our 
# dataset we suspect that we are overfitting
ARCHITECTURE = 'Baseline'
AUGMENT     = True

# Step 0: Compute balancing weights
balanced_weights = balanced_weights_per_class(labels_train)

# Step 1: Model definition
with tf.device('/device:GPU:0'):
    #tf.reset_default_graph()   
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    X, Y, is_train             = create_placeholders(height=RESIZED_H, width=RESIZED_W,channels=len(CHANNELS))
    
    # choose architecture
    if ARCHITECTURE == 'Baseline':
        parameters             = initialize_parameters(channels=len(CHANNELS))
        shared_layers          = forward_propagation_shared_layers(X, parameters, is_train)
        mlt_logits             = forward_propagation_task_specific_layers(shared_layers)
    elif ARCHITECTURE == 'Version1':
        parameters             = initialize_parameters_v1(channels=len(CHANNELS))
        shared_layers          = forward_propagation_shared_layers_v1(X, parameters, is_train)
        mlt_logits             = forward_propagation_task_specific_layers(shared_layers)
    elif ARCHITECTURE == 'Version2':
        # this one is very similar to baseline however does not have task specific layers
        parameters             = initialize_parameters(channels=len(CHANNELS))    
        shared_layers          = forward_propagation_shared_layers(X, parameters, is_train)
        last_layer             = tf.contrib.layers.fully_connected(shared_layers, 28, activation_fn=None)
        mlt_logits             = [tf.expand_dims(tf.gather(last_layer,indices=i,axis=1),axis=-1) for i in range(28)]
        
    # choose loss 
    if LOSS_TO_USE == 'Focal':
        predict_proba, loss    = multitask_heads_focal(mlt_logits,balanced_weights,Y, gamma=6)
    if LOSS_TO_USE == 'FocalDifferentGammas':
        gammas                 = compute_task_specific_gamma(labels)
        predict_proba, loss    = multitask_heads_weighted_focal_different_gammas(mlt_logits,balanced_weights,Y,gammas)
    elif LOSS_TO_USE == 'WBCE':
        predict_proba, loss    = multitask_heads(mlt_logits,balanced_weights,Y)
    elif LOSS_TO_USE == 'WeightedFocal':
        predict_proba, loss    = multitask_heads_weighted_focal(mlt_logits,balanced_weights,Y)
    elif LOSS_TO_USE == 'WeightedHinge':
        predict_proba, loss    = multitask_heads_weighted_hinge(mlt_logits,balanced_weights,Y)
    elif LOSS_TO_USE == 'WeightedHingeSquared':
        predict_proba, loss    = multitask_heads_weighted_hinge_squared(mlt_logits,balanced_weights,Y)
    
    predict_probabilities  = partial(_predict_probabilities,X=X,sess=sess,predict_proba=predict_proba,is_train=is_train)
    #loss                   = tf.reduce_sum(weighted_task_losses)
    optimizer              = tf.train.AdamOptimizer(learning_rate=0.001)
    # account for batch normalization layers
    if ARCHITECTURE == 'Version1':
        update_ops             = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            opt = optimizer.minimize(loss)
    else:
        opt = optimizer.minimize(loss)
    init                   = tf.global_variables_initializer()
    sess.run(init)

# Step 2: Run Model for specified 
n_epochs = 15
loss_values_by_iteration = []
for j in range(n_epochs):
    #out = evaluate(paths_val,labels_val,predict_probabilities,batch_size=BATCH,row_resize_ratio=RESIZE_ROWS,
    #               col_resize_ratio=RESIZE_COLS,width=W,height=H,channels_to_include=['green'])
    #print('\n Epoch eval {0} \n'.format(out))
    print('****** Epoch {0} started! *****'.format(j))
    train_gen = train_generator_single_epoch(paths_train,labels_train,batch_size=BATCH,row_resize_ratio=RESIZE_ROWS,
                                             col_resize_ratio=RESIZE_COLS,width=W,height=H,channels_to_include=CHANNELS)
    for i,(batch_images,batch_labels) in enumerate(train_gen): 
        if AUGMENT:
            augmenter = iaa.Sequential([ iaa.OneOf([ iaa.Affine(rotate=0), iaa.Affine(rotate=90), iaa.Affine(rotate=180),
                                                     iaa.Affine(rotate=270),     iaa.Fliplr(0.5), iaa.Flipud(0.5)])], random_order=True)
            batch_images = np.concatenate((batch_images, augmenter.augment_images(batch_images)), 0)
            batch_labels = np.concatenate((batch_labels, batch_labels), 0)
        # normalize images 
        batch_images = batch_images/255
        _, loss_val = sess.run([opt,loss], feed_dict={X:batch_images,Y:batch_labels,is_train:True})
        if i%100==0:
            print('Iteration {0}, value of loss function {1}'.format(i,loss_val))
        loss_values_by_iteration.append(loss_val)
probs = eval_predict_proba_generator(paths_val,predict_probabilities,batch_size=BATCH,row_resize_ratio=RESIZE_ROWS,
                                     col_resize_ratio=RESIZE_COLS,width=W,height=H,channels_to_include=CHANNELS)
def predict_from_probs(predict_proba,cutoffs):
    y_hat = np.zeros(predict_proba.shape)
    for j,cutoff in enumerate(cutoffs):
        y_hat[predict_proba[:,j]>cutoff,j]=1
    return y_hat

def evaluate_given_thresholds(probs,thresholds,labels):
    ''' Evaluate '''
    y_preds   = np.zeros(probs.shape)
    for j,thresh in enumerate(thresholds):
        y_preds[probs[:,j]>thresh,j] = 1
    f1_macro = f1_score(labels,y_preds,average='macro')
    return f1_macro

# 1) Strategy 1: Search single constant threshold
best_threshold  = None
best_macro_f1   = float('-inf')
n_trials        = 100
thresholds      = np.linspace(0,1,n_trials)
for i in range(n_trials):
    threshold = thresholds[i]*np.ones(28)
    f1_new = evaluate_given_thresholds(probs,threshold,labels_val)
    if f1_new > best_macro_f1:
        print('New best F-1 {0}'.format(f1_new))
        best_threshold = threshold
        best_macro_f1  = f1_new
print('Best F-1 macro achieved with first strategy is {0}'.format(best_macro_f1))
print(best_threshold)
    
# 2) Trying to improve upon startegy 1: Search individual thresholds that corrspond to proportion of labels in train set
trials_per_column = 100
true_props      = np.sum(labels_train,axis=0) / labels_train.shape[0]
col_indexes     = np.argsort(true_props)
threshs = np.linspace(0,1,trials_per_column)
for col in col_indexes:
    best_thresh_val = best_threshold[col]
    for t in threshs:
        best_threshold[col] = t
        f1 = evaluate_given_thresholds(probs,best_threshold,labels_val)
        if f1 > best_macro_f1:
            best_thresh_val = t
            best_macro_f1   = f1
            print('New Best F-1 is {0}'.format(best_macro_f1))
    best_threshold[col] = best_thresh_val    

    
print('FINAL BEST F-1 is {0}'.format(evaluate_given_thresholds(probs,best_threshold,labels_val)))
import tqdm

submit           = pd.read_csv('../input/sample_submission.csv')
path_test_data,_ = test_data()
submission       = []
probs_test_data  = eval_predict_proba_generator(path_test_data,predict_probabilities,batch_size=BATCH,row_resize_ratio=RESIZE_ROWS,
                                                 col_resize_ratio=RESIZE_COLS,width=W,height=H,channels_to_include=CHANNELS)
label_predict    = predict_from_probs(probs_test_data,best_threshold)
for j in range(label_predict.shape[0]):
    pos_labels   = np.where(label_predict[j,:]==1)[0]
    submission.append(' '.join(str(l) for l in pos_labels))
submit['Predicted'] = submission
np.save('draw_predict_proba_baseline.npy', probs_test_data)
submit.to_csv('submit_baseline.csv', index=False)
# record loss function
loss_function_values = np.asarray(loss_values_by_iteration)
np.save('loss_function_values.npy',loss_function_values)
lv_save = np.asarray(loss_values_by_iteration)
np.save('loss_values.npy',lv_save)
best_f1 = np.array([best_f1])
np.save('val_best_f1.npy',best_f1)
from sklearn.ensemble import RandomForestClassifier

label_predictors = [0]*28
inclusion_mask = np.array([True]*28)
models = [0]*28

for j in range(28):
    model = RandomForestClassifier
    