import tensorflow as tf

import pandas as pd

# tf.enable_eager_execution()

import numpy as np
dataframe_testing = pd.read_csv("../input/test-data/test_dataset/test_dataset.csv")

dataframe_training = pd.read_csv("../input/training-set/final_training.csv")

dataframe_training = dataframe_training.reindex(np.random.permutation(dataframe_training.index))
dataframe_training.index
train = dataframe_training[:28960]

t = train.values

dir_path = "../input/imagedata/images/images/"

p = dir_path + t[:,0]
test = dataframe_testing.values

te = dir_path + test[:,0]
testing = np.ndarray.tolist(te)
train.shape

# dir_path = "../input/imagedata/images/images/"

# p = dir_path + train[:,0]
training = np.ndarray.tolist(p)
type(training)
len(training)
dataframe_validation = dataframe_training[28960:]
val = dataframe_validation.values
val.shape
v = dir_path + val[:,0]
validation = np.ndarray.tolist(v)
type(validation)
len(validation)
training_label = np.ndarray.tolist(t[:,1:])

testing_label = np.ndarray.tolist(test[:,1:])

type(testing_label

    )
len(testing_label

   )
validation_label = np.ndarray.tolist(val[:,1:])
len(validation_label)
print(dataframe_validation[:1])
print(validation[0],validation_label[0])
delta = tf.random.uniform(

    [1],

    minval=0,

    maxval=0.5,

    dtype=tf.float32,

    seed=None,

    name=None

)

d = tf.Session().run(delta)
d[0]
def load_and_preprocess_image(path,label):

    delta = tf.random.uniform(

    [1],

    minval=0,

    maxval=1,

    dtype=tf.float32,

    seed=None,

    name=None

)

    delta = delta[0]

    def f1(): 

        return tf.cond(delta<0.25,lambda:image,lambda:tf.image.adjust_brightness(

    image,

    0.1

))





    def f2(): return tf.cond(delta>0.75,lambda:tf.image.adjust_contrast(image,0.5),lambda:tf.image.adjust_gamma(image,0.95))

    

    image = tf.read_file(path)

    image = tf.image.decode_png(image, channels=3)

    image = tf.image.convert_image_dtype(image, tf.float32)

    image = tf.reshape(image,[1,480,640,3])

    image = tf.image.resize_nearest_neighbor(image,[256,256])

    image = tf.reshape(image,[256,256,3])

    image = tf.cond(delta>0.5,f2,f1)

    image = tf.image.convert_image_dtype(image, tf.float32)

    return image,label
with tf.device('/cpu:0'):

    dataset_train = tf.data.Dataset.from_tensor_slices((training, training_label))

    dataset_train = dataset_train.repeat()

# dataset = dataset.shuffle(len(filenames))

    dataset_train = dataset_train.map(load_and_preprocess_image, num_parallel_calls=4)

#     dataset_train = dataset_train.cache(filename='/home/shubham/Videos/cache_tf/train_cache')

# dataset = dataset.map(train_preprocess, num_parallel_calls=4)

    dataset_train = dataset_train.batch(32)

    dataset_train = dataset_train.prefetch(1)
with tf.device('/cpu:0'):

    dataset_validation = tf.data.Dataset.from_tensor_slices((validation, validation_label))

    dataset_validation = dataset_validation.repeat()

# dataset = dataset.shuffle(len(filenames))

    dataset_validation = dataset_validation.map(load_and_preprocess_image, num_parallel_calls=4)

#     dataset_validation = dataset_validation.cache(filename='/home/shubham/Videos/cache_tf/val_cache')

# dataset = dataset.map(train_preprocess, num_parallel_calls=4)

    dataset_validation = dataset_validation.batch(32)

    dataset_validation = dataset_validation.prefetch(1)
# from tf.keras.layers import Input,Dropout, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D

# from keras.models import Model, load_model

# from keras.preprocessing import image

# from keras.utils import layer_utils

# from keras.utils.data_utils import get_file

# from keras.utils.vis_utils import model_to_dot

# from keras.utils import plot_model

# from keras.initializers import glorot_uniform

# K.set_image_data_format('channels_last')

# K.set_learning_phase(1)
def identity_block(X, f, filters, stage, block):

    """

    Implementation of the identity block as defined in Figure 3

    

    Arguments:

    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)

    f -- integer, specifying the shape of the middle CONV's window for the main path

    filters -- python list of integers, defining the number of filters in the CONV layers of the main path

    stage -- integer, used to name the layers, depending on their position in the network

    block -- string/character, used to name the layers, depending on their position in the network

    

    Returns:

    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)

    """

    

    # defining name basis

    conv_name_base = 'res' + str(stage) + block + '_branch'

    bn_name_base = 'bn' + str(stage) + block + '_branch'

    

    # Retrieve Filters

    F1, F2= filters

    

    # Save the input value. You'll need this later to add back to the main path. 

    X_shortcut = X

    

    # First component of main path

    X = tf.keras.layers.Conv2D(filters = F1, kernel_size = (3, 3), strides = (1,1), padding = 'same', name = conv_name_base + '2a', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(X)

    X = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)

    X = tf.keras.layers.Activation('relu')(X)



    

    # Second component of main path (≈3 lines)

    X = tf.keras.layers.Conv2D(filters = F2, kernel_size = (3, 3), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(X)

    X = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)

#     X = tf.keras.layers.Activation('relu')(X)



    # Third component of main path (≈2 lines)

#     X = tf.keras.layers.Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(X)

#     X = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)



    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)

    X = tf.keras.layers.Add()([X, X_shortcut])

    X = tf.keras.layers.Activation('relu')(X)

    

    

    return X

def convolutional_block(X, f, filters, stage, block, s = 2):

    """

    Implementation of the convolutional block as defined in Figure 4

    

    Arguments:

    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)

    f -- integer, specifying the shape of the middle CONV's window for the main path

    filters -- python list of integers, defining the number of filters in the CONV layers of the main path

    stage -- integer, used to name the layers, depending on their position in the network

    block -- string/character, used to name the layers, depending on their position in the network

    s -- Integer, specifying the stride to be used

    

    Returns:

    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)

    """

    

    # defining name basis

    conv_name_base = 'res' + str(stage) + block + '_branch'

    bn_name_base = 'bn' + str(stage) + block + '_branch'

    

    # Retrieve Filters

    F1, F2,F3 = filters

    

    # Save the input value

    X_shortcut = X





    ##### MAIN PATH #####

    # First component of main path 

    X = tf.keras.layers.Conv2D(F1, (2, 2), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(X)

    X = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)

    X = tf.keras.layers.Activation('relu')(X)



    # Second component of main path (≈3 lines)

    X = tf.keras.layers.Conv2D(filters = F2, kernel_size = (3, 3), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(X)

    X = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)

#     X = tf.keras.layers.Activation('relu')(X)





    # Third component of main path (≈2 lines)

#     X = tf.keras.layers.Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(X)

#     X = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)





    ##### SHORTCUT PATH #### (≈2 lines)

    X_shortcut = tf.keras.layers.Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1',

                        kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(X_shortcut)

    X_shortcut = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)



    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)

    X = tf.keras.layers.Add()([X, X_shortcut])

    X = tf.keras.layers.Activation('relu')(X)

    

    

    return X

# from keras.callbacks import Callback

import os

# import keras.backend as K

# import numpy as np



class SGDRScheduler(tf.keras.callbacks.Callback):

    '''Cosine annealing learning rate scheduler with periodic restarts.

    # Usage

        ```python

            schedule = SGDRScheduler(min_lr=1e-5,

                                     max_lr=1e-2,

                                     steps_per_epoch=np.ceil(epoch_size/batch_size),

                                     lr_decay=0.9,

                                     cycle_length=5,

                                     mult_factor=1.5)

            model.fit(X_train, Y_train, epochs=100, callbacks=[schedule])

        ```

    # Arguments

        min_lr: The lower bound of the learning rate range for the experiment.

        max_lr: The upper bound of the learning rate range for the experiment.

        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`. 

        lr_decay: Reduce the max_lr after the completion of each cycle.

                  Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.

        cycle_length: Initial number of epochs in a cycle.

        mult_factor: Scale epochs_to_restart after each full cycle completion.

    # References

        Blog post: jeremyjordan.me/nn-learning-rate

        Original paper: http://arxiv.org/abs/1608.03983

    '''

    def __init__(self,

                 min_lr,

                 max_lr,

                 steps_per_epoch,

                 lr_decay=1,

                 cycle_length=10,

                 mult_factor=2):



        self.min_lr = min_lr

        self.max_lr = max_lr

        self.lr_decay = lr_decay



        self.batch_since_restart = 0

        self.next_restart = cycle_length



        self.steps_per_epoch = steps_per_epoch



        self.cycle_length = cycle_length

        self.mult_factor = mult_factor

        self.path_format = os.path.join('', 'weights_cycle_{}.h5')



        self.history = {}



    def clr(self):

        '''Calculate the learning rate.'''

        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)

        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))

        return lr



    def on_train_begin(self, logs={}):

        '''Initialize the learning rate to the minimum value at the start of training.'''

        logs = logs or {}

        tf.keras.backend.set_value(self.model.optimizer.lr, self.max_lr)



    def on_batch_end(self, batch, logs={}):

        '''Record previous batch statistics and update the learning rate.'''

        logs = logs or {}

        self.history.setdefault('lr', []).append(tf.keras.backend.get_value(self.model.optimizer.lr))

        for k, v in logs.items():

            self.history.setdefault(k, []).append(v)



        self.batch_since_restart += 1

        tf.keras.backend.set_value(self.model.optimizer.lr, self.clr())



    def on_epoch_end(self, epoch, logs={}):

        '''Check for end of current cycle, apply restarts when necessary.'''

        if epoch + 1 == self.next_restart:

            self.batch_since_restart = 0

            self.model.save_weights(self.path_format.format(self.cycle_length), overwrite=True)

            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)

            self.next_restart += self.cycle_length

            self.max_lr *= self.lr_decay

#             self.model.save_weights(self.path_format.format(self.cycle_length), overwrite=True)

#             self.best_weights = self.model.get_weights()



    def on_train_end(self, logs={}):

        '''Set weights to the values from the end of the most recent cycle for best performance.'''

#         self.model.set_weights(self.best_weights)
# sgdr = SGDRScheduler(min_lr=5e-4,max_lr=5e-2,steps_per_epoch=905,lr_decay=0.5,cycle_length = 1,mult_factor=2)
def metrics_custom(y_true,y_pred):

#     print(type(y_true))

#     print(type(y_pred))

    xi1 = tf.math.maximum(y_pred[:,0],y_true[:,0])

    xi2 = tf.math.minimum(y_pred[:,1],y_true[:,1])

    yi1 = tf.math.maximum(y_pred[:,2],y_true[:,2])

    yi2 = tf.math.minimum(y_pred[:,3],y_true[:,3])

    x_inter = tf.math.maximum(tf.math.subtract(xi2,xi1),0)

    y_inter = tf.math.maximum(tf.math.subtract(yi2,yi1),0)

    area_inter = tf.math.multiply(x_inter,y_inter)

    area_true = tf.math.multiply(tf.math.subtract(y_true[:,1],y_true[:,0]),tf.math.subtract(y_true[:,3],y_true[:,2]))

    area_pred = tf.math.multiply(tf.math.subtract(y_pred[:,1],y_pred[:,0]),tf.math.subtract(y_pred[:,3],y_pred[:,2]))

    iou = tf.math.abs(tf.math.divide(area_inter,tf.math.subtract(tf.math.add(area_true,area_pred),area_inter)))

    ans = tf.math.reduce_mean(iou)

#     print(xi1)

#     print(xi2)

    return ans







def loss_custom(y_true,y_pred):

#     print(type(y_true))

#     print(type(y_pred))

    xi1 = tf.math.maximum(y_pred[:,0],y_true[:,0])

    xi2 = tf.math.minimum(y_pred[:,1],y_true[:,1])

    yi1 = tf.math.maximum(y_pred[:,2],y_true[:,2])

    yi2 = tf.math.minimum(y_pred[:,3],y_true[:,3])

    x_inter = tf.math.maximum(tf.math.subtract(xi2,xi1),0)

    y_inter = tf.math.maximum(tf.math.subtract(yi2,yi1),0)

    area_inter = tf.math.multiply(x_inter,y_inter)

    area_true = tf.math.multiply(tf.math.subtract(y_true[:,1],y_true[:,0]),tf.math.subtract(y_true[:,3],y_true[:,2]))

    area_pred = tf.math.multiply(tf.math.subtract(y_pred[:,1],y_pred[:,0]),tf.math.subtract(y_pred[:,3],y_pred[:,2]))

    iou = tf.math.subtract(tf.constant(1.0,dtype=tf.float32),tf.math.divide(area_inter,tf.math.subtract(tf.math.add(area_true,area_pred),area_inter)))

    ans = tf.math.reduce_sum(iou)

#     print(xi1)

#     print(xi2)

    return ans

def cnn_model():

    input_layer = tf.keras.layers.Input((256, 256, 3))

    use_bias = True 

    # Layer 1

    conv = tf.keras.layers.Conv2D(32,

                                  kernel_size=(3, 3),

                                  padding='same',

                                  use_bias=use_bias,

                                  activation=None,kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(input_layer)

    bn = tf.keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(conv)

    activation = tf.keras.layers.ReLU()(bn)

    

 

    conv = tf.keras.layers.Conv2D(32,

                                  kernel_size=(3, 3),

                                  padding='same',

                                  use_bias=use_bias,

                                  activation=None,kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(activation)

    bn = tf.keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(conv)

    activation = tf.keras.layers.ReLU()(bn)

 

    max_pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(activation)

    

    

    

    X = convolutional_block(max_pool, f=3, filters=[64, 64, 64], stage=2, block='a', s=2)

    X = identity_block(X, 3, [64, 64], stage=2, block='b')

    X = identity_block(X, 3, [64, 64], stage=2, block='c')

   

    X = convolutional_block(X, f=3, filters=[128, 128, 128], stage=3, block='b', s=2)

    

    X = identity_block(X, 3, [128, 128], stage=3, block='d')

#   X = convolutional_block(X, f=3, filters=[128, 128], stage=3, block='b', s=2)

    X = identity_block(X, 3, [128, 128], stage=3, block='e')

    

#     X = identity_block(X, 3, [128, 128], stage=2, block='b')

#     X = identity_block(X, 3, [128, 128], stage=2, block='c')

   



#     X = tf.keras.layers.Conv2D(256, (1, 1), strides = (2,2), name = 'pooling_2' + '2a', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(X)

#     X = tf.keras.layers.BatchNormalization(axis = 3, name = 'pooling_bn_2' + '2a')(X)

#     X = tf.keras.layers.Activation('relu')(X)

#     X = convolutional_block(X, f=3, filters=[256, 256, 256], stage=4, block='b', s=2)

#     X = identity_block(X, 3, [256, 256], stage=4, block='d')

#     X = identity_block(X, 3, [256, 256], stage=4, block='e')

#     X = identity_block(X, 3, [256, 256], stage=4, block='f')

#     X = identity_block(X, 3, [256, 256], stage=3, block='g')

        

#     X = tf.keras.layers.GlobalAvgPool2D()(X)

# #     X = tf.keras.layers.Flatten()(X)

#     X = tf.keras.layers.Dense(4, activation='relu' , name='fc' + str(4), kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(X)

    

  

    model = tf.keras.Model(inputs = input_layer, outputs = X, name='resnet_stage_1')

 

    

    

 

    return model
model_1 = cnn_model()

model_1.summary()

model_1.load_weights("../input/stage-4-res/bmodel1_weight.h5")

model_1.trainable = False

# model_1.summary()
def get_stage_2():

    Input_layer = tf.keras.layers.Input(shape = (256,256,3))

    X = model_1(Input_layer)

#     X = convolutional_block(X, f=3, filters=[128, 128, 128], stage=3, block='b', s=2)

    

    X = identity_block(X, 3, [128, 128], stage=4, block='d')

#   X = convolutional_block(X, f=3, filters=[128, 128], stage=3, block='b', s=2)

    X = identity_block(X, 3, [128, 128], stage=4, block='e')

    X = convolutional_block(X, f=3, filters=[256, 256, 256], stage=4, block='b', s=2)

    

    X = identity_block(X, 3, [256, 256], stage=5, block='d')

#   X = convolutional_block(X, f=3, filters=[128, 128], stage=3, block='b', s=2)

    X = identity_block(X, 3, [256, 256], stage=5, block='e')

    

    model = tf.keras.Model(inputs = Input_layer,outputs = X,name = "second2")

    return model

    
# model_stage_1.layers[-11]

# model_stage_1    = cnn_model()

# model_stage_1.trainable = False

# model_stage_1.summary()

# # model_stage_1.get_weights[1]

# bottleneck_input  = model_stage_1.get_layer(index=0).input

# bottleneck_output = model_stage_1.get_layer(index=-3).output

# bottleneck_model  = tf.keras.Model(inputs=bottleneck_input,outputs=bottleneck_output)

# bottleneck_model.trainable = False

# bottleneck_model.summary()

model_2 = get_stage_2()

# model_2.summary()
model_2.load_weights("../input/stage-4-res/bmodel2_weight.h5")

model_2.summary()
# model_2.trainable =  True

# model_2.summary()
def get_model_3():

    Input_layer = tf.keras.layers.Input(shape = (256,256,3))

    X = model_2(Input_layer)

    X = identity_block(X, 3, [256, 256], stage=5, block='d')

#   X = convolutional_block(X, f=3, filters=[128, 128], stage=3, block='b', s=2)

    X = identity_block(X, 3, [256, 256], stage=5, block='e')

    X = convolutional_block(X, f=3, filters=[512, 512, 512], stage=4, block='b', s=2)

    

    X = identity_block(X, 3, [512, 512], stage=6, block='d')

#   X = convolutional_block(X, f=3, filters=[128, 128], stage=3, block='b', s=2)

    X = identity_block(X, 3, [512, 512], stage=6, block='e')

    

    model = tf.keras.Model(inputs = Input_layer,outputs = X,name = "third3")

    return model

    

    
model_3 = get_model_3()

model_3.summary()
# model_1.trainable = False

# model_2.trainable = False
model_3.load_weights("../input/stage-4-res/bmodel3_weight.h5")
def get_model():

    Input_layer = tf.keras.layers.Input(shape = (256,256,3))

    X = model_3(Input_layer)

    X = tf.keras.layers.Flatten()(X)

#     X = tf.keras.layers.Dense(1024,kernel_initializer=tf.keras.initializers.glorot_uniform,kernel_regularizer=tf.keras.regularizers.l2(0.001))(X)

#     X = tf.keras.layers.Dropout(0.3)(X)

#     X = tf.keras.layers.ReLU()(X)

#     X = tf.keras.layers.Dense(64,kernel_initializer=tf.keras.initializers.glorot_uniform,kernel_regularizer=tf.keras.regularizers.l2(0.001))(X)

#     X = tf.keras.layers.ReLU()(X)

#     X = tf.keras.layers.Dropout(0.)(X)

    X = tf.keras.layers.Dense(4,kernel_initializer=tf.keras.initializers.glorot_uniform,kernel_regularizer=tf.keras.regularizers.l2(0.008))(X)

#     tf.keras.layers.Input((256,256,3)),

    

    model = tf.keras.Model(inputs = Input_layer,outputs = X,name = "first1")

    return model
# model_stage_1.load_weights("../input/weights-stage-1/weights_cycle_8.0.h5")

# model_stage_1.trainable = False
# def model_2():

#     input_layer = input_layer = tf.keras.layers.Input((256, 256, 3))

#     X = 

#     X = identity_block(X, 3, [128, 128], stage=2, block='b')

#     X = identity_block(X, 3, [128, 128], stage=2, block='c')

   



#     X = tf.keras.layers.Conv2D(256, (1, 1), strides = (2,2), name = 'pooling_2' + '2a', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(X)

#     X = tf.keras.layers.BatchNormalization(axis = 3, name = 'pooling_bn_2' + '2a')(X)

#     X = tf.keras.layers.Activation('relu')(X)

#     X = identity_block(X, 3, [256, 256], stage=3, block='d')

#     X = identity_block(X, 3, [256, 256], stage=3, block='e')

#     X = identity_block(X, 3, [256, 256], stage=3, block='f')

#     X = identity_block(X, 3, [256, 256], stage=3, block='g')

    

#     model = tf.keras.Model(inputs = input_layer, outputs = X, name='resnet_stage_2')

 

    

    

 

#     return model

    

    
model = get_model()
model_2.trainable= False

model.load_weights("../input/stage-4-res/bmodel_stage4.h5")

model.summary()

# model.load_weights("../input/stage-2-res/stage_2_resnet/model_stage2.h5")
# model_1.trainable = True
# model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0002), 

                loss=tf.keras.losses.mean_absolute_error,

                metrics=[metrics_custom])
def pre(path,label):

    image = tf.read_file(path)

    image = tf.image.decode_png(image, channels=3)

    image = tf.image.convert_image_dtype(image, tf.float32)

    image = tf.reshape(image,[1,480,640,3])

    image = tf.image.resize_nearest_neighbor(image,[256,256])

    image = tf.reshape(image,[256,256,3])

#     image = tf.cond(delta>0.5,f2,f1)

    image = tf.image.convert_image_dtype(image, tf.float32)

    return image,label



path = tf.placeholder(dtype=tf.string)
with tf.device('/cpu:0'):

    dataset_test = tf.data.Dataset.from_tensor_slices((testing,testing_label))

#     dataset_test = dataset_validation.repeat()

# dataset = dataset.shuffle(len(filenames))

    dataset_test = dataset_test.map(pre, num_parallel_calls=4)

#     dataset_validation = dataset_validation.cache(filename='/home/shubham/Videos/cache_tf/val_cache')

# dataset = dataset.map(train_preprocess, num_parallel_calls=4)

    dataset_test = dataset_test.batch(15)

    dataset_test = dataset_test.prefetch(1)
# next = dataset_test.make_one_shot_iterator().get_next()

# next
ans = model.predict(dataset_test,steps=1603)
test[:,1:] = ans

# test[:1,1:] = ans[:,:]
# test[:15]
# test_path = test[:,0]

# root_path = '../input/imagedata/images/images/'

# final_path = []

# leng = len(test_path)

# for i in range(leng):

#         final_path.append(root_path + test_path[i])

        

# for i in range(100):

#     print(i)

#     pat = final_path[i]

#     with tf.Session() as sess:

#         img = sess.run(pre(path),feed_dict = {path:pat})

#     predict = model.predict_on_batch(img.reshape(1,256,256,3))

#     test[i,1:] = predict[0]

# #     gc.collect()

    

data = pd.DataFrame(test,columns=['image_name', 'x1', 'x2', 'y1', 'y2'])

data.to_csv(path_or_buf="test_output.csv",index = False)
dataf = pd.read_csv("test_output.csv")

dataf
# ans = model.predict_on_batch(dataset_test)

# model.fit(dataset_train,epochs=4,shuffle=False,steps_per_epoch=905,validation_steps=50,validation_data=dataset_validation)
# tf.keras.backend.eval(model.optimizer.lr)
# sgdr = SGDRScheduler(min_lr=5e-4,max_lr=5e-2,steps_per_epoch=1810,lr_decay=0.8,cycle_length = 1,mult_factor=2)
# model.fit(dataset_train,epochs=15,shuffle=False,steps_per_epoch=1810,validation_steps=100,validation_data=dataset_validation,callbacks=[sgd])
# model.save_weights("model_stage4.h5",overwrite=True)
# model_1.save_weights("model1_weight.h5",overwrite=True)
# model_2.save_weights("model2_weight.h5",overwrite=True)
# model_3.save_weights("model3_weight.h5",overwrite=True)