# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#importing libraries

import tensorflow as tf

from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test =  pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
print(train.shape)

train.head()
print(test.shape)

test.head()
#distribution of label class in training dataset

train['label'].value_counts()
def load_mnist():

    Y = train['label'].values #returns numpy array of shape (42000,)

    X = train.drop(columns='label').values #returns numpy array of shape (42000,784)

    

    X = X.reshape(-1,28,28,1).astype('float32') * 1000000.

    Y = to_categorical(Y,num_classes=10,dtype='float32') #tf.keras.utils.to_categorical

    

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,stratify = Y) #sklearn.model_selection.train_test_split

    return (X_train,Y_train),(X_test,Y_test)
(X_train,Y_train),(X_test,Y_test) = load_mnist()

test = test.values.reshape(-1,28,28,1).astype('float32') / 255.

#for freeing up some space

del train
#printing shapes of training and validation set

print('X_train shape =',X_train.shape)

print('Y_train shape =',Y_train.shape)

print('X_test shape =',X_test.shape)

print('Y_test shape =',Y_test.shape)
#verifying that labels are equally distributed in training and validation set

print(np.unique(np.argmax(Y_train,1),return_counts = True))

print(np.unique(np.argmax(Y_test,1),return_counts = True))
EPSILON = tf.keras.backend.epsilon()



def squasher(vectors):

    squared_norm = tf.math.reduce_sum(tf.square(vectors), axis = -1,keepdims =True)

    scalar_factor = squared_norm / (1 + squared_norm ) / tf.math.sqrt(squared_norm + EPSILON)

    squashed = scalar_factor * vectors

    return(squashed)
def compute_length(vectors):

    vectors_length = tf.math.sqrt(tf.math.reduce_sum(tf.square(vectors), axis = -1))

    return vectors_length
#Capsule Layer where input are capsules and ouput are capsules

#Dynamic routing is performed herein

class CapsuleLayer(tf.keras.layers.Layer):

    

    def __init__(self, num_capsules, dim_vector, num_routing = 3,kernel_initializer='glorot_uniform',**kwargs):

        super(CapsuleLayer, self).__init__(**kwargs)

        self.num_capsules = num_capsules

        self.dim_vector = dim_vector

        self.num_routing = num_routing

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)



    def build(self, inputs_shape):

        assert(len(inputs_shape) >= 3)

        self.input_num_capsules = inputs_shape[1]

        self.input_dim_vector = inputs_shape[2]

        self.W = self.add_weight(shape=[self.num_capsules, self.input_num_capsules,self.input_dim_vector,self.dim_vector],

                                 initializer=self.kernel_initializer,name='W')

        self.built = True



    def call(self, inputs, training=None):

        inputs_expand = tf.expand_dims(inputs, axis = 1)

        inputs_tiled = tf.tile(inputs_expand, [1, self.num_capsules, 1, 1])

        inputs_tiled_expand = tf.expand_dims(inputs_tiled,axis = 3)

        inputs_hat = tf.map_fn(lambda x: tf.squeeze(tf.matmul(x,self.W)), elems=inputs_tiled_expand)



        b = tf.zeros(shape=[tf.keras.backend.shape(inputs_hat)[0], self.num_capsules,1, self.input_num_capsules])

        assert self.num_routing > 0

        for i in range(self.num_routing):

            c = tf.nn.softmax(b, axis=1)

            outputs = squasher(tf.squeeze(tf.matmul(c, inputs_hat),axis = -2))

            if i < self.num_routing - 1:

                b += tf.expand_dims(tf.squeeze(tf.matmul(inputs_hat,tf.expand_dims(outputs,axis = -1)),axis =-1),axis = -2)

        return outputs



    def compute_output_shape(self, input_shape):

        return tuple([None, self.num_capsules, self.dim_vector])



    def get_config(self):

        config = {

            'num_capsule': self.num_capsules,

            'dim_capsule': self.dim_vector,

            'routings': self.num_routing

        }

        base_config = super(CapsuleLayer, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
def PrimaryCaps(inputs,num_capsules,dim_vector,kernel_size,strides,padding):

    conv2 = tf.keras.layers.Conv2D(filters = num_capsules*dim_vector,kernel_size = kernel_size,

                                   strides = strides,padding = padding,name = 'conv2')(inputs)

    conv2_reshaped = tf.keras.layers.Reshape(target_shape = (conv2.shape[1]*conv2.shape[2]*num_capsules,dim_vector),

                                             name = 'conv2_reshaped')(conv2)

    primarycaps = tf.keras.layers.Lambda(squasher,name = 'primarycaps')(conv2_reshaped)

    return primarycaps
def CAPSNET(input_shape,n_class,num_routing = 3):

    

    x = tf.keras.Input(shape = input_shape,name = 'image28x28')

    

    conv1 = tf.keras.layers.Conv2D(filters = 256,kernel_size = 9,strides = 1,padding = 'valid',activation = 'relu',name = 'conv1')(x)

    

    primarycaps = PrimaryCaps(inputs = conv1,num_capsules = 32,dim_vector = 8, kernel_size = 9,strides = 2,padding = 'valid')

    

    hiddencaps = CapsuleLayer(num_capsules = 32,dim_vector=8,num_routing = num_routing,name = 'hiddencaps')(primarycaps)

    

    combined = tf.keras.layers.concatenate([primarycaps,hiddencaps],axis = 1)

    

    digitcaps = CapsuleLayer(num_capsules = n_class, dim_vector=16,num_routing = num_routing,name = 'digitcaps')(combined)

    

    outcaps = tf.keras.layers.Lambda(compute_length,name = 'outcaps')(digitcaps)

    

    model = tf.keras.Model(inputs = x , outputs = outcaps)

    

    return model

model = CAPSNET(input_shape=[28,28,1],n_class = 10)

model.summary()
def margin_loss(y_true, y_pred):

    L = y_true * tf.math.square(tf.keras.backend.maximum(0., 0.9 - y_pred)) + 0.5 * (1 - y_true) * tf.math.square(tf.keras.backend.maximum(0., y_pred - 0.1))

    return tf.math.reduce_mean(tf.math.reduce_sum(L,1))
def train(model, data):

    (x_train, y_train), (x_test, y_test) = data



    log = tf.keras.callbacks.CSVLogger('log.csv')

    checkpoint = tf.keras.callbacks.ModelCheckpoint('weights-{epoch:02d}.h5',

                                           save_best_only=True, save_weights_only=True, verbose=1)

    lr_decay = tf.keras.callbacks.LearningRateScheduler(schedule=lambda epoch: 0.001 * np.exp(-epoch / 10.))

    

    model.compile(optimizer='adam',loss=margin_loss,metrics=[tf.keras.metrics.CategoricalAccuracy()])



    model.fit(x_train, y_train,

              epochs=30,

              batch_size = 64,

              validation_data=[x_test, y_test],

              callbacks=[log, checkpoint, lr_decay])

    

    model.save_weights('trained_model.h5')

    print('Trained model saved at trained_model.h5')
train(model = model,data = ((X_train,Y_train),(X_test,Y_test)))
def show_wrongly_predicted_images(x,y_true):

    y_pred = model.predict(x,batch_size = 64)

    print('y_pred shape --->',y_pred.shape)

    ypred = np.argmax(y_pred,1)

    ytrue = np.argmax(y_true,1)

    confusion_mtx = confusion_matrix(ytrue, ypred)

    print('\n\t\tCONFUSION MATRIX',confusion_mtx,sep='\n')

    images = x[ypred!=ytrue]

    yp = ypred[ypred!=ytrue]

    yt = ytrue[ypred!=ytrue]

    print(f'\nNumber of Images wrongly guessed are {images.shape[0]}\n')

    for i in range(images.shape[0]):

        plt.imshow(images[i].reshape(28,28))

        plt.title(f'Predicted {yp[i]}\nTrue{yt[i]}')

        plt.show()
print('These are image from testing set which are wrongly guessed')

show_wrongly_predicted_images(X_test,Y_test)
print('These are image from training set which are wrongly guessed')

show_wrongly_predicted_images(X_train,Y_train)