# Basic CNN for classifying dogs and cats pictures



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import random

import math



# Input data files are available in the "../input/" directory.

import os

print(os.listdir("../input/dogs-vs-cats"))
#Prepare training data

filenames = os.listdir("../input/dogs-vs-cats/train/train")

categories = []

for filename in filenames:

    category = filename.split('.')[0]

    if category == 'dog':

        categories.append(1)

    else:

        categories.append(0)



print('1=dog; 0=cat')

df = pd.DataFrame({

    'filename': filenames,

    'category': categories

})

df.head()
#How are the pictures distributed ?

df['category'].value_counts().plot.bar()
#See a random sample:

from keras.preprocessing.image import load_img



sample = random.choice(filenames)

image = load_img("../input/dogs-vs-cats/train/train/"+sample)

plt.imshow(image)
#define image shape:

height = 150

width = 150

channels = 3

image_shape = (height, width, channels)
#Because we'll use a generator with binary classification for the training set, we must pass from 'int' to 'string' for the y_col="category" column

df["category"] = df["category"].replace({0: 'cat', 1: 'dog'}) 
#splitting into training and validation sets:

from sklearn.model_selection import train_test_split



#validation set:

train_df_tmp, validate_df = train_test_split(df, test_size=0.1, random_state=2)

#validation set:

train_df, test_df = train_test_split(train_df_tmp, test_size=0.1, random_state=2)



#print the number of samples in each set:

print('Number of samples in train_df:', len(train_df), 

      '\nNumber of samples in validate_df:', len(validate_df),

      '\nNumber of samples in validate_df:', len(test_df)

     )
#Reduce the sizes of the training and validation sets when testing code to save some GPU time.

#validate_df = validate_df.sample(n=1000).reset_index() # use for fast testing code purpose

#train_df = train_df.sample(n=5000).reset_index() # use for fast testing code purpose



#For training on the full set, comment the lines above and uncomment the ones that follows:

train_df = train_df.reset_index()

validate_df = validate_df.reset_index()
batch_size = 32

total_train = train_df.shape[0]

total_validate = validate_df.shape[0]

epochs = 100
#Data preprocessing:

from keras.preprocessing.image import ImageDataGenerator



#ImageDataGenerator with data augmentation:

train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=10,

    width_shift_range=0.1,

    height_shift_range=0.1,

    shear_range=0.1,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest')

#Note: We choose "small" value of the parapeters for Data augmentation.

#Indeed, Data augmentation is done only on the training set, and thus might lead to a validation set containing "easier" cases to predict.



#Generator for training:

train_generator = train_datagen.flow_from_dataframe(

    dataframe=train_df,

    directory='../input/dogs-vs-cats/train/train/', 

    x_col='filename',

    y_col='category',

    class_mode='binary',

    target_size=(height, width),

    batch_size=batch_size

)



#The images outputed by the validation generator should not be augmented:

validation_datagen = ImageDataGenerator(rescale=1./255)



validation_generator = validation_datagen.flow_from_dataframe(

    dataframe=validate_df, 

    directory='../input/dogs-vs-cats/train/train/',

    x_col='filename',

    y_col='category',

    class_mode='binary',

    target_size=(height, width),

    batch_size=batch_size

)
#Displaying some randomly augmented training images

example_df = train_df.sample(n=1).reset_index(drop=True)

example_generator = train_datagen.flow_from_dataframe(

    dataframe=example_df,

    directory='../input/dogs-vs-cats/train/train/', 

    x_col='filename',

    y_col='category',

    class_mode='categorical'

)

plt.figure(figsize=(12, 12))

for i in range(0, 9):

    plt.subplot(3, 3, i+1)

    for X_batch, Y_batch in example_generator:

        image = X_batch[0]

        plt.imshow(image)

        break

plt.tight_layout()

plt.show()
from keras import backend as K



#Definition of the Maxout layer:

def Maxout(inputs, num_units, axis=None):

    """

    Maxout OP as in the paper https://arxiv.org/pdf/1302.4389.pdf

    

    Max pooling is performed on the filter/channel dimension. This can also be

    used after fully-connected layers to reduce number of features.



    Args:

        inputs (Tensor): A NHWC (or NCHW) tensor on which maxout will be performed. The number of channels C has to be known.

        num_units (int): Specifies how many features will remain after the max pooling operation performed in the filter/channel dimension. num_unit must be a multiple of num_channels, so that num_channels=num_unit*K where K stands for the number of layers (in the channel dimension) where the Max pooling operation is done. Typically, K=2, 3, or 4

        axis: The dimension where max pooling will be performed. Default is the last dimension.

        

    Returns:

        A 'Tensor' of shape (B,H,W,num_units) named ``output``.

    

    Raises:

        ValueError: if num_units is not multiple of number of features.

"""

    

    input_shape = inputs.get_shape().as_list()

    ndim = len(input_shape)

    assert ndim == 4

    

    if axis is None:  # Assume that channel is the last dimension

        axis = -1

    num_channels = input_shape[axis]

    assert num_channels is not None 

    

    if num_channels % num_units:

        raise ValueError('number of features({}) is not a multiple of num_units({})'.format(num_channels, num_units))

    

    input_shape[axis] = num_units

    input_shape+= [num_channels // num_units]

    

    outputs = K.reshape(inputs, (-1, input_shape[1], input_shape[2], input_shape[3], input_shape[4]))

    outputs = K.max(outputs, axis=-1, keepdims=False)



    return outputs
import tensorflow as tf



#Testing function Maxout

if __name__ == '__main__':

    with tf.Session() as sess:

        x = tf.Variable(np.random.uniform(size=(1, 25, 10, 500)))

        y = tf.square(x)

        mo = Maxout(x, 50, axis=None)

        sess.run(tf.global_variables_initializer())



        print(mo.eval().shape)
#Building a Maxout network for efficient Dropout

from keras.models import Model

from keras import Input, optimizers

from keras.layers import Dropout, Lambda, Conv2D, MaxPooling2D, Flatten, Dense

from keras.constraints import max_norm



# To adapt the network with Dropout, we'll follow these steps:

# 1. use maxout instead of RelU (https://arxiv.org/pdf/1302.4389.pdf)

# 2. setting a maxnorm constraint on the weights 

# 3. use Stochastic Gradient Descent with high scheduled decaying learning rate, and large momentum



max_norm4 = max_norm(max_value=4, axis=[0, 1, 2])



input_tensor = Input(shape=image_shape)



#We use a week value (p=0.1) for the dropout units that act before Conv2D layers. Two reasons for that: 

# 1) Conv2D layers are not the more prone to overfitting, and 

# 2) it will provide some noisy inputs for the higher fully connected layers.

x = Conv2D(32, (3, 3), padding='same', kernel_constraint=max_norm4)(input_tensor)

x = Lambda(Maxout, arguments={'num_units':16}, name='act1')(x)

x = MaxPooling2D((2, 2))(x)

x = Dropout(0.1)(x)



x = Conv2D(64, (3, 3), padding='same', kernel_constraint=max_norm4)(x)

x = Lambda(Maxout, arguments={'num_units':32}, name='act2')(x)

x = MaxPooling2D((2, 2))(x)

x = Dropout(0.1)(x)



x = Conv2D(128, (3, 3), padding='same', kernel_constraint=max_norm4)(x)

x = Lambda(Maxout, arguments={'num_units':64}, name='act3')(x)

x = MaxPooling2D((2, 2))(x)

x = Dropout(0.1)(x)



x = Conv2D(128, (3, 3), padding='same', kernel_constraint=max_norm4)(x)

x = Lambda(Maxout, arguments={'num_units':64}, name='act4')(x)

x = MaxPooling2D((2, 2))(x)

x = Dropout(0.1)(x)



x = Flatten()(x)



x = Dense(512, activation='relu', kernel_constraint=max_norm(4))(x)

x = Dropout(0.5)(x) #We use p=0.5 for this dropout unit to fight overfitting

output_tensor = Dense(1, activation="sigmoid")(x)



Maxout_model = Model(input_tensor, output_tensor)

model_name = Maxout_model.name



#Dropout is most effective when taking relatively large steps in parameter space: lr=0.01

#Note: be careful with the momentum value: indeed, at the beginning, the optimizer may go in same direction as the gradient (which is good) some long time. 

#However, this may cause a very big momentum if the "momentum" parameter is set too high.

#It can results in "climbing hills" with the optimizer, and thus a possible increase of the training loss.

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) 



Maxout_model.compile(loss='binary_crossentropy',

                     optimizer=sgd,

                     metrics=['acc'])



Maxout_model.summary()
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler



#def step_decay(epoch):

#    initial_lrate = 0.01

#    drop = 0.1

#    epochs_drop = 40.0

#    lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))

#    return lrate

#lrate = LearningRateScheduler(step_decay)



#filepath = 'best_{0}'.format(model_name) + '-{epoch:03d}-{val_acc:03f}.h5'

#mcp = ModelCheckpoint(filepath, monitor='val_loss', mode='min', save_best_only=True, verbose=1)

earlystop = EarlyStopping(monitor='val_loss',

                          mode='min',

                          patience=20,

                          verbose=0)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=10, 

                                            verbose=1, 

                                            factor=0.1, 

                                            min_lr=1e-5)

callbacks = [earlystop, learning_rate_reduction]
#Fitting the model using a batch generator:

history = Maxout_model.fit_generator(

    train_generator,

    steps_per_epoch=total_train//batch_size,

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=total_validate//batch_size,

    callbacks=callbacks,

    verbose=2)
#Define a smooth function to display the training and validation curves

def plot_smoothed_learning_curves(history):

    val_loss = history.history['val_loss']#[-30:-1] #Uncomment if you want to see only the last epochs

    loss = history.history['loss']#[-30:-1]

    acc = history.history['acc']#[-30:-1]

    val_acc = history.history['val_acc']#[-30:-1]

    

    epochs = range(1, len(acc)+1 )

    

    # Plot the loss and accuracy curves for training and validation 

    fig, ax = plt.subplots(2,1, figsize=(12, 12))

    ax[0].plot(epochs, smooth_curve(loss), 'bo', label="Smoothed training loss")

    ax[0].plot(epochs, smooth_curve(val_loss), 'b', label="Smoothed validation loss",axes =ax[0])

    legend = ax[0].legend(loc='best', shadow=True)

    ax[0].set_xlabel('Epochs')

    ax[0].set_ylabel('Loss')



    ax[1].plot(epochs, smooth_curve(acc), 'bo', label="Smoothed training accuracy")

    ax[1].plot(epochs, smooth_curve(val_acc), 'b',label="Smoothed validation accuracy")

    legend = ax[1].legend(loc='best', shadow=True)

    ax[1].set_xlabel('Epochs')

    ax[1].set_ylabel('Accuracy')

    return



def smooth_curve(points, factor=0.8):

    smoothed_points = []

    for point in points:

        if smoothed_points:

            previous = smoothed_points[-1]

            smoothed_points.append(previous*factor + point*(1-factor))

        else:

            smoothed_points.append(point)

    return smoothed_points
# Visualisation:

plot_smoothed_learning_curves(history)
#Define test_generator:

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(

    dataframe = test_df, 

    directory="../input/dogs-vs-cats/train/train/", 

    x_col='filename',

    y_col='category',

    class_mode='binary',

    target_size=(height, width),

    batch_size=batch_size,

    shuffle=False

)



nb_samples = test_df.shape[0]
#Evaluate the model:

test_loss, test_acc = Maxout_model.evaluate_generator(test_generator, steps=np.ceil(nb_samples/batch_size))

print('test acc:', test_acc)
#Load test data:

test_filenames = os.listdir("../input/dogs-vs-cats/test1/test1")

real_test_df = pd.DataFrame({

    'filename': test_filenames

})



real_test_df = real_test_df

nb_samples = real_test_df.shape[0]
#Generator:

real_test_generator = test_datagen.flow_from_dataframe(

    dataframe = real_test_df, 

    directory="../input/dogs-vs-cats/test1/test1/", 

    x_col='filename',

    y_col=None,

    class_mode=None,

    target_size=(height, width),

    batch_size=batch_size,

    shuffle=False

)
#Prediction:

predict = Maxout_model.predict_generator(real_test_generator, steps=np.ceil(nb_samples/batch_size))
real_test_df['category'] = predict.round().astype(int)
real_test_df['category'].value_counts().plot.bar()
submission_df = real_test_df.copy()

submission_df['id'] = submission_df['filename'].str.split('.').str[0]

submission_df['label'] = submission_df['category']

submission_df.drop(['filename', 'category'], axis=1, inplace=True)

submission_df.to_csv('submission_MaxOut.csv', index=False)