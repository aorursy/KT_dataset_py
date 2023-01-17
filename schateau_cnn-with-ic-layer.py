#Loading librairies:

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import random



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
#define image_shape:

height = 150

width = 150

channels = 3

image_shape = (height, width, channels)
#Because we'll use a generator with binary classification for the training set, we must pass from 'int' to 'string' for the y_col="category" column

df["category"] = df["category"].replace({0: 'cat', 1: 'dog'}) 
#splitting into training, validation, and test sets:

from sklearn.model_selection import train_test_split



#validation set:

train_df_tmp, validate_df = train_test_split(df, test_size=0.1, random_state=2)

#training and test sets:

train_df, test_df = train_test_split(train_df_tmp, test_size=0.1, random_state=2)



#print the number of samples in each set:

print('Number of samples in train_df:', len(train_df), 

      '\nNumber of samples in validate_df:', len(validate_df)

     ,'\nNumber of samples in test_df:', len(test_df))
#Since i will train and test different models, i choose to reduce the sizes of the training set to save some GPU time.

#Here, i cut train_df in half

train_df = train_df.sample(n=10000).reset_index() # use for fast testing code purpose

validate_df = validate_df.reset_index()

test_df = test_df.reset_index()



#If you want to train on the full set, uncomment the following line:

#train_df = train_df.reset_index()
batch_size = 64

total_train = train_df.shape[0]

total_validate = validate_df.shape[0]

epochs = 100
#Data preprocessing:

from keras.preprocessing.image import ImageDataGenerator



#ImageDataGenerator with data augmentation:

train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest')



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
#Building a network using Batch_normalization for fast training

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, BatchNormalization

from keras import optimizers



BN_model = Sequential(name='BN_model')



#Note: a common practice is to place BN before the activation if RelU is employed. This is questionned by https://arxiv.org/pdf/1905.05928.pdf

BN_model.add(Conv2D(32, (3, 3), input_shape=image_shape, use_bias=False)) #Batchnorm layers already include a bias term.

BN_model.add(BatchNormalization()) 

BN_model.add(Activation("relu"))

BN_model.add(MaxPooling2D((2, 2)))



BN_model.add(Conv2D(64, (3, 3), use_bias=False))

BN_model.add(BatchNormalization())

BN_model.add(Activation("relu"))

BN_model.add(MaxPooling2D((2, 2)))



BN_model.add(Conv2D(128, (3, 3), use_bias=False))

BN_model.add(BatchNormalization())

BN_model.add(Activation("relu"))

BN_model.add(MaxPooling2D((2, 2)))



BN_model.add(Conv2D(128, (3, 3), use_bias=False))

BN_model.add(BatchNormalization())

BN_model.add(Activation("relu"))

BN_model.add(MaxPooling2D((2, 2)))



BN_model.add(Flatten())



BN_model.add(Dense(512, use_bias=False))

BN_model.add(BatchNormalization())

BN_model.add(Activation("relu"))



BN_model.add(Dense(1, use_bias=False))

BN_model.add(BatchNormalization())

BN_model.add(Activation("sigmoid"))



model_name = BN_model.name

BN_model.compile(loss='binary_crossentropy',

                 optimizer=optimizers.RMSprop(lr=1e-3), #Because of BN, we can use larger learning rate

                 metrics=['acc'])



BN_model.summary()
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau



patience_earlystop = 20

patience_ReduceLROnPlateau = 10



filepath = 'best_{0}'.format(model_name) + '-{epoch:03d}-{val_acc:03f}.h5'

mcp = ModelCheckpoint(filepath, monitor='val_loss', mode='min', save_best_only=True, verbose=1)

earlystop = EarlyStopping(monitor='val_loss',

                          mode='min',

                          patience=patience_earlystop,

                          verbose=1)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=patience_ReduceLROnPlateau, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=1e-5)

callbacks = [mcp, earlystop, learning_rate_reduction]
#Fitting the model using a batch generator:

history = BN_model.fit_generator(

    train_generator,

    steps_per_epoch=total_train//batch_size,

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=total_validate//batch_size,

    callbacks=callbacks)
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

test_loss, test_acc = BN_model.evaluate_generator(test_generator, steps=np.ceil(nb_samples/batch_size))

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

predict = BN_model.predict_generator(real_test_generator, steps=np.ceil(nb_samples/batch_size))
real_test_df['category'] = predict.round().astype(int)
real_test_df['category'].value_counts().plot.bar()
submission_df = real_test_df.copy()

submission_df['id'] = submission_df['filename'].str.split('.').str[0]

submission_df['label'] = submission_df['category']

submission_df.drop(['filename', 'category'], axis=1, inplace=True)

submission_df.to_csv('submission_BN.csv', index=False)
#Building a networl with IC layers:

from keras import Input

from keras.models import Model

from keras.constraints import max_norm

from keras.layers import Dropout



max_norm4 = max_norm(max_value=4, axis=[0, 1, 2])



def IC(inputs, p):

    x = BatchNormalization()(inputs)

    x = Dropout(p)(x)

    return x



input_tensor = Input(shape=image_shape)



x = Activation("relu")(input_tensor)

x = IC(x, 0.03) #We use a very small value of dropout, as advised in Chen et al. (2019)

x = Conv2D(32, (3, 3), use_bias=False, kernel_constraint=max_norm4)(x)

x = MaxPooling2D((2, 2))(x)



x = Activation("relu")(x)

x = IC(x, 0.03)

x = Conv2D(64, (3, 3), use_bias=False, kernel_constraint=max_norm4)(x)

x = MaxPooling2D((2, 2))(x)



x = Activation("relu")(x)

x = IC(x, 0.03)

x = Conv2D(64, (3, 3), use_bias=False, kernel_constraint=max_norm4)(x)

x = MaxPooling2D((2, 2))(x)



x = Activation("relu")(x)

x = IC(x, 0.03)

x = Conv2D(128, (3, 3), use_bias=False, kernel_constraint=max_norm4)(x)

x = MaxPooling2D((2, 2))(x)



x = Flatten()(x)



x = Activation("relu")(x)

x = IC(x, 0.03)

x = Dense(512, use_bias=False, kernel_constraint=max_norm(4))(x)



output_tensor = Dense(1, activation="sigmoid")(x)



IC_model = Model(input_tensor, output_tensor)

model_name = IC_model.name



IC_model.compile(loss='binary_crossentropy',

                 optimizer=optimizers.rmsprop(lr=1e-3),

                 metrics=['acc'])



IC_model.summary()
#Here, we just modify the patience value:

patience_earlystop = 30

patience_ReduceLROnPlateau = 10



earlystop = EarlyStopping(monitor='val_loss',

                          mode='min',

                          patience=patience_earlystop,

                          verbose=1)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=patience_ReduceLROnPlateau, 

                                            verbose=1, 

                                            factor=0.1, 

                                            min_lr=1e-5)
#Fitting the model using a batch generator:

history = IC_model.fit_generator(

    train_generator,

    steps_per_epoch=total_train//batch_size,

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=total_validate//batch_size,

    callbacks=callbacks)
#Visualization:

plot_smoothed_learning_curves(history)
#Evaluate the model:

test_loss, test_acc = IC_model.evaluate_generator(test_generator, steps=np.ceil(nb_samples/batch_size))

print('test acc:', test_acc)
#Prediction:

predict = IC_model.predict_generator(real_test_generator, steps=np.ceil(nb_samples/batch_size))
real_test_df['category'] = predict.round().astype(int)
real_test_df['category'].value_counts().plot.bar()
submission_df = real_test_df.copy()

submission_df['id'] = submission_df['filename'].str.split('.').str[0]

submission_df['label'] = submission_df['category']

submission_df.drop(['filename', 'category'], axis=1, inplace=True)

submission_df.to_csv('submission_IC.csv', index=False)