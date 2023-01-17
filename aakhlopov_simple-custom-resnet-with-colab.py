from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Lambda
from keras.layers import AveragePooling2D, Input, Flatten, MaxPooling2D, Add, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.models import Model, load_model
import numpy as np
import pandas as pd
import os
from sklearn.utils import shuffle
def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, n_blocks, num_classes=10, n_stages=3,
                avg_pooling_size=4, dropout_rate=0.2):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.

    # Arguments
        input_shape (tensor): shape of input image tensor
        n_blocks (int): number of blocks in core convolutional layers
        num_classes (int): number of classes
        n_stages (int): number of stages
        avg_pooling_size (int): size of average pooling
        dropout_rate (float): rate of droput in final connections

    # Returns
        model (Model): Keras model instance
    """
    # Computed depth from supplied model parameter n
    depth = n_blocks * 6 + 2
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n_blocks+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(n_stages):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample

            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)

            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            # x = keras.layers.add([x, y])
            x = Add()([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    x = AveragePooling2D(pool_size=avg_pooling_size)(x)
    x = Flatten()(x)
    y = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model
from google.colab import drive
drive.mount('/content/drive')
# Project path
root_path = '/content/drive/My Drive/dog-breed-identification/'
# root_path = './'

# Pathes of the train and test images
train_path = root_path + 'train/'
test_path = root_path + 'test/'
labels_csv = pd.read_csv(root_path + 'labels.csv')
labels_csv.head()
if len(os.listdir(train_path)) == len(labels_csv):
    print('Filenames match actual amount of files!')
else:
    print('Filenames do not match actual amount of files, check the target directory.')
train_df = labels_csv.assign(img_path=lambda x: x['id'] +'.jpg')
train_df.head()
n_blocks = 3
n_stages = 5
avg_pooling_size = 4
dropout_rate = 0.2

# Input image dimensions
height = 320
width = 320
color = 3
input_shape = (height, width, color)

# Number of classes
unique_breeds = labels_csv.breed.unique().tolist()
num_classes = len(unique_breeds)

# Model name
model_name = 'dog_breeds'
load_path = root_path + model_name + '.h5'
if os.path.exists(load_path):
    print(f"loading the trained model: {load_path}")
    model = load_model(load_path)
else:
    model = resnet_v1(input_shape=input_shape,
                    n_blocks=n_blocks,
                    num_classes=num_classes,
                    n_stages=n_stages,
                    avg_pooling_size=avg_pooling_size,
                    dropout_rate=dropout_rate)

model.compile(loss='categorical_crossentropy',
            optimizer=Adam(lr=0.0001),
            metrics=['accuracy'])
model.summary()
epochs = 200
batch_size = 32
num_of_train_images = int(0.8 * len(train_df))
# Shuffle training DataFrame.
train_df = shuffle(train_df)

train_datagen = ImageDataGenerator(rotation_range=30,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                rescale=1./255,
                                shear_range=0.2,
                                zoom_range=0.3,
                                horizontal_flip=True,
                                fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(dataframe=train_df[:num_of_train_images],
                                                    shuffle=True,
                                                    directory=train_path,
                                                    x_col='img_path',
                                                    y_col='breed',
                                                    classes=unique_breeds,
                                                    class_mode='categorical',
                                                    target_size=(height, width),
                                                    batch_size=batch_size)

val_generator = val_datagen.flow_from_dataframe(dataframe=train_df[num_of_train_images:],
                                                shuffle=False,
                                                directory=train_path,
                                                x_col='img_path',
                                                y_col='breed',
                                                classes=unique_breeds,
                                                class_mode='categorical',
                                                target_size=(height, width),
                                                batch_size=batch_size)
# We'll stop training if no improvement after some epochs
earlystopper = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1)

# And reduce learning rate when val_accuracy no improvement after some epochs
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=3, 
                            verbose=1, mode='max', min_lr=0.00001)

# Save the best model during the traning
checkpoint = ModelCheckpoint(filepath=root_path + model_name + '.{epoch:02d}-{val_accuracy:.2f}.h5',
                            monitor='val_accuracy',
                            verbose=1,
                            save_best_only=True)

callbacks = [checkpoint, earlystopper, reduce_lr]
model.fit_generator(train_generator,
                    steps_per_epoch=train_generator.n//train_generator.batch_size,
                    epochs=epochs,
                    validation_data=val_generator,
                    validation_steps=val_generator.n//val_generator.batch_size,
                    verbose=1,
                    callbacks=callbacks)
preds_df = pd.DataFrame(columns=['id'] + list(unique_breeds))
preds_df.head()
preds_df['id'] = [path for path in os.listdir(test_path)]
preds_df.head()
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(dataframe=preds_df,
                                                shuffle=False,
                                                directory=test_path,
                                                x_col='id',
                                                y_col=None,
                                                class_mode=None,
                                                target_size=(height, width),
                                                batch_size=1)

test_predictions = model.predict_generator(test_generator, steps = test_generator.n, verbose=1)
preds_df[list(unique_breeds)] = test_predictions
preds_df.head()
preds_df['id'] = preds_df['id'].apply(lambda x: x.split('.')[0])
preds_df.head()
preds_df.to_csv(root_path + 'MySubmission.csv', index=False)