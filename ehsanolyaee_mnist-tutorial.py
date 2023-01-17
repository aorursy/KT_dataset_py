import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import Input
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization, average, GlobalAveragePooling2D
from keras.layers import PReLU, add, AveragePooling2D, UpSampling3D
from keras.models import Model
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from IPython.display import Image
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from keras.applications.inception_v3 import InceptionV3

print(os.listdir("../input/"))
print(os.listdir("../"))

# create the training & test sets, skipping the header row with [1:]
train = pd.read_csv("../input/digit-recognizer/train.csv")
print(train.shape)
train.head()
test= pd.read_csv("../input/digit-recognizer/test.csv")
print(test.shape)
test.head()
x_train = (train.iloc[:,1:].values).astype('float32') # all pixel values
y_train = (train.iloc[:,0].values).astype('int32') # only labels i.e targets digits
x_test = test.values.astype('float32')
print(x_train.shape)
print(x_test.shape)
#expand 1 more dimention as 1 for colour channel gray
x_train = x_train.reshape(x_train.shape[0], 28, 28,1)
print(x_train.shape)
x_test = x_test.reshape(x_test.shape[0], 28, 28,1)
print(x_test.shape)
y_train= to_categorical(y_train)
num_classes = y_train.shape[1]
y_train.shape
for i in range(6, 9):
    plt.subplot(330 + (i+1))
    plt.imshow(x_train[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
    plt.title(y_train[i]);
# Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0


# Set the random seed
random_seed = 2
# Split the train and the validation set for the fitting
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=random_seed)

# Create a model
input = Input(shape = (28,28,1), dtype='float32', name='m1input')
input_inception = Input(shape = (28,28,3), dtype='float32', name='m3input')

# Create a model
# lenet_input = Input(shape = (28,28,1), dtype='float32', name='m1input')

# Add the first convolution layer
x = Conv2D(filters=20, 
           kernel_size=(5, 5), 
           activation='relu', 
           input_shape=(28, 28, 1), 
           data_format='channels_last', 
           name='lenet_conv1',
           padding="same")(input)
# Add a pooling layer
x = MaxPooling2D(pool_size=(2, 2),
                strides =  (2, 2), 
                name='lenet_maxpool1')(x)

# Add the second convolution layer
x = Conv2D(filters = 50,
           kernel_size = (5, 5), 
           activation='relu',
           padding = "same",
           name='lenet_conv2')(x)

# Add a second pooling layer
x =MaxPooling2D(pool_size=(2, 2),
                strides =  (2, 2), 
                name='lenet_maxpool2')(x)

# Flatten the network
x = Flatten(name='lenet_flatten')(x)


# Add a fully-connected hidden layer
x =Dense(500, activation='relu', name='lenet_dense1')(x)

# Add a fully-connected output layer
prediction_lenet = Dense(10, activation='softmax', name='lenet_output')(x)

def name_builder(type, stage, block, name):
        return "{}{}{}_branch{}".format(type, stage, block, name)
    
def identity_block(input_tensor, kernel_size, filters, stage, block):
    F1, F2, F3 = filters

    def name_fn(type, name):
        return name_builder(type, stage, block, name)

    x = Conv2D(F1, (1, 1), name=name_fn('res', '2a'))(input_tensor)
    x = BatchNormalization(name=name_fn('bn', '2a'))(x)
    x = PReLU()(x)

    x = Conv2D(F2, kernel_size, padding='same', name=name_fn('res', '2b'))(x)
    x = BatchNormalization(name=name_fn('bn', '2b'))(x)
    x = PReLU()(x)

    x = Conv2D(F3, (1, 1), name=name_fn('res', '2c'))(x)
    x = BatchNormalization(name=name_fn('bn', '2c'))(x)
    x = PReLU()(x)

    x = add([x, input_tensor])
    x = PReLU()(x)

    return x
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    def name_fn(type, name):
        return name_builder(type, stage, block, name)

    F1, F2, F3 = filters

    x = Conv2D(F1, (1, 1), strides=strides, name=name_fn("res", "2a"))(input_tensor)
    x = BatchNormalization(name=name_fn("bn", "2a"))(x)
    x = PReLU()(x)

    x = Conv2D(F2, kernel_size, padding='same', name=name_fn("res", "2b"))(x)
    x = BatchNormalization(name=name_fn("bn", "2b"))(x)
    x = PReLU()(x)

    x = Conv2D(F3, (1, 1), name=name_fn("res", "2c"))(x)
    x = BatchNormalization(name=name_fn("bn", "2c"))(x)

    sc = Conv2D(F3, (1, 1), strides=strides, name=name_fn("res", "1"))(input_tensor)
    sc = BatchNormalization(name=name_fn("bn", "1"))(sc)

    x = add([x, sc])
    x = PReLU()(x)

    return x

# Create a model
# resnet_input = Input(shape = (28,28,1), dtype='float32', name='m2input')

net = ZeroPadding2D((3, 3))(input)
net = Conv2D(64, (7, 7), strides=(2, 2), name="conv1")(net)
net = BatchNormalization(name="bn_conv1")(net)
net = PReLU()(net)
net = MaxPooling2D((3, 3), strides=(2, 2))(net)

net = conv_block(net, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
net = identity_block(net, 3, [64, 64, 256], stage=2, block='b')
net = identity_block(net, 3, [64, 64, 256], stage=2, block='c')

net = conv_block(net, 3, [128, 128, 512], stage=3, block='a')
net = identity_block(net, 3, [128, 128, 512], stage=3, block='b')
net = identity_block(net, 3, [128, 128, 512], stage=3, block='c')
net = identity_block(net, 3, [128, 128, 512], stage=3, block='d')

net = conv_block(net, 3, [256, 256, 1024], stage=4, block='a')
net = identity_block(net, 3, [256, 256, 1024], stage=4, block='b')
net = identity_block(net, 3, [256, 256, 1024], stage=4, block='c')
net = identity_block(net, 3, [256, 256, 1024], stage=4, block='d')
net = identity_block(net, 3, [256, 256, 1024], stage=4, block='e')
net = identity_block(net, 3, [256, 256, 1024], stage=4, block='f')
net = AveragePooling2D((2, 2))(net)

net = Flatten()(net)
prediction_resnet = Dense(10, activation="softmax", name="softmax")(net)
prediction_combined = average([prediction_resnet, prediction_lenet])
# create the base pre-trained model
x = UpSampling3D(size=(3,3,1), data_format="channels_last")(input_inception)
base_model = InceptionV3(weights=None, include_top=False, input_tensor=x)
base_model.load_weights('../input/inceptionv3-weights-notop/inception_v3_weights_tf_dim_ordering_tf_kernels_notop (1).h5')
# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
x = Dense(200, activation='softmax')(x)
perdiction_inceptionv3 = Dense(10, activation='softmax')(x)



# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(x_train)

# Compile the model
lenet_model = Model(inputs=input, outputs=prediction_lenet)
lenet_model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

#plot the model
plot_model(lenet_model, to_file='lenet_model.png')
Image(filename = 'lenet_model.png', width=200, height=200, unconfined=True)
# lenet_model.summary()

#start training
# history = lenet_model.fit(x_train,y_train,validation_split=0.10,epochs=15, batch_size=86, verbose=1)
lenet_history = lenet_model.fit_generator(datagen.flow(x_train,y_train, batch_size=86),
                                    validation_data = (x_val,y_val),
                                    epochs=30, 
                                    callbacks=[learning_rate_reduction])
# Compile the resnet model
resnet_model = Model(inputs=input, outputs=prediction_resnet)
resnet_model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
#plot the model
plot_model(resnet_model, to_file='resnet_model.png')
Image(filename = 'resnet_model.png', width=200, height=200, unconfined=True)
#start training
# history = lenet_model.fit(x_train,y_train,validation_split=0.10,epochs=15, batch_size=86, verbose=1)
resnet_history = resnet_model.fit_generator(datagen.flow(x_train,y_train, batch_size=86),
                                    validation_data = (x_val,y_val),
                                    epochs=30, 
                                    callbacks=[learning_rate_reduction])
print('salam')
combined_model = Model(inputs= input, outputs=prediction_combined)
combined_model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
#plot the model
plot_model(combined_model, to_file='combined_model.png')
Image(filename = 'combined_model.png', width=200, height=200, unconfined=True)
#start training
# history = lenet_model.fit(x_train,y_train,validation_split=0.10,epochs=15, batch_size=86, verbose=1)
combined_history = combined_model.fit_generator(datagen.flow(x_train,y_train, batch_size=86),
                                    validation_data = (x_val,y_val),
                                    epochs=50, 
                                    callbacks=[learning_rate_reduction])
a = np.array([[[1],[2],[3]], [[1],[2],[3]]])
b = np.concatenate((a,a), axis = 2)
print(a.shape, b.shape)
inceptionv3_x_train = np.concatenate((x_train, x_train, x_train), axis=3)
inceptionv3_x_val = np.concatenate((x_val, x_val, x_val), axis=3)
print(inceptionv3_x_val.shape)

inceptionv3_model = Model(inputs= input_inception, outputs=perdiction_inceptionv3)
for layer in base_model.layers:
    layer.trainable = True
inceptionv3_model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

# history = lenet_model.fit(x_train,y_train,validation_split=0.10,epochs=15, batch_size=86, verbose=1)
inceptionv3_history = inceptionv3_model.fit_generator(datagen.flow(inceptionv3_x_train,y_train, batch_size=86),
                                    validation_data = (inceptionv3_x_val,y_val),
                                    epochs=100, 
                                    callbacks=[learning_rate_reduction])
# Plot training & validation accuracy values
plt.plot(lenet_history.history['acc'])
plt.plot(lenet_history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(lenet_history.history['loss'])
plt.plot(lenet_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation accuracy values
plt.plot(resnet_history.history['acc'])
plt.plot(resnet_history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(resnet_history.history['loss'])
plt.plot(resnet_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation accuracy values
plt.plot(combined_history.history['acc'])
plt.plot(combined_history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(combined_history.history['loss'])
plt.plot(combined_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
# Plot training & validation accuracy values
plt.plot(combined_history.history['acc'])
plt.plot(combined_history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(combined_history.history['loss'])
plt.plot(combined_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
# Evaluate test file
val_loss, val_acc = lenet_model.evaluate(x_val,y_val)
print('Test accuracy:', val_acc)
test= pd.read_csv("../input/sample_submission.csv")
print(test.shape)
test.head()
x = np.array(lenet_model.predict(x_test))

# print(x[0,:], np.argmax(x,axis=1))
print(np.argmax(x,axis=1))
ans =np.reshape(np.argmax(x,axis=1),(28000))
print(ans.shape)
test.loc[:28000,'Label'] = np.transpose(np.argmax(x,axis=1))
test.head()

test.to_csv('./lenet_output.csv', index = False)
# Evaluate test file
val_loss, val_acc = resnet_model.evaluate(x_val,y_val)
print('Test accuracy:', val_acc)
test= pd.read_csv("../input/sample_submission.csv")
print(test.shape)
test.head()
x = np.array(resnet_model.predict(x_test))

# print(x[0,:], np.argmax(x,axis=1))
print(np.argmax(x,axis=1))
ans =np.reshape(np.argmax(x,axis=1),(28000))
print(ans.shape)
test.loc[:28000,'Label'] = np.transpose(np.argmax(x,axis=1))
test.head()

test.to_csv('./resnet_output.csv', index = False)
# Evaluate test file
val_loss, val_acc = combined_model.evaluate(x_val,y_val)
print('Test accuracy:', val_acc)

test= pd.read_csv("../input/sample_submission.csv")
print(test.shape)
test.head()

x = np.array(combined_model.predict(x_test))
# print(x[0,:], np.argmax(x,axis=1))
print(np.argmax(x,axis=1))
ans =np.reshape(np.argmax(x,axis=1),(28000))
print(ans.shape)
test.loc[:28000,'Label'] = np.transpose(np.argmax(x,axis=1))
test.head()

test.to_csv('./combi_output.csv', index = False)
test= pd.read_csv("../input/sample_submission.csv")
print(test.shape)
test.head()

x1 = np.array(lenet_model.predict(x_test))
x2 = np.array(resnet_model.predict(x_test))

# print(x[0,:], np.argmax(x,axis=1))
print(np.argmax(x,axis=1))
ans =np.reshape(np.argmax(x,axis=1),(28000))
print(ans.shape)
test.loc[:28000,'Label'] = np.transpose(np.argmax(x,axis=1))
test.head()

test.to_csv('./combined_output.csv', index = False)
print(os.listdir("./"))