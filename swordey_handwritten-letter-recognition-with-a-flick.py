import numpy as np

from tensorflow import keras



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, BatchNormalization

from matplotlib import pyplot as plt
from numpy.random import seed

seed(1)
# Image params

img_rows, img_cols = 28, 28



# Data params

letter_file = "../input/emnist/emnist-letters-train.csv"

test_file = "../input/emnist/emnist-letters-test.csv"

num_classes = 37

classes = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'
## Prepare input data

def prep_data(raw):

    y = raw[:, 0]

    out_y = keras.utils.to_categorical(y, num_classes)



    x = raw[:, 1:]

    num_images = raw.shape[0]

    out_x = x.reshape(num_images, img_rows, img_cols, 1)

    out_x = out_x / 255

    return out_x, out_y
## Convert One-Hot-Encoded values back to real values

def decode_label(binary_encoded_label):

    return np.argmax(binary_encoded_label)-1
## Plot an image with it's correct value

def show_img(img,label):

    img_flip = np.transpose(img, axes=[1,0])

    plt.title('Label: ' + str(classes[decode_label(label)]))

    plt.imshow(img_flip, cmap='Greys_r')
## Evaluate model with the test dataset

def eval_model(model,test_x,test_y):

    result = model.evaluate(test_x, test_y)

    print("The accuracy of the model is: ",result[1])

    return result
## Plot the training history

def plot_history(history):

    acc = history.history['acc']

    val_acc = history.history['val_acc']

    loss = history.history['loss']

    val_loss = history.history['val_loss']



    epochs = range(1, len(acc) + 1)



    # "bo" is for "blue dot"

    plt.plot(epochs, loss, 'bo', label='Training loss')

    plt.plot(epochs, acc, 'b', label='Training accuracy')

    # b is for "solid blue line"

    plt.plot(epochs, val_loss, 'ro', label='Validation loss')

    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')

    plt.title('Training and validation loss')

    plt.xlabel('Epochs')

    plt.ylabel('Loss')

    plt.legend()



    plt.show()
letter_data = np.loadtxt(letter_file, skiprows=1, delimiter=',')

x, y = prep_data(letter_data)
print(x.shape)

print(y.shape)
fig = plt.figure(figsize=(17,4.5))

for idx in range(30):

    fig.add_subplot(3,10,idx+1)

    plt.axis('off')

    show_img(np.squeeze(x[idx]),y[idx])

plt.subplots_adjust(wspace=0.3, hspace=0.3)
test_data = np.loadtxt(test_file, skiprows=1, delimiter=',')

test_x, test_y = prep_data(test_data)
print(test_x.shape)

print(test_y.shape)
# Create a basic short CNN model

# As a first trial for the project

def create_basic_model():

    batch_size = 64



    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), strides=1,activation='relu'))

    model.add(Conv2D(32, (3, 3), activation='relu', strides=1))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))



    model.compile(loss=keras.losses.categorical_crossentropy,

                  optimizer='adam',

                  metrics=['accuracy'])

    

    return model
batch_size = 128



basic_model = create_basic_model()

basic_history = basic_model.fit(x, y,

          batch_size = batch_size,

          epochs = 10,

          validation_split = 0.2)
plot_history(basic_history)
eval_model(basic_model,test_x,test_y)
# Create a basic short CNN model with regularization

# As a second trial for the project

def create_basic_model_with_reg():

    batch_size = 64



    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), strides=1,activation='relu', input_shape = (28, 28, 1)))

    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), activation='relu', strides=1))

    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))





    model.compile(loss=keras.losses.categorical_crossentropy,

                  optimizer='adam',

                  metrics=['accuracy'])

    

    return model
batch_size = 128



basic_model_reg = create_basic_model_with_reg()

basic_reg_history = basic_model_reg.fit(x, y,

          batch_size = batch_size,

          epochs = 10,

          validation_split = 0.2)
plot_history(basic_reg_history)
eval_model(basic_model_reg,test_x,test_y)
# Create a basic short CNN model with regularization

# As a second trial for the project

def create_basic_model_with_reg2():

    batch_size = 64



    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), strides=1,activation='relu', input_shape = (28, 28, 1)))

    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), activation='relu', strides=1))

    model.add(BatchNormalization())

    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dropout(0.4))

    model.add(Dense(128, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))





    model.compile(loss=keras.losses.categorical_crossentropy,

                  optimizer='adam',

                  metrics=['accuracy'])

    

    return model
batch_size = 128



basic_model_reg2 = create_basic_model_with_reg2()

basic_reg2_history = basic_model_reg2.fit(x, y,

          batch_size = batch_size,

          epochs = 10,

          validation_split = 0.2)
plot_history(basic_history)

plot_history(basic_reg2_history)
eval_model(basic_model_reg2,test_x,test_y)
# Create a more complex model

# As the architectura decision for the project

def create_complex_model(input_size,output_size):

    model = Sequential()



    model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (input_size[0], input_size[1], input_size[2])))

    model.add(BatchNormalization())

    model.add(Conv2D(32, kernel_size = 3, activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.4))



    model.add(Conv2D(64, kernel_size = 3, activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size = 3, activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.4))



    model.add(Conv2D(128, kernel_size = 4, activation='relu'))

    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dropout(0.4))

    model.add(Dense(output_size, activation='softmax'))



    model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer='adam',

              metrics=['accuracy'])

    

    return model
batch_size = 64



complex_model = create_complex_model([img_rows, img_cols,1],len(classes))

complex_history = complex_model.fit(x, y,

          batch_size = batch_size,

          epochs = 15,

          validation_split = 0.1)
plot_history(complex_history)
eval_model(complex_model,test_x,test_y)
data_generator_with_aug = keras.preprocessing.image.ImageDataGenerator(validation_split=.2,

                                            width_shift_range=.1, 

                                            height_shift_range=.1,

                                            rotation_range=10, 

                                            zoom_range=.1)

train_generator = data_generator_with_aug.flow(x, y, subset='training')

validation_data_generator = data_generator_with_aug.flow(x, y, subset='validation')



model_with_aug = create_complex_model([img_rows, img_cols,1],len(classes))



history_with_aug = model_with_aug.fit_generator(train_generator, 

                              steps_per_epoch=20000, epochs=15, # can change epochs to 10

                              validation_data=validation_data_generator)

plot_history(history_with_aug)
eval_model(model_with_aug,test_x,test_y)
batch_size = 32

tengwar_test = "../input/handwritten-tengwar-letters/tengwar/tengwar/test/"

tengwar_train = "../input/handwritten-tengwar-letters/tengwar/tengwar/train/"

input_size = [64,64,3]

output_classes = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
data_generator = keras.preprocessing.image.ImageDataGenerator(validation_split=.2,

                                            width_shift_range=.1, 

                                            height_shift_range=.1,

                                            rotation_range=10, 

                                            zoom_range=.1)



train_generator = data_generator.flow_from_directory(

        tengwar_train,

        target_size=(input_size[0], input_size[1]),

        batch_size=batch_size,

        class_mode='categorical')



validation_generator = data_generator.flow_from_directory(

        tengwar_test,

        target_size=(input_size[0], input_size[1]),

        class_mode='categorical')
tengwar_model = create_complex_model(input_size,len(output_classes))

tengwar_history = tengwar_model.fit_generator(train_generator, 

                              steps_per_epoch=1000, epochs=10,

                              validation_data=validation_generator)

plot_history(tengwar_history)
import glob

import cv2

tengwar_final = "../input/handwritten-tengwar-letters/tengwar/tengwar/output/*"

img_names = [img for img in sorted(glob.glob(tengwar_final+"*.png"))]

print(img_names)

x = np.array([cv2.imread(img) for img in img_names])
plt.imshow(x[5])

plt.show()
result = tengwar_model.predict(x)
out_res = ''

for res in result:

    out_res += output_classes[np.argmax(res)]

print(out_res.lower())
