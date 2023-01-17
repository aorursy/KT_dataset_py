import numpy as np

import pandas as pd

import cv2

import csv



import tensorflow as tf

from keras.engine.input_layer import Input

from keras.layers import *

from keras.models import Model

import keras



def cnn_model():

    inp = tf.keras.Input(shape=(28,28,1))

    x1 = tf.keras.layers.Conv2D(128, (1,1), strides=(1,1), activation='relu')(inp)

    x1 = tf.keras.layers.BatchNormalization()(x1)

    x3 = tf.keras.layers.Conv2D(128, (3,3), padding='same', strides=(1,1), activation='relu')(inp)

    x3 = tf.keras.layers.BatchNormalization()(x3)

    x5 = tf.keras.layers.Conv2D(128, (5,5), padding='same', strides=(1,1), activation='relu')(inp)

    x5 = tf.keras.layers.BatchNormalization()(x5)



    averaged = tf.keras.layers.Average()([x1,x3,x5])

    averaged = tf.keras.layers.Activation('relu')(averaged)

    averaged = tf.keras.layers.BatchNormalization()(averaged)



    x = tf.keras.layers.Conv2D(128, (5,5), strides=(1,1), activation='relu')(averaged)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Dropout(rate=0.25)(x)

    first_block_out = tf.keras.layers.Flatten()(x)





    x1 = tf.keras.layers.Conv2D(128, (1,1), strides=(1,1), activation='relu')(x)

    x1 = tf.keras.layers.BatchNormalization()(x1)

    x3 = tf.keras.layers.Conv2D(128, (3,3), padding='same', strides=(1,1), activation='relu')(x)

    x3 = tf.keras.layers.BatchNormalization()(x3)

    x5 = tf.keras.layers.Conv2D(128, (5,5), padding='same', strides=(1,1), activation='relu')(x)

    x5 = tf.keras.layers.BatchNormalization()(x5)



    averaged = tf.keras.layers.Average()([x1,x3,x5])

    averaged = tf.keras.layers.Activation('relu')(averaged)

    averaged = tf.keras.layers.BatchNormalization()(averaged)



    x = tf.keras.layers.Conv2D(128, (3,3), strides=(1,1), activation='relu')(averaged)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Dropout(rate=0.25)(x)

    second_block_out = tf.keras.layers.Flatten()(x)



    x1 = tf.keras.layers.Conv2D(256, (1,1), strides=(1,1), activation='relu')(x)

    x1 = tf.keras.layers.BatchNormalization()(x1)

    x3 = tf.keras.layers.Conv2D(256, (3,3), padding='same', strides=(1,1), activation='relu')(x)

    x3 = tf.keras.layers.BatchNormalization()(x3)

    x5 = tf.keras.layers.Conv2D(256, (5,5), padding='same', strides=(1,1), activation='relu')(x)

    x5 = tf.keras.layers.BatchNormalization()(x5)



    averaged = tf.keras.layers.Average()([x1,x3,x5])

    averaged = tf.keras.layers.Activation('relu')(averaged)

    averaged = tf.keras.layers.BatchNormalization()(averaged)



    x = tf.keras.layers.Conv2D(256, (1,1), strides=(1,1), activation='relu')(averaged)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Dropout(rate=0.25)(x)

    third_block_out = tf.keras.layers.Flatten()(x)

    

    x1 = tf.keras.layers.Conv2D(256, (1,1), strides=(1,1), activation='relu')(x)

    x1 = tf.keras.layers.BatchNormalization()(x1)

    x3 = tf.keras.layers.Conv2D(256, (3,3), padding='same', strides=(1,1), activation='relu')(x)

    x3 = tf.keras.layers.BatchNormalization()(x3)

    x5 = tf.keras.layers.Conv2D(256, (5,5), padding='same', strides=(1,1), activation='relu')(x)

    x5 = tf.keras.layers.BatchNormalization()(x5)



    averaged = tf.keras.layers.Average()([x1,x3,x5])

    averaged = tf.keras.layers.Activation('relu')(averaged)

    averaged = tf.keras.layers.BatchNormalization()(averaged)



    x = tf.keras.layers.Conv2D(256, (1,1), strides=(1,1), activation='relu')(averaged)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Dropout(rate=0.25)(x)

    fourth_block_out = tf.keras.layers.Flatten()(x)

    

    print(first_block_out.shape, second_block_out.shape, third_block_out.shape, fourth_block_out.shape)

    #x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Concatenate()([second_block_out, third_block_out, fourth_block_out])

    x = tf.keras.layers.Dense(512, activation=tf.nn.relu)(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(256, activation=tf.nn.relu)(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(64, activation=tf.nn.relu)(x)

    x = tf.keras.layers.BatchNormalization()(x)



    

    output = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(x)



    model = tf.keras.Model(inputs=inp, outputs=output)

    return model



# here the paths are

# /kaggle/input/Kannada-MNIST/train.csv

# /kaggle/input/Kannada-MNIST/Dig-MNIST.csv

# /kaggle/input/Kannada-MNIST/test.csv

# /kaggle/input/Kannada-MNIST/sample_submission.csv



# file_paths

train_file = "/kaggle/input/Kannada-MNIST/train.csv"

validate_file = "/kaggle/input/Kannada-MNIST/Dig-MNIST.csv"

test_file = "/kaggle/input/Kannada-MNIST/test.csv"



#train

data = pd.read_csv(train_file)

data_numpy = np.array(data)

print(data_numpy.shape)



# (60000, 785) -> (60000, 1)

# pull one column out and whole image

data_numpy_label = data_numpy[:,0]

data_numpy_image = data_numpy[:,1:]



data_numpy_label = data_numpy_label.reshape(60000, 1)

print(data_numpy_label.shape)

print(data_numpy_image.shape)



# (60000, 784) => (60000, 28, 28)

data_numpy_image_28x28 = data_numpy_image.reshape(60000, 28, 28)

print(data_numpy_image_28x28.shape)



# visualize the images formed

# image -> (0, 255) (unsigned integer 8)



print(type(data_numpy_image_28x28[0][0][0]))

data_numpy_image_28x28 = data_numpy_image_28x28.astype(np.uint8)



print(type(data_numpy_image_28x28[0][0][0]))

# now, (60000, 28, 28) => which is of <class 'numpy.uint8'>



# for cnt, i in enumerate(data_numpy_image_28x28):

#     # just see this is from image processing

#     label = data_numpy_label[cnt][0]

#     #print(label)

#     cv2.imshow("image_window", i)

#     if cv2.waitKey(0) == 27:

#         break

# cv2.destroyAllWindows()



#validate_

validate_data = pd.read_csv(validate_file)

validate_data_numpy = np.array(validate_data)

print(validate_data_numpy.shape)



# (10240, 785) -> (10240, 1)

# pull one column out and whole image

validate_data_numpy_label = validate_data_numpy[:,0]

validate_data_numpy_image = validate_data_numpy[:,1:]



validate_data_numpy_label = validate_data_numpy_label.reshape(10240, 1)

print(validate_data_numpy_label.shape)

print(validate_data_numpy_image.shape)



# (10240, 784) => (10240, 28, 28)

validate_data_numpy_image_28x28 = validate_data_numpy_image.reshape(10240, 28, 28)

print(validate_data_numpy_image_28x28.shape)



validate_data_numpy_image = validate_data_numpy_image.astype(np.float32)

validate_data_numpy_label = validate_data_numpy_label.astype(np.float32)



#test

test_data = pd.read_csv(test_file)

test_data_numpy = np.array(test_data)

print(test_data_numpy.shape)



# (5000, 785) -> (5000, 1)

# pull one column out and whole image

test_data_numpy_label = test_data_numpy[:,0]

test_data_numpy_image = test_data_numpy[:,1:]



test_data_numpy_label = test_data_numpy_label.reshape(-1, 1)

print(test_data_numpy_label.shape)

print(test_data_numpy_image.shape)



# (60000, 784) => (5000, 28, 28)

test_data_numpy_image_28x28 = test_data_numpy_image.reshape(-1, 28, 28)

print(test_data_numpy_image_28x28.shape)



test_data_numpy_image = test_data_numpy_image.astype(np.float32)

test_data_numpy_label = test_data_numpy_label.astype(np.float32)



data_numpy_image_28x28 = data_numpy_image_28x28.astype(np.float32)

data_numpy_image = data_numpy_image.astype(np.float32)



# now what we have,....

# data_numpy_image -> (60000, 784) -> (float32)

# data_numpy_image_28x28 -> (60000, 28, 28) -> (float32)

# data_numpy_label -> (60000, 1) -> (int)





# machine learning starts from here



# 1. Linear regression

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import ElasticNet

from sklearn.svm import LinearSVC





#reg = LinearRegression().fit(data_numpy_image, data_numpy_label)

#reg = ElasticNet(random_state=0).fit(data_numpy_image, data_numpy_label)

reg = LinearSVC(random_state=0, tol=1e-5).fit(data_numpy_image, data_numpy_label)



score = reg.score(data_numpy_image, data_numpy_label)

print(score)







# 1. Linear regression

print(reg.get_params())



predict_label = reg.predict(validate_data_numpy_image)

print(predict_label.shape)



correct = 0

total = 0

for i in range(len(predict_label)):

    pred_label = predict_label[i]

    ground_truth_label = validate_data_numpy_label[i][0]

    #print(round(pred_label),"\t", ground_truth_label)

    

    if int(round(pred_label)) == int(ground_truth_label):

        correct += 1

    total += 1

print("correct / total, %", correct, total, correct*100./total, "%")



test_label = reg.predict(test_data_numpy_image)

print(test_label.shape)



# result = []

# for i in range(len(test_label)):

#     pred_label = test_label[i]

#     output_got = int(round(pred_label))

#     result.append([i, output_got])

# result = np.append(np.array([['id', 'label']]), result, axis=0)

# print(result.shape)



# with open("submission.csv", 'w', newline='') as csvfile:

#     spamwriter = csv.writer(csvfile)

#     for r in result:

#         spamwriter.writerow(r)



from keras.engine.input_layer import Input

from keras.layers import *

from keras.models import Model

import keras



# input : (60000, 28, 28) => output (60000, 1)

x_inp = Input(shape=(28,28,1)) # shape -> (?, 28, 28)

x = Conv2D(filters=64, kernel_size=((3,3)), strides=(1, 1), padding='valid', activation='tanh')(x_inp)

x = Conv2D(filters=128, kernel_size=(3,3), strides=(1, 1), padding='valid', activation='tanh')(x)

x = Conv2D(filters=256, kernel_size=(3,3), strides=(1, 1), padding='valid', activation='tanh')(x)

x = Conv2D(filters=256, kernel_size=(3,3), strides=(1, 1), padding='valid', activation='tanh')(x)



x_flat = Flatten()(x)





x_flat = Dense(256, activation='tanh')(x_flat)

x_flat = Dropout(rate=0.3)(x_flat)

x_flat = Dense(256, activation='tanh')(x_flat)

x_flat = Dropout(rate=0.3)(x_flat)

x4 = Dense(10, activation='softmax')(x_flat)



# upto here, input: (?, 28, 28), output: (?, 10)



#model = Model(inputs=x_inp, outputs=x4)

model = cnn_model()



data_numpy_label_cate = keras.utils.to_categorical(data_numpy_label, num_classes=10, dtype='float32')

data_numpy_image_28x28 = np.expand_dims(data_numpy_image_28x28, axis=-1)



# Sanity check

# what you have till now

print(data_numpy_image_28x28.shape, data_numpy_image_28x28.dtype)

print(data_numpy_label_cate.shape, data_numpy_label_cate.dtype)



# required for the model

print(model.input_shape, model.output_shape)



model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy', 'mse'])



# fit the model

model.fit(x=data_numpy_image_28x28, y=data_numpy_label_cate, batch_size=64, epochs=15, verbose=1, validation_split=0.1)



# sanity check of shapes, of validation data

validate_data_numpy_image_28x28 = validate_data_numpy_image_28x28.astype(np.float32)

validate_data_numpy_image_28x28 = np.expand_dims(validate_data_numpy_image_28x28, axis=-1)

validate_data_numpy_label_cate = keras.utils.to_categorical(validate_data_numpy_label, num_classes=10, dtype='float32')



print(validate_data_numpy_image_28x28.shape, validate_data_numpy_image_28x28.dtype)

print(validate_data_numpy_label_cate.shape, validate_data_numpy_label_cate.dtype)



resultss_validation=model.evaluate(validate_data_numpy_image_28x28, validate_data_numpy_label_cate, verbose=1)

print(model.metrics_names)

print(resultss_validation)



test_data_numpy_image_28x28 = test_data_numpy_image_28x28.astype(np.float32)

test_data_numpy_image_28x28_expnd = np.expand_dims(test_data_numpy_image_28x28, axis=-1)



test_label = model.predict(test_data_numpy_image_28x28_expnd)

print(test_label.shape)



result = []

for i in range(len(test_label)):

    pred_label = np.argmax(test_label[i])

    output_got = int(round(pred_label))

    result.append([i, output_got])

result = np.append(np.array([['id', 'label']]), result, axis=0)

print(result.shape)



with open("submission.csv", 'w', newline='') as csvfile:

    spamwriter = csv.writer(csvfile)

    for r in result:

        spamwriter.writerow(r)
