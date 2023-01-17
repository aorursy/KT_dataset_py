### Setting seeds and disabling multithreading to provide reproducible results ###

from os import environ

environ['PYTHONHASHSEED'] = '0'



import random as rn

import numpy as np

import tensorflow as tf



SEED = 1234

np.random.seed(SEED)

rn.seed(SEED)



session_conf = tf.ConfigProto(

    intra_op_parallelism_threads=1,

    inter_op_parallelism_threads=1)

tf.set_random_seed(SEED)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)



from keras import backend as K



K.set_session(sess)



### Necessary imports and settings ###

import functools

import operator

import shutil

import timeit

from os import listdir, mkdir, path

import pandas as pd

from keras.applications.inception_v3 import InceptionV3

from keras.applications.mobilenet import MobileNet

from keras.applications.resnet50 import ResNet50

from keras.applications.vgg16 import VGG16

from keras.applications.xception import Xception

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.initializers import glorot_uniform

from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,

                          Dropout, Flatten, MaxPooling2D)

from keras.losses import binary_crossentropy

from keras.models import Sequential

from keras.optimizers import SGD, Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.regularizers import l2

from matplotlib import image

from matplotlib import pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix



root_dir = "../input/all/All/"  # original path for the dataset (relational, from kernel)
### File with true labels ###

gtruth = pd.read_csv(root_dir + 'GTruth.csv', header=0)

df = pd.DataFrame(gtruth.head())

df
### Distribution of data in classes ###

labels = ('Healthy', 'Pneumonia')

counts = (

    np.count_nonzero(gtruth["Ground_Truth"]),

    np.count_nonzero(gtruth["Ground_Truth"] == 0))



print(counts)



plt.pie(counts, labels=labels, colors=('#BFBFBF', '#808080'), autopct='%1.f%%')

plt.axis('equal')

plt.show()
### Loading dataset sample (first 10 images) ###

sample_size = 10

img_sample = [(fname, image.imread(root_dir + fname))

              for i, fname in enumerate(listdir(root_dir))

              if fname.endswith('.jpeg') and 5 < i < sample_size]



### Show sample images along with their filename and shape ###

rows, columns = 2, 2

fig = plt.figure(figsize=(6, 6))

for index, img in enumerate(img_sample):

    print('Filename:', img[0], ' shape:', img[1].shape)

    fig.add_subplot(rows, columns, index + 1)

    # matplotlib displays single-channel images in greenish color, so it's necessary to choose a gray colormap

    plt.imshow(img[1], cmap=plt.cm.gray)

plt.subplots_adjust(left=1, right=2)

plt.show()
### Creating directories for train-validation-test sets ###

base_dir = '../pneumonia-chest-x-ray'



try:

    mkdir(base_dir)

except FileExistsError:

    shutil.rmtree(base_dir)

    mkdir(base_dir)



train_dir = path.join(base_dir, 'train')

validation_dir = path.join(base_dir, 'validation')

test_dir = path.join(base_dir, 'test')



try:

    mkdir(train_dir)

    mkdir(validation_dir)

    mkdir(test_dir)

except FileExistsError:

    pass



train_1_dir = path.join(train_dir, 'healthy')

train_0_dir = path.join(train_dir, 'pneumonia')

validation_1_dir = path.join(validation_dir, 'healthy')

validation_0_dir = path.join(validation_dir, 'pneumonia')

test_1_dir = path.join(test_dir, 'healthy')

test_0_dir = path.join(test_dir, 'pneumonia')



try:

    mkdir(train_1_dir)

    mkdir(train_0_dir)

    mkdir(validation_1_dir)

    mkdir(validation_0_dir)

    mkdir(test_1_dir)

    mkdir(test_0_dir)

except FileExistsError:

    pass



### Determine lists of id's of images in classes ###

class_0_full = [np.array2string(row[0]) for row in gtruth.values if row[1] == 0]

class_1_full = [np.array2string(row[0]) for row in gtruth.values if row[1] != 0]



### Take first 1280 images from every class ###

class_0 = rn.sample(class_0_full, 1280)

class_1 = rn.sample(class_1_full, len(class_0))



print("Number of images in classes: \nclass 0 - pneumonia:", len(class_0),

      "\nclass 1 - healthy:", len(class_1))
### Splitting the data into train-val-test sets/directories ###

for i, (img_0, img_1) in enumerate(zip(class_0, class_1)):

    fname_0 = img_0 + '.jpeg'

    fname_1 = img_1 + '.jpeg'

    if i < 0.8 * len(class_0):

        shutil.copyfile(path.join(root_dir, fname_0),

                        path.join(train_0_dir, fname_0))

        shutil.copyfile(path.join(root_dir, fname_1),

                        path.join(train_1_dir, fname_1))

    elif i < 0.9 * len(class_0):

        shutil.copyfile(path.join(root_dir, fname_0),

                        path.join(validation_0_dir, fname_0))

        shutil.copyfile(path.join(root_dir, fname_1),

                        path.join(validation_1_dir, fname_1))

    else:

        shutil.copyfile(path.join(root_dir, fname_0),

                        path.join(test_0_dir, fname_0))

        shutil.copyfile(path.join(root_dir, fname_1),

                        path.join(test_1_dir, fname_1))



### Number of images in train-validation-test sets ###

n_train = len(listdir(train_1_dir)) + len(listdir(train_0_dir))

n_val = len(listdir(validation_1_dir)) + len(listdir(validation_0_dir))

n_test = len(listdir(test_1_dir)) + len(listdir(test_0_dir))

print('Train images:', n_train)

print('Validation images:', n_val)

print('Test images:', n_test)
### Preparing image generators with rescaling, resizing ###

batch_size = 64

v_batch_size = 64

input_size = (128, 128)

input_shape = input_size + (3, )



train_datagen = ImageDataGenerator(rescale=1. / 255)

test_datagen = ImageDataGenerator(rescale=1. / 255)



train_generator = train_datagen.flow_from_directory(

    train_dir,

    target_size=input_size,

    batch_size=batch_size,

    class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

    validation_dir,

    target_size=input_size,

    batch_size=v_batch_size,

    class_mode='binary')
### CNN model with batch normalization and dropout ###

model = Sequential()



model.add(Conv2D(16, (3, 3), input_shape=input_shape, kernel_initializer=glorot_uniform(seed=SEED)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D((2, 2)))



model.add(Conv2D(32, (3, 3), kernel_regularizer=l2(0.01), kernel_initializer=glorot_uniform(seed=SEED)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D((2, 2)))



model.add(Conv2D(64, (3, 3), kernel_regularizer=l2(0.01), kernel_initializer=glorot_uniform(seed=SEED)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D((2, 2)))

model.add(Dropout(rate=0.3, seed=SEED))



model.add(Flatten())

model.add(Dense(512, kernel_regularizer=l2(0.01), kernel_initializer=glorot_uniform(seed=SEED)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dense(1, activation='sigmoid', kernel_initializer=glorot_uniform(seed=SEED)))



model.compile(optimizer=SGD(lr=0.01, nesterov=True),

              loss=binary_crossentropy,

              metrics=['accuracy'])



### Details of the model ###

model.summary()
### Defining callbacks ###

reduce_lr = ReduceLROnPlateau(

    monitor='val_loss',

    factor=0.5,

    patience=3,

    verbose=1,

    mode='auto',

    min_lr=0.0001)

early_stopping = EarlyStopping(

    monitor='val_loss',

    patience=5,

    verbose=1,

    mode='auto')

model_checkpoint = ModelCheckpoint(

    filepath='weights.h5',

    monitor='val_loss',

    verbose=1,

    save_best_only=True,

    save_weights_only=True,

    mode='auto')



### Importing module for execution timing

start_time = timeit.default_timer()



### Fitting model to the data ###

history = model.fit_generator(

    train_generator,

    steps_per_epoch=train_generator.n // batch_size,

    epochs=50,

    shuffle=False,

    validation_data=validation_generator,

    validation_steps=validation_generator.n // v_batch_size,

    callbacks=[reduce_lr, early_stopping, model_checkpoint])



cnn_training_time = timeit.default_timer() - start_time



model.load_weights('weights.h5')

model.save('pneumonia-chest-x-ray-cnn.h5')
### Accuracy and loss plots ###

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = [n + 1 for n in range(len(acc))]

fig = plt.figure(figsize=(12, 4))



fig.add_subplot(1, 2, 1)

plt.plot(epochs, acc, 'k', label='Training accuracy')

plt.plot(epochs, val_acc, 'b:', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()



fig.add_subplot(1, 2, 2)

plt.plot(epochs, loss, 'k', label='Training loss')

plt.plot(epochs, val_loss, 'b:', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
### Predict classes for test images ###

test_generator = test_datagen.flow_from_directory(

    test_dir,

    target_size=input_size,

    shuffle=False,

    batch_size=v_batch_size,

    class_mode='binary')

cnn_test_score = model.evaluate_generator(

    test_generator,

    steps=test_generator.n // v_batch_size)



print("Test set:\n loss: %.4f, accuracy: %.4f\nTraining time: %.0fs" %

      (cnn_test_score[0], cnn_test_score[1], cnn_training_time))
### Get numerical predictions for test set images ###

test_generator.reset()

predictions = model.predict_generator(

    test_generator,

    steps=test_generator.n // v_batch_size)



### True and predicted labels ###

true_labels = test_generator.labels.tolist()

pred_labels = [1 if p > 0.5 else 0 for p in predictions.ravel()]



### Confusion matrix ###

tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()

print("TP: %3i   FP: %3i\nFN: %3i   TN: %3i\n" % (tp, fp, fn, tn))



### Classification metrics ###

print(

    classification_report(

        true_labels,

        pred_labels,

        target_names=['pneumonia', 'healthy']))
### Prepare list of predictions and corresponding filenames ###

pred_with_filenames = {}

files = test_generator.filenames

files.sort()

for filename, pred in zip(files, predictions):

    pred_with_filenames[filename.split('/')[1]] = pred[0]



### Show sample test images from class 0 (pneumonia) with their predictions ###

imagelist = listdir(test_0_dir)

rn.shuffle(imagelist)

test_img_sample = [(filename, image.imread(test_0_dir + '/' + filename))

                   for i, filename in enumerate(imagelist) if i < 10]

rows, columns = 2, 5

fig = plt.figure(figsize=(16, 8))

for index, img in enumerate(test_img_sample):

    fig.add_subplot(rows, columns, index + 1)

    plt.imshow(img[1], cmap=plt.cm.gray)

    value = pred_with_filenames[img[0]]

    label = "pneumonia" if value > 0.5 else "healthy"

    title = "Predicted value: %.2f\nPredicted label: %s" % (value, label)

    print(title)

    plt.title(title)

plt.subplots_adjust(left=1, right=2)

plt.show()
### Transfer learning - MobileNet(16 MB), Xception (88MB), InceptionV3 (92 MB), ResNet50 (98MB), VGG16 (528 MB) with feature extraction ###

models_dict = {}

model_names = ["MobileNet", "Xception", "InceptionV3", "ResNet50", "VGG16"]

input_sizes = [(224, 224), (299, 299), (299, 299), (224, 224), (224, 224)]

features_dim = [[7, 7, 1024], [10, 10, 2048], [8, 8, 2048], [7, 7, 2048],

                [7, 7, 512]]

model_list = [MobileNet, Xception, InceptionV3, ResNet50, VGG16]

for i, m in enumerate(model_names):

    models_dict[m] = {}

    models_dict[m]['object'] = model_list[i]

    models_dict[m]['input_size'] = input_sizes[i]

    models_dict[m]['input_shape'] = input_sizes[i] + (3, )

    models_dict[m]['features_dim'] = features_dim[i]

    models_dict[m]['dense_input_dim'] = functools.reduce(operator.mul, features_dim[i], 1)

    models_dict[m]['weights_filename'] = m.lower() + '_weights.h5'

    models_dict[m]['model_filename'] = 'pneumonia-chest-x-ray-' + m.lower() + '.h5'
datagen = ImageDataGenerator(rescale=1. / 255)

batch_size = 64



for m, data in models_dict.items():

    base_model = data['object'](

        input_shape=data['input_shape'],

        weights='imagenet',

        include_top=False)



    def extract_features(directory, sample_count):

        shape = tuple([sample_count] + data['features_dim'])

        features = np.zeros(shape=shape)

        labels = np.zeros(shape=(sample_count))

        generator = datagen.flow_from_directory(

            directory,

            target_size=data['input_size'],

            batch_size=batch_size,

            class_mode='binary')

        i = 0

        for inputs_batch, labels_batch in generator:

            features_batch = base_model.predict(inputs_batch)

            features[i * batch_size:(i + 1) * batch_size] = features_batch

            labels[i * batch_size:(i + 1) * batch_size] = labels_batch

            i += 1

            if i * batch_size >= sample_count:

                break

        return features, labels



    start_time = timeit.default_timer()



    train_features, train_labels = extract_features(train_dir, n_train)

    validation_features, validation_labels = extract_features(validation_dir, n_val)

    test_features, test_labels = extract_features(test_dir, n_test)



    feature_extraction_time = timeit.default_timer() - start_time

    data['feature_extraction_time'] = feature_extraction_time



    train_features = np.reshape(train_features, (n_train, data['dense_input_dim']))

    validation_features = np.reshape(validation_features, (n_val, data['dense_input_dim']))

    test_features = np.reshape(test_features, (n_test, data['dense_input_dim']))



    model = Sequential()

    model.add(Dense(512, activation='relu', input_dim=data['dense_input_dim']))

    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))



    model.compile(

        optimizer=Adam(lr=0.00001),

        loss=binary_crossentropy,

        metrics=['accuracy'])



    model_checkpoint = ModelCheckpoint(

        filepath=data['weights_filename'],

        monitor='val_loss',

        verbose=1,

        save_best_only=True,

        save_weights_only=True,

        mode='auto')

    early_stopping = EarlyStopping(

        monitor='val_loss',

        patience=10,

        verbose=1,

        mode='auto')



    start_time = timeit.default_timer()



    history = model.fit(

        train_features,

        train_labels,

        epochs=50,

        batch_size=batch_size,

        validation_data=(validation_features, validation_labels),

        callbacks=[early_stopping, model_checkpoint])

    model.load_weights(data['weights_filename'])

    model.save(data['model_filename'])



    training_time = timeit.default_timer() - start_time

    data['training_time'] = training_time



    ### Accuracy and loss plots ###

    acc = history.history['acc']

    val_acc = history.history['val_acc']

    loss = history.history['loss']

    val_loss = history.history['val_loss']



    epochs = [n + 1 for n in range(len(acc))]

    fig = plt.figure(figsize=(12, 4))



    fig.add_subplot(1, 2, 1)

    plt.plot(epochs, acc, 'k', label='Training accuracy')

    plt.plot(epochs, val_acc, 'b:', label='Validation accuracy')

    plt.title('Training and validation accuracy')

    plt.legend()



    fig.add_subplot(1, 2, 2)

    plt.plot(epochs, loss, 'k', label='Training loss')

    plt.plot(epochs, val_loss, 'b:', label='Validation loss')

    plt.title('Training and validation loss')

    plt.legend()



    plt.show()



    ### Predict classes for test images ###

    test_score = model.evaluate(

        test_features, 

        test_labels,

        steps= n_test // v_batch_size)



    data['test_score'] = test_score

    

    ### Get numerical predictions for test set images ###

    predictions = model.predict(

        test_features,

        batch_size= n_test // v_batch_size)



    ### True and predicted labels ###

    pred_labels = [1 if p > 0.5 else 0 for p in predictions.ravel()]



    ### Confusion matrix ###

    tn, fp, fn, tp = confusion_matrix(test_labels, pred_labels).ravel()

    print("TP: %3i   FP: %3i\nFN: %3i   TN: %3i\n" % (tp, fp, fn, tn))



    ### Classification metrics ###

    print(

        classification_report(

            test_labels,

            pred_labels,

            target_names=['pneumonia', 'healthy']))
### Print results of CNN and transfer learning models ###

print("Model name: Custom CNN\n\tTraining time: {:.0f}s\n\tAccuracy: {:.2%}".format(cnn_training_time, cnn_test_score[1]))

for m, data in models_dict.items():

    print(

        "Model name: {}\n\tFeature extraction time: {:.0f}s\n\tTraining time: {:.0f}s\n\tTotal computing time: {:.0f}s\n\tAccuracy: {:.2%}"

        .format(m, data['feature_extraction_time'], data['training_time'],

                data['feature_extraction_time']+data['training_time'],

                data['test_score'][1]))