from keras.layers import Dense, Conv2D, Flatten
from keras.layers import MaxPooling2D, Activation, Dropout
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.backend import image_data_format
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from glob import glob
import datetime as dt
import numpy as np
import cv2
import os
%matplotlib inline
train_path = r'../input/petclassdata/pet_class_data/train'
test_path = r'../input/petclassdata/pet_class_data/test'
os.listdir(train_path), os.listdir(test_path)
# creating Image generator to create more data samples
train_gen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            rescale=1./255,
            horizontal_flip=True,
            vertical_flip=True,
            validation_split=0.25,
            fill_mode='nearest')

test_gen = ImageDataGenerator(rescale=1./255)

def create_sample_dataset(train_path, test_path, train_samples=0, test_samples=0):
    # creating samples for training data
    for fold in os.listdir(train_path):
        for t_img in glob(train_path + '/' + fold + '/*.jpg'):
            img = img_to_array(load_img(t_img))
            img = img.reshape((1,) + img.shape)
            i = 0
            for batch in train_gen.flow(img, 
                                         batch_size=1,
                                         save_to_dir=train_path + '/' + fold,
                                         save_prefix=fold,
                                         save_format='jpg'):
                if i == new_train_samples:
                    break
                i += 1


    # creating samples for testing data
    for fold in os.listdir(test_path):
        for t_img in glob(test_path + '/' + fold + '/*.jpg'):
            img = img_to_array(load_img(t_img))
            img = img.reshape((1, ) + img.shape)
            i = 0
            for batch in test_gen.flow(img, 
                                         batch_size=1,
                                         save_to_dir=test_path + '/' + fold,
                                         save_prefix=fold,
                                         save_format='jpg'):
                if i == new_test_samples:
                    break
                i += 1
# new_train_samples = 50
# new_test_samples = 20
# create_sample_dataset(train_path, test_path, new_train_samples, new_test_samples)
batch_size = 32
train_image = train_gen.flow_from_directory(train_path,
                                           target_size=(150, 150),
                                           batch_size=batch_size,
                                           subset='training',
                                           class_mode='binary')

validation_image = train_gen.flow_from_directory(train_path,
                                           target_size=(150, 150),
                                           batch_size=batch_size,
                                           subset='validation',
                                           class_mode='binary')

if image_data_format == 'channels_first':
    input_shape = (3, 150, 150)
else:
    input_shape = (150, 150, 3)
def get_model():
    model = Sequential()

    # First Convolutional layer
    model.add(Conv2D(64, (3, 3), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Second Convolutional layer
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Flattening the input
    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
             optimizer='adam',
            #  optimizer='sgd',
            #  optimizer='rmsprop',
             metrics=['accuracy'])

    print('Model building done')
    return model


def run_model(model, train_samples, test_samples, train_image, test_image, iteration=0, batch_size=8):
    my_callbacks = [
#     EarlyStopping(patience=8),
    ModelCheckpoint(filepath='../model_cp.{epoch:02d}-{val_loss:.2f}.h5')
#     TensorBoard(log_dir='/content/drive/My Drive/datasets/pet_class_data/model_checkpoint/logs')
    ]
    model.fit_generator(train_image,
                   steps_per_epoch=train_samples//batch_size,
                   epochs=iteration,
                   validation_steps=test_samples//batch_size,
                   callbacks = my_callbacks,
                   validation_data=validation_image
                   )
    # list all data in history
    history = model.history
    # print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    plt.show()  
train_samples, validation_samples = train_image.samples, validation_image.samples
print('Total train samples: {} and classes {}\nTotal test samples: {} and classes {}'.format(train_samples, np.unique(train_image.classes), validation_samples, np.unique(validation_image.classes)))
my_model = get_model()
my_model.summary()
# running model
epochs = 150
my_model = get_model()
run_model(my_model, train_samples, validation_samples, train_image, validation_image, iteration=epochs, batch_size=batch_size)
my_model.save(r'/content/drive/My Drive/datasets/pet_class_data/model/model_'+str(epochs)+'_'+str(dt.datetime.now()))
history = my_model.history
max(history.history['accuracy']), max(history.history['val_accuracy'])
min(history.history['loss']), min(history.history['val_loss'])
image = '../input/petclassdata/pet_class_data/test/cats/cats_0_1140.jpg'
img = cv2.imread(image)
plt.imshow(img)
plt.show()
img = cv2.resize(img,(150,150))
img = np.reshape(img, (1, 150, 150, 3))
classes = my_model.predict_classes(img)
print(classes)
image = '../input/petclassdata/pet_class_data/test/dogs/101.jpg'
img = cv2.imread(image)
plt.imshow(img)
plt.show()
img = cv2.resize(img,(150,150))
img = np.reshape(img, (1, 150, 150, 3))
classes = my_model.predict_classes(img)
print(classes)
