import os
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

import cv2
import matplotlib.pyplot as plt
import numpy as np
device_name = tf.test.gpu_device_name()
if "GPU" not in device_name:
    print("GPU device not found")
print('Found GPU at: {}'.format(device_name))
train_dir = '/kaggle/input/isl-dataset/'
for folder in os.listdir(train_dir):
    print(folder) 
# display the datset

def load_unique(train_dir):
    images_for_plot = []
    labels_for_plot = []
    for folder in os.listdir(train_dir):
        for file in os.listdir(train_dir + '/' + folder):
            filepath = train_dir + '/' + folder + '/' + file
            if filepath.endswith('txt'):
                continue
            image = cv2.imread(filepath)
            image=cv2.resize(image,(64,64))
           # blurred_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
           # print(blurred_img.shape)
            images_for_plot.append(image)
            labels_for_plot.append(folder)
            break
    return images_for_plot, labels_for_plot


images_for_plot, labels_for_plot = load_unique(train_dir+'Digits')
images_tmp, labels_tmp = load_unique(train_dir+'Letters')

images_for_plot.extend(images_tmp)
labels_for_plot.extend(labels_tmp)
print("unique_labels = ", labels_for_plot)

fig = plt.figure(figsize = (15,15))

row = 6
col = 6
for i in range(1, 33):
    fig.add_subplot(row, col, i)
    plt.imshow(images_for_plot[i])
    plt.title(labels_for_plot[i])
    plt.axis('off')

plt.show()
labels_dict = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'a':10,'b':11,'c':12,
                   'd':13,'e':14,'f':15,'g':16,'i':17,'k':18,'l':19,'m':20,'n':21,'o':22,'p':23,'q':24,
                   'r':25,'s':26,'t':27,'u':28,'w':29,'x':30,'y':31,'z':32}

image_size=(128,128)

def read_from_folder(train_dir):
    """
    Loads data and preprocess. Returns train and test data along with labels.
    """
    images = []
    labels = []
    print("LOADING DATA FROM ",end = "")
    for folder in os.listdir(train_dir):
        print(folder, end = ' | ')
        for image in os.listdir(train_dir + "/" + folder):
            temp_img = cv2.imread(train_dir + '/' + folder + '/' + image)
            temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
            img_np = cv2.resize(temp_img,image_size) 
       #    img_np = cv2.normalize(img_np, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            images.append(img_np)
            labels.append(labels_dict[folder])  
            
    return images, labels


def load_data():
    images, labels = read_from_folder(train_dir+'Digits')
    img_tmp, label_tmp = read_from_folder( train_dir +'Letters')
    images.extend(img_tmp)
    labels.extend(label_tmp)   
    images = np.array(images)
    print(images.shape)
    labels = tf.keras.utils.to_categorical(labels)
    X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size = 0.05)
    print(labels.shape)
    print('Loaded', len(X_train),'images for training,','Train data shape =',X_train.shape)
    print('Loaded', len(X_test),'images for testing','Test data shape =',X_test.shape)
    
    return X_train, X_test, Y_train, Y_test


X_train, X_test, Y_train, Y_test = load_data()
# reshaping images 
X_train = X_train.reshape(X_train.shape[0],128, 128, 1)
X_test = X_test.reshape(X_test.shape[0], 128, 128, 1)

#normalizing
X_train.astype('float32')
X_test.astype('float32')

X_train = X_train/255.0
X_test = X_test/ 255.0

X_train.shape
#Y_train.shape
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


datagen.fit(X_train)
train_generator = datagen.flow(X_train, Y_train, batch_size=64)
test_gen = ImageDataGenerator()
test_generator = test_gen.flow(X_test, Y_test, batch_size=64)

def create_model():
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), input_shape=(128,128,1)))
    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    
    model.add(layers.Conv2D(128,(3, 3)))
    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(128, (3, 3)))
    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

        
    model.add(layers.Flatten())

    # Fully connected layer
    model.add(layers.Dense(512))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(33))
    model.add(layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def fit_model():
    model_hist = model.fit_generator(train_generator, epochs = 10, validation_data=test_generator)
    return model_hist 
with tf.device('/gpu:0'):
    model = create_model()
    curr_model_hist = fit_model()
plt.plot(curr_model_hist.history['accuracy'])
plt.plot(curr_model_hist.history['val_accuracy'])
plt.legend(['train', 'test'], loc='lower right')
plt.title('accuracy plot - train vs test')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

plt.plot(curr_model_hist.history['loss'])
plt.plot(curr_model_hist.history['val_loss'])
plt.legend(['training loss', 'validation loss'], loc = 'upper right')
plt.title('loss plot - training vs vaidation')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

evaluate_metrics = model.evaluate(X_test, Y_test)
print("\nEvaluation Accuracy = ", "{:.2f}%".format(evaluate_metrics[1]*100),"\nEvaluation loss = " ,"{:.6f}".format(evaluate_metrics[0]))
#saving the model
from keras.models import load_model
model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
# returns a compiled model

from IPython.display import FileLink, FileLinks
FileLinks('.') #lists all downloadable files on server
predictions = [model.predict_classes(image.reshape(1,64,64,1))[0] for image in X_test]
Y_test[0]
predfigure = plt.figure(figsize = (13,13))
def plot_image_1(fig, image, label, prediction, predictions_label, row, col, index):
    fig.add_subplot(row, col, index)
    plt.axis('off')
    plt.imshow(image)
    title = "prediction : [" + str(predictions_label) + "] "+ "\n" + label
    plt.title(title)
    return

image_index = 0
row = 5
col = 6
for i in range(1,(row*col-1)):
    plot_image_1(predfigure, X_test[image_index], Y_test[image_index], predictions[image_index], predictions_labels_plot[image_index], row, col, i)
    image_index = image_index + 1
plt.show()
