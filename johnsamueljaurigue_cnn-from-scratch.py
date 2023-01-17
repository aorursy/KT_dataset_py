from glob import glob

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import random

import os



from sklearn.datasets import load_files 

from sklearn.model_selection import train_test_split

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras.layers import Dropout, Flatten, Dense

from keras.models import Sequential

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint

from keras.utils import np_utils

from keras.preprocessing import image                  

from tqdm import tqdm

from PIL import ImageFile 

from shutil import copyfile
%rm -rf data 



%mkdir -p data/train/daisy

%mkdir -p data/train/tulip

%mkdir -p data/train/sunflower

%mkdir -p data/train/rose

%mkdir -p data/train/dandelion



%mkdir -p data/valid/daisy

%mkdir -p data/valid/tulip

%mkdir -p data/valid/sunflower

%mkdir -p data/valid/rose

%mkdir -p data/valid/dandelion



%mkdir -p data/test/daisy

%mkdir -p data/test/tulip

%mkdir -p data/test/sunflower

%mkdir -p data/test/rose

%mkdir -p data/test/dandelion





%ls data/train

%ls data/valid

%ls data/test
base_dir = "../input/flowers/flowers"

categories = os.listdir(base_dir)
%matplotlib inline

plt.rcParams["figure.figsize"] = (20,3)
def train_valid_test(files):

    train_fles = files[:int(len(files)*0.6)]

    valid_files = files[int(len(files)*0.6):int(len(files)*0.8)]

    test_files = files[int(len(files)*0.8):]

    return train_fles, valid_files, test_files
def copy_files(files, src, dest):

    for file in files:

        copyfile("{}/{}".format(src, file), "{}/{}".format(dest, file))
def plot_images(category, images):

    for i in range(len(images)):

        plt.subplot(1,5,i+1)

        plt.title(category)

        image = mpimg.imread("{}/{}/{}".format(base_dir, category, images[i]))

        plt.imshow(image)

    plt.show()
total_images = []

for category in categories:

    images = os.listdir("{}/{}".format(base_dir, category))

    random.shuffle(images)

    filtered_images = [image for image in images if image not in ['flickr.py', 'flickr.pyc', 'run_me.py']]

    

    total_images.append(len(filtered_images))

    

    

    train_images, valid_images, test_images = train_valid_test(filtered_images)

    

    copy_files(train_images, "{}/{}".format(base_dir, category), "./data/train/{}".format(category))

    copy_files(valid_images, "{}/{}".format(base_dir, category), "./data/valid/{}".format(category))

    copy_files(test_images, "{}/{}".format(base_dir, category), "./data/test/{}".format(category))

    plot_images(category, images[:5])

    

        
print("Total images: {}".format(np.sum(total_images)))

for i in range(len(categories)):

    print("{}: {}".format(categories[i], total_images[i]))
def load_dataset(path):

    data = load_files(path)

    flower_files = np.array(data['filenames'])

    print(data['target_names'])

    flower_targets = np_utils.to_categorical(np.array(data['target']), 5)

    return flower_files, flower_targets



train_files, train_targets = load_dataset('data/train')

valid_files, valid_targets = load_dataset('data/valid')

test_files, test_targets = load_dataset('data/test')



print('There are %d total flower categories.' % len(categories))

print('There are %s total flower images.\n' % len(np.hstack([train_files, valid_files, test_files])))

print('There are %d training flower images.' % len(train_files))

print('There are %d validation flower images.' % len(valid_files))

print('There are %d test flower images.' % len(test_files))

def path_to_tensor(img_path):

    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)

    return np.expand_dims(x, axis=0)
def paths_to_tensor(img_paths):

    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]

    return np.vstack(list_of_tensors)
ImageFile.LOAD_TRUNCATED_IMAGES = True                 



train_tensors = paths_to_tensor(train_files).astype('float32')/255

valid_tensors = paths_to_tensor(valid_files).astype('float32')/255

test_tensors = paths_to_tensor(test_files).astype('float32')/255
model = Sequential()

print(train_tensors.shape)

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (224,224,3)))

model.add(MaxPooling2D(pool_size=(2,2)))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

 



model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))





model.add(GlobalAveragePooling2D())

model.add(Dense(5, activation='softmax'))

model.summary()
model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='best_model', 

                               verbose=1, save_best_only=True)



History = model.fit(train_tensors, train_targets, 

          validation_data=(valid_tensors, valid_targets),

          epochs=50, batch_size=32, callbacks=[checkpointer], verbose=1)
model.load_weights('best_model')
flower_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]



test_accuracy = 100*np.sum(np.array(flower_predictions)==np.argmax(test_targets, axis=1))/len(flower_predictions)

print('Test accuracy: %.4f%%' % test_accuracy)
plt.plot(History.history['loss'])

plt.plot(History.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epochs')

plt.legend(['train', 'validation'])

plt.show()
plt.plot(History.history['acc'])

plt.plot(History.history['val_acc'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epochs')

plt.legend(['train', 'validation'])

plt.show()
for i in range(5):

    predicted = np.argmax(model.predict(np.expand_dims(test_tensors[i], axis=0)))

    actual = np.argmax(test_targets[i])

    print("Predicted: {}, Actual: {}, Name: {}".format(predicted, actual, test_files[i].split("/")[2]))

    image = mpimg.imread(test_files[i])

    plt.imshow(image)

    plt.show()
%rm -rf data