import numpy as np                      # For Scientific Computations

import pandas as pd                     # For Data Ananlysis

import os                               # For directly transferring the Data to/fro  from any directories

import cv2                              # For Image Augmentation and Displaying the Image

import matplotlib.pyplot as plt         # For Plotting the Diagrams

from tqdm import tqdm_notebook as tqdm  # For fancy display of the training

from sklearn.utils import shuffle       # For Shuffling the dataset



from keras import applications          # Useful for loading the already pretrained model

from keras import optimizers            # Optimizers

from keras.utils import to_categorical  # Converts a class vector (integers) to binary class matrix.

from keras.models import Sequential, Model, load_model  # For Model generating, loading and Saving

from keras.layers import Dropout, Flatten, Dense  # Flatten, Dense and Dropout

from keras.preprocessing.image import ImageDataGenerator   # A useful library, which we will be seeing 

from keras.callbacks import ModelCheckpoint  # For checkpointing the model, for a specified accuracy
# Creating a datafram which contains the path of the image and the category of the image (in terms of number i.e 0 for cat,etc)



foldernames = os.listdir('/kaggle/input/animals10/raw-img')

categories = []

files = []

i = 0

for k, folder in enumerate(foldernames):

    filenames = os.listdir("../input/animals10/raw-img/" + folder);

    for file in filenames:

        files.append("../input/animals10/raw-img/" + folder + "/" + file)

        categories.append(k)

        

df = pd.DataFrame({

    'filename': files,

    'category': categories

})

train_df = pd.DataFrame(columns=['filename', 'category'])

for i in range(10):

    train_df = train_df.append(df[df.category == i].iloc[:500,:])



train_df.head()

train_df = train_df.reset_index(drop=True)

train_df
y = train_df['category']

x = train_df['filename']

y = train_df['category']



x, y = shuffle(x, y, random_state=8)         # Randomly shuffling the dataframe, so as to improvise the accuracy of the model on unseen data
# Image Augmentation



def centering_image(img):

    size = [256,256]

    

    img_size = img.shape[:2]

    

    # centering

    row = (size[1] - img_size[0]) // 2

    col = (size[0] - img_size[1]) // 2

    resized = np.zeros(list(size) + [img.shape[2]], dtype=np.uint8)

    resized[row:(row + img.shape[0]), col:(col + img.shape[1])] = img



    return resized



images = []

with tqdm(total=len(train_df)) as pbar:

    for i, file_path in enumerate(train_df.filename.values):

        #read image

        img = cv2.imread(file_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)             # Converting BGR to RGB Image , default is in BGR form



        #resize

        if(img.shape[0] > img.shape[1]):

            tile_size = (int(img.shape[1]*256/img.shape[0]),256)

        else:

            tile_size = (256, int(img.shape[0]*256/img.shape[1]))



        #centering

        img = centering_image(cv2.resize(img, dsize=tile_size))



        #out put 224*224px 

        img = img[16:240, 16:240]

        images.append(img)

        pbar.update(1)



images = np.array(images)
# Plotting some of the images of our training dataset



rows,cols = 2,5

fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20,20))

for i in range(10):

    path = train_df[train_df.category == i].values[2]

    # image = cv2.imread(path[0])/

    axes[i//cols, i%cols].set_title(path[0].split('/')[-2] + str(path[1]))

    axes[i//cols, i%cols].imshow(images[train_df[train_df.filename == path[0]].index[0]])
data_num = len(y)

random_index = np.random.permutation(data_num)       # Creating a random list of number from 0 to len(y) - 1, useful for shuffling



x_shuffle = []

y_shuffle = []

for i in range(data_num):

    x_shuffle.append(images[random_index[i]])

    y_shuffle.append(y[random_index[i]])

    

x = np.array(x_shuffle)                              # Converting images into Numpy Array

y = np.array(y_shuffle)                              # Labels or Class

val_split_num = int(round(0.2*len(y)))               

x_train = x[val_split_num:]                          # Converting into Training Data and Test Data

y_train = y[val_split_num:]

x_test = x[:val_split_num] 

y_test = y[:val_split_num]



print('x_train', x_train.shape)

print('y_train', y_train.shape)

print('x_test', x_test.shape)

print('y_test', y_test.shape)

y_train = to_categorical(y_train)                    # Categorical Data

y_test = to_categorical(y_test)                      # Categorical Data



x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train /= 255

x_test /= 255



img_rows, img_cols, img_channel = 224, 224, 3

name_animal = []

for i in range(10):

    path = train_df[train_df.category == i].values[2]

    if path[0].split('/')[-2] == 'scoiattolo':

        name_animal.append('squirrel')

    elif path[0].split('/')[-2] == 'cavallo':

        name_animal.append('horse')

    elif path[0].split('/')[-2] == 'farfalla':

        name_animal.append('butterfly')

    elif path[0].split('/')[-2] == 'mucca':

        name_animal.append('cow')

    elif path[0].split('/')[-2] == 'gatto':

        name_animal.append('cat')

    elif path[0].split('/')[-2] == 'pecora':

        name_animal.append('sheep')

    elif path[0].split('/')[-2] == 'gallina':

        name_animal.append('chicken')

    elif path[0].split('/')[-2] == 'elefante':

        name_animal.append('elephant')

    elif path[0].split('/')[-2] == 'ragno':

        name_animal.append('spider')

    elif path[0].split('/')[-2] == 'cane':

        name_animal.append('dog')
# Importing the already designed Model, means the model has already defined all the convolutional layers, max pooling layers, and dimensions of those layers

base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, img_channel))



# add model will be added to the last part, where the input from the base_model will be used to predict the class of the image

add_model = Sequential()

add_model.add(Flatten(input_shape=base_model.output_shape[1:]))

add_model.add(Dense(256, activation='relu'))

add_model.add(Dense(10, activation='softmax'))



# Combining the base_model and add_model to make the final model

model = Model(inputs=base_model.input, outputs=add_model(base_model.output))

model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),

              metrics=['accuracy'])



#Optimizers - Stochastic Gradient Descent, with specified learning parameters and momentum



model.summary()
batch_size = 32                       # Specifying the batch size, which means after 32 images, the correction or the back propagation will be mage

epochs = 50                           # Number of Epochs



# ImageDataGenerator

train_datagen = ImageDataGenerator(

        rotation_range=30, 

        width_shift_range=0.1,

        height_shift_range=0.1, 

        horizontal_flip=True)

train_datagen.fit(x_train)





history = model.fit_generator(

    train_datagen.flow(x_train, y_train, batch_size=batch_size),

    steps_per_epoch=x_train.shape[0] // batch_size,

    epochs=epochs,

    validation_data=(x_test, y_test),

    callbacks=[ModelCheckpoint('VGG16-transferlearning.model', monitor='val_acc')]

)
print("CNN: Epochs={0:d}, Train accuracy={1:.5f}, Validation accuracy={2:.5f}".format(epochs,history.history['accuracy'][epochs-1],history.history['val_accuracy'][epochs-1]))

def show_plots(history):

    """ Useful function to view plot of loss values & accuracies across the various epochs """

    loss_vals = history['loss']

    val_loss_vals = history['val_loss']

    epochs = range(1, len(history['accuracy'])+1)

    

    f, ax = plt.subplots(nrows=1,ncols=2,figsize=(16,4))

    

    # plot losses on ax[0]

    ax[0].plot(epochs, loss_vals, color='navy',marker='o', linestyle=' ', label='Training Loss')

    ax[0].plot(epochs, val_loss_vals, color='firebrick', marker='*', label='Validation Loss')

    ax[0].set_title('Training & Validation Loss')

    ax[0].set_xlabel('Epochs')

    ax[0].set_ylabel('Loss')

    ax[0].legend(loc='best')

    ax[0].grid(True)

    

    # plot accuracies

    acc_vals = history['accuracy']

    val_acc_vals = history['val_accuracy']



    ax[1].plot(epochs, acc_vals, color='navy', marker='o', ls=' ', label='Training Accuracy')

    ax[1].plot(epochs, val_acc_vals, color='firebrick', marker='*', label='Validation Accuracy')

    ax[1].set_title('Training & Validation Accuracy')

    ax[1].set_xlabel('Epochs')

    ax[1].set_ylabel('Accuracy')

    ax[1].legend(loc='best')

    ax[1].grid(True)

    

    plt.show()

    plt.close()

    

    # delete locals from heap before exiting

    del loss_vals, val_loss_vals, epochs, acc_vals, val_acc_vals

show_plots(history.history)
test_images = []

j = 39 # change this to get different images

for i in range(10):

    path = train_df[train_df.category == i].values[j]

    a = images[train_df[train_df.filename == path[0]].index[0]]

    img = np.array(a)

    img = img[:, :, ::-1].copy() 

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if(img.shape[0] > img.shape[1]):

        tile_size = (int(img.shape[1]*256/img.shape[0]),256)

    else:

        tile_size = (256, int(img.shape[0]*256/img.shape[1]))

    img = centering_image(cv2.resize(img, dsize=tile_size))

    img = img[16:240, 16:240]

    test_images.append(img)



test_images = np.array(test_images).reshape(-1,224,224,3)

something = model.predict(test_images)

animals = name_animal

i = 0

for pred in something:

    path = train_df[train_df.category == i].values[2]

    plt.imshow(test_images[i])

    plt.show()

    print('Actual  :', animals[i])

    print('Predict :', animals[np.where(pred.max() == pred)[0][0]])

    i += 1
