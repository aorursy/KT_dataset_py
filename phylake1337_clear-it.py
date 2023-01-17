import keras, os

import tensorflow as tf

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



from tqdm import tqdm

from tensorflow.keras.preprocessing.image import load_img



from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
#Set some directories

train_zip_path = '/kaggle/input/denoising-dirty-documents/train.zip'

test_zip_path = '/kaggle/input/denoising-dirty-documents/test.zip'

sample_zip_path = '/kaggle/input/denoising-dirty-documents/sampleSubmission.csv.zip'

trainclean_zip_path = '/kaggle/input/denoising-dirty-documents/train_cleaned.zip'

extracting_path = '/kaggle/working'

train_path = extracting_path + '/train'

test_path = extracting_path + '/test'

train_cleaned_path = extracting_path + '/train_cleaned'
#Extract data files to '/kaggle/working' 

import zipfile

with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:

    zip_ref.extractall(extracting_path)

    

with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:

    zip_ref.extractall(extracting_path)

    

with zipfile.ZipFile(sample_zip_path, 'r') as zip_ref:

    zip_ref.extractall(extracting_path)

    

with zipfile.ZipFile(trainclean_zip_path, 'r') as zip_ref:

    zip_ref.extractall(extracting_path)
#Figure out images shape.

image_names = os.listdir(extracting_path + '/train')

data_size = len(image_names)

#initailize output arrays.

X = np.zeros([data_size, 2], dtype=np.uint16)

for i in tqdm(range(data_size)):

    image_name = image_names[i]

    img_dir = os.path.join(extracting_path + '/train', image_name)

    img_pixels = mpimg.imread(img_dir)

    X[i] = img_pixels.shape



print('Number of training images:', data_size)

print('Differnet image hights: {}'.format(set(X[:,0])))

print('Differnet image widths: {}'.format(set(X[:,1])))
def images_to_array(data_dir, label_dir=None, img_size=(420, 560)):

    '''

    1- Read image samples from certain directory.

    2- Resize and Stack them into one big numpy array.

    -- And if there are labels images ..

    3- Read sample's label form the labels directory.

    4- Resize and Stack them into one big numpy array.

    5- Shuffle Data and label arrays.

    '''

    image_names = os.listdir(data_dir)

    data_size = len(image_names)

    #initailize data arrays.

    X = np.zeros([data_size, img_size[0], img_size[1]], dtype=np.uint8)

    #read data.

    for i in tqdm(range(data_size)):

        image_name = image_names[i]

        img_dir = os.path.join(data_dir, image_name)

        img_pixels = load_img(img_dir, color_mode='grayscale', target_size=img_size)

        X[i] = img_pixels

    #reshape into 4-d array    

    X = X.reshape(data_size, img_size[0], img_size[1], 1) 

    

    if label_dir:

        label_names = os.listdir(label_dir)

        data_size = len(label_names)

        #initailize labels arrays.

        y = np.zeros([data_size, img_size[0], img_size[1]], dtype=np.uint8)

        #read lables.

        for i in tqdm(range(data_size)):

            image_name = label_names[i]

            img_dir = os.path.join(label_dir, image_name)

            img_pixels = load_img(img_dir, color_mode='grayscale', target_size=img_size)

            y[i] = img_pixels

        #reshape into 4-d array    

        y = y.reshape(data_size, img_size[0], img_size[1], 1) 

        #shuffle    

        ind = np.random.permutation(data_size)

        X = X[ind]

        y = y[ind]

        print('Data Array Shape: ', X.shape)

        print('Label Array Shape: ', y.shape)

        return X/255., y/255.

    

    print('Ouptut Data Size: ', X.shape)

    return X/255.
X, y = images_to_array(extracting_path + '/train', extracting_path + '/train_cleaned')
#Divide our data to train and validation data.

val_split = int(.15 * data_size)

X_val, y_val = X[:val_split], y[:val_split]

X_train, y_train = X[val_split:], y[val_split:]

print('Train data shape: ', X_train.shape)

print('Test data shape: ', X_val.shape)
# First row will be raw data, second row will be the corresponding label images

samples = np.concatenate((X_train[:3], y_train[:3]), axis=0) 



f, ax = plt.subplots(2, 3, figsize=(20,10))

f.subplots_adjust(hspace = .05, wspace=.05)

for i, img in enumerate(samples):

    ax[i//3, i%3].imshow(img[:,:,0], cmap='gray')

    ax[i//3, i%3].axis('off')

plt.show() 
#Perpare the Autoencoder and compile it.

def create_model():

    input_layer = Input(shape=(None, None, 1))

    # encoder

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    x = MaxPooling2D((2, 2), padding='same')(x)



    # decoder

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

    x = UpSampling2D((2, 2))(x)

    output_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = keras.models.Model(inputs=[input_layer], outputs=[output_layer])

    autoencoder.compile(optimizer = 'adam' , loss = "mean_squared_error")

    return autoencoder



autoencoder = create_model()
history = autoencoder.fit(X_train, y_train, validation_data = (X_val, y_val), epochs=200, batch_size=16, verbose=False)
fig = plt.figure()

ax = plt.subplot(111)



ax.plot(history.epoch, history.history['loss'], label='Training Loss')

ax.plot(history.epoch, history.history['val_loss'], label='Validation Loss')



ax.legend()

plt.show()
print('Final validation loss: ',autoencoder.evaluate(X_val, y_val))
#Retrain same model on the whole dataset.

autoencoder = create_model()

autoencoder.compile(optimizer = 'adam' , loss = "mean_squared_error")

history = autoencoder.fit(X, y, epochs=300, batch_size=16, verbose=True)
print('Final loss: ',autoencoder.evaluate(X, y))
#Pick sample images from validation dataset.

test_samples, test_labels = X_val[:3], y_val[:3]

test_pred = autoencoder.predict(X_val[:3])



# First row will be raw data, second row will be the corresponding cleaned images

samples = np.concatenate((test_samples, test_labels, test_pred), axis=0) 

map_dict = {0:'Original Image', 1:'Label Image', 2:'Predicted Image'}

f, ax = plt.subplots(3, 3, figsize=(18,15))

f.tight_layout(pad=1.0)

for i, img in enumerate(samples):

    ax[i//3, i%3].imshow(img[:,:,0], cmap='gray')

    ax[i//3, i%3].title.set_text(map_dict[i//3])

    ax[i//3, i%3].axis('off')

plt.show() 
#Load and Scale test images into one big list.

image_names = sorted(os.listdir(extracting_path + '/test'))

data_size = len(image_names)

#initailize data arrays.

X_test = []

#read data.

for i in tqdm(range(data_size)):

    image_name = image_names[i]

    img_dir = os.path.join(extracting_path + '/test', image_name)

    img_pixels = load_img(img_dir, color_mode='grayscale')

    w, h = img_pixels.size

    X_test.append(np.array(img_pixels).reshape(1, h, w, 1) / 255.)

    

print('Test sample shape: ', X_test[0].shape)

print('Test sample dtype: ', X_test[0].dtype)
#Predict test images one by one and store them into a list.

yh_test = []

for img in X_test:

    yh_test.append(autoencoder.predict(img)[0, :, :, 0])
# First column will be raw data, second column will be the corresponding cleaned images.

f, ax = plt.subplots(2,3, figsize=(20,10))

f.subplots_adjust(hspace = .1, wspace=.05)

for i, (img, lbl) in enumerate(zip(X_test[:3], yh_test[:3])):

    ax[0, i].imshow(img[0,:,:,0], cmap='gray')

    ax[0, i].title.set_text('Original Image')

    ax[0, i].axis('off')



    ax[1, i].imshow(lbl, cmap='gray')

    ax[1, i].title.set_text('Cleaned Image')

    ax[1, i].axis('off')

plt.show() 
#Flatten the 'yh_test' list into 1-d list for submission.

submit_vector = []

for img in yh_test:

    h, w = img.shape

    for i in range(w):

        for j in range(h):

            submit_vector.append(img[j,i])

print(len(submit_vector))
#Make sure that we got the proper length.

c = 0

for img in yh_test:

    hi, wi = img.shape

    c += (hi * wi)

print('Total values :', c)
sample_csv = pd.read_csv(extracting_path + '/sampleSubmission.csv')

sample_csv.head(10)
id_col = sample_csv['id']

value_col = pd.Series(submit_vector, name='value')

submission = pd.concat([id_col, value_col], axis=1)

submission.head(10)
submission.to_csv('Cleared.csv',index = False)
#Always clear the output directory for faster committing.

import shutil

shutil.rmtree(extracting_path + '/train')

shutil.rmtree(extracting_path + '/test')

shutil.rmtree(extracting_path + '/train_cleaned')