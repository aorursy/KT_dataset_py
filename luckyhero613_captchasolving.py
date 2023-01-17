# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings
warnings.filterwarnings("ignore") #Matching warnings does not display again
import os #for interacting with operating system importing os
import gc #importing garbage collection module
import cv2 #importing OpenCV cv2 module
import glob #importing glob for finding all pathnames matching a specified pattern
import h5py #helps store huge amount of numerical data
import shutil #enables us to operate with file object easily
import itertools #provides various functions to work on iterators to produce complex iterators
import random as rn #random number generation

import imgaug as aug #for image augmentation
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #for stastical graphics in Python
from pathlib import Path #offers classes representing filesystem paths with semantics appropriate for different operating systems
from collections import Counter #container to stores elements as dictionary keys and count as dictionary values
import matplotlib.pyplot as plt #collection of work to make matplotlib work like MATLAB
import imgaug.augmenters as iaa #import augmenters for aritmetic changes, image color changes, etc.

from skimage.io import imread #image reading and writing
from skimage.transform import resize #for image resiszing to a certain size
from sklearn.metrics import confusion_matrix #used to describe the performance of classifier model
from mlxtend.plotting import plot_confusion_matrix #for visualizing confusion matrices via matplotlib
from sklearn.model_selection import train_test_split #for splitting matrices into random train and test subsets

from keras.models import Sequential, Model, load_model #sequential for grouping linear stack of layers in Model,and load_model for loading model 
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout #imorting 2D convolution layer, Max pooling operation for 2D data,
#regular densely neural networks,applies dropout to the input
from keras.layers import Input, Flatten, BatchNormalization, Lambda #flatten the input, normalization of scale inputs, wraps arbitrary expression as layer object
from keras.layers import GRU, LSTM, Bidirectional #Gated recurrent unit, Long Short term memory layer, Bidirectional wrapper for recurrent neural network
from keras.layers import Add, Concatenate, Reshape # layer add, concatenate a list of inputs and reshape input into a given shape
from keras.optimizers import Adam, SGD, RMSprop #compile using Adam algorithm, Gradient descent with momentum  optimizer, RMSprop algoritm optimizer 
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping #to save keras model at some frequency, abstract base class used to build new callbacks, stop training when a monitored metric has stopped working
from keras.utils import to_categorical #for conversion of class vector to binary class matrix

from keras import backend as K #import keras backend API for low level operations
import tensorflow as tf 


color = sns.color_palette()
#sets backend of matlplotlib to the inline backend
%matplotlib inline 
%config InlineBackend.figure_format="svg" #enables scalable graphics inline

# Set the seed for hash based operations in python
os.environ['PYTHONHASHSEED'] = '0'

seed=1234

# set the seed for random number generator
rn.seed(seed)

# Set the numpy seed
np.random.seed(seed)

# Set the random seed in tensorflow at graph level
tf.random.set_seed(seed)

#Using CuDNN implementation of RNNs which already is non-reproducible
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.get_session()

# Make the augmentation sequence deterministic
aug.seed(seed)
print("Seeding Successful")
data_dir = Path("../input/captcha6letterspng/samples/")

# getting list of all images
images = list(data_dir.glob("*.png"))
print("Number of images found: ", len(images))
sample_images = images[:4]

f,ax = plt.subplots(2,2, figsize=(5,3))
for i in range(4):
    img = imread(sample_images[i])
    print("Shape of image: ", img.shape)
    ax[i//2, i%2].imshow(img)
    ax[i//2, i%2].axis('off')
plt.show()
# make a set of all unique characters. .
letters = set()

# A list to store the max length for each catcha
lengths = []

# Iterate over each image. The name of the image is the 
# text contained in it. 
for image in images:
    image_name = str(image.name).split(".")[0]
    lengths.append(len(image_name))
    for ch in image_name:
        letters.add(ch)

# Sort the letters        
letters = sorted(letters)
print("Number of unqiue letters in the whole dataset: ", len(letters))
print("Maximum length of any captcha: ", max(Counter(lengths).keys()))
print("\nAll letters to be considered: ")
print(letters)
dataset = []

for image in images:
    image_path = str(image)
    label = str(image.name).split(".")[0]
    dataset.append((image_path, label))

dataset = pd.DataFrame(dataset, columns=["img_path", "label"], index=None)
dataset = dataset.sample(frac=1.).reset_index(drop=True)
print("Total number of samples in the dataset: ", len(dataset))
dataset.head(10)
# split into train and validation sets
training_data, validation_data = train_test_split(dataset, test_size=0.1, random_state=seed)

training_data = training_data.reset_index(drop=True)
validation_data = validation_data.reset_index(drop=True)

print("Number of training samples: ", len(training_data))
print("Number of validation samples: ", len(validation_data))
# function to create labels from text
def text_to_labels(text):
    return list(map(lambda x: letters.index(x), text))

# function to convert labels back to texts
def labels_to_text(label):
    return ''.join(list(map(lambda x: letters[int(x)], label)))

# sanity-check for letters
def is_valid_str(s):
    for ch in s:
        if not ch in letters:
            return False
    return True
def build_data(df, resize=True, img_height=50, img_width=200):
    """This function reads samples from a dataframe and store
    the image values and labels in two separate arrays.
    
    Args:
        df        : dataframe from which we want to read the data
        resize    : whether to resize images or not
        img_width : width of images to be considered
        img_height: height of images to be considered
        
    Returns:
        images    : numpy array of images
        labels    : numpy array of encoded labels
    """
    n = len(df)
    images = np.zeros((n, img_height, img_width), dtype=np.float32)
    labels = [0]*n
    for i in range(n):
        img = cv2.imread(df["img_path"][i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if resize:
            img = cv2.resize(img, (img_width, img_height))
        
        img = (img/255.).astype(np.float32)
        label = df["label"][i]
        
        # only add to if all the charaters are valid
        if is_valid_str(label):
            images[i, :, :] = img
            labels[i] = label
    
    return images, np.array(labels)
training_images, training_labels = build_data(training_data)
print("Number of training images: ", training_images.shape)
print("Number of training labels: ", training_labels.shape)
validation_images, validation_labels = build_data(validation_data)
print("Number of validation images: ", validation_images.shape)
print("Number of validation labels: ", validation_labels.shape)
f,ax = plt.subplots(4,2, figsize=(8,5))
for i in range(4):
    ax[i//2, i%2].imshow(training_images[i], cmap='gray')
    ax[i//2, i%2].set_title(training_labels[i])
    ax[i//2, i%2].axis('off')

for i in range(4, 8):
    ax[i//2, i%2].imshow(validation_images[i], cmap='gray')
    ax[i//2, i%2].set_title(validation_labels[i])
    ax[i//2, i%2].axis('off')
    
plt.show()
def data_generator(df, 
                   batch_size, 
                   img_width, 
                   img_height, 
                   downsample_factor, 
                   max_text_len, 
                   is_validation_data=False):
    """This is a data generator which yields batches 
    of (image, label) pairs.
    
    Args:
        df                : training or validation dataframe
        batch_size        : batch size to be used during training
        img_width         : width of images to be considered  
        img_height        : height of images to be considered
        downsample_factor : by what factor the CNN has downsampled the images
        max_text_len      : maximum length of the text in your data
        is_validation_data: is the data being considered a validation data?
        
    Returns:
        inputs: numpy array containg inputs that are required for the final model
        outputs: a dummy array of zeros 
    """
    n = len(df)
    indices = np.arange(n)
    np.random.shuffle(indices)
    nb_batches = int(np.ceil(n/batch_size))
    
    if not is_validation_data:
        images, texts = training_images, training_labels
    else:
        images, texts = validation_images, validation_labels
    
    batch_images = np.ones((batch_size, img_width, img_height, 1), dtype=np.float32)
    batch_labels = np.ones((batch_size, max_text_len), dtype=np.float32)
    input_length = np.ones((batch_size, 1), dtype=np.int64) * \
                                            (img_width // downsample_factor - 2)
    label_length = np.zeros((batch_size, 1), dtype=np.int64)
    
    while True:
        for i in range(nb_batches):
            idx_to_consider = indices[i*batch_size:(i+1)*batch_size]
            
            for j, idx in enumerate(idx_to_consider):
                img = images[idx].T
                img = np.expand_dims(img, axis=-1)
                text = texts[idx]
                
                if is_valid_str(text):
                    label = text_to_labels(texts[idx])
                    batch_images[j] = img
                    batch_labels[j] = label
                    label_length[j] = len(text)

            inputs = {
            'input_data': batch_images,
            'input_label': batch_labels,
            'input_length': input_length,
            'label_length': label_length,
            }
            
            outputs = {'ctc_loss': np.zeros([batch_size], dtype=np.float32)}
            yield inputs, outputs
# batch size to be used for training
batch_size = 32

# image dimensions
img_width=200
img_height=50 

# by what factor the image has been downsampled by the CNN part?
downsample_factor=5

# maximum length of any text in the data
max_text_len=5
# Get a generator object for the training data
train_data_generator = data_generator(training_data, 
                                      batch_size=batch_size, 
                                      img_width=img_width, 
                                      img_height=img_height, 
                                      downsample_factor=downsample_factor, 
                                      max_text_len=max_text_len, 
                                      is_validation_data=False)

# Get a generator object for the validation data 
valid_data_generator = data_generator(validation_data, 
                                      batch_size=batch_size, 
                                      img_width=img_width, 
                                      img_height=img_height, 
                                      downsample_factor=downsample_factor, 
                                      max_text_len=max_text_len, 
                                      is_validation_data=True)
# A handy-dandy function for checking the generator output
# always sanity-check the data before passing it to the model
def visualize_data_gen_output(data_gen, samples_to_visualize=2):
    for i, (inp, out) in enumerate(data_gen):
        print('Text generator output (data which will be fed into the neutral network):')
        print('1)the_input (image)')
        img = (inp['input_data'][i, :, :, 0]*255).astype(np.uint8)
        plt.imshow(img.T, cmap='gray')
        plt.show()
        print(f"2) the_labels(captcha) {labels_to_text(inp['input_label'][i])} is encoded as {list(map(int, inp['input_label'][i]))}") 
        print(f"3) input_length (width of image that is fed to the network after CNN): {inp['input_length'][i][0]} == (200/5 - 2)")
        print(f"4) label_length (length of captcha): {inp['label_length'][i][0]}")
        print(" ")
        if i==samples_to_visualize:
            break
visualize_data_gen_output(train_data_generator)
# Using this loss function as the output
# The loss function in model.compile(..) will be a dummy one
# This is different from a normal scenario where we pass an actual 
# loss function when compile the model
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
def build_model():
    # Inputs to the model
    input_img = Input(shape=(img_width, img_height, 1), name='input_data', dtype='float32')
    labels = Input(name='input_label', shape=[max_text_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    
    # Convolution part for feaure extraction
    x = Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same', name='Conv1')(input_img)
    x = MaxPooling2D((2,2), name='pool1')(x)
    x = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same', name='Conv2')(x)
    x = MaxPooling2D((2,2), name='pool2')(x)
    
    # Reshape the features for passing to RNN
    # We have used two max pool with pool size and strides of 2. Hence, downsampled is 4x smaller
    # Also, the number of filters in the last layer is 64.
    new_shape = ((img_width // 4), (img_height // 4)*64)
    x = Reshape(target_shape=new_shape, name='reshape')(x)
    x = Dense(64, activation='relu', name='dense1')(x)
    
    # RNNs
    x = Bidirectional(LSTM(128, return_sequences=True,  name='lstm_1'), name='bi_1')(x)
    x = Bidirectional(LSTM(128, return_sequences=True,  name='lstm_2'), name='bi_2')(x)
    
    # final part
    x = Dense(len(letters)+1, activation='softmax', name='dense2', kernel_initializer='he_normal')(x)
    
    # Get the CTC loss and represent it in a layer
    output = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc_loss')([x, labels, input_length, label_length])
    
    # define the final model
    model = Model([input_img, labels, input_length, label_length], output, name='ocr_model_v1')
    
    # optimizer
    sgd = SGD(lr=0.002, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    
    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    # this is the reason we have this ctc_loss array of zeros in our generator
    model.compile(loss={'ctc_loss': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    return model
model = build_model()
model.summary()
# things required for starting the training 
nb_epochs = 50
nb_train_steps = training_data.shape[0] // batch_size
nb_validation_steps = validation_data.shape[0] // batch_size
es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
ckpt = ModelCheckpoint(filepath='ocr_v2.h5', save_best_only=True, monitor='val_loss')
# Train the model
history = model.fit_generator(train_data_generator, 
                    epochs=nb_epochs, 
                    steps_per_epoch=nb_train_steps, 
                    validation_data=valid_data_generator, 
                    validation_steps=nb_validation_steps,
                    callbacks=[es, ckpt])
def decode_batch_predictions(pred):
    pred = pred[:, 2:]
    input_len = np.ones(pred.shape[0])*pred.shape[1]
    
    # Use greedy search. For complex tasks, you can use beam search
    results = K.get_value(K.ctc_decode(pred, 
                                   input_length=input_len,
                                   greedy=True)[0][0])
    
    # Iterate over the results and get back the text
    texts = []
    for res in results:
        outstr = ''
        for c in res:
            if c < len(letters):
                outstr += letters[c]
        texts.append(outstr)
    
    # return final text results
    return texts
# Get the input output layer and define a Keras function
# It is similar to getting layers in tensorflow and 
# passing the information to the session.
output_func = K.function([model.get_layer(name='input_data').input],
                        [model.get_layer(name='dense2').output])


#  Let's check results on some validation samples
for p, (inp_value, _) in enumerate(valid_data_generator):
    bs = inp_value['input_data'].shape[0]
    X_data = inp_value['input_data']
    labels = inp_value['input_label']
    
    preds = output_func([X_data])[0]
    pred_texts = decode_batch_predictions(preds)
    
    
    orig_texts = []
    for label in labels:
        text = ''.join(list(map(lambda x: letters[int(x)], label)))
        orig_texts.append(text)
        
    for i in range(bs):
        print(f'GT: {orig_texts[i]} \t Predicted: {pred_texts[i]}')
    break