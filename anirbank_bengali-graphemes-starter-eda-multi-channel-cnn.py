!pip install -U tensorflow
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm.auto import tqdm

from glob import glob

import time, gc

import cv2



import tensorflow as tf

import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont

from matplotlib import pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
print(tf.__version__)
train_df_ = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')

test_df_ = pd.read_csv('/kaggle/input/bengaliai-cv19/test.csv')

class_map_df = pd.read_csv('/kaggle/input/bengaliai-cv19/class_map.csv')

sample_sub_df = pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')
train_df_.head()
test_df_.head()
sample_sub_df.head()
class_map_df.head()
print(f'Size of training data: {train_df_.shape}')

print(f'Size of test data: {test_df_.shape}')

print(f'Size of class map: {class_map_df.shape}')
HEIGHT = 236

WIDTH = 236



def get_n(df, field, n, top=True):

    top_graphemes = df.groupby([field]).size().reset_index(name='counts')['counts'].sort_values(ascending=not top)[:n]

    top_grapheme_roots = top_graphemes.index

    top_grapheme_counts = top_graphemes.values

    top_graphemes = class_map_df[class_map_df['component_type'] == field].reset_index().iloc[top_grapheme_roots]

    top_graphemes.drop(['component_type', 'label'], axis=1, inplace=True)

    top_graphemes.loc[:, 'count'] = top_grapheme_counts

    return top_graphemes



def image_from_char(char):

    image = Image.new('RGB', (WIDTH, HEIGHT))

    draw = ImageDraw.Draw(image)

    myfont = ImageFont.truetype('/kaggle/input/kalpurush-fonts/kalpurush-2.ttf', 120)

    w, h = draw.textsize(char, font=myfont)

    draw.text(((WIDTH - w) / 2,(HEIGHT - h) / 3), char, font=myfont)



    return image
print(f'Number of unique grapheme roots: {train_df_["grapheme_root"].nunique()}')

print(f'Number of unique vowel diacritic: {train_df_["vowel_diacritic"].nunique()}')

print(f'Number of unique consonant diacritic: {train_df_["consonant_diacritic"].nunique()}')
top_10_roots = get_n(train_df_, 'grapheme_root', 10)

top_10_roots
f, ax = plt.subplots(2, 5, figsize=(16, 8))

ax = ax.flatten()



for i in range(10):

    ax[i].imshow(image_from_char(top_10_roots['component'].iloc[i]), cmap='Greys')
bottom_10_roots = get_n(train_df_, 'grapheme_root', 10, False)

bottom_10_roots
f, ax = plt.subplots(2, 5, figsize=(16, 8))

ax = ax.flatten()



for i in range(10):

    ax[i].imshow(image_from_char(bottom_10_roots['component'].iloc[i]), cmap='Greys')
top_5_vowels = get_n(train_df_, 'vowel_diacritic', 5)

top_5_vowels
f, ax = plt.subplots(1, 5, figsize=(16, 8))

ax = ax.flatten()



for i in range(5):

    ax[i].imshow(image_from_char(top_5_vowels['component'].iloc[i]), cmap='Greys')
top_5_consonants = get_n(train_df_, 'consonant_diacritic', 5)

top_5_consonants
f, ax = plt.subplots(1, 5, figsize=(16, 8))

ax = ax.flatten()



for i in range(5):

    ax[i].imshow(image_from_char(top_5_consonants['component'].iloc[i]), cmap='Greys')
train_df_ = train_df_.drop(['grapheme'], axis=1, inplace=False)
train_df_[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']] = train_df_[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].astype('uint8')
IMG_SIZE=64

N_CHANNELS=1
def resize(df, size=IMG_SIZE, need_progress_bar=True):

    resized = {}

    if need_progress_bar:

        for i in tqdm(range(df.shape[0])):

            image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size, size))

            resized[df.index[i]] = image.reshape(-1)

    else:

        for i in range(df.shape[0]):

            image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size, size))

            resized[df.index[i]] = image.reshape(-1)

    resized = pd.DataFrame(resized).T

    return resized
def get_dummies(df):

    cols = []

    for col in df:

        cols.append(pd.get_dummies(df[col].astype(str)))

    return pd.concat(cols, axis=1)
class BengaliNet:

    @staticmethod

    def build_grapheme_branch(inputs, numGraphemes,finalAct="softmax", chanDim=-1):

 

        x = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(inputs)

        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.BatchNormalization(axis=chanDim)(x)

        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3))(x)

        x = tf.keras.layers.Dropout(0.25)(x)

        

        x = tf.keras.layers.Conv2D(64, (3, 3), padding="same")(x)

        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.BatchNormalization(axis=chanDim)(x)

        x = tf.keras.layers.Conv2D(64, (3, 3), padding="same")(x)

        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.BatchNormalization(axis=chanDim)(x)

        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        x = tf.keras.layers.Dropout(0.25)(x)



        x = tf.keras.layers.Conv2D(128, (3, 3), padding="same")(x)

        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.BatchNormalization(axis=chanDim)(x)

        x = tf.keras.layers.Conv2D(128, (3, 3), padding="same")(x)

        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.BatchNormalization(axis=chanDim)(x)

        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        x = tf.keras.layers.Dropout(0.25)(x)



        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(256)(x)

        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Dropout(0.5)(x)

        x = tf.keras.layers.Dense(numGraphemes)(x)

        x = tf.keras.layers.Activation(finalAct, name="grapheme_output")(x)

 

        # return the Grapheme prediction sub-network

        return x

    

    @staticmethod

    def build_vowel_branch(inputs, numVowels, finalAct="softmax",chanDim=-1):



        x = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(inputs)

        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.BatchNormalization(axis=chanDim)(x)

        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3))(x)

        x = tf.keras.layers.Dropout(0.25)(x)

        

        x = tf.keras.layers.Conv2D(64, (3, 3), padding="same")(x)

        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.BatchNormalization(axis=chanDim)(x)

        x = tf.keras.layers.Conv2D(64, (3, 3), padding="same")(x)

        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.BatchNormalization(axis=chanDim)(x)

        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        x = tf.keras.layers.Dropout(0.25)(x)



        x = tf.keras.layers.Conv2D(128, (3, 3), padding="same")(x)

        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.BatchNormalization(axis=chanDim)(x)

        x = tf.keras.layers.Conv2D(128, (3, 3), padding="same")(x)

        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.BatchNormalization(axis=chanDim)(x)

        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        x = tf.keras.layers.Dropout(0.25)(x)



        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(128)(x)

        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Dropout(0.5)(x)

        x = tf.keras.layers.Dense(numVowels)(x)

        x = tf.keras.layers.Activation(finalAct, name="vowel_output")(x)



        # return the vowel prediction sub-network

        return x

    

    @staticmethod

    def build_consonant_branch(inputs, numConsonants, finalAct="softmax",chanDim=-1):



        x = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(inputs)

        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.BatchNormalization(axis=chanDim)(x)

        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3))(x)

        x = tf.keras.layers.Dropout(0.25)(x)

        

        x = tf.keras.layers.Conv2D(64, (3, 3), padding="same")(x)

        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.BatchNormalization(axis=chanDim)(x)

        x = tf.keras.layers.Conv2D(64, (3, 3), padding="same")(x)

        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.BatchNormalization(axis=chanDim)(x)

        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        x = tf.keras.layers.Dropout(0.25)(x)



        x = tf.keras.layers.Conv2D(128, (3, 3), padding="same")(x)

        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.BatchNormalization(axis=chanDim)(x)

        x = tf.keras.layers.Conv2D(128, (3, 3), padding="same")(x)

        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.BatchNormalization(axis=chanDim)(x)

        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        x = tf.keras.layers.Dropout(0.25)(x)



        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(128)(x)

        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Dropout(0.5)(x)

        x = tf.keras.layers.Dense(numConsonants)(x)

        x = tf.keras.layers.Activation(finalAct, name="consonant_output")(x)



        # return the consonant prediction sub-network

        return x

    

    @staticmethod

    def build(width, height, numGraphemes, numVowels, numConsonants, finalAct="softmax"):

        # initialize the input shape and channel dimension (this code

        # assumes you are using TensorFlow which utilizes channels

        # last ordering)

        inputShape = (height, width,1)

        chanDim = -1



        # construct both the "grapheme" , "vowel", and "consonant" sub-networks

        inputs = tf.keras.layers.Input(shape=inputShape)

        graphemeBranch = BengaliNet.build_grapheme_branch(inputs,

            numGraphemes, finalAct=finalAct, chanDim=chanDim)

        vowelBranch = BengaliNet.build_vowel_branch(inputs,

            numVowels, finalAct=finalAct, chanDim=chanDim)

        consonantBranch = BengaliNet.build_consonant_branch(inputs,

            numConsonants, finalAct=finalAct, chanDim=chanDim)



        # create the model using our input (the batch of images) and

        # three separate outputs -- one for the grapheme

        # branch, the vowel branch, and consonant branch respectively

        model = tf.keras.models.Model(

            inputs=inputs,

            outputs=[graphemeBranch, vowelBranch, consonantBranch],

            name="Bengalinet")



        # return the constructed network architecture

        return model

EPOCHS = 25

INIT_LR = 1e-3

BS = 32
model = BengaliNet.build(64, 64,numGraphemes=168,numVowels=11,numConsonants=7,finalAct="softmax")



# define two dictionaries: one that specifies the loss method for

# each output of the network along with a second dictionary that

# specifies the weight per loss

losses = {

    "grapheme_output": "categorical_crossentropy",

    "vowel_output": "categorical_crossentropy",

    "consonant_output": "categorical_crossentropy"

}

lossWeights = {"grapheme_output": 1.0, "vowel_output": 1.0, "consonant_output":1.0}



# initialize the optimizer and compile the model

print("[INFO] compiling model...")

opt = tf.keras.optimizers.Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,metrics=["accuracy"])
model.summary()
HEIGHT = 137

WIDTH = 236
histories = []

for i in range(4):

    train_df = pd.merge(pd.read_parquet(f'/kaggle/input/bengaliai-cv19/train_image_data_{i}.parquet'), train_df_, on='image_id').drop(['image_id'], axis=1)

    

    # Visualize few samples of current training dataset

    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(16, 8))

    count=0

    for row in ax:

        for col in row:

            col.imshow(resize(train_df.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1).iloc[[count]], need_progress_bar=False).values.reshape(-1).reshape(IMG_SIZE, IMG_SIZE).astype(np.float64))

            count += 1

    plt.show()

    

    X_train = train_df.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1)

    X_train = resize(X_train)

    

    # CNN takes images in shape `(batch_size, h, w, channels)`, so reshape the images

    X_train = X_train.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)

    

    Y_train_root = pd.get_dummies(train_df['grapheme_root']).values

    Y_train_vowel = pd.get_dummies(train_df['vowel_diacritic']).values

    Y_train_consonant = pd.get_dummies(train_df['consonant_diacritic']).values



    print(f'Training images: {X_train.shape}')

    print(f'Training labels root: {Y_train_root.shape}')

    print(f'Training labels vowel: {Y_train_vowel.shape}')

    print(f'Training labels consonants: {Y_train_consonant.shape}')



    # Divide the data into training and validation set

    x_train, x_test, y_train_root, y_test_root, y_train_vowel, y_test_vowel, y_train_consonant, y_test_consonant = train_test_split(X_train, Y_train_root, Y_train_vowel, Y_train_consonant, test_size=0.08, random_state=666)

    del train_df

    del X_train

    del Y_train_root, Y_train_vowel, Y_train_consonant



    # This will just calculate parameters required to augment the given data. This won't perform any augmentations

    print("train_size:"+str(x_train.shape[0]))



    # Fit the model

    history = model.fit((x_train, {"grapheme_output": y_train_root, "vowel_output": y_train_vowel, "consonant_output": y_train_consonant}),batch_size=BS,

                              epochs = EPOCHS, validation_data = (x_test, [y_test_root, y_test_vowel, y_test_consonant]), 

                              steps_per_epoch=x_train.shape[0] // BS, verbose=1 )

    histories.append(history)

    

    # Delete to reduce memory usage

    del x_train

    del x_test

    del y_train_root

    del y_test_root

    del y_train_vowel

    del y_test_vowel

    del y_train_consonant

    del y_test_consonant

    gc.collect()
%matplotlib inline

def plot_loss(his, epoch, title):

    plt.style.use('ggplot')

    plt.figure()

    plt.plot(np.arange(0, epoch), his.history['loss'], label='train_loss')

    plt.plot(np.arange(0, epoch), his.history['grapheme_output_loss'], label='train_root_loss')

    plt.plot(np.arange(0, epoch), his.history['vowel_output_loss'], label='train_vowel_loss')

    plt.plot(np.arange(0, epoch), his.history['consonant_output_loss'], label='train_consonant_loss')

    

    plt.plot(np.arange(0, epoch), his.history['val_grapheme_output_loss'], label='val_train_root_loss')

    plt.plot(np.arange(0, epoch), his.history['val_vowel_output_loss'], label='val_train_vowel_loss')

    plt.plot(np.arange(0, epoch), his.history['val_consonant_output_loss'], label='val_train_consonant_loss')

    

    plt.title(title)

    plt.xlabel('Epoch #')

    plt.ylabel('Loss')

    plt.legend(loc='upper right')

    plt.show()



def plot_acc(his, epoch, title):

    plt.style.use('ggplot')

    plt.figure()

    plt.plot(np.arange(0, epoch), his.history['grapheme_output_accuracy'], label='train_root_acc')

    plt.plot(np.arange(0, epoch), his.history['vowel_output_accuracy'], label='train_vowel_accuracy')

    plt.plot(np.arange(0, epoch), his.history['consonant_output_accuracy'], label='train_consonant_accuracy')

    

    plt.plot(np.arange(0, epoch), his.history['val_grapheme_output_accuracy'], label='val_root_acc')

    plt.plot(np.arange(0, epoch), his.history['val_vowel_output_accuracy'], label='val_vowel_accuracy')

    plt.plot(np.arange(0, epoch), his.history['val_consonant_output_accuracy'], label='val_consonant_accuracy')

    plt.title(title)

    plt.xlabel('Epoch #')

    plt.ylabel('Accuracy')

    plt.legend(loc='upper right')

    plt.show()
for dataset in range(4):

    plot_loss(histories[dataset], epochs, f'Training Dataset: {dataset}')

    plot_acc(histories[dataset], epochs, f'Training Dataset: {dataset}')
del histories

gc.collect()
preds_dict = {

    'grapheme_root': [],

    'vowel_diacritic': [],

    'consonant_diacritic': []

}
components = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']

target=[] # model predictions placeholder

row_id=[] # row_id place holder

for i in range(4):

    df_test_img = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_{}.parquet'.format(i)) 

    df_test_img.set_index('image_id', inplace=True)



    X_test = resize(df_test_img, need_progress_bar=False)

    X_test = X_test.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)

    

    preds = model.predict(X_test)



    for i, p in enumerate(preds_dict):

        preds_dict[p] = np.argmax(preds[i], axis=1)



    for k,id in enumerate(df_test_img.index.values):  

        for i,comp in enumerate(components):

            id_sample=id+'_'+comp

            row_id.append(id_sample)

            target.append(preds_dict[comp][k])

    del df_test_img

    del X_test

    gc.collect()



df_sample = pd.DataFrame(

    {

        'row_id': row_id,

        'target':target

    },

    columns = ['row_id','target'] 

)

df_sample.to_csv('submission.csv',index=False)

df_sample.head()