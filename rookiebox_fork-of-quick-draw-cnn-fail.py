import os

import re

from glob import glob

from tqdm import tqdm

import numpy as np

import pandas as pd

import ast

import matplotlib.pyplot as plt

%matplotlib inline
fnames = glob('../input/train_simplified/*.csv') #<class 'list'>

cnames = ['countrycode', 'drawing', 'key_id', 'recognized', 'timestamp', 'word']

drawlist = []

for f in fnames[0:6]: # num of word : 5

    first = pd.read_csv(f, nrows=10) # make sure we get a recognized drawing

    first = first[first.recognized==True].head(2) # top head 2 get 

    drawlist.append(first)

draw_df = pd.DataFrame(np.concatenate(drawlist), columns=cnames) # <class 'pandas.core.frame.DataFrame'>

draw_df
draw_df.drawing.values[0]
evens = range(0,11,2)

odds = range(1,12, 2)

# We have drawing images, 2 per label, consecutively

df1 = draw_df[draw_df.index.isin(evens)]

df2 = draw_df[draw_df.index.isin(odds)]



example1s = [ast.literal_eval(pts) for pts in df1.drawing.values]

example2s = [ast.literal_eval(pts) for pts in df2.drawing.values]

labels = df2.word.tolist()



for i, example in enumerate(example1s):

    plt.figure(figsize=(6,3))

    

    for x,y in example:

        plt.subplot(1,2,1)

        plt.plot(x, y, marker='.')

        plt.axis('off')



    for x,y, in example2s[i]:

        plt.subplot(1,2,2)

        plt.plot(x, y, marker='.')

        plt.axis('off')

        label = labels[i]

        plt.title(label, fontsize=10)



    plt.show()  
%reset -f

# reset this program, deleting all pre-made variables, modules, functions, etc, that were before this cell


import os

from glob import glob

import re

import ast

import numpy as np 

import pandas as pd

from PIL import Image, ImageDraw 

from tqdm import tqdm

from dask import bag



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.models import Sequential

from keras.utils import to_categorical

from tensorflow.keras.layers import Dense, Dropout, Flatten

from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.metrics import top_k_categorical_accuracy

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
def pyth_test (x1, x2):

   

    print (x1 + x2)



pyth_test(1,2)
path = '../input/train_simplified/'

classfiles = os.listdir(path)



numstonames = {i: v[:-4].replace(" ", "_") for i, v in enumerate(classfiles)} # sleeping bag -> sleeping_bag

files = [os.path.join(path, file) for i, file in enumerate(classfiles)]

word_mapping = {file.split('/')[-1][:-4]:i for i, file in enumerate(files)}



num_classes = len(files)    #340

imheight, imwidth = 32, 32 # size of an image

ims_per_class = 2000  #max? # in the code above and above, there existed more than 100 thousand images per class(/label)

def draw_it(strokes):

    image = Image.new("P", (256,256), color=255) #"P":(8-bit pixels, mapped to any other mode using a color palette)

    image_draw = ImageDraw.Draw(image)

    for stroke in ast.literal_eval(strokes):

        for i in range(len(stroke[0])-1):

            image_draw.line([stroke[0][i], 

                             stroke[1][i],

                             stroke[0][i+1], 

                             stroke[1][i+1]],

                            fill=0, width=5)

    image = image.resize((imheight, imwidth))

    return np.array(image)/255.



def imageGenerator2(batchsize, validation=False):

    print(batchsize)

    df = []

    check = []

    T2 = []

    for file in files:

        if validation:

            df.append(pd.read_csv(file, nrows=110000, usecols=[1, 5]).tail(10000).sample(1000))

        else:

            df.append(pd.read_csv(file, nrows=100000, usecols=[1, 5]).sample(1000))

                

    df = pd.concat(df) 

    df['word'] = df['word'].map(word_mapping)

    df = df.sample(frac=1).reset_index(drop=True)

    y = to_categorical(df['word'].values, num_classes)

    print(y.shape)

    imagebag = bag.from_sequence(df['drawing'].values).map(draw_it)

    X = np.array(imagebag.compute())

    X = X.reshape(X.shape[0],imheight, imwidth, 1)

    print(X.shape)

    i = 0

    while True:

        if i+batchsize<=y.shape[0]:

#             print("if",i+batchsize)

            y_yield = y[i:i+batchsize]

            X_yield = X[i:i+batchsize]

#             print(y_yield.shape)

#             print(y_yield)

#             print(X_yield.shape)

#             print(X_yield)

            i += batchsize

            #yield (X_yield, y_yield)

        else:

            break



train_generator = imageGenerator2(batchsize=1000)
def draw_it(strokes):

    image = Image.new("P", (256,256), color=255) #"P":(8-bit pixels, mapped to any other mode using a color palette)

    image_draw = ImageDraw.Draw(image)

    for stroke in ast.literal_eval(strokes):

        for i in range(len(stroke[0])-1):

            image_draw.line([stroke[0][i], 

                             stroke[1][i],

                             stroke[0][i+1], 

                             stroke[1][i+1]],

                            fill=0, width=5)

    image = image.resize((imheight, imwidth))

    return np.array(image)/255.



def imageGenerator(batchsize, validation=False):

    #print(batchsize)

    while True:

        df = []

        check = []

        T2 = []

        for file in files:

            if validation:

                df.append(pd.read_csv(file, nrows=1100, usecols=[1, 5]).tail(100).sample(100))

            else:

                df.append(pd.read_csv(file, nrows=1000, usecols=[1, 5]).sample(100))

                

        df = pd.concat(df) 

        df['word'] = df['word'].map(word_mapping)

        df = df.sample(frac=1).reset_index(drop=True)

        y = to_categorical(df['word'].values, num_classes)

  

        imagebag = bag.from_sequence(df['drawing'].values).map(draw_it)

        X = np.array(imagebag.compute())

        X = X.reshape(X.shape[0],imheight, imwidth, 1)

    

        i = 0

        while True:

            if i+batchsize<=y.shape[0]:

                y_yield = y[i:i+batchsize]

                X_yield = X[i:i+batchsize]

                i += batchsize

                yield (X_yield, y_yield)

            else:

                break

    


# model = Sequential()

# model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(imheight, imwidth, 1)))

# model.add(MaxPooling2D(pool_size=(2, 2)))



# model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))

# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.2))



# model.add(Flatten())

# model.add(Dense(680, activation='relu'))

# model.add(Dropout(0.5))

# model.add(Dense(num_classes, activation='softmax'))



# model.summary()
# def top_3_accuracy(x,y): 

#     t3 = top_k_categorical_accuracy(x,y, 3)

#     return t3



# reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, 

#                                    verbose=1, mode='auto', min_delta=0.005, cooldown=5, min_lr=0.0001)

# earlystop = EarlyStopping(monitor='val_top_3_accuracy', mode='max', patience=5) 

# callbacks = [reduceLROnPlat, earlystop]



# model.compile(loss='categorical_crossentropy',

#               optimizer='adam',

#               metrics=['accuracy', top_3_accuracy])



# model.fit(x=X_train, y=y_train,

#           batch_size = 128,

#           epochs = 100,

#           validation_data = (X_val, y_val),

#           callbacks = callbacks,

#           verbose = 1)

# train_generator = imageGenerator(batchsize=1000)

# valid_generator = imageGenerator(batchsize=1000, validation=True)



# model.fit_generator(train_generator, steps_per_epoch=350, epochs=130, validation_data=valid_generator, validation_steps=5)
#%% get test set

# ttvlist = []

# reader = pd.read_csv('../input/test_simplified.csv', index_col=['key_id'],

#     chunksize=2048)

# for chunk in tqdm(reader, total=55):

#     imagebag = bag.from_sequence(chunk.drawing.values).map(draw_it)

#     testarray = np.array(imagebag.compute())

#     testarray = np.reshape(testarray, (testarray.shape[0], imheight, imwidth, 1))

#     testpreds = model.predict(testarray, verbose=0)

#     ttvs = np.argsort(-testpreds)[:, 0:3]  # top 3

#     ttvlist.append(ttvs)

    

# ttvarray = np.concatenate(ttvlist)
# preds_df = pd.DataFrame({'first': ttvarray[:,0], 'second': ttvarray[:,1], 'third': ttvarray[:,2]})

# preds_df = preds_df.replace(numstonames)

# preds_df['words'] = preds_df['first'] + " " + preds_df['second'] + " " + preds_df['third']



# sub = pd.read_csv('../input/sample_submission.csv', index_col=['key_id'])

# sub['word'] = preds_df.words.values

# sub.to_csv('subcnn_small.csv')

# sub.head()