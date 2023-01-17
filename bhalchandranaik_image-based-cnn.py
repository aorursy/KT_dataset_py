import os
import re
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
%matplotlib inline
fnames = glob('../input/train_simplified/*.csv')
cnames = ['countrycode', 'drawing', 'key_id', 'recognized', 'timestamp', 'word']
drawlist = []
for f in fnames[0:6]:
    first = pd.read_csv(f, nrows=10) # make sure we get a recognized drawing
    first = first[first.recognized==True].head(2)
    drawlist.append(first)
draw_df = pd.DataFrame(np.concatenate(drawlist), columns=cnames)
draw_df
evens = range(0,11,2)
odds = range(1,12, 2)
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
# # commented out to save memory
# import urllib

# LABELS = np.array(['baseball', 'bowtie', 'clock', 'hand', 'hat'])
# for b in LABELS:
#     url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy".format(b)
#     urllib.request.urlretrieve(url, "{}.npy".format(b))
#     nb = np.load("{}.npy".format(b))
#     print("\n Class '{0}' has {1} examples of {2}x{2} images".format(b, nb.shape[0], int(nb.shape[1]**0.5)))
%reset -f 
#%% import
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
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, concatenate, Input
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
#%% set label dictionary and params
classfiles = os.listdir('../input/train_simplified/')
numstonames = {i: v[:-4].replace(" ", "_") for i, v in enumerate(classfiles)} #adds underscores

num_classes = 20    #340 max 
imheight, imwidth = 32, 32  
ims_per_class = 10000  #max?
# faster conversion function
def draw_it(strokes):
    image = Image.new("P", (256,256), color=255)
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

#%% get train arrays
train_grand = []
num_classes = 20
class_paths = glob('../input/train_simplified/*.csv')
for i,c in enumerate(tqdm(class_paths[0: num_classes])):
    train = pd.read_csv(c, usecols=['drawing', 'recognized'], nrows=ims_per_class*5//4)
    train = train[train.recognized == True].head(ims_per_class)
    imagebag = bag.from_sequence(train.drawing.values).map(draw_it) 
    trainarray = np.array(imagebag.compute())  # PARALLELIZE
    trainarray = np.reshape(trainarray, (ims_per_class, -1))    
    labelarray = np.full((train.shape[0], 1), i)
    trainarray = np.concatenate((labelarray, trainarray), axis=1)
    train_grand.append(trainarray)
    

train_grand = np.array([train_grand.pop() for i in np.arange(num_classes)]) #less memory than np.concatenate
train_grand = train_grand.reshape((-1, (imheight*imwidth+1)))
del trainarray
del train
# memory-friendly alternative to train_test_split?
valfrac = 0.1
cutpt = int(valfrac * train_grand.shape[0])

np.random.shuffle(train_grand)
y_train, X_train = train_grand[cutpt: , 0], train_grand[cutpt: , 1:]
y_val, X_val = train_grand[0:cutpt, 0], train_grand[0:cutpt, 1:] #validation set is recognized==True

# del train_grand
y_train = keras.utils.to_categorical(y_train, num_classes)
X_train = X_train.reshape(X_train.shape[0], imheight, imwidth, 1)
y_val = keras.utils.to_categorical(y_val, num_classes)
X_val = X_val.reshape(X_val.shape[0], imheight, imwidth, 1)
print(y_train.shape, "\n",
      X_train.shape, "\n",
      y_val.shape, "\n",
      X_val.shape)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(imheight, imwidth, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(680, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()
def top_3_accuracy(x,y): 
    t3 = top_k_categorical_accuracy(x,y, 3)
    return t3
model_1 = Sequential()
model_1.add(Conv2D(12, kernel_size = (3,3), padding='same', activation='tanh', strides=1, input_shape=(32,32,1)))
model_1.add(AveragePooling2D(pool_size=(2, 2), strides=2))
model_1.add(Conv2D(32, kernel_size = (3,3), padding='same', activation='tanh', strides=1))
model_1.add(AveragePooling2D(pool_size=(2, 2), strides=2))
model_1.add(Conv2D(32, kernel_size = (3,3), padding='same', activation='tanh', strides=1))
model_1.add(AveragePooling2D(pool_size=(2, 2), strides=2))
model_1.add(Conv2D(240, kernel_size = (3,3), padding='same', activation='tanh', strides=1))
model_1.add(Flatten())
model_1.add(Dense(84, activation='tanh'))
model_1.add(Dense(num_classes, activation='softmax'))
model_1.summary()
model_1.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', top_3_accuracy])

model_1.fit(x=X_train, y=y_train,
          batch_size = 32,
          epochs = 20,
          validation_data = (X_val, y_val),
#           callbacks = callbacks,
          verbose = 1)
model_2 = Sequential()
model_2.add(Conv2D(32, kernel_size = (3,3), padding='same', activation='relu', strides=1, input_shape=(32,32,1)))
model_2.add(MaxPooling2D(pool_size=(2, 2), strides=1))
model_2.add(BatchNormalization(axis=1))

model_2.add(Conv2D(86, kernel_size = (3,3), padding='same', activation='relu', strides=1))
model_2.add(MaxPooling2D(pool_size=(3, 3), strides=2))
model_2.add(BatchNormalization(axis=1))

model_2.add(Conv2D(128, kernel_size = (3,3), padding='same', activation='relu', strides=1))
model_2.add(Conv2D(128, kernel_size = (3,3), padding='same', activation='relu', strides=1))
model_2.add(Conv2D(128, kernel_size = (3,3), padding='same', activation='relu', strides=1))
model_2.add(MaxPooling2D(pool_size=(3, 3), strides=2))
model_2.add(Flatten())

model_2.add(Dense(64, activation='relu'))
model_2.add(Dense(64, activation='relu'))
model_2.add(Dense(num_classes, activation='softmax'))
model_2.summary()
model_2.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', top_3_accuracy])

model_2.fit(x=X_train, y=y_train,
          batch_size = 32,
          epochs = 20,
          validation_data = (X_val, y_val),
#           callbacks = callbacks,
          verbose = 1)
from keras.layers import merge
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, concatenate, Input
from keras.metrics import top_k_categorical_accuracy
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

input_layer = Input(shape=(32,32,1))

stem = Conv2D(32, kernel_size = (4,4), padding='same', activation='relu', strides=1)(input_layer)
stem = MaxPooling2D(pool_size=(3, 3), strides=2)(stem)
stem = BatchNormalization(axis=1)(stem)

stem = Conv2D(32, kernel_size = (1,1), padding='same', activation='relu', strides=1)(stem)
stem = Conv2D(32, kernel_size = (3,3), padding='same', activation='relu', strides=1)(stem)
stem = BatchNormalization(axis=1)(stem)

tower1_1 = Conv2D(64, kernel_size = (1,1), padding='same', activation='relu', strides=1)(stem)

tower1_2 = Conv2D(64, kernel_size = (1,1), padding='same', activation='relu', strides=1)(stem)
tower1_2 = Conv2D(64, kernel_size = (3,3), padding='same', activation='relu', strides=1)(tower1_2)

tower1_3 = Conv2D(64, kernel_size = (1,1), padding='same', activation='relu', strides=1)(stem)
tower1_3 = Conv2D(64, kernel_size = (5,5), padding='same', activation='relu', strides=1)(tower1_3)

tower1_4 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(stem)
tower1_4 = Conv2D(64, kernel_size = (1,1), padding='same', activation='relu', strides=1)(tower1_4)

output_inception_1 = concatenate([tower1_1, tower1_2, tower1_3, tower1_4])


# INCEPTION MODULE 2
tower2_1 = Conv2D(64, kernel_size = (1,1), padding='same', activation='relu', strides=1)(output_inception_1)

tower2_2 = Conv2D(64, kernel_size = (1,1), padding='same', activation='relu', strides=1)(output_inception_1)
tower2_2 = Conv2D(64, kernel_size = (3,3), padding='same', activation='relu', strides=1)(tower1_2)

tower2_3 = Conv2D(64, kernel_size = (1,1), padding='same', activation='relu', strides=1)(output_inception_1)
tower2_3 = Conv2D(64, kernel_size = (5,5), padding='same', activation='relu', strides=1)(tower1_3)

tower2_4 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(output_inception_1)
tower2_4 = Conv2D(64, kernel_size = (1,1), padding='same', activation='relu', strides=1)(tower1_4)

output_inception_2 = concatenate([tower2_1, tower2_2, tower2_3, tower2_4])

#FINAL OUTPUT APEX
apex =  AveragePooling2D(pool_size=(5,5), strides=1)(output_inception_2)
apex = Flatten()(apex)
apex = Dropout(0.5)(apex)
apex = Dense(num_classes, activation='softmax')(apex)

model_3 = Model([input_layer], outputs = apex)
model_3.summary()

model_3.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', top_3_accuracy])

model_3.fit(x=X_train, y=y_train,
          batch_size = 32,
          epochs = 20,
          validation_data = (X_val, y_val),
#           callbacks = callbacks,
          verbose = 1)
# #%% get test set
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
test_images = 20000
test_grand_x = list()
test_grand_y = list()
for i,c in enumerate(tqdm(class_paths[0: num_classes])):
    test = pd.read_csv(c, usecols=['drawing', 'recognized'], nrows=(test_images+ims_per_class)*5//4)
    test = test.tail(test_images)
    imagebag = bag.from_sequence(test.drawing.values).map(draw_it) 
    testarray = np.array(imagebag.compute())  # PARALLELIZE
    testarray = np.reshape(testarray, (test_images, -1))    
    labelarray = np.full((test.shape[0], 1), i)
    test_grand_x.append(testarray)
    test_grand_y.append(labelarray)
    
test_grand_x = np.concatenate(test_grand_x)
test_grand_y = np.concatenate(test_grand_y)
test_grand_x = test_grand_x.reshape(test_grand_x.shape[0], imheight, imwidth, 1)
test_predicted_1 = model_1.predict(test_grand_x, verbose=0)
print('done prediction 1')
test_predicted_2 = model_2.predict(test_grand_x, verbose=0)
print('done prediction 2')
test_predicted_3 = model_3.predict(test_grand_x, verbose=0)
print('done prediction 3')

top_three_classes_1 = np.argsort(-test_predicted_1)[:, 0:3]
top_three_classes_2 = np.argsort(-test_predicted_2)[:, 0:3]
top_three_classes_3 = np.argsort(-test_predicted_3)[:, 0:3]

count_1 = 0
count_2 = 0
count_3 = 0
for i, o_1 in enumerate(tqdm(top_three_classes_1)):
    if test_grand_y[i] in top_three_classes_1[i]:
        count_1 = count_1+1
    if test_grand_y[i] in top_three_classes_2[i]:
        count_2 = count_2+1 
    if test_grand_y[i] in top_three_classes_3[i]:
        count_3 = count_3+1
        
print(count_1/top_three_classes_1.shape[0])
print(count_2/top_three_classes_2.shape[0])
print(count_3/top_three_classes_3.shape[0])


