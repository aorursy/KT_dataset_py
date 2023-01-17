# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt  # plotting 

#import matplotlib.font_manager as fm # to plot the font

from tqdm.auto import tqdm # see progress bar

#import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont # Darw picture from font

from skimage.transform import resize # Resizing of image

from cv2 import resize as cv2_resize # resizng of image

from keras.preprocessing.image import ImageDataGenerator  # image augmentation on training images ONLY

#from sklearn.model_selection import train_test_split  # splitting the data

import keras.backend as K # for custom metrices implementations and other processes that we define

from keras.layers import Dense,BatchNormalization,Input,Dropout,Conv2D,Flatten,MaxPool2D,LeakyReLU,Activation,Concatenate # keras layers

from keras.models import Model #Model class

from keras.optimizers import Adam #optimizer

from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping

from keras import  models

from keras import optimizers

from keras import applications

from keras.utils import to_categorical

# Call backs acts like milestones and if/else while model is being trained

import gc # garbage collector

import sys

import cv2

import time

from keras.preprocessing.image import ImageDataGenerator



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
BATCH_SIZE = 100



TEST_IMG_PATH = '../input/bengaliai-cv19/test_image_data_'

FILE_TYPE = '.parquet'



train_classes = pd.read_csv("../input/bengaliai-cv19/train.csv")

train_classes['g_num'] = train_classes.apply(lambda row: row.grapheme_root + 168*row.vowel_diacritic+168*11*row.consonant_diacritic, axis=1)



train_classes.head()
testdata = np.load('/kaggle/input/data128c/dataC/data0.npy')

testdata = testdata.reshape(testdata.shape[0],128,128,1)



index = train_classes.loc[train_classes['g_num'] == 15].index.values.astype(int)[0:30]



subset = testdata[index,:,:]

subset = subset.reshape(subset.shape[0],128,128,1)
w=10

h=10

fig=plt.figure(figsize=(12, 12))

columns = 5

rows = 4

for i in range(0, columns*rows):

    img = testdata[i,:,:,0]

    fig.add_subplot(rows, columns, i+1)

    plt.imshow(img)

plt.show()
w=10

h=10

fig=plt.figure(figsize=(12, 12))

columns = 5

rows = 4

for i in range(0, columns*rows):

    img = subset[i,:,:,0]

    fig.add_subplot(rows, columns, i+1)

    plt.imshow(img)

plt.show()
del testdata

gc.collect()
class CustomDataGenerator(ImageDataGenerator):

    '''

    This class extends the ImageDataGenerator but as the parent class only map 1 class label to each image

    For example it can only map if a picture of car is black or white but we are trying to map it to

    N classes so that it can override the default flow() and provide a mapping of a car to color,model,

    company etc. Specially useful if you have different losses for each class so you have to pass a dict

    of y_labels

    

    This code's credit goes to - https://github.com/keras-team/keras/issues/12639

    '''

    

    def flow(self,x,y=None,batch_size=BATCH_SIZE,shuffle=True,sample_weight=None,seed=None,save_to_dir=None,

             save_prefix='',save_format='png',subset=None): 

        '''

        Function takes data & label arrays, generates batches of augmented data (#official keras Documents)

        Input:

            x: Flow method looks for Rank-4 numpy array. i.e (number_of_images,width,height,channels)

            y: dictonary which maps each picture to its ONE-HOT ENCODES respective classes such as  

            if Image1 is associated to 3 classes in a way ->[0,1,2] and Image2 is associated as [3,4,5] so

            the y will be as y={'y1':to_categorical([0,3]),'y2':to_categorical([1,4])...and so on} 

            others: default settings of parameters in the original flow() method

        Output:

            Just like the default flow(), it'll generate an instance of image array x  but instead of a 

            single y-label/class mapping it'll produce a a dictonary as label_dict that contains mapping 

            of all the classes for that image

        '''



        labels_array = None # all the labels array will be concatenated in this single array

        key_lengths = {} 

        # define a dict which maps the 'key' (y1,y2 etc) to lengths of corresponding label_array

        ordered_labels = [] # to store the ordering in which the labels Y were passed in this class

        for key, label_value in y.items():

            if labels_array is None:

                labels_array = label_value 

                # for the first time loop, it's empty, so insert first element

            else:

                labels_array = np.concatenate((labels_array, label_value), axis=1) 

                # concat each array of y_labels 

                

            key_lengths[key] = label_value.shape[1] 

            # key lengths will be different for different range of classes in each class due to_categorical 

            # ONE-HOT encodings. Ex- some have 2 classes (red,yellow) but other can have 4 

            # (Audi,BMW,Ferrari,Toyota) so we have to keep track  due to inner working of super().flow()

            ordered_labels.append(key)





        for x_out, y_out in super().flow(x, labels_array, batch_size=batch_size):

            label_dict = {} # final dictonary that'll be yielded

            i = 0 # keeps count of the ordering of the labels and their lengths

            for label in ordered_labels:

                target_length = key_lengths[label]

                label_dict[label] = y_out[:, i: i + target_length] 

                # Extract to-from the range of length of labels values. That is why we had ordered_labels

                # and key_lengths It'll extract the elements ordering vise else there will be conflict

                i += target_length



            yield x_out, label_dict
import sys

def sizeof_fmt(num, suffix='B'):

    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''

    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:

        if abs(num) < 1024.0:

            return "%3.1f %s%s" % (num, unit, suffix)

        num /= 1024.0

    return "%.1f %s%s" % (num, 'Yi', suffix)



for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),

                         key= lambda x: -x[1])[:10]:

    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
img_input = Input(shape=(128,128,1))

img_conc = Concatenate()([img_input, img_input, img_input]) 



densenet_base = applications.InceptionResNetV2(input_tensor=img_conc, include_top=False, weights='imagenet')



x = densenet_base.output

x = Flatten()(x)

x = Dropout(0.35)(x)



x = Dense(800,activation='relu')(x)

x = BatchNormalization()(x)

x = Dropout(rate = 0.35)(x)



x = Dense(800,activation='relu')(x)

x = BatchNormalization()(x)

x = Dropout(rate = 0.35)(x)



x = Dense(800,activation='relu')(x)

x = BatchNormalization()(x)

x = Dropout(rate = 0.35)(x)



out1 = Dense(168, activation = 'softmax',name='out_1')(x) # names of output layers. We need these names

out2 = Dense(11, activation = 'softmax',name='out_2')(x)  # as they act as the keys for mapping output

out3 = Dense(7, activation = 'softmax',name='out_3')(x)   # to each later. See in the model.fit()



model = models.Model(inputs=img_input, outputs=[out1,out2,out3])

#model = models.Model(inputs=resnet_base.input, outputs=[out1,out2,out3])

model.summary()
model.load_weights('/kaggle/input/weights19/network6_128input_Augmentation5.h5')
from random import randrange



def create_training_batch(j):

    #j = randrange(3)

    print(j)

    #train_X = pd.read_parquet(TRAIN_IMG_PATH+str(j)+FILE_TYPE) # import any random image file given in input

    #train_X = pd.read_feather('train_data_'+str(j)+'.feather')

    #train_X = train_X.iloc[:,1:].values.reshape(train_X.shape[0],64,64,1).astype('float32')/255.

    train_X = np.load('/kaggle/input/data128c/dataC/data'+str(j)+'.npy')

    train_X = train_X.reshape(train_X.shape[0],128,128,1).astype('float32')/255.

    



    

    if j == 0:

        train_Y = train_classes[['grapheme_root','vowel_diacritic','consonant_diacritic']].iloc[:50210]

    if j == 1:

        train_Y = train_classes[['grapheme_root','vowel_diacritic','consonant_diacritic']].iloc[50210:100420]

    if j == 2:

        train_Y = train_classes[['grapheme_root','vowel_diacritic','consonant_diacritic']].iloc[100420:150630]

    if j == 3:

        train_Y = train_classes[['grapheme_root','vowel_diacritic','consonant_diacritic']].iloc[150630:200840]

    train_Y1 = to_categorical(train_Y['grapheme_root'],num_classes=168)

    train_Y2 = to_categorical(train_Y['vowel_diacritic'],num_classes=11)

    train_Y3 = to_categorical(train_Y['consonant_diacritic'],num_classes=7)

    

    

    return train_X, train_Y1, train_Y2, train_Y3
adam = optimizers.Adam(lr=0.00002)

model.compile(adam, loss='categorical_crossentropy', metrics=["accuracy"])



for i in range(0):

    print(i)

    train_X, train_Y1, train_Y2, train_Y3 = create_training_batch(i%4)

    

    datagen = CustomDataGenerator(

        featurewise_center=False,

        featurewise_std_normalization=False,

        rotation_range=7,

        width_shift_range=6,

        height_shift_range=8,

        #brightness_range= [0.1,1],

        shear_range = 5,

        zoom_range = 0.08,

        horizontal_flip=False)



    datagen.fit(train_X)

    

    if i%4 != 0:

        history = model.fit_generator(datagen.flow(train_X, {'out_1': train_Y1,'out_2': train_Y2,'out_3': train_Y3},shuffle = False, batch_size=128),

                    steps_per_epoch=train_X.shape[0] / 128, epochs=1)

        

    if i%4 == 0:

        test_X = train_X[40000:50210,:,:,:]

        test_Y1 = train_Y1[40000:50210,:]

        test_Y2 = train_Y2[40000:50210,:]

        test_Y3 = train_Y3[40000:50210,:]

        

        train_X = train_X[:40000,:,:,:]

        train_Y1 = train_Y1[:40000,:]

        train_Y2 = train_Y2[:40000,:]

        train_Y3 = train_Y3[:40000,:]

        

        history = model.fit_generator(datagen.flow(train_X, {'out_1': train_Y1,'out_2': train_Y2,'out_3': train_Y3},shuffle = False, batch_size=128),

                    steps_per_epoch=train_X.shape[0] / 128, epochs=1, validation_data = (test_X, [test_Y1, test_Y2, test_Y3]))

        

        del test_X

        del test_Y1

        del test_Y2

        del test_Y3

    

    

    del train_X

    del train_Y1

    del train_Y2

    del train_Y3

    gc.collect()
adam = optimizers.Adam(lr=0.0001)

model.compile(adam, loss='categorical_crossentropy', metrics=["accuracy"])



for i in range(0):

    train_X, train_Y1, train_Y2, train_Y3 = create_training_batch(i%4)

    print(i)

    

    datagen = CustomDataGenerator(

        featurewise_center=False,

        featurewise_std_normalization=False,

        rotation_range=7,

        width_shift_range=6,

        height_shift_range=8,

        #brightness_range= [0.1,1],

        shear_range = 5,

        zoom_range = 0.08,

        horizontal_flip=False)



    datagen.fit(train_X)

    

    history = model.fit_generator(datagen.flow(train_X, {'out_1': train_Y1,'out_2': train_Y2,'out_3': train_Y3},shuffle = False, batch_size=128),

                    steps_per_epoch=train_X.shape[0] / 128, epochs=1)



    

    

    del train_X

    del train_Y1

    del train_Y2

    del train_Y3

    gc.collect()
# Save the model

#model.save('network6_128input_Augmentation5.h5')
#from IPython.display import FileLink

#FileLink('network6_128input_Augmentation5.h5')
# serialize model to JSON

#model_json = model.to_json()

#with open("network6.json", "w") as json_file:

#    json_file.write(model_json)
#FileLink('network6.json')
train_X, train_Y1, train_Y2, train_Y3 = create_training_batch(0)



test_X = train_X[40000:50210,:,:,:]

test_Y1 = train_Y1[40000:50210,:]

test_Y2 = train_Y2[40000:50210,:]

test_Y3 = train_Y3[40000:50210,:]
datagen = ImageDataGenerator(

    featurewise_center=False,

    featurewise_std_normalization=False,

    rotation_range=7,

    width_shift_range=6,

    height_shift_range=8,

    #brightness_range= [0.1,1],

    shear_range = 5,

    zoom_range = 0.08,

    horizontal_flip=False)



datagen.fit(test_X)



#test_Xaug = datagen.flow(test_X,shuffle = False, batch_size=10210)
pred_Y1,pred_Y2,pred_Y3 = model.predict_generator(datagen.flow(test_X, shuffle = False,batch_size=128))
#pred_Y1_temp,pred_Y2_temp,pred_Y3_temp = model.predict(test_Xaug)

#pred_Y1 += pred_Y1_temp

#pred_Y2 += pred_Y2_temp

#pred_Y3 += pred_Y3_temp



#del pred_Y1_temp

#del pred_Y2_temp

#del pred_Y3_temp

#gc.collect()
from sklearn.metrics import accuracy_score

p1 = accuracy_score(np.argmax(test_Y1,axis=1), np.argmax(pred_Y1,axis=1))

p2 = accuracy_score(np.argmax(test_Y2,axis=1), np.argmax(pred_Y2,axis=1))

p3 = accuracy_score(np.argmax(test_Y3,axis=1), np.argmax(pred_Y3,axis=1))

print(p1)

print(p2)

print(p3)



0.5*p1 + 0.25*p2 + 0.25*p3
del test_X

del test_Y1

del test_Y2

del test_Y3

        

del train_X

del train_Y1

del train_Y2

del train_Y3



del pred_Y1

del pred_Y2

del pred_Y3

gc.collect()
HEIGHT = 137

WIDTH = 236

SIZE = 128
def bbox(img):

    rows = np.any(img, axis=1)

    cols = np.any(img, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]

    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax



def crop_resize(img0, size=SIZE, pad=16):

    #crop a box around pixels large than the threshold 

    #some images contain line at the sides

    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)

    #cropping may cut too much, so we need to add it back

    xmin = xmin - 13 if (xmin > 13) else 0

    ymin = ymin - 10 if (ymin > 10) else 0

    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH

    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT

    img = img0[ymin:ymax,xmin:xmax]

    #remove lo intensity pixels as noise

    img[img < 28] = 0

    lx, ly = xmax-xmin,ymax-ymin

    l = max(lx,ly) + pad

    #make sure that the aspect ratio is kept in rescaling

    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')

    return cv2.resize(img,(size,size))
def Resize(df,size=128):

    resized = {} 

    df = df.set_index('image_id')

    for i in tqdm(range(df.shape[0])):

        image = 255 - df.loc[df.index[i]].values.reshape(137,236)

        image = (image*(255.0/image.max())).astype(np.uint8)

        image = crop_resize(image)

        resized[df.index[i]] = image.reshape(-1)

    resized = pd.DataFrame(resized).T.reset_index()

    resized.columns = resized.columns.astype(str)

    resized.rename(columns={'index':'image_id'},inplace=True)

    return resized
target=[] # model predictions placeholder

row_id=[] # row_id place holder



datagen = ImageDataGenerator(

    featurewise_center=False,

    featurewise_std_normalization=False,

    rotation_range=7,

    width_shift_range=6,

    height_shift_range=8,

    #brightness_range= [0.1,1],

    shear_range = 5,

    zoom_range = 0.08,

    horizontal_flip=False)



for j in range(4):

    test_X = pd.read_parquet(TEST_IMG_PATH+str(j)+FILE_TYPE)

    test_X = Resize(test_X)

    index = test_X.iloc[:,0]

    test_X = test_X.iloc[:,1:].values.reshape(test_X.shape[0],128,128,1).astype('float32')/255.

    

    datagen.fit(test_X)

    grapheme_root_all,vowel_diacritic_all,consonant_diacritic_all = model.predict_generator(datagen.flow(test_X, shuffle = False))

    

    for a in range(9):



        datagen.fit(test_X)



        grapheme_root_all_temp,vowel_diacritic_all_temp,consonant_diacritic_all_temp = model.predict_generator(datagen.flow(test_X, shuffle = False))

        grapheme_root_all += grapheme_root_all_temp

        vowel_diacritic_all += vowel_diacritic_all_temp

        consonant_diacritic_all += consonant_diacritic_all_temp

        

        del grapheme_root_all_temp

        del vowel_diacritic_all_temp

        del consonant_diacritic_all_temp

        gc.collect()

    

    for i in range(consonant_diacritic_all.shape[0]):

        #pred = np.argmax(preds[i])

        #consonant_diacritic = int(pred/(11*168))

        #vowel_diacritic = int((pred - consonant_diacritic*11*168)/168)

        #grapheme_root = pred - consonant_diacritic*11*168 - vowel_diacritic*168

        consonant_diacritic = np.argmax(consonant_diacritic_all[i])

        vowel_diacritic = np.argmax(vowel_diacritic_all[i])

        grapheme_root = np.argmax(grapheme_root_all[i])

        row_id.append(index[i]+'_consonant_diacritic')

        target.append(consonant_diacritic)

        row_id.append(index[i]+'_vowel_diacritic')

        target.append(vowel_diacritic)

        row_id.append(index[i]+'_grapheme_root')

        target.append(grapheme_root)



    del test_X

    del index

    del consonant_diacritic

    del vowel_diacritic

    del grapheme_root

    del consonant_diacritic_all

    del vowel_diacritic_all

    del grapheme_root_all

    

    gc.collect()
submission = pd.DataFrame({'row_id': row_id,'target':target},columns = ['row_id','target'])

submission.to_csv('submission.csv',index=False)

submission.head()