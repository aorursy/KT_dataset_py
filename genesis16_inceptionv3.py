# GPU details

!nvidia-smi
# CPU details

!cat /proc/cpuinfo
# suppress warnings

import warnings

warnings.filterwarnings("ignore")
import os # for basic operations like folder creation, directory validation etc

import glob # finds file according to wildcard given

import shutil # for moving files

import matplotlib.pyplot as plt # for plotting graphs and viewing images
from numpy import uint8

import numpy as np

import base64



#sbox formation using affine transform

from numpy import uint8

def make_sbox():

    s=[]

    for i in range(0,256):

        q=uint8(i)

        x=q ^ leftrotater(q, 1) ^ leftrotater(q, 2) ^ leftrotater(q, 3) ^ leftrotater(q, 4)^99

        x=hex(x)

        f=x.split('x')[1]

        if len(f)==1:

            f='0'+f

        s.append(f)

    return s

sbox=(make_sbox())

print(sbox)



# circular left shift in array

def leftrotater(x,shift):

    x=(x<<shift)|(x>>(8-shift))

    return uint8(x)



#function to convert float values to decimal

def floatdec_convert(my_number, places = 18):

   res=[]

   my_whole, my_dec = str(my_number).split(".")

   my_whole = int(my_whole)

   my_dec = int (my_dec)

   if my_dec==0:

        res ='0.000000000000000000'

   else:

       res = bin(my_whole).lstrip("0b") + "."

       rang=1

       while (my_dec!=0) and (rang<=places):

          my_whole, my_dec = str((my_decimal_converter(my_dec)) * 2).split(".")

          

          my_dec = int(my_dec)

          res += my_whole

          rang+=1



       while rang<=places:

          res+='0'

          rang+=1

            

   return res



def my_decimal_converter(num):

   while num > 1:

      num /= 10

   return num



# function to recieve only binary values after decimal point 

def arr2bin(a=[]):

    lst=[]

    

    for i in range(0,len(a)):

        print("{0:.9f}".format(float(a[i])))

        bin_val=floatdec_convert(str(a[i]))

        c=bin_val.split('.')[1]

        

        lst+=(c)

       

    return (lst)



def appendingzeros(b): ## to make length of list as multiple of 8

    lnth=len(b)

    if lnth%8!=0:

        c=8-(lnth%8)

        

        while(c>0):

            b.append('0')

            c-=1

        return b



def getcode(b):  ###to get code word generated from sbox and binary values of features

    lnth=len(b)

    code=''

    for i in range(0,lnth,8):

        d_val=128*int(b[i])+64*int(b[i+1])+32*int(b[i+2])+16*int(b[i+3])+8*int(b[i+4])+4*int(b[i+5])+2*int(b[i+6])+1*int(b[i+7])

        code+=(sbox[d_val])

    ###for extra security base64 encoding of code

    data=code

    keyword=base64.b64encode(data.encode('utf-8'))

    keyword1=str(keyword,"utf-8")

    return keyword1  



def callforcode(a):

    a/=100 ####normalizing input 

    b=arr2bin(a)

    c= appendingzeros(b)

    word=getcode(c)

    return word



a=np.array([256.5625,6.500,0.0123])###specify input array



word=callforcode(a)

print(word)

print(len(word))
import cv2 as cv

import numpy as np

from matplotlib import pyplot as plt



img = cv.imread('/kaggle/input/fingerprint-database/fvc2006SETA/train/1/1_3.bmp',0)

print(type(img))



# Otsu's thresholding

ret, thresh_img = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)



def intarray(binstring):

    '''Change a 2D matrix of 01 chars into a list of lists of ints'''

    return [[1 if ch == '1' else 0 for ch in line] 

            for line in binstring.strip().split()]

 

def chararray(intmatrix):

    '''Change a 2d list of lists of 1/0 ints into lines of 1/0 chars'''

    return '\n'.join(''.join(str(p) for p in row) for row in intmatrix)

 

def toTxt(intmatrix):

    '''Change a 2d list of lists of 1/0 ints into lines of '#' and '.' chars'''

    return '\n'.join(''.join(('#' if p else '.') for p in row) for row in intmatrix)

 

def neighbours(x, y, image):

    '''Return 8-neighbours of point p1 of picture, in order'''

    i = image

    x1, y1, x_1, y_1 = x+1, y-1, x-1, y+1

    #print ((x,y))

    return [i[y1][x],  i[y1][x1],   i[y][x1],  i[y_1][x1],  # P2,P3,P4,P5

            i[y_1][x], i[y_1][x_1], i[y][x_1], i[y1][x_1]]  # P6,P7,P8,P9

 

def transitions(neighbours):

    n = neighbours + neighbours[0:1]    # P2, ... P9, P2

    return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))

 

def zhangSuen(image):

    changing1 = changing2 = [(-1, -1)]

    while changing1 or changing2:

        # Step 1

        changing1 = []

        for y in range(1, len(image) - 1):

            for x in range(1, len(image[0]) - 1):

                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, image)

                if (image[y][x] == 1 and    # (Condition 0)

                    P4 * P6 * P8 == 0 and   # Condition 4

                    P2 * P4 * P6 == 0 and   # Condition 3

                    transitions(n) == 1 and # Condition 2

                    2 <= sum(n) <= 6):      # Condition 1

                    changing1.append((x,y))

        for x, y in changing1: image[y][x] = 0

        # Step 2

        changing2 = []

        for y in range(1, len(image) - 1):

            for x in range(1, len(image[0]) - 1):

                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, image)

                if (image[y][x] == 1 and    # (Condition 0)

                    P2 * P6 * P8 == 0 and   # Condition 4

                    P2 * P4 * P8 == 0 and   # Condition 3

                    transitions(n) == 1 and # Condition 2

                    2 <= sum(n) <= 6):      # Condition 1

                    changing2.append((x,y))

        for x, y in changing2: image[y][x] = 0

        #print changing1

        #print changing2

    return image





invert = cv.bitwise_not(thresh_img/255)

skeleton = zhangSuen(invert)



plt.figure(figsize=(10,10), dpi=100)

plt.subplot(3, 1, 1)

plt.imshow(img, 'gray')

plt.subplot(3, 1, 2)

plt.imshow(thresh_img,'gray')

plt.subplot(3, 1, 3)

plt.imshow(skeleton, 'gray')



plt.show()
import cv2 as cv



def intarray(binstring):

    '''Change a 2D matrix of 01 chars into a list of lists of ints'''

    return [[1 if ch == '1' else 0 for ch in line] 

            for line in binstring.strip().split()]

 

def chararray(intmatrix):

    '''Change a 2d list of lists of 1/0 ints into lines of 1/0 chars'''

    return '\n'.join(''.join(str(p) for p in row) for row in intmatrix)

 

def toTxt(intmatrix):

    '''Change a 2d list of lists of 1/0 ints into lines of '#' and '.' chars'''

    return '\n'.join(''.join(('#' if p else '.') for p in row) for row in intmatrix)

 

def neighbours(x, y, image):

    '''Return 8-neighbours of point p1 of picture, in order'''

    i = image

    x1, y1, x_1, y_1 = x+1, y-1, x-1, y+1

    #print ((x,y))

    return [i[y1][x],  i[y1][x1],   i[y][x1],  i[y_1][x1],  # P2,P3,P4,P5

            i[y_1][x], i[y_1][x_1], i[y][x_1], i[y1][x_1]]  # P6,P7,P8,P9

 

def transitions(neighbours):

    n = neighbours + neighbours[0:1]    # P2, ... P9, P2

    return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))

 

def zhangSuen(image):

    changing1 = changing2 = [(-1, -1)]

    while changing1 or changing2:

        # Step 1

        changing1 = []

        for y in range(1, len(image) - 1):

            for x in range(1, len(image[0]) - 1):

                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, image)

                if (image[y][x] == 1 and    # (Condition 0)

                    P4 * P6 * P8 == 0 and   # Condition 4

                    P2 * P4 * P6 == 0 and   # Condition 3

                    transitions(n) == 1 and # Condition 2

                    2 <= sum(n) <= 6):      # Condition 1

                    changing1.append((x,y))

        for x, y in changing1: image[y][x] = 0

        # Step 2

        changing2 = []

        for y in range(1, len(image) - 1):

            for x in range(1, len(image[0]) - 1):

                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, image)

                if (image[y][x] == 1 and    # (Condition 0)

                    P2 * P6 * P8 == 0 and   # Condition 4

                    P2 * P4 * P8 == 0 and   # Condition 3

                    transitions(n) == 1 and # Condition 2

                    2 <= sum(n) <= 6):      # Condition 1

                    changing2.append((x,y))

        for x, y in changing2: image[y][x] = 0

        #print changing1

        #print changing2

    return image





def fingerprint_preprocess(image):

    '''

        this function accepts a numpy array and return the same

    '''

#     img = cv.imread(image, 0)

#     print(type(image))

#     gray = cv.normalize(src=image, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)

    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    print(type(gray), gray.shape)

    # Otsu's thresholding

    ret, thresh_img = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    #invert image color

    invert = cv.bitwise_not(thresh_img/255)

    #skeletonize the fingerprint

    skeleton = zhangSuen(invert)

    

    return skeleton
from keras.models import Model

from keras import backend as K

from keras.layers import Input

from keras.layers import Dense, GlobalAveragePooling2D, Dropout, MaxPool1D

from keras.applications.inception_v3 import InceptionV3

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import CSVLogger, ModelCheckpoint

from keras.optimizers import Adam, RMSprop, SGD
img_width, img_height = 96, 96

train_data_dir = '/kaggle/input/fingerprint-database/fvc2006SETA/train'

validation_data_dir = '/kaggle/input/fingerprint-database/fvc2006SETA/train/'

nb_train_samples = 3024

nb_validation_samples = 2016

batch_size = 64

epochs = 100



if K.image_data_format() == 'channels_last':

    input_tensor = Input(shape=(img_width, img_height, 3))

else:

    input_tensor = Input(shape=(3, img_width, img_height))





train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=30,

#     horizontal_flip=True,

#     width_shift_range=0.2,

#     height_shift_range=0.2,

#     preprocessing_function=fingerprint_preprocess,

    validation_split=0.40,

)



train_generator = train_datagen.flow_from_directory(

        train_data_dir,

        target_size=(img_width, img_height),

        batch_size=batch_size,

        class_mode='categorical',

        subset='training')



validation_generator = train_datagen.flow_from_directory(

        train_data_dir,

        target_size=(img_width, img_height),

        batch_size=batch_size,

        class_mode='categorical',

        subset='validation')



# imports the inceptionv3 pretained model

inception_model = InceptionV3(input_tensor=input_tensor, weights=None, include_top=False)



x = inception_model.output

x = GlobalAveragePooling2D()(x)

# x = Dense(1024, activation='relu')(x)

# x = Dropout(0.20)(x)

# x = Dense(512, activation='relu')(x)

# x = Dropout(0.20)(x)

# x = Dense(256, activation='relu')(x)

# x = Dropout(0.20)(x)

x = Dense(256, activation='relu')(x)

x = Dropout(0.20)(x)

x = Dense(256, activation='relu')(x)

# x = Dropout(0.20)(x)

predictions = Dense(140, activation='softmax')(x)



model = Model(inputs=inception_model.input, outputs=predictions)



# functions to calculate f1score

def recall_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



def precision_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision



def f1score(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))



optimizer = Adam()



model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', f1score, recall_m, precision_m])

# model.summary()



filepath_accuracy = 'accuracy_weights.{epoch:02d}-{val_accuracy:.2f}.hdf5'

filepath_f1score = 'f1score_weights.{epoch:02d}-{val_f1score:.2f}.hdf5'



csv_logger = CSVLogger('training.log')



accuracy_checkpoint = ModelCheckpoint(filepath_accuracy, monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)

f1score_checkpoint = ModelCheckpoint(filepath_f1score, monitor='val_f1score', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)



history = model.fit_generator(

        train_generator,

        steps_per_epoch=nb_train_samples // batch_size,

        epochs=epochs,

        shuffle=True,

        validation_data=validation_generator,

        validation_steps=nb_validation_samples // batch_size,

        callbacks=[csv_logger, accuracy_checkpoint]

        )
model.save('inceptionv3.h5')
print(history.history.keys())

# summarize history for accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for recall

plt.plot(history.history['recall_m'])

plt.plot(history.history['val_recall_m'])

plt.title('model recall_m')

plt.ylabel('recall_m')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for precision

plt.plot(history.history['precision_m'])

plt.plot(history.history['val_precision_m'])

plt.title('model precision_m')

plt.ylabel('precision_m')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for f1score

plt.plot(history.history['f1score'])

plt.plot(history.history['val_f1score'])

plt.title('model f1score')

plt.ylabel('f1score')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()