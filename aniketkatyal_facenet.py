# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import backend

#import keras.backend as K

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization

from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Concatenate, Lambda

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from tensorflow.keras.optimizers import Adam

from tensorflow.keras import regularizers, activations

import os

import pathlib

from sklearn.utils import shuffle

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from sklearn.model_selection import train_test_split

import cv2

from tensorflow.keras.models import load_model

#import face_recognition

import random

from tensorflow.keras.models import model_from_json



from tqdm import tqdm

import matplotlib.pyplot as plt

from keras.utils import to_categorical



#import itertools

#import shutil

%matplotlib inline
def image_preprocessing():

    image_text_open = open('/kaggle/input/dataset_B_FacialImages/EyeCoordinatesInfo_OpenFace.txt','r')

    image_text_closed = open('/kaggle/input/dataset_B_FacialImages/EyeCoordinatesInfo_ClosedFace.txt','r')

    image_text_open_ = [x.split(' ')[0] for x in image_text_open]

    image_text_closed_ = [x.split(' ')[0] for x in image_text_closed]



    image_info_closed = [ os.path.join('/kaggle/input/dataset_B_FacialImages/ClosedFace',b) for b in image_text_closed_] 

    image_info_open =  [ os.path.join('/kaggle/input/dataset_B_FacialImages/OpenFace',b) for b in image_text_open_] 

    image_final = image_info_closed + image_info_open



    y = [str(0) if x in image_info_closed else str(1) for x in image_final]



    image_data_input = [ cv2.imread(x) for x in image_final]

    image_data_input = [i/255 for i in image_data_input]



    X_train, X_test, y_train, y_test = train_test_split(image_data_input, y, test_size=0.33, random_state=0)

    X_train = np.array(X_train)

    y_train = np.array(y_train)



    image_df = pd.DataFrame({'Paths': image_final,'target': y})

    image_df = image_df.sample(frac=1).reset_index(drop=True)

    train_image_df = image_df.iloc[:1623,:]

    test_image_df = image_df.iloc[1623:,:]

    

    return (train_image_df,test_image_df)
### CONVOLUTIONAL RESNET BLOCK ###



def convolutional_block(x,numfilt,filtsz,strides=1,pad='same',act=True,name=None):

    x = Conv2D(numfilt,filtsz,strides,padding=pad,data_format='channels_last',kernel_initializer = 'he_normal',use_bias=False,name=name+'conv2d')(x)

    x = BatchNormalization(axis=3,scale=False,name=name+'conv2d'+'bn')(x)

    if act:

        x = Activation('relu',name=name+'conv2d'+'act')(x)

    return x
def incresA(x,scale,name=None):

    pad = 'same'

    branch0 = convolutional_block(x,32,1,1,pad,True,name=name+'b0')

    branch1 = convolutional_block(x,32,1,1,pad,True,name=name+'b1_1')

    branch1 = convolutional_block(branch1,32,3,1,pad,True,name=name+'b1_2')

    branch2 = convolutional_block(x,32,1,1,pad,True,name=name+'b2_1')

    branch2 = convolutional_block(branch2,48,3,1,pad,True,name=name+'b2_2')

    branch2 = convolutional_block(branch2,64,3,1,pad,True,name=name+'b2_3')

    branches = [branch0,branch1,branch2]

    mixed = Concatenate(axis=3, name=name + '_concat')(branches)

    filt_exp_1x1 = convolutional_block(mixed,384,1,1,pad,False,name=name+'filt_exp_1x1')

    final_lay = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,

                      output_shape=backend.int_shape(x)[1:],

                      arguments={'scale': scale},

                      name=name+'act_scaling')([x, filt_exp_1x1])

    return final_lay



def incresB(x,scale,name=None):

    pad = 'same'

    branch0 = convolutional_block(x,192,1,1,pad,True,name=name+'b0')

    branch1 = convolutional_block(x,128,1,1,pad,True,name=name+'b1_1')

    branch1 = convolutional_block(branch1,160,[1,7],1,pad,True,name=name+'b1_2')

    branch1 = convolutional_block(branch1,192,[7,1],1,pad,True,name=name+'b1_3')

    branches = [branch0,branch1]

    mixed = Concatenate(axis=3, name=name + '_mixed')(branches)

    filt_exp_1x1 = convolutional_block(mixed,1152,1,1,pad,False,name=name+'filt_exp_1x1')

    final_lay = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,

                      output_shape=backend.int_shape(x)[1:],

                      arguments={'scale': scale},

                      name=name+'act_scaling')([x, filt_exp_1x1])

    return final_lay



def incresC(x,scale,name=None):

    pad = 'same'

    branch0 = convolutional_block(x,192,1,1,pad,True,name=name+'b0')

    branch1 = convolutional_block(x,192,1,1,pad,True,name=name+'b1_1')

    branch1 = convolutional_block(branch1,224,[1,3],1,pad,True,name=name+'b1_2')

    branch1 = convolutional_block(branch1,256,[3,1],1,pad,True,name=name+'b1_3')

    branches = [branch0,branch1]

    mixed = Concatenate(axis=3, name=name + '_mixed')(branches)

    filt_exp_1x1 = convolutional_block(mixed,2048,1,1,pad,False,name=name+'fin1x1')

    final_lay = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,

                      output_shape=backend.int_shape(x)[1:],

                      arguments={'scale': scale},

                      name=name+'act_scaling')([x, filt_exp_1x1])

    return final_lay



#### ResNet NETWORK ## 

def resNet(x):

#Inception-ResNet-A modules

    x = incresA(x,0.15,name='incresA_1')

    x = incresA(x,0.15,name='incresA_2')

    x = incresA(x,0.15,name='incresA_3')

    x = incresA(x,0.15,name='incresA_4')



#35 × 35 to 17 × 17 reduction module.

    x_red_11 = MaxPooling2D(3,strides=2,padding='valid',name='red_maxpool_1')(x)



    x_red_12 = convolutional_block(x,384,3,2,'valid',True,name='x_red1_c1')



    x_red_13 = convolutional_block(x,256,1,1,'same',True,name='x_red1_c2_1')

    x_red_13 = convolutional_block(x_red_13,256,3,1,'same',True,name='x_red1_c2_2')

    x_red_13 = convolutional_block(x_red_13,384,3,2,'valid',True,name='x_red1_c2_3')



    x = Concatenate(axis=3, name='red_concat_1')([x_red_11,x_red_12,x_red_13])



    #Inception-ResNet-B modules

    x = incresB(x,0.1,name='incresB_1')

    x = incresB(x,0.1,name='incresB_2')

    x = incresB(x,0.1,name='incresB_3')

    x = incresB(x,0.1,name='incresB_4')

    x = incresB(x,0.1,name='incresB_5')

    x = incresB(x,0.1,name='incresB_6')

    x = incresB(x,0.1,name='incresB_7')



    #17 × 17 to 8 × 8 reduction module.

    x_red_21 = MaxPooling2D(3,strides=2,padding='valid',name='red_maxpool_2')(x)



    x_red_22 = convolutional_block(x,256,1,1,'same',True,name='x_red2_c11')

    x_red_22 = convolutional_block(x_red_22,384,3,2,'valid',True,name='x_red2_c12')



    x_red_23 = convolutional_block(x,256,1,1,'same',True,name='x_red2_c21')

    x_red_23 = convolutional_block(x_red_23,256,3,2,'valid',True,name='x_red2_c22')



    x_red_24 = convolutional_block(x,256,1,1,'same',True,name='x_red2_c31')

    x_red_24 = convolutional_block(x_red_24,256,3,1,'same',True,name='x_red2_c32')

    x_red_24 = convolutional_block(x_red_24,256,3,2,'valid',True,name='x_red2_c33')



    x = Concatenate(axis=3, name='red_concat_2')([x_red_21,x_red_22,x_red_23,x_red_24])



    #Inception-ResNet-C modules

    x = incresC(x,0.2,name='incresC_1')

    x = incresC(x,0.2,name='incresC_2')

    x = incresC(x,0.2,name='incresC_3')



    x = GlobalAveragePooling2D(data_format='channels_last')(x)

#   x = Dropout(0.1)(x)

    x = Activation('sigmoid')(x)

    

    

    return x



## STEM BLOCK #

def stem(img_input):



    x = convolutional_block(img_input,32,3,2,'valid',True,name='conv1')

    x = convolutional_block(x,32,3,1,'valid',True,name='conv2')

    x = convolutional_block(x,64,3,1,'valid',True,name='conv3')



    x_11 = MaxPooling2D(3,strides=1,padding='valid',name='stem_br_11'+'_maxpool_1')(x)

    x_12 = convolutional_block(x,64,3,1,'valid',True,name='stem_br_12')



    x = Concatenate(axis=3, name = 'stem_concat_1')([x_11,x_12])



    x_21 = convolutional_block(x,64,1,1,'same',True,name='stem_br_211')

    x_21 = convolutional_block(x_21,64,[1,7],1,'same',True,name='stem_br_212')

    x_21 = convolutional_block(x_21,64,[7,1],1,'same',True,name='stem_br_213')

    x_21 = convolutional_block(x_21,96,3,1,'valid',True,name='stem_br_214')



    x_22 = convolutional_block(x,64,1,1,'same',True,name='stem_br_221')

    x_22 = convolutional_block(x_22,96,3,1,'valid',True,name='stem_br_222')



    x = Concatenate(axis=3, name = 'stem_concat_2')([x_21,x_22])



    x_31 = convolutional_block(x,192,3,1,'valid',True,name='stem_br_31')

    x_32 = MaxPooling2D(3,strides=1,padding='valid',name='stem_br_32'+'_maxpool_2')(x)

    x = Concatenate(axis=3, name = 'stem_concat_3')([x_31,x_32])

    

    return x

### Custom accuracy ###

from tensorflow.python.ops import math_ops

from tensorflow.python.framework import ops

from tensorflow.python.keras import backend as K

from tensorflow.python.ops import array_ops

    

def new_sparse_categorical_accuracy(y_true, y_pred):

    y_pred_rank = ops.convert_to_tensor(y_pred).get_shape().ndims

    y_true_rank = ops.convert_to_tensor(y_true).get_shape().ndims

    # If the shape of y_true is (num_samples, 1), squeeze to (num_samples,)

    if (y_true_rank is not None) and (y_pred_rank is not None) and (len(K.int_shape(y_true)) == len(K.int_shape(y_pred))):

        y_true = array_ops.squeeze(y_true, [-1])

    y_pred = math_ops.argmax(y_pred, axis=-1)

    # If the predicted output and actual output types don't match, force cast them

    # to match.

    if K.dtype(y_pred) != K.dtype(y_true):

        y_pred = math_ops.cast(y_pred, K.dtype(y_true))

    return math_ops.cast(math_ops.equal(y_true, y_pred), K.floatx())
def save_model(model):

    

    model_json = model.to_json()

    with open("model.json", "w") as json_file:

        json_file.write(model_json)

    # serialize weights to HDF5

    model.save_weights("model.h5")



def augment_data(train_df,test_df):

    tr_datagen =  ImageDataGenerator(rescale=1.0/255, 

                                 horizontal_flip=True,

                                 vertical_flip=True)



    ts_datagen = ImageDataGenerator(rescale=1.0/255)



    train_gen = tr_datagen.flow_from_dataframe(train_df,x_col = 'Paths', y_col = 'target',

                                target_size=(100,100),

                                batch_size=32,

                                class_mode="binary")



    test_gen = ts_datagen.flow_from_dataframe(test_df,x_col = 'Paths', y_col = 'target',

                               target_size=(100,100),

                               batch_size=32,

                               class_mode="binary",

                               shuffle=False)

    return train_gen,test_gen



def train_model(train_gen,lr,loss,metric,epochs):

    

    ##COMPILE MODEL ##

    img_input = Input(shape=(100,100,3))

    x = stem(img_input)

    output = resNet(x)

    model=Model(img_input,output)

    opt = keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=True)

    model.compile(optimizer = opt ,loss = loss,metrics = [metric])

    

    

#     filepath = 'model.h5'

#     checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 

#                         verbose=1, save_best_only=True, mode='max')



#     early = EarlyStopping(monitor="val_loss", 

#                       mode="min", 

#                       patience=5, restore_best_weights=True)

#     callbacks_list = [checkpoint, early]

    

    

    # ## Fitting the model

    model.fit_generator(train_gen,epochs = epochs,verbose = 1)

    

    save_model(model)

    

    return model

    

    

    

def predict(img, model):

    

    img = Image.fromarray(img, 'RGB').convert('L')

    img = imresize(img, (IMG_SIZE,IMG_SIZE)).astype('float32')

    img /= 255

    img = img.reshape(1,IMG_SIZE,IMG_SIZE,1)

    prediction = model.predict(img)

    if prediction < 0.1:

        prediction = 'closed'

    elif prediction > 0.9:

        prediction = 'open'

    else:

        prediction = 'idk'

    return prediction









    

    

train_image_df,val_image_df = image_preprocessing()

train_gen,test_gen = augment_data(train_image_df,val_image_df)

model_ = train_model(train_gen,0.01,'sparse_categorical_crossentropy','accuracy',40)
def load_model():

    json_file = open('model.json', 'r')

    loaded_model_json = json_file.read()

    json_file.close()

    loaded_model = model_from_json(loaded_model_json)

# load weights into new model

    loaded_model.load_weights("model.h5")

    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return loaded_model
def evaluate(model,test_gen,steps):

    print('Evaluate model')

    loss, acc = model.evaluate_generator(test_gen,steps = steps,verbose = 1)

    print(acc * 100)
evaluate(model_,test_gen,800)
#Image encoding

def process_and_encode(images):

    known_encodings = []

    known_names = []

    print("[LOG] Encoding dataset ...")



    for image_path in tqdm(images):

        # Load image

        image = cv2.imread(image_path)

        # Convert it from BGR to RGB

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

     

        # detect face in the image and get its location (square boxes coordinates)

        boxes = face_recognition.face_locations(image, model='hog')



        # Encode the face into a 128-d embeddings vector

        encoding = face_recognition.face_encodings(image, boxes)



        # the person's name is the name of the folder where the image comes from

        name = image_path.split(' ')[0]



        if len(encoding) > 0 : 

            known_encodings.append(encoding[0])

            known_names.append(name)



    return {"encodings": known_encodings, "names": known_names}

def init():

    face_cascPath = 'haarcascade_frontalface_alt.xml'

    # face_cascPath = 'lbpcascade_frontalface.xml'



    open_eye_cascPath = 'haarcascade_eye_tree_eyeglasses.xml'

    left_eye_cascPath = 'haarcascade_lefteye_2splits.xml'

    right_eye_cascPath ='haarcascade_righteye_2splits.xml'

    dataset = 'faces'



    face_detector = cv2.CascadeClassifier(face_cascPath)

    open_eyes_detector = cv2.CascadeClassifier(open_eye_cascPath)

    left_eye_detector = cv2.CascadeClassifier(left_eye_cascPath)

    right_eye_detector = cv2.CascadeClassifier(right_eye_cascPath)



    print("[LOG] Opening webcam ...")

    video_capture = VideoStream(src=0).start()



    model = load_model()





    print("[LOG] Collecting images ...")

    images = []

    for direc, _, files in tqdm(os.walk(dataset)):

        for file in files:

            if file.endswith("jpg"):

                images.append(os.path.join(direc,file))

    return (model,face_detector, open_eyes_detector, left_eye_detector,right_eye_detector, video_capture, images) 
def isBlinking(history, maxFrames):

    """ @history: A string containing the history of eyes status 

         where a '1' means that the eyes were closed and '0' open.

        @maxFrames: The maximal number of successive frames where an eye is closed """

    for i in range(maxFrames):

        pattern = '1' + '0'*(i+1) + '1'

        if pattern in history:

            return True

    return False


def detect_and_display(model, video_capture, face_detector, open_eyes_detector, left_eye_detector, right_eye_detector, data, eyes_detected):

        frame = video_capture.read()

        # resize the frame

        frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)



        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        

        # Detect faces

        faces = face_detector.detectMultiScale(

            gray,

            scaleFactor=1.2,

            minNeighbors=5,

            minSize=(50, 50),

            flags=cv2.CASCADE_SCALE_IMAGE

        )



        # for each detected face

        for (x,y,w,h) in faces:

            # Encode the face into a 128-d embeddings vector

            encoding = face_recognition.face_encodings(rgb, [(y, x+w, y+h, x)])[0]



            # Compare the vector with all known faces encodings

            matches = face_recognition.compare_faces(data["encodings"], encoding)



            # For now we don't know the person name

            name = "Unknown"



            # If there is at least one match:

            if True in matches:

                matchedIdxs = [i for (i, b) in enumerate(matches) if b]

                counts = {}

                for i in matchedIdxs:

                    name = data["names"][i]

                    counts[name] = counts.get(name, 0) + 1



                # The known encoding with the most number of matches corresponds to the detected face name

                name = max(counts, key=counts.get)



            face = frame[y:y+h,x:x+w]

            gray_face = gray[y:y+h,x:x+w]



            eyes = []

            

            # Eyes detection

            # check first if eyes are open (with glasses taking into account)

            open_eyes_glasses = open_eyes_detector.detectMultiScale(

                gray_face,

                scaleFactor=1.1,

                minNeighbors=5,

                minSize=(30, 30),

                flags = cv2.CASCADE_SCALE_IMAGE

            )

            # if open_eyes_glasses detect eyes then they are open 

            if len(open_eyes_glasses) == 2:

                eyes_detected[name]+='1'

                for (ex,ey,ew,eh) in open_eyes_glasses:

                    cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

            

            # otherwise try detecting eyes using left and right_eye_detector

            # which can detect open and closed eyes                

            else:

                # separate the face into left and right sides

                left_face = frame[y:y+h, x+int(w/2):x+w]

                left_face_gray = gray[y:y+h, x+int(w/2):x+w]



                right_face = frame[y:y+h, x:x+int(w/2)]

                right_face_gray = gray[y:y+h, x:x+int(w/2)]



                # Detect the left eye

                left_eye = left_eye_detector.detectMultiScale(

                    left_face_gray,

                    scaleFactor=1.1,

                    minNeighbors=5,

                    minSize=(30, 30),

                    flags = cv2.CASCADE_SCALE_IMAGE

                )



                # Detect the right eye

                right_eye = right_eye_detector.detectMultiScale(

                    right_face_gray,

                    scaleFactor=1.1,

                    minNeighbors=5,

                    minSize=(30, 30),

                    flags = cv2.CASCADE_SCALE_IMAGE

                )



                eye_status = '1' # we suppose the eyes are open



                # For each eye check wether the eye is closed.

                # If one is closed we conclude the eyes are closed

                for (ex,ey,ew,eh) in right_eye:

                    color = (0,255,0)

                    pred = predict(right_face[ey:ey+eh,ex:ex+ew],model)

                    if pred == 'closed':

                        eye_status='0'

                        color = (0,0,255)

                    cv2.rectangle(right_face,(ex,ey),(ex+ew,ey+eh),color,2)

                for (ex,ey,ew,eh) in left_eye:

                    color = (0,255,0)

                    pred = predict(left_face[ey:ey+eh,ex:ex+ew],model)

                    if pred == 'closed':

                        eye_status='0'

                        color = (0,0,255)

                    cv2.rectangle(left_face,(ex,ey),(ex+ew,ey+eh),color,2)

                eyes_detected[name] += eye_status



            # Each time, we check if the person has blinked

            # If yes, we display its name

            if isBlinking(eyes_detected[name],3):

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Display name

                y = y - 15 if y - 15 > 15 else y + 15

                cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)



        return frame