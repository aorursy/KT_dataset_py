'''from mtcnn import MTCNN

import tqdm

import datetime

import smtplib

import os

import cv2

import numpy as np

import sys

import shutil

d_num=sys.argv[1]

if len(d_num)==1:

    a_num = d_num

    d_num='0'+d_num

else:

    a_num=d_num

detector = MTCNN()

def detect_face(img):

    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    final = []

    detected_faces_raw = detector.detect_faces(img)

    if detected_faces_raw==[]:

        #print('no faces found')

        return []

    confidences=[]

    for n in detected_faces_raw:

        x,y,w,h=n['box']

        final.append([x,y,w,h])

        confidences.append(n['confidence'])

    if max(confidences)<0.7:

        return []

    max_conf_coord=final[confidences.index(max(confidences))]

    #return final

    return max_conf_coord

def crop(img,x,y,w,h):

    x-=40

    y-=40

    w+=80

    h+=80

    if x<0:

        x=0

    if y<=0:

        y=0

    return cv2.cvtColor(cv2.resize(img[y:y+h,x:x+w],(256,256)),cv2.COLOR_BGR2RGB)

def detect_video(video):

    v_cap = cv2.VideoCapture(video)

    v_cap.set(1, NUM_FRAME)

    success, vframe = v_cap.read()

    vframe = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)

    bounding_box=detect_face(vframe)

    if bounding_box==[]:

        count=0

        current=NUM_FRAME

        while bounding_box==[] and count<MAX_SKIP:

            current+=1

            v_cap.set(1,current)

            success, vframe = v_cap.read()

            vframe = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)

            bounding_box=detect_face(vframe)

            count+=1

        if bounding_box==[]:

            print('hi')

            return None

    x,y,w,h=bounding_box

    v_cap.release()

    return crop(vframe,x,y,w,h)

test_dir = './dfdc_train_part_' + a_num + '/'

test_video_files = [test_dir + x for x in os.listdir(test_dir)]

os.makedirs('./DeepFake' + d_num,exist_ok=True)

MAX_SKIP=10

NUM_FRAME=150

count=0

for video in tqdm.tqdm(test_video_files):

    try:

        if video=='./dfdc_train_part_'+a_num+'/metadata.json':

            shutil.copyfile(video,'./metadata'+str(a_num)+'.json')

        img_file=detect_video(video)

        os.remove(video)

        if img_file is None:

            count+=1

            continue

        cv2.imwrite('./DeepFake'+d_num+'/'+video.replace('.mp4','').replace(test_dir,'')+'.jpg',img_file)

    except Exception as err:

      print(err)'''
!pip install ../input/mtcnn-package/mtcnn-0.1.0-py3-none-any.whl
import pandas as pd

import keras

import os

import numpy as np

from sklearn.metrics import log_loss

from keras import Sequential

from keras.layers import *

from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

import cv2

from mtcnn import MTCNN

from tqdm.notebook import tqdm
df_train0 = pd.read_json('../input/deepfake/metadata0.json')

df_train1 = pd.read_json('../input/deepfake/metadata1.json')

df_train2 = pd.read_json('../input/deepfake/metadata2.json')

df_train3 = pd.read_json('../input/deepfake/metadata3.json')

df_train4 = pd.read_json('../input/deepfake/metadata4.json')

df_train5 = pd.read_json('../input/deepfake/metadata5.json')

df_train6 = pd.read_json('../input/deepfake/metadata6.json')

df_train7 = pd.read_json('../input/deepfake/metadata7.json')

df_train8 = pd.read_json('../input/deepfake/metadata8.json')

df_train9 = pd.read_json('../input/deepfake/metadata9.json')

df_train10 = pd.read_json('../input/deepfake/metadata10.json')

df_train11 = pd.read_json('../input/deepfake/metadata11.json')

df_train12 = pd.read_json('../input/deepfake/metadata12.json')

df_train13 = pd.read_json('../input/deepfake/metadata13.json')

df_train14 = pd.read_json('../input/deepfake/metadata14.json')

df_train15 = pd.read_json('../input/deepfake/metadata15.json')

df_train16 = pd.read_json('../input/deepfake/metadata16.json')

LABELS = ['REAL','FAKE']

df_trains = [df_train0 ,df_train1, df_train2, df_train3, df_train4,

             df_train5, df_train6, df_train7, df_train8, df_train9,

            df_train11, df_train12, df_train13, df_train14, df_train15,

            df_train16]

nums = list(range(len(df_trains)))
from tqdm import tqdm_notebook

def read_image(num,name):

    num=str(num)

    if len(num)==2:

        path='../input/deepfake/DeepFake'+num+'/DeepFake'+num+'/' + x.replace('.mp4', '') + '.jpg'

        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    else:

        path='../input/deepfake/DeepFake0'+num+'/DeepFake0'+num+'/' + x.replace('.mp4', '') + '.jpg'

        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

        

X = []

y = []

for df_train,num in tqdm_notebook(zip(df_trains,nums),total=len(df_trains)):

    images = list(df_train.columns.values)

    for x in images:

        try:

            X.append(read_image(num,x))

            y.append(LABELS.index(df_train[x]['label']))

        except Exception as err:

            pass

            #print(x)
print(np.shape(X))

print(np.shape(y))

print(type(X))
print('There are '+str(y.count(1))+' fake samples')

print('There are '+str(y.count(0))+' real samples')
import random

real=[]

fake=[]

for m,n in zip(X,y):

    if n==0:

        real.append(m)

    else:

        fake.append(m)

fake=random.sample(fake,len(real))

X,y=[],[]

for x in real:

    X.append(x)

    y.append(0)

for x in fake:

    X.append(x)

    y.append(1)
print('There are '+str(y.count(1))+' fake samples')

print('There are '+str(y.count(0))+' real samples')
train_X,val_X,train_y,val_y = train_test_split(X, y, test_size=0.15,shuffle=True)
def define_model():

    model = Sequential(

        [

            Conv2D(8, (3, 3), padding="same", activation = 'elu', input_shape=(256, 256,3)),

            BatchNormalization(),

            MaxPooling2D(2, 2),

            Conv2D(8, (5, 5), padding="same", activation = 'elu'),

            BatchNormalization(),

            MaxPooling2D(2, 2),

            Conv2D(16, (5, 5), padding="same", activation = 'elu'),

            BatchNormalization(),

            MaxPooling2D(2, 2),

            Conv2D(16, (5, 5), padding="same", activation = 'elu'),

            BatchNormalization(),

            MaxPooling2D(4, 4),

            Flatten(),

            Dropout(0.5),

            Dense(16,activation='relu'),

            Dropout(0.5),

            Dense(1, activation="sigmoid"),

        ]

    )

    # Define the optimizer

    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model
from keras.callbacks import ReduceLROnPlateau

from keras.optimizers import RMSprop

# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='accuracy', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)



# Training hyperparameters

epochs = 4

batch_size = 20



model=define_model()

history = model.fit([train_X], [train_y], batch_size = batch_size, epochs = 4, verbose = 1,

                    callbacks=[learning_rate_reduction])

answer=[LABELS[n] for n in val_y]

pred=np.random.random(len(val_X))

print('random loss: ' + str(log_loss(answer,pred.clip(0.0001,0.99999))))

pred=np.array([1 for _ in range(len(val_X))])

print('1 loss: ' + str(log_loss(answer,pred)))

pred=np.array([0 for _ in range(len(val_X))])

print('0 loss: ' + str(log_loss(answer,pred)))

pred=np.array([0.5 for _ in range(len(val_X))])

print('0.5 loss: ' + str(log_loss(answer,pred)))
pred=model.predict([val_X])

print('model loss: '+str(log_loss(answer,pred.clip(0.1,0.9))))
print(pred.mean())

print(pred[:10])
model.save('model.h5')
MAX_SKIP=10

NUM_FRAME=150

test_dir = '/kaggle/input/deepfake-detection-challenge/test_videos/'

filenames = os.listdir(test_dir)

prediction_filenames = filenames

test_video_files = [test_dir + x for x in filenames]

detector = MTCNN()

def detect_face(img):

    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    final = []

    detected_faces_raw = detector.detect_faces(img)

    if detected_faces_raw==[]:

        #print('no faces found')

        return []

    confidences=[]

    for n in detected_faces_raw:

        x,y,w,h=n['box']

        final.append([x,y,w,h])

        confidences.append(n['confidence'])

    if max(confidences)<0.7:

        return []

    max_conf_coord=final[confidences.index(max(confidences))]

    #return final

    return max_conf_coord

def crop(img,x,y,w,h):

    x-=40

    y-=40

    w+=80

    h+=80

    if x<0:

        x=0

    if y<=0:

        y=0

    return cv2.cvtColor(cv2.resize(img[y:y+h,x:x+w],(256,256)),cv2.COLOR_BGR2RGB)

def detect_video(video):

    v_cap = cv2.VideoCapture(video)

    v_cap.set(1, NUM_FRAME)

    success, vframe = v_cap.read()

    vframe = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)

    bounding_box=detect_face(vframe)

    if bounding_box==[]:

        count=0

        current=NUM_FRAME

        while bounding_box==[] and count<MAX_SKIP:

            current+=1

            v_cap.set(1,current)

            success, vframe = v_cap.read()

            vframe = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)

            bounding_box=detect_face(vframe)

            count+=1

        if bounding_box==[]:

            print('no faces found')

            prediction_filenames.remove(video.replace('/kaggle/input/deepfake-detection-challenge/test_videos/',''))

            return None

    x,y,w,h=bounding_box

    v_cap.release()

    return crop(vframe,x,y,w,h)

test_X = []

for video in tqdm(test_video_files):

    x=detect_video(video)

    if x is None:

        continue

    test_X.append(x)
df_test=pd.read_csv('/kaggle/input/deepfake-detection-challenge/sample_submission.csv')

df_test['label']=0.5

preds=model.predict([test_X]).clip(0.1,0.9)

for pred,name in zip(preds,prediction_filenames):

    name=name.replace('/kaggle/input/deepfake-detection-challenge/test_videos/','')

    df_test.iloc[list(df_test['filename']).index(name),1]=pred
preds[:10]
df_test.head()
df_test.to_csv('submission.csv',index=False)