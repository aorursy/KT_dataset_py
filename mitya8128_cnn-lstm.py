!pip install ../input/mtcnn-package/mtcnn-0.1.0-py3-none-any.whl
import pandas as pd

import keras

import os

import numpy as np

from sklearn.metrics import log_loss

from keras import Model,Sequential

from keras.layers import *

from keras.optimizers import *

from sklearn.model_selection import train_test_split

import cv2

from tqdm.notebook import tqdm

import glob

#from mtcnn import MTCNN
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

df_train17 = pd.read_json('../input/deepfake/metadata17.json')

df_train18 = pd.read_json('../input/deepfake/metadata18.json')

df_train19 = pd.read_json('../input/deepfake/metadata19.json')

df_train20 = pd.read_json('../input/deepfake/metadata20.json')

df_train21 = pd.read_json('../input/deepfake/metadata21.json')

df_train22 = pd.read_json('../input/deepfake/metadata22.json')

df_train23 = pd.read_json('../input/deepfake/metadata23.json')

df_train24 = pd.read_json('../input/deepfake/metadata24.json')

df_train25 = pd.read_json('../input/deepfake/metadata25.json')

df_train26 = pd.read_json('../input/deepfake/metadata26.json')

df_train27 = pd.read_json('../input/deepfake/metadata27.json')

df_train28 = pd.read_json('../input/deepfake/metadata28.json')

df_train29 = pd.read_json('../input/deepfake/metadata29.json')

df_train30 = pd.read_json('../input/deepfake/metadata30.json')

df_train31 = pd.read_json('../input/deepfake/metadata31.json')

df_train32 = pd.read_json('../input/deepfake/metadata32.json')

df_train33 = pd.read_json('../input/deepfake/metadata33.json')

df_train34 = pd.read_json('../input/deepfake/metadata34.json')

df_train35 = pd.read_json('../input/deepfake/metadata35.json')

df_train36 = pd.read_json('../input/deepfake/metadata36.json')

df_train37 = pd.read_json('../input/deepfake/metadata37.json')

df_train38 = pd.read_json('../input/deepfake/metadata38.json')

df_train39 = pd.read_json('../input/deepfake/metadata39.json')

df_train40 = pd.read_json('../input/deepfake/metadata40.json')

df_train41 = pd.read_json('../input/deepfake/metadata41.json')

df_train42 = pd.read_json('../input/deepfake/metadata42.json')

df_train43 = pd.read_json('../input/deepfake/metadata43.json')

df_train44 = pd.read_json('../input/deepfake/metadata44.json')

df_train45 = pd.read_json('../input/deepfake/metadata45.json')

df_train46 = pd.read_json('../input/deepfake/metadata46.json')

df_val1 = pd.read_json('../input/deepfake/metadata47.json')

df_val2 = pd.read_json('../input/deepfake/metadata48.json')

df_val3 = pd.read_json('../input/deepfake/metadata49.json')

df_trains = [df_train0 ,df_train1, df_train2, df_train3, df_train4,

             df_train5, df_train6, df_train7, df_train8, df_train9,df_train10,

            df_train11, df_train12, df_train13, df_train14, df_train15,df_train16, 

            df_train17, df_train18, df_train19, df_train20, df_train21, df_train22, 

            df_train23, df_train24, df_train25, df_train26, df_train27, df_train28, 

            df_train29, df_train30, df_train31, df_train32, df_train33, df_train34,

            df_train34, df_train35, df_train36, df_train37, df_train38, df_train39,

            df_train40, df_train41, df_train42, df_train43, df_train44, df_train45,

            df_train46]

df_vals=[df_val1, df_val2, df_val3]

nums = list(range(len(df_trains)+1))

LABELS = ['REAL','FAKE']

val_nums=[47, 48, 49]
def get_path(num,n):

    num=str(num)

    if len(num)==2:

        path='../input/deepfake/DeepFake'+num+'/DeepFake'+num+'/' + x.replace('.mp4', '') + '.jpg'

    else:

        path='../input/deepfake/DeepFake0'+num+'/DeepFake0'+num+'/' + x.replace('.mp4', '') + '.jpg'

    if not os.path.exists(path):

       raise Exception

    return path

paths=[]

y=[]

for df_train,num in tqdm(zip(df_trains,nums),total=len(df_trains)):

    images = list(df_train.columns.values)

    for x in images:

        try:

            paths.append(get_path(num,x))

            y.append(LABELS.index(df_train[x]['label']))

        except Exception as err:

            #print(err)

            pass



val_paths=[]

val_y=[]

for df_val,num in tqdm(zip(df_vals,val_nums),total=len(df_vals)):

    images = list(df_val.columns.values)

    for x in images:

        try:

            val_paths.append(get_path(num,x))

            val_y.append(LABELS.index(df_val[x]['label']))

        except Exception as err:

            #print(err)

            pass
print('There are '+str(y.count(1))+' fake train samples')

print('There are '+str(y.count(0))+' real train samples')

print('There are '+str(val_y.count(1))+' fake val samples')

print('There are '+str(val_y.count(0))+' real val samples')
import random

real=[]

fake=[]

for m,n in zip(paths,y):

    if n==0:

        real.append(m)

    else:

        fake.append(m)

fake=random.sample(fake,len(real))

paths,y=[],[]

for x in real:

    paths.append(x)

    y.append(0)

for x in fake:

    paths.append(x)

    y.append(1)
real=[]

fake=[]

for m,n in zip(val_paths,val_y):

    if n==0:

        real.append(m)

    else:

        fake.append(m)

fake=random.sample(fake,len(real))

val_paths,val_y=[],[]

for x in real:

    val_paths.append(x)

    val_y.append(0)

for x in fake:

    val_paths.append(x)

    val_y.append(1)
print('There are '+str(y.count(1))+' fake train samples')

print('There are '+str(y.count(0))+' real train samples')

print('There are '+str(val_y.count(1))+' fake val samples')

print('There are '+str(val_y.count(0))+' real val samples')
def read_img(path):

    return cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)

X=[]

for img in tqdm(paths):

    X.append(read_img(img))

val_X=[]

for img in tqdm(val_paths):

    val_X.append(read_img(img))
import random

def shuffle(X,y):

    new_train=[]

    for m,n in zip(X,y):

        new_train.append([m,n])

    random.shuffle(new_train)

    X,y=[],[]

    for x in new_train:

        X.append(x[0])

        y.append(x[1])

    return X,y
def InceptionLayer(a, b, c, d):

    def func(x):

        x1 = Conv2D(a, (1, 1), padding='same', activation='relu')(x)

        

        x2 = Conv2D(b, (1, 1), padding='same', activation='relu')(x)

        x2 = Conv2D(b, (3, 3), padding='same', activation='relu')(x2)

            

        x3 = Conv2D(c, (1, 1), padding='same', activation='relu')(x)

        x3 = Conv2D(c, (3, 3), dilation_rate = 2, strides = 1, padding='same', activation='relu')(x3)

        

        x4 = Conv2D(d, (1, 1), padding='same', activation='relu')(x)

        x4 = Conv2D(d, (3, 3), dilation_rate = 3, strides = 1, padding='same', activation='relu')(x4)

        y = Concatenate(axis = -1)([x1, x2, x3, x4])

            

        return y

    return func

    

def define_model():

    x = Input(shape = (150, 150, 3))

    

    x1 = InceptionLayer(1, 4, 4, 2)(x)

    x1 = BatchNormalization()(x1)

    x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

    

    x2 = InceptionLayer(2, 4, 4, 2)(x1)

    x2 = BatchNormalization()(x2)        

    x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

    

    x2 = InceptionLayer(2, 4, 4, 2)(x2)

    x2 = BatchNormalization()(x2)        

    x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)        

    

    x3 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x2)

    x3 = BatchNormalization()(x3)

    x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

    

    #x4 = Dense(128)(x4)

    #x4 = Dense(64)(x4)

    

    x4 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x3)

    x4 = BatchNormalization()(x4)

    x4 = MaxPooling2D(pool_size = 4, padding='same')(x4)

    

    x4 = Reshape((16,16))(x4)

    x4 = LSTM(2048, return_sequences=False)(x4)

    #x4 = LSTM(1024)(x4)

    

    #y = Flatten()(x4)

    y = Dropout(0.5)(x4)

    y = Dense(32)(y)

    y = Dense(16)(y)

    y = LeakyReLU(alpha=0.1)(y)

    y = Dropout(0.5)(y)

    #y = Reshape((3,3,16))(y)

    y = Dense(1, activation = 'sigmoid')(y)

    model=Model(inputs = x, outputs = y)

    model.compile(loss='mean_squared_error',optimizer=Adam(lr=2e-4))

    model.summary()

    return model
model=define_model()

model.summary()

#model.load_weights('../input/meso-pretrain/MesoInception_F2F')
model.fit([X],[y],epochs=3)
answer=[LABELS[n] for n in val_y]

pred=np.random.random(len(val_X))

print('random loss: ' + str(log_loss(answer,pred.clip(0.45,0.65))))

pred=np.array([1 for _ in range(len(val_X))])

print('1 loss: ' + str(log_loss(answer,pred)))

pred=np.array([0 for _ in range(len(val_X))])

print('0 loss: ' + str(log_loss(answer,pred)))

pred=np.array([0.5 for _ in range(len(val_X))])

print('0.5 loss: ' + str(log_loss(answer,pred)))
pred=model.predict([val_X])

print('model loss: '+str(log_loss(answer,pred)))
print(pred.mean())

print(pred.std())

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

preds=model.predict([test_X],batch_size=32).clip(0.15,0.85)

for pred,name in zip(preds,prediction_filenames):

    name=name.replace('/kaggle/input/deepfake-detection-challenge/test_videos/','')

    df_test.iloc[list(df_test['filename']).index(name),1]=pred
preds[:10]
df_test.head()
df_test.to_csv('submission.csv',index=False)