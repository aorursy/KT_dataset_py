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
!pip install keras
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

from mtcnn import MTCNN
sorted(glob.glob('../input/deepfake/meta*'))
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
def get_path(num,x):

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
X,y=shuffle(X,y)

val_X,val_y=shuffle(val_X,val_y)
def InceptionLayer(a, b, c, d):

    def func(x):

        x1 = Conv2D(a, (1, 1), padding='same', activation='elu')(x)

        

        x2 = Conv2D(b, (1, 1), padding='same', activation='elu')(x)

        x2 = Conv2D(b, (3, 3), padding='same', activation='elu')(x2)

            

        x3 = Conv2D(c, (1, 1), padding='same', activation='elu')(x)

        x3 = Conv2D(c, (3, 3), dilation_rate = 2, strides = 1, padding='same', activation='elu')(x3)

        

        x4 = Conv2D(d, (1, 1), padding='same', activation='elu')(x)

        x4 = Conv2D(d, (3, 3), dilation_rate = 3, strides = 1, padding='same', activation='elu')(x4)

        y = Concatenate(axis = -1)([x1, x2, x3, x4])

            

        return y

    return func

    

def define_model(shape=(256,256,3)):

    x = Input(shape = shape)

    

    x1 = InceptionLayer(1, 4, 4, 2)(x)

    x1 = BatchNormalization()(x1)

    x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

    

    x2 = InceptionLayer(2, 4, 4, 2)(x1)

    x2 = BatchNormalization()(x2)        

    x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)        

        

    x3 = Conv2D(16, (5, 5), padding='same', activation = 'elu')(x2)

    x3 = BatchNormalization()(x3)

    x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

        

    x4 = Conv2D(16, (5, 5), padding='same', activation = 'elu')(x3)

    x4 = BatchNormalization()(x4)

    if shape==(256,256,3):

        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

    else:

        x4 = MaxPooling2D(pool_size=(2, 2), padding='same')(x4)

    y = Flatten()(x4)

    y = Dropout(0.5)(y)

    y = Dense(16)(y)

    y = LeakyReLU(alpha=0.1)(y)

    y = Dropout(0.5)(y)

    y = Dense(1, activation = 'sigmoid')(y)

    model=Model(inputs = x, outputs = y)

    model.compile(loss='binary_crossentropy',optimizer=Adam(lr=1e-4))

    #model.summary()

    return model

df_model=define_model()

df_model.load_weights('../input/meso-pretrain/MesoInception_DF')

f2f_model=define_model()

f2f_model.load_weights('../input/meso-pretrain/MesoInception_F2F')
from keras.callbacks import LearningRateScheduler



# we list out > 25 LRs

lrs=[1e-2,9e-3,8e-3,7e-3,6e-3,5e-3,4e-3,3e-3,2e-3,

     1e-3,9e-4,8e-4,7e-4,6e-4,5e-4,4e-4,3e-4,2e-4,

     1e-4,9e-5,8e-5,7e-5,6e-5,5e-5,4e-5,3e-5,2e-5,

     1e-5,9e-6,8e-6,7e-6,6e-6,5e-6,4e-6,3e-6,2e-6,

     1e-6,9e-7,8e-7,7e-7,6e-7,5e-7,4e-7,3e-7,2e-7,

     1e-7,9e-8,8e-8,7e-8,6e-8,5e-8,4e-8,3e-8,2e-8]

def schedule(epoch):

    return lrs[epoch]
LOAD_PRETRAIN = False # originally True
import gc

kfolds=5

losses=[]

if LOAD_PRETRAIN:

    # import keras.backend as K

    df_models=[]

    f2f_models=[]

    i=0

    while len(df_models)<kfolds:

        model=define_model((150,150,3))

        if i==0:

            model.summary()

        #model.load_weights('../input/meso-pretrain/MesoInception_DF')

        for new_layer, layer in zip(model.layers[1:-8], df_model.layers[1:-8]):

            new_layer.set_weights(layer.get_weights())

        model.fit([X],[y],epochs=25,callbacks=[LearningRateScheduler(schedule)]) #original 2 epochs

        pred=model.predict([val_X])

        loss=log_loss(val_y,pred)

        losses.append(loss)

        print('fold '+str(i)+' model loss: '+str(loss))

        df_models.append(model)

        K.clear_session()

        del model

        gc.collect()

        i+=1

    i=0

    while len(f2f_models)<kfolds:

        model=define_model((150,150,3))

        #model.load_weights('../input/meso-pretrain/MesoInception_DF')

        for new_layer, layer in zip(model.layers[1:-8], f2f_model.layers[1:-8]):

            new_layer.set_weights(layer.get_weights())

        model.fit([X],[y],epochs=25,callbacks=[LearningRateScheduler(schedule)]) #original 2 epochs

        pred=model.predict([val_X])

        loss=log_loss(val_y,pred)

        losses.append(loss)

        print('fold '+str(i)+' model loss: '+str(loss))

        f2f_models.append(model)

        K.clear_session()

        del model

        gc.collect()

        i+=1

        models=f2f_models+df_models

else:

    models=[]

    i=0

    while len(models)<kfolds:

        model=define_model((150,150,3))

        if i==0:

            model.summary()

        model.fit([X],[y],epochs=50,callbacks=[LearningRateScheduler(schedule)]) #original 2 epochs

        pred=model.predict([val_X])

        loss=log_loss(val_y,pred)

        losses.append(loss)

        print('fold '+str(i)+' model loss: '+str(loss))

        if loss<0.68:

            models.append(model)

        else:

            print('loss too bad, retrain!')

        K.clear_session()

        del model

        gc.collect()

        i+=1
def prediction_pipline(X,two_times=False):

    preds=[]

    for model in tqdm(models):

        pred=model.predict([X])

        preds.append(pred)

    preds=sum(preds)/len(preds)

    if two_times:

        return larger_range(preds,2)

    else:

        return preds

def larger_range(model_pred,time):

    return (((model_pred-0.5)*time)+0.5)
best_model_pred=models[losses.index(min(losses))].predict([val_X])
model_pred=prediction_pipline(val_X)
random_pred=np.random.random(len(val_X))

print('random loss: ' + str(log_loss(val_y,random_pred.clip(0.35,0.65))))

allone_pred=np.array([1 for _ in range(len(val_X))])

print('1 loss: ' + str(log_loss(val_y,allone_pred)))

allzero_pred=np.array([0 for _ in range(len(val_X))])

print('0 loss: ' + str(log_loss(val_y,allzero_pred)))

allpoint5_pred=np.array([0.5 for _ in range(len(val_X))])

print('0.5 loss: ' + str(log_loss(val_y,allpoint5_pred)))
print('Simple Averaging Loss: '+str(log_loss(val_y,model_pred.clip(0.35,0.65))))

print('Two Times Larger Range(Averaging) Loss: '+str(log_loss(val_y,larger_range(model_pred,2).clip(0.35,0.65))))

print('Best Single Model Loss: '+str(log_loss(val_y,best_model_pred.clip(0.35,0.65))))

print('Two Times Larger Range(Single Model) Loss: '+str(log_loss(val_y,larger_range(best_model_pred,2).clip(0.35,0.65))))

if log_loss(val_y,model_pred.clip(0.35,0.65))<log_loss(val_y,larger_range(model_pred,2).clip(0.35,0.65)):

    two_times=False

    print('simple averaging is better')

else:

    two_times=True

    print('two times larger range is better')

two_times=False #This is not a bug. I did this intentionally because the model can't get most of the private validation set right(based on LB)
import scipy

print(model_pred.clip(0.35,0.65).mean())

print(scipy.stats.median_absolute_deviation(model_pred.clip(0.35,0.65))[0])
def check_answers(pred,real,num):

    for i,(x,y) in enumerate(zip(pred,real)):

        correct_incorrect='correct ✅ ' if round(float(x),0)==round(float(y),0) else 'incorrect❌'

        print(correct_incorrect+' prediction: '+str(x[0])+', answer: '+str(y))

        if i>num:

            return

def correct_precentile(pred,real):

    correct=0

    incorrect=0

    for x,y in zip(pred,real):

        if round(float(x),0)==round(float(y),0):

            correct+=1

        else:

            incorrect+=1

    print('number correct: '+str(correct)+', number incorrect: '+str(incorrect))

    print(str(round(correct/len(real)*100,1))+'% correct'+', '+str(round(incorrect/len(real)*100,1))+'% incorrect')

check_answers(model_pred,val_y,15)

correct_precentile(model_pred,val_y)
del X,y,val_X,val_y
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

    if max(confidences)<0.9:

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

    return cv2.cvtColor(cv2.resize(img[y:y+h,x:x+w],(150,150)),cv2.COLOR_BGR2RGB)

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

preds=prediction_pipline(test_X,two_times=two_times).clip(0.35,0.65)

for pred,name in zip(preds,prediction_filenames):

    name=name.replace('/kaggle/input/deepfake-detection-challenge/test_videos/','')

    df_test.iloc[list(df_test['filename']).index(name),1]=pred
print(preds.clip(0.35,0.65).mean())

print(scipy.stats.median_absolute_deviation(preds.clip(0.35,0.65))[0])

print(preds[:10])
df_test.head()
df_test.to_csv('submission.csv',index=False)