# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# Any results you write to the current directory are saved as output.
PARAMS_VIDEO={

    'shape':(256,256),

    'is_face':True,

    'is_first_face':False,

    'is_first':False,

    'on_each':30,

}
!pip install /kaggle/input/efficientnet/efficientnet-1.0.0-py3-none-any.whl
!pip install /kaggle/input/mtcnn-package/mtcnn-0.1.0-py3-none-any.whl
import efficientnet.keras as efn
import os

import cv2

import sys

import pandas as pd

from skimage.metrics import mean_squared_error as mse

from skimage.metrics import structural_similarity as ssim

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
DATA_DIR='/kaggle/input/deepfake-detection-challenge/'

TRAIN_DIR=f'{DATA_DIR}train_sample_videos/'

TEST_DIR=f'{DATA_DIR}test_videos/'

SUB_DIR=f'{DATA_DIR}sample_submission.csv'



FACE_DETECTION_FOLDER=f'/kaggle/input/haar-cascades-for-face-detection/'

FACENET_DIR=f'/kaggle/input/facenet-keras/'
class ObjectDetector():

    '''

    Class for Object Detection

    '''

    def __init__(self,object_cascade_path):

        '''

        param: object_cascade_path - path for the *.xml defining the parameters for {face, eye, smile, profile}

        detection algorithm

        source of the haarcascade resource is: https://github.com/opencv/opencv/tree/master/data/haarcascades

        '''



        self.objectCascade=cv2.CascadeClassifier(object_cascade_path)





    def detect(self, image, scale_factor=1.3,

               min_neighbors=5,

               min_size=(20,20)):

        '''

        Function return rectangle coordinates of object for given image

        param: image - image to process

        param: scale_factor - scale factor used for object detection

        param: min_neighbors - minimum number of parameters considered during object detection

        param: min_size - minimum size of bounding box for object detected

        '''

        rects=self.objectCascade.detectMultiScale(image,

                                                scaleFactor=scale_factor,

                                                minNeighbors=min_neighbors,

                                                minSize=min_size)

        return rects
import json

f=open(TRAIN_DIR+'metadata.json')

train_labels=json.loads(f.read())



def json2pd(jdata):

    res=[]

    for k in jdata.keys():

        jdata[k]['name']=k

        res.append(jdata[k])

    return pd.DataFrame(res)



train_labels=json2pd(train_labels)
import gc

import sys



def get_obj_size(obj):

    marked = {id(obj)}

    obj_q = [obj]

    sz = 0



    while obj_q:

        sz += sum(map(sys.getsizeof, obj_q))



        # Lookup all the object referred to by the object in obj_q.

        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents

        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))



        # Filter object that are already marked.

        # Using dict notation will prevent repeated objects.

        new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}



        # The new obj_q will be the ones that were not marked,

        # and we will update marked with their ids so we will

        # not traverse them again.

        obj_q = new_refr.values()

        marked.update(new_refr.keys())



    return sz
from mtcnn import MTCNN
face_detect=MTCNN()
dir(face_detect)
def detect_objects(image, scale_factor=1.3, min_neighbors=5, min_size=(50,50)):

    

    image_gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)





    eyes=eye_detector.detect(image_gray,

                   scale_factor=scale_factor,

                   min_neighbors=min_neighbors,

                   min_size=(int(min_size[0]/2), int(min_size[1]/2)))



    for x, y, w, h in eyes:

        #detected eyes shown in color image

        cv2.circle(image,(int(x+w/2),int(y+h/2)),(int((w + h)/4)),(0, 0,255),3)

 

    # deactivated due to many false positive

    #smiles=sd.detect(image_gray,

    #               scale_factor=scale_factor,

    #               min_neighbors=min_neighbors,

    #               min_size=(int(min_size[0]/2), int(min_size[1]/2)))



    #for x, y, w, h in smiles:

    #    #detected smiles shown in color image

    #    cv.rectangle(image,(x,y),(x+w, y+h),(0, 0,255),3)





    profiles=profile_detector.detect(image_gray,

                   scale_factor=scale_factor,

                   min_neighbors=min_neighbors,

                   min_size=min_size)



    for x, y, w, h in profiles:

        #detected profiles shown in color image

        cv2.rectangle(image,(x,y),(x+w, y+h),(255, 0,0),3)



    faces=front_detector.detect(image_gray,

                   scale_factor=scale_factor,

                   min_neighbors=min_neighbors,

                   min_size=min_size)



    for x, y, w, h in faces:

        #detected faces shown in color image

        cv2.rectangle(image,(x,y),(x+w, y+h),(0, 255,0),3)



    # image

    fig = plt.figure(figsize=(10,10))

    ax = fig.add_subplot(111)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ax.imshow(image)
def get_face(image, scale_factor=2,preshape=(512,512), min_neighbors=5, min_size=(30,30),target_shape=(256,256)):

    

    image_gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



    if(preshape):

        image_gray=cv2.resize(image_gray,preshape)



    profiles=profile_detector.detect(image_gray,

                   scale_factor=scale_factor,

                   min_neighbors=min_neighbors,

                   min_size=min_size)



    faces=front_detector.detect(image_gray,

                   scale_factor=scale_factor,

                   min_neighbors=min_neighbors,

                   min_size=min_size)

    res_faces=[]

    if(len(profiles)!=0 or len(faces)!=0):

        for p in profiles:

            im=image[p[1]:p[1]+p[3],p[0]:p[0]+p[2]]

            if(target_shape):

                im=cv2.resize(im,target_shape, interpolation = cv2.INTER_AREA)

            res_faces.append(im)

        for p in faces:

            im=image[p[1]:p[1]+p[3],p[0]:p[0]+p[2]]

            if(target_shape):

                im=cv2.resize(im,target_shape, interpolation = cv2.INTER_AREA)

            res_faces.append(im)

    filtered_faces=[]        

    if(len(res_faces)!=0):

        for f in res_faces:

            eyes=eye_detector.detect(f,

                   scale_factor=scale_factor,

                   min_neighbors=min_neighbors,

                   min_size=(int(min_size[0]/2), int(min_size[1]/2)))

            if(len(eyes)!=0):

                filtered_faces.append(f)

    #features=model.predict(np.array(filtered_faces))

    return filtered_faces
def get_face(image, scale_factor=2,preshape=(256,256), target_shape=(256,256)):



    original_shape=image.shape

    if(preshape):

        scale_y=original_shape[0]/preshape[0]

        scale_x=original_shape[1]/preshape[1]

        reshape_image=cv2.resize(image,preshape)

    else:

        scale_y=1

        scale_x=1

        reshape_image=image

    

    par_faces=face_detect.detect_faces(reshape_image)

    

    faces=[]

    

    

    for p in par_faces:

        width=int(p['box'][2]*scale_x)

        height=int(p['box'][3]*scale_y)

        new_width=int(width*scale_factor)

        new_height=int(height*scale_factor)

        x=int(p['box'][0]*scale_x)-(new_width-width)//2

        y=int(p['box'][1]*scale_y)-(new_height-height)//2

        if(x<0):

            x=0

        if(y<0):

            y=0

        

        try:

            if(x+width<image.shape[1] and y+height<image.shape[0] and new_width!=0 and new_height!=0):

                face=image[y:y+new_width,x:x+new_width]

                if(target_shape):

                    face=cv2.resize(face,target_shape)

                faces.append(face)

            #else:

            #    print('error',p)

        except:

            print('error',p)



    return faces
class VideoReader():

    def __init__(self,video_path,shape=None,is_gray=False,is_face=False,is_dif=False,is_first=False,is_first_face=False,on_each=1,offset=0):

        self.video_path=video_path

        self.codec=cv2.VideoCapture(self.video_path)

        self.shape=shape

        

        self.stop_read=False

        

        self.cur_ind=0

        

        self.is_gray=is_gray

        self.is_face=is_face

        self.is_dif=is_dif

        self.is_first=is_first

        self.is_first_face=is_first_face

        self.on_each=on_each

        self.offset=offset

        

        self.is_error=False

        

        self.frames=[]

        self.frames_vector=[]

        

        

        self.dif_frames=[]

        self.first_frame=None

        self.last_frame=None

        

        self.faces=[]

        self.miss_faces=[]

        

        

        self.params={}

        

        self.dif_params={

            'mse':[],

            'ssim':[],

        }

    

    def get_video(self):

        while(self.codec.isOpened() and not self.stop_read):

            self.on_frame()

        #self.get_params()

        

    def on_frame(self):

        ret, frame = self.codec.read()

        if (ret==True and self.cur_ind>=self.offset):

            if(type(frame)!=type(None) ):

                if((self.cur_ind+1)%self.on_each==0):

                    #frame=self.preprocess_frame(frame)

                    if(self.is_face):

                        self.get_face(frame)

                    if(self.is_first):

                        self.stop_read=True

                    self.frames.append(frame)

        else:

            self.stop_read=True

        self.cur_ind+=1

            

    def get_face(self,frame):

        faces=get_face(frame,preshape=None,target_shape=self.shape)

        if(len(faces)>0):

            self.faces.append(faces)

            if(self.is_first_face):

                self.stop_read=True

        else:

            self.miss_faces.append(self.cur_ind)

        

            

    

    def get_params(self):

        self.params['fname']=self.video_path

        self.params['length']=len(self.frames)

        self.params['size_obj']=get_obj_size(self)

        

    def preprocess_frame(self,frame):

        if(self.shape):

            frame=cv2.resize(frame,(self.shape[1],self.shape[0]))

        if(self.is_gray):

            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        return frame

    
class VideoGroupReader():

    def __init__(self,original_file,list_fakes=[]):

        self.original_file=original_file

        self.list_fakes=list_fakes

    def dif_videos(self):

        for i in range(len(self.list_fakes)):

            fake=self.list_fakes[i]

            vr=VideoReader(fake,is_face=True,is_first_face=False,on_each=30)

        
frontal_cascade_path= os.path.join(FACE_DETECTION_FOLDER,'haarcascade_frontalface_default.xml')

eye_cascade_path= os.path.join(FACE_DETECTION_FOLDER,'haarcascade_eye.xml')

profile_cascade_path= os.path.join(FACE_DETECTION_FOLDER,'haarcascade_profileface.xml')

smile_cascade_path= os.path.join(FACE_DETECTION_FOLDER,'haarcascade_smile.xml')



print(eye_cascade_path)

#Detector object created

# frontal face

front_detector=ObjectDetector(frontal_cascade_path)

# eye

eye_detector=ObjectDetector(eye_cascade_path)

# profile face

profile_detector=ObjectDetector(profile_cascade_path)

# smile

smile_detector=ObjectDetector(smile_cascade_path)
train_files=os.listdir(TRAIN_DIR)

test_files=os.listdir(TEST_DIR)

for k in train_files:

    if('.json' in k):

        json_file=k

        train_files.remove(k)
len(train_files),len(test_files),json_file
import json

f=open(TRAIN_DIR+json_file)

train_label=json.loads(f.read())
train_label['dkzvdrzcnr.mp4']
import cv2

import matplotlib.pyplot as plt

import keras
from keras.utils import Sequence

from skimage.measure import compare_ssim
"""train_data={}

bad_names=[]

for i in tqdm(range(len(train_files))):

    video = VideoReader(TRAIN_DIR+train_files[i],shape=(256,256),is_face=True,is_first_face=True,is_first=False,on_each=30)

    video.get_video()

    if(len(video.faces)>0):

        train_data[train_files[i]]=video.faces[0]

    else:

        bad_names.append(train_files[i])"""
!mkdir face_data
"""#train_data_x=[]

train_data_y=[]

bad_names=[]

train_decode_data={}

for i in tqdm(range(len(train_files))):

    video = VideoReader(TRAIN_DIR+train_files[i],shape=PARAMS_VIDEO['shape'],is_face=PARAMS_VIDEO['is_face'],is_first_face=PARAMS_VIDEO['is_first_face'],is_first=PARAMS_VIDEO['is_first'],on_each=PARAMS_VIDEO['on_each'])

    video.get_video()

    label=int(train_label[train_files[i]]['label']=='FAKE')

    start_ind=len(train_data_y)

    for j in range(len(video.faces)):

        if(len(video.faces[j])==1):

            

            #train_data_x+=video.faces[j]

            for t in range(len(video.faces[j])):

                #print(video.faces[j][t].shape)

                cv2.imwrite(f'face_data/{train_files[i]}_{j}_{t}_{label}.png',video.faces[j][t])

            train_data_y+=[label] * (len(video.faces[j]))

            

            train_decode_data[train_files[i]]=(start_ind,len(train_data_y))

            

        #print(i,len(train_data_x),len(train_data_y))

    else:

        bad_names.append(train_files[i])"""

    
#train_data_x=[]

train_data_y=[]

bad_names=[]

train_decode_data={}

for i in tqdm(range(len(train_files))):

    video = VideoReader(TRAIN_DIR+train_files[i],shape=PARAMS_VIDEO['shape'],is_face=PARAMS_VIDEO['is_face'],is_first_face=PARAMS_VIDEO['is_first_face'],is_first=PARAMS_VIDEO['is_first'],on_each=PARAMS_VIDEO['on_each'])

    video.get_video()

    label=int(train_label[train_files[i]]['label']=='FAKE')

    start_ind=len(train_data_y)

    

    for j in range(len(video.faces)):

        if(len(video.faces[j])==1):

            

            #train_data_x+=video.faces[j]

            for t in range(len(video.faces[j])):

                #print(video.faces[j][t].shape)

                fname=f'face_data/{train_files[i]}_{j}_{t}_{label}.png'

                cv2.imwrite(fname,video.faces[j][t])

                if(train_files[i] in train_decode_data.keys()):

                    train_decode_data[train_files[i]].append(fname)

                else:

                    train_decode_data[train_files[i]]=[fname]

            #train_data_y+=[label] * (len(video.faces[j]))

    if(len(video.faces)==0):

        bad_names.append(train_files[i])

    
bad_names
"""test_data={}

bad_names_test=[]

for i in tqdm(range(len(test_files))):

    video = VideoReader(TEST_DIR+test_files[i])

    face=video.get_first_face()

    test_data[test_files[i]]=face

    if(np.max(face)!=0):

        bad_names_test.append(test_files[i])"""
import skimage.filters

class DataGeneratorFull(Sequence):

    'Generates data for Keras'

    def __init__(self, files,data,jdata=None,len_frames=300,batch_size=4,shuffle=True,dim=(1024,1024),channels=3,mode='fit'):

        self.dim = dim

        self.files=files

        self.data=data

        self.jdata=jdata

        self.len_frames=len_frames

        self.batch_size=batch_size

        self.shuffle=shuffle

        self.dim=dim

        self.channels=channels

        self.mode=mode



    def __len__(self):

        'Denotes the number of batches per epoch'

        return int((len(self.files) / self.batch_size))



    def __getitem__(self, index):



        batch_files = self.files[index*self.batch_size:(index+1)*self.batch_size]

        

        X = self.__generate_X(batch_files)

        

        if self.mode == 'fit':

            y = self.__generate_y(batch_files)

            return X, y

        

        elif self.mode == 'predict':

            return X

        else:

            raise AttributeError('The parameter mode should be set to "fit" or "predict".')

        

    def on_epoch_end(self):

        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.files))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)

    

    def __generate_X(self, batch_files):

        x=np.zeros((self.batch_size,*self.dim,3),dtype=np.float32)

        for i in range(len(batch_files)):

            #if(self.mode=='fit'):

            face = self.data[batch_files[i]][0]

            #else:

                #face = VideoReader(TEST_DIR+batch_files[i],shape=(256,256))

            #print(face)

            x[i,:,:]=face/255

        return x

    

    def __generate_y(self, batch_files):

        y=np.zeros((self.batch_size,1))

        for i in range(len(batch_files)):

            val=self.jdata[batch_files[i]]['label']=='FAKE'

            y[i]=val#keras.utils.to_categorical(val,2)

            #print(val)

        return y
import skimage.filters

class DataGeneratorFull(Sequence):

    'Generates data for Keras'

    def __init__(self, data_files,data_dir='face_data',len_frames=300,batch_size=4,shuffle=True,dim=(1024,1024),channels=3,mode='fit'):

        self.dim = dim

        self.data_files=data_files

        #print(len(self.data_files))

        self.len_frames=len_frames

        self.batch_size=batch_size

        self.shuffle=shuffle

        self.dim=dim

        self.channels=channels

        self.mode=mode

        self.data_dir=data_dir

        self.indexes = np.arange(len(self.data_files))



    def __len__(self):

        'Denotes the number of batches per epoch'

        return (len(self.data_files) // self.batch_size)



    def __getitem__(self, index):



        batch_files = self.data_files[index*self.batch_size:(index+1)*self.batch_size]

        #print(index,len(batch_files),self.batch_size)

        X = self.__generate_X(batch_files)

        

        if self.mode == 'fit':

            y = self.__generate_y(batch_files)

            return X, y

        

        elif self.mode == 'predict':

            return X

        else:

            raise AttributeError('The parameter mode should be set to "fit" or "predict".')

        

    def on_epoch_end(self):

        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.data_files))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)

    

    def __generate_X(self, batch_files):

        x=np.zeros((self.batch_size,*self.dim,3),dtype=np.float32)

        #print('len=',len(batch_files))

        for i in range(len(batch_files)):

            face = cv2.imread(f'{self.data_dir}/'+batch_files[i])

            #print('max=',np.max(face))

            x[i,]=face/255

        return x

    

    def __generate_y(self, batch_files):

        y=np.zeros((self.batch_size,2))

        for i in range(len(batch_files)):

            #val=self.jdata[batch_files[i]]['label']=='FAKE'

            indxs=train_decode_data[batch_files[i].split('_')[0]]

            label=train_data_y[indxs[0]]

            y[i]=keras.utils.to_categorical(label,2)

            #print(val)

        return y
import skimage.filters

class DataGeneratorFull(Sequence):

    'Generates data for Keras'

    def __init__(self, data_files,data_dir='face_data',len_frames=300,batch_size=4,shuffle=True,dim=(1024,1024),channels=3,mode='fit'):

        self.dim = dim

        self.data_files=data_files

        #print(len(self.data_files))

        self.len_frames=len_frames

        self.batch_size=batch_size

        self.shuffle=shuffle

        self.dim=dim

        self.channels=channels

        self.mode=mode

        self.data_dir=data_dir

        self.indexes = np.arange(len(self.data_files))



    def __len__(self):

        'Denotes the number of batches per epoch'

        return (len(self.data_files) // self.batch_size)



    def __getitem__(self, index):



        batch_files = self.data_files[index*self.batch_size:(index+1)*self.batch_size]

        #print(index,len(batch_files),self.batch_size)

        X = self.__generate_X(batch_files)

        

        if self.mode == 'fit':

            y = self.__generate_y(batch_files)

            return X, y

        

        elif self.mode == 'predict':

            return X

        else:

            raise AttributeError('The parameter mode should be set to "fit" or "predict".')

        

    def on_epoch_end(self):

        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.data_files))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)

    

    def __generate_X(self, batch_files):

        #x=np.zeros((self.batch_size,*self.dim,3),dtype=np.float32)

        #print('len=',len(batch_files))

        x=[]

        for i in range(len(batch_files)):

            cur_files=train_decode_data[batch_files[i]]

            for c in cur_files:

                #print(c)

                face = cv2.imread(c)

                x.append(face/255)

            #print('max=',np.max(face))

            #x[i,]=face/255

        return np.array([x])

    

    def __generate_y(self, batch_files):

        y=np.zeros((self.batch_size,2))

        for i in range(len(batch_files)):

            cur_files=train_decode_data[batch_files[i]]

            #val=self.jdata[batch_files[i]]['label']=='FAKE'

            #indxs=train_decode_data[batch_files[i].split('_')[0]]

            #try:

            label=int(cur_files[0].split('_')[-1].split('.')[0])

            #except:

                #print(cur_files,batch_files)

            #    label=1

            y[i]=keras.utils.to_categorical(label,2)

            #print(val)

        return y
#train_decode_data['efdyrflcpg.mp4']
bad_names
#new_train_files=os.listdir('face_data')
new_train_files=[]

for i in range(len(train_files)):

    if(not (train_files[i] in bad_names)):

        if(train_files[i] in train_decode_data.keys()):

            if(len(train_decode_data[train_files[i]])!=0):

                new_train_files.append(train_files[i])
len(new_train_files)
"""data_hist=[]

for i in range(len(train_files)):

    data_hist.append(train_label[train_files[i]]['label']=='FAKE')

data_hist=np.array(data_hist)

coef_fake=data_hist.sum()/len(data_hist)

coef_real=1-coef_fake

print(coef_fake,coef_real)"""
from sklearn.model_selection import train_test_split

train_x,val_x  = train_test_split(new_train_files, test_size=0.15, random_state=42)
len(train_x),len(val_x)
#from sklearn.model_selection import train_test_split

#train_x,val_x,train_y,val_y  = train_test_split(train_data_x,train_data_y, test_size=0.15, random_state=42)
#len(train_x),len(val_x)
gen=DataGeneratorFull(train_x,dim=(256,256),batch_size=1)

val=DataGeneratorFull(val_x,dim=(256,256),batch_size=1)

#test_gen=DataGeneratorFull(test_files,test_data,train_label,dim=(256,256),batch_size=1,mode='predict')
#gen=DataGeneratorFull(train_x,train_y,dim=(256,256),batch_size=4)

#val=DataGeneratorFull(val_x,val_y,dim=(256,256),batch_size=4)

#test_gen=DataGeneratorFull(test_files,test_data,train_label,dim=(256,256),batch_size=1,mode='predict')
%%time

g=gen.__getitem__(30)
g[0].shape
#plt.hist(g[0][0].flatten())
#plt.imshow(g[0][7])
#g[1]
def build_lstm_model(input_shape=(300,256,256,1)):

    inp=keras.layers.Input(input_shape)

    #out1=keras.layers.ConvLSTM2D(16,3,return_sequences=True,activation='relu',padding='same')(inp)

    #out2=keras.layers.ConvLSTM2D(16,3,return_sequences=True,activation='relu',padding='same')(inp)

   # out=keras.layers.concatenate([out1,out2])



    #print(out.shape)

    #out=keras.layers.MaxPooling3D((1,2,2))(inp)

    out1=keras.layers.Conv3D(16,3,activation='relu',padding='same')(inp)

    out2=keras.layers.Conv3D(16,3,activation='relu',padding='same')(inp)

    out=keras.layers.concatenate([out1,out2])



    out=keras.layers.MaxPooling3D((1,2,2))(out)

    out1=keras.layers.Conv3D(16,3,activation='relu',padding='same')(out)

    out2=keras.layers.Conv3D(16,3,activation='relu',padding='same')(out)

    out=keras.layers.concatenate([out1,out2])



    out=keras.layers.GlobalAveragePooling3D()(out)

    out=keras.layers.Dense(1,activation='sigmoid')(out)



    model=keras.models.Model(input=inp,output=out)

    return model
#os.listdir('/kaggle/input/densenet-keras/')
#os.listdir('/kaggle/input/efficientnet-keras-weights-b0b5')
def unet(input_size = (32,32,1),descr=1,classes=1,activation='sigmoid'):

    #descr=2

    inputs = keras.layers.Input(input_size)

    conv1 = keras.layers.Conv2D(int(64/descr), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)

    conv1 = keras.layers.Conv2D(int(64/descr), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)

    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = keras.layers.Conv2D(int(128/descr), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)

    conv2 = keras.layers.Conv2D(int(128/descr), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)

    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = keras.layers.Conv2D(int(256/descr), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)

    conv3 = keras.layers.Conv2D(int(256/descr), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)

    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = keras.layers.Conv2D(int(512/descr), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)

    conv4 = keras.layers.Conv2D(int(512/descr), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)

    drop4 = keras.layers.Dropout(0.5)(conv4)

    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)



    conv5 = keras.layers.Conv2D(int(1024/descr), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)

    conv5 = keras.layers.Conv2D(int(1024/descr), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)

    drop5 = keras.layers.Dropout(0.5)(conv5)



    up6 = keras.layers.Conv2D(int(512/descr), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(drop5))

    merge6 = keras.layers.concatenate([drop4,up6], axis = 3)

    conv6 = keras.layers.Conv2D(int(512/descr), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)

    conv6 = keras.layers.Conv2D(int(512/descr), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)



    up7 = keras.layers.Conv2D(int(256/descr), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(conv6))

    merge7 = keras.layers.concatenate([conv3,up7], axis = 3)

    conv7 = keras.layers.Conv2D(int(256/descr), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)

    conv7 = keras.layers.Conv2D(int(256/descr), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)



    up8 = keras.layers.Conv2D(int(128/descr), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(conv7))

    merge8 = keras.layers.concatenate([conv2,up8], axis = 3)

    conv8 = keras.layers.Conv2D(int(128/descr), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)

    conv8 = keras.layers.Conv2D(int(128/descr), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)



    up9 = keras.layers.Conv2D(int(64/descr), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(conv8))

    merge9 = keras.layers.concatenate([conv1,up9], axis = 3)

    conv9 = keras.layers.Conv2D(int(64/descr), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)

    conv9 = keras.layers.Conv2D(int(64/descr), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    conv9 = keras.layers.Conv2D(classes*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    conv10 = keras.layers.Conv2D(classes, 1, activation = activation)(conv9)

    #conv10=tf.keras.layers.Attention()(conv10)

    #glpool=GlobalAveragePooling2D()(conv9)

    #x = Dense(512, activation="relu")(glpool)

    #x = Dropout(0.5)(x)

    #x = Dense(256, activation="relu")(x)

    #predictions = Dense(1107, activation="softmax")(x)

    model = keras.models.Model(input = inputs, output = conv10)

    

    #model.summary()



    #if(pretrained_weights):

    	#model.load_weights(pretrained_weights)



    return model
def build_lstm_model(input_shape=(256,256,3)):

    inp=keras.layers.Input(input_shape)

    

    unet_back=unet(input_size = input_shape,descr=8,classes=3,activation='relu')

    back=keras.applications.mobilenet.MobileNet(input_shape=input_shape,pooling='avg',include_top=False,weights=None)

    back.load_weights('/kaggle/input/mobilenet/mobilenet_1_0_224_tf_no_top.h5')

    out=unet_back(inp)

    out=back(out)

    #out1=keras.layers.Conv2D(16,3,activation='relu',padding='same')(inp)

    #out2=keras.layers.Conv2D(16,3,activation='relu',padding='same')(inp)

    #out=keras.layers.concatenate([out1,out2])



    #out=keras.layers.MaxPooling2D((2,2))(out)

    #out1=keras.layers.Conv2D(16,3,activation='relu',padding='same')(out)

    #out2=keras.layers.Conv2D(16,3,activation='relu',padding='same')(out)

    #out=keras.layers.concatenate([out1,out2])



    #out=keras.layers.GlobalAveragePooling2D()(out)

    #out=keras.layers.Dense(1024,activation='relu')(out)

    

    out=keras.layers.Dense(512,activation='relu')(out)

    out=keras.layers.Dropout(0.5)(out)

    out=keras.layers.Dense(2,activation='softmax')(out)



    model=keras.models.Model(input=inp,output=out)

    return model
def build_lstm_model(input_shape=(256,256,3)):

    inp=keras.layers.Input(input_shape)

    back=efn.EfficientNetB0(input_shape=input_shape,include_top=False,weights=None,pooling='avg')

    back.load_weights('/kaggle/input/efficientnet-keras-weights-b0b5/efficientnet-b0_imagenet_1000_notop.h5')

    #back.trainable = False

    out=back(inp)

    #out1=keras.layers.Conv2D(16,3,activation='relu',padding='same')(inp)

    #out2=keras.layers.Conv2D(16,3,activation='relu',padding='same')(inp)

    #out=keras.layers.concatenate([out1,out2])



    #out=keras.layers.MaxPooling2D((2,2))(out)

    #out1=keras.layers.Conv2D(16,3,activation='relu',padding='same')(out)

    #out2=keras.layers.Conv2D(16,3,activation='relu',padding='same')(out)

    #out=keras.layers.concatenate([out1,out2])



    #out=keras.layers.GlobalAveragePooling2D()(out)

    #out=keras.layers.Dense(1024,activation='relu')(out)

    #out=keras.layers.BatchNormalization()(out)

    #out=keras.layers.Dropout(0.5)(out)

    out=keras.layers.Dense(256,activation='relu')(out)

    out=keras.layers.Dense(2,activation='softmax')(out)



    model=keras.models.Model(input=inp,output=out)

    return model
def build_lstm_model(input_shape=(None,256,256,3)):

    inp=keras.layers.Input(input_shape)

    back=efn.EfficientNetB0(input_shape=(256,256,3),include_top=False,weights=None,pooling=None)

    back.load_weights('/kaggle/input/efficientnet-keras-weights-b0b5/efficientnet-b0_imagenet_1000_notop.h5')

    #back.trainable = False

    

    #out=back(inp)

    #out1=keras.layers.ConvLSTM2D(32,3,activation='relu',padding='same')(inp)

    #out2=keras.layers.ConvLSTM2D(32,3,activation='relu',go_backwards=True,padding='same')(inp)

    #out=keras.layers.concatenate([out1,out2])

    

    out=keras.layers.TimeDistributed(back)(inp)

    

    out1=keras.layers.ConvLSTM2D(128,3,activation='relu')(out)

    out2=keras.layers.ConvLSTM2D(128,3,go_backwards=True,activation='relu')(out)

    out=keras.layers.concatenate([out1,out2])



    out=keras.layers.GlobalAveragePooling2D()(out)

    #out=keras.layers.Dense(1024,activation='relu')(out)

    #out=keras.layers.BatchNormalization()(out)

    #out=keras.layers.Dropout(0.5)(out)

    #out=keras.layers.Dense(256,activation='relu')(out)

    out=keras.layers.Dense(2,activation='softmax')(out)



    model=keras.models.Model(input=inp,output=out)

    return model
model=build_lstm_model()
model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
check=keras.callbacks.ModelCheckpoint('res_weights.h5', monitor='val_loss',save_best_only=True)
history=model.fit_generator(gen,validation_data=val,verbose=1,epochs=50,callbacks=[check])

#history=model.fit(np.array(train_data_x),keras.utils.to_categorical(train_data_y,2),validation_split=0.15,verbose=1,epochs=20,callbacks=[check],shuffle=True)
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])
model.load_weights('res_weights.h5')
import gc

gc.collect()
!mkdir test_data
"""test_data_x=[]



bad_names_test=[]

test_decode_data={}

for i in tqdm(range(len(test_files))):

    video = VideoReader(TEST_DIR+test_files[i],shape=PARAMS_VIDEO['shape'],is_face=PARAMS_VIDEO['is_face'],is_first_face=PARAMS_VIDEO['is_first_face'],is_first=PARAMS_VIDEO['is_first'],on_each=PARAMS_VIDEO['on_each'])

    video.get_video()

    start_ind=len(test_data_x)

    for j in range(len(video.faces)):

        if(len(video.faces[j])==1):

            

            for t in range(len(video.faces[j])):

                #print(video.faces[j][t].shape)

                cv2.imwrite(f'test_data/{test_files[i]}_{t}.png',video.faces[j][t])

                test_data_x.append(test_files[i])

                test_decode_data[test_files[i]]=(start_ind,len(test_data_x))    

                

        #print(i,len(train_data_x),len(train_data_y))

        else:

            bad_names_test.append(test_files[i])

    

    """
test_x=os.listdir('test_data')
test_x
test_gen=DataGeneratorFull(test_x,data_dir='test_data',dim=(256,256),batch_size=1,mode='predict')
res=model.predict_generator(test_gen,verbose=1)

#res=model.predict(np.array(test_data_x))
len(res)
pred_res=np.argmax(res,axis=-1)

plt.hist(pred_res)
sub_df=pd.read_csv(SUB_DIR)
full_res=[]

sub_test_files=list(sub_df['filename'].values)

for i in (range(len(sub_test_files))):

    if(sub_test_files[i] in test_decode_data.keys()):

        indxs=test_decode_data[sub_test_files[i]] 

        val=np.sum(res[indxs[0]:indxs[1]],axis=0)#,axis=-1)

        if(val[0]==0 and val[1]==0):

            val=np.array([0,1],dtype='float32')

        val=np.argmax(val)

        full_res.append(val)

        #print(indxs,val)

    else:

        full_res.append(1)

        
len(full_res)
full_res
plt.hist(full_res)
sub_df.head()
#model.load_weights('res_weights.h5')
#res=model.predict_generator(test_gen,verbose=1)
sub_df['label']=full_res
sub_df.head()
sub_df.to_csv('submission.csv',index=False)
pd.read_csv('submission.csv').head()