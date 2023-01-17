# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import load_model

from keras.layers import LSTM

from sklearn.model_selection import train_test_split

from keras.applications.xception import Xception

from keras.applications.inception_v3 import preprocess_input

import cv2 as cv

import math

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# Any results you write to the current directory are saved as output.
!pip install mtcnn
# personal imports

import json

from mtcnn.mtcnn import MTCNN

import matplotlib.pyplot as plt
from os import listdir

from os.path import join



methods_name = ['Deepfakes', 'FaceSwap', 'Face2Face', 'NeuralTextures']

methods_size = []

submission = []



videos = '/kaggle/input/faceforensics/original_sequences/youtube/c23/videos'

files = [join(videos, filename) for filename in listdir(videos)]

methods_size.append(len(files))

submission.extend(sorted(files))

print(len(files), 'Original')



for method in methods_name:

    videos = f'/kaggle/input/faceforensics/manipulated_sequences/{method}/c23/videos'

    files = [join(videos, filename) for filename in listdir(videos)]

    methods_size.append(len(files))

    submission.extend(sorted(files))

    print(len(files), method)
model = load_model('/kaggle/input/facenet/keras-facenet/model/facenet_keras.h5')

# summarize input and output shape

print(model.inputs)

print(model.outputs)
# Xception

model = Xception(include_top=False, weights='imagenet', input_shape=(160,160,3), pooling='avg')

print(model.summary())
def get_frames(video_path,offset=20,nframes=30):

    cap = cv.VideoCapture(video_path)

    length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    if length<nframes:

        print(f"Video too short {video_path}")

        return []

    if offset+nframes>=length:

        offset=length-nframes

    cap.set(1,offset)

    images=[]

    for i in range(nframes):

        success, image = cap.read()

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        images.append(image)

    cap.release()

    return images



def extract_face(image,detector, required_size=(160, 160)):

    results = detector.detect_faces(image)

    # extract the bounding box from the first face

    if len(results)==0:

        return cv.resize(image,required_size)

    x1, y1, width, height = results[0]['box']

    # bug fix

    x1, y1 = abs(x1), abs(y1)

    x2, y2 = x1 + width, y1 + height

    #print(x1,x2)

    # add padding

    x1-=min(20,x1,y1)

    y1-=min(20,x1,y1)

    x2+=min(20,image.shape[1]-x2-1,image.shape[0]-y2-1)

    y2+=min(20,image.shape[1]-x2-1,image.shape[0]-y2-1)

    # extract the face

    face = image[y1:y2, x1:x2]

    

    face = cv.resize(face,required_size)

   

    # standardize for the facenet model

    return preprocess_input(face)





def save_originals(src_dir,out_dir):

    detector = MTCNN()

    saved_ids=[]

    for dirname, _, filenames in os.walk(src_dir):

        for idx,filename in enumerate(sorted(filenames)):

            nframes=15

            frames=get_frames(os.path.join(dirname, filename),nframes=nframes)

            if len(frames)<nframes:

                continue

            #ignore videos with more than one face

            results = detector.detect_faces(frames[0])

            if len(results)>1:

                continue

            face_frames=[extract_face(frame,detector) for frame in frames]

            embeds = model.predict(np.array(face_frames))

            original_embeds[filename[:3]]=embeds

            #os.mkdir(f'{out_dir}/{filename[:3]}')

            saved_ids.append(filename[:3])

    return saved_ids



def save_fakes(ids,src_dir,out_dir):

    detector = MTCNN()

    saved_ids=[]

    for file_id in ids:

        for dirname, _, filenames in os.walk(src_dir):

            for idx,filename in enumerate(sorted(filenames)):

                if file_id != filename[:3]:

                    continue

                nframes=15

                frames=get_frames(os.path.join(dirname, filename),nframes=nframes)

                if len(frames)<nframes:

                    continue

                #ignore videos with more than one face

                results = detector.detect_faces(frames[0])

                if len(results)>1:

                    continue

                face_frames=[extract_face(frame,detector) for frame in frames]

                embeds = model.predict(np.array(face_frames))

                deepfakes_embeds[file_id]=embeds

                saved_ids.append(file_id)

    return saved_ids
orig_dir='original'

deep_dir='deepfake'

os.mkdir(orig_dir)

os.mkdir(deep_dir)

original_embeds={}

deepfakes_embeds={}
ids=save_originals('/kaggle/input/faceforensics/original_sequences/youtube/c23/videos',orig_dir)
def serializeArray(arr):

    out=''

    for i in range(arr.shape[0]):

        out+=','.join([str(val) for val in arr[i].tolist()])

        out+='\n'

    return out

with open('original/orig.csv', 'a') as file:

    for key in sorted(original_embeds.keys()):

        file.write(serializeArray(original_embeds[key]))
ids_fakes=save_fakes(ids,'/kaggle/input/faceforensics/manipulated_sequences/Deepfakes/c23/videos',deep_dir)

ids=sorted(original_embeds.keys())

ids_fakes=sorted(deepfakes_embeds.keys())
with open('deepfake/deep.csv', 'a') as file:

    for key in sorted(deepfakes_embeds.keys()):

        file.write(serializeArray(deepfakes_embeds[key]))
with open('original/orig_index.txt', 'a') as file:

    file.write('\n'.join(ids))

with open('deepfake/deep_index.txt', 'a') as file:

    file.write('\n'.join(ids_fakes))
def tmpfunc():

    detector = MTCNN()

    original_embeds={}

    plt.figure(figsize=(20,10))

    columns=5

    num_vids=1000

    for dirname, _, filenames in os.walk('/kaggle/input/faceforensics/original_sequences/youtube/c23/videos'):

        for idx,filename in enumerate(sorted(filenames)):

            # grab first 30 frames from each video

            nframes=30

            frames=get_frames(os.path.join(dirname, filename),nframes=nframes)

            if len(frames)<nframes:

                continue

            #ignore videos with more than one face

            results = detector.detect_faces(frames[0])

            if len(results)>1:

                continue

            face_frames=[extract_face(frame,detector) for frame in frames]

            embeddings=model.predict(np.array(face_frames))

            original_embeds[filename]=embeddings

            print(f'Derived embeddings for {filename}')

            #plt.subplot(num_vids / columns, columns, len(original_embeds.keys()))

            #plt.imshow(face_frames[0])

            if len(original_embeds.keys())==num_vids:

                print("Videos processed till",idx)

                break
def tmpfunc2():

    os.mkdir('embeds')

    os.mkdir('embeds/original')

    os.mkdir('embeds/deepfakes')



    def serializeArray(arr):

        out=''

        for i in range(arr.shape[0]):

            out+=','.join([str(val) for val in arr[i].tolist()])

            out+='\n'

        return out



    with open('embeds/original/orig.csv', 'a') as file:

        for key in original_embeds.keys():

            file.write(serializeArray(original_embeds[key]))



    deepfakes_embeds={}

    # get corresponding deepfakes

    for key in original_embeds.keys():

        for dirname, _, filenames in os.walk('/kaggle/input/faceforensics/manipulated_sequences/Deepfakes/c23/videos'):

            for filename in filenames:

                if filename[:3] in key:

                    frames=get_frames(os.path.join(dirname, filename))

                    face_frames=[extract_face(frame,detector) for frame in frames]

                    embeddings=model.predict(np.array(face_frames))

                    deepfakes_embeds[key]=embeddings



    # write to disk

    with open('embeds/deepfakes/deep.csv', 'a') as file:

        for key in original_embeds.keys():

            file.write(serializeArray(deepfakes_embeds[key]))



    #for key in original_embeds.keys():

        #with open('embeds/original/'+key[:3]+"_orig.csv", 'w') as file:

            #file.write(serializeArray(original_embeds[key]))

        #with open('embeds/deepfakes/'+key[:3]+"_deep.csv", 'w') as file:

            #file.write(serializeArray(deepfakes_embeds[key]))
