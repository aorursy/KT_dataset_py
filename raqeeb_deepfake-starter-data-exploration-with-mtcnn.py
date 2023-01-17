!pip install imageio

!pip install imageio-ffmpeg

!pip install mtcnn
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

from collections import defaultdict

import cv2

import numpy as np

import pylab

import imageio

from mtcnn import MTCNN

import cv2

import json

from tqdm import tqdm
def train_test_data(path,complete_path=False):

    '''get file names of train and test data folder

       complete_path - get list with the full path  

        '''

    dataset = []

    folderseq = []

    for dirname, _, filenames in os.walk(path):

        

        folderseq.append(_)



        for folder in _:

            

            for dire,_,file in os.walk(os.path.join(dirname,folder)):

                if complete_path == True:

                    full_path = []

                    for val in file:

                        full_path.append(os.path.join(dire,val))

                    dataset.append(full_path)

                else:

                    dataset.append(file)

#     print(dataset)

   

    if folderseq[0][0] == 'train_sample_videos':

        return dataset[0],dataset[1]

        

    else :

        return dataset[1],dataset[0]

        

    

def count_file(file_list):

    '''Check type of file format in each train test folder'''

    count = defaultdict(int)

    for file in file_list:

        fileformat = file.split('.')[1]

        count[fileformat] += 1

    return count
TRAIN_FOLDER = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/'

TEST_FOLDER = '/kaggle/input/deepfake-detection-challenge/test_videos/'

train_file,test_file = train_test_data('/kaggle/input/deepfake-detection-challenge/',complete_path=False)
count_file(train_file),count_file(test_file)
def get_jsonpath(train_file):

    '''get the path on json file in train data'''

    for file in train_file:        

        if file.split('.')[1] == 'json':

            json_index = train_file.index(file)

    metadata = os.path.join(TRAIN_FOLDER,train_file[json_index])

    return metadata

metadata = get_jsonpath(train_file)
'''remove metadata form train folder '''

train_file.remove('metadata.json')

print(len(train_file))
json_meta = pd.read_json(metadata)

json_meta = json_meta.T.reset_index()

json_meta.columns = ['input', 'label', 'split', 'original']

json_meta.head()
json_meta['label'].value_counts()
def check_datacoverage(json_meta,train_file):

    '''check how may .mp4 files of metadata.json present in our dataset'''

    original = json_meta.dropna().values.tolist()

    test1 = []

    for val in original:

        if val in train_file:

            test1.append(val)

    return print('Out of {} videos {} files available in dataset'.format(len(original),len(test1)))

    
# All  input videos of json  file present in train folder

check_datacoverage(json_meta['input'],train_file)
# Out of 323 original videos only 58 vides files available in train dataset

check_datacoverage(json_meta['original'],train_file)
def generate_data(metadata,train_file):

    '''generate dataset of only those FAKE - REAL file pairs that are available in the dataset'''

    json_data = json.load(open(metadata))

    dataset = []

    for val in json_data:

        if val in train_file:

            if json_data[val]['label'] == 'FAKE':

                if json_data[val]['original'] in train_file:

                    dataset.append((val,json_data[val]['original']))

    print('Total File Pairs - {}'.format(len(dataset)))

    return dataset

    
fake_original_data = generate_data(metadata,train_file)
def get_data_imgio(data,frames):

    '''Return fake and real images  for single video file

    with numbers of frames set to scan for faces. All images are resized to

        160*160 size

    '''

    test0 = []

    test1 = []

    vid = imageio.get_reader(os.path.join(TRAIN_FOLDER,data[0]),  'ffmpeg')

    vid2 = imageio.get_reader(os.path.join(TRAIN_FOLDER,data[1]),  'ffmpeg')

    nums = np.linspace(0,vid.count_frames() - 1,frames,dtype=int)

    for num in nums:

        try:

            image = vid.get_data(num)

            image2 = vid2.get_data(num)

    #         print(image)

            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

            detector = MTCNN()

            x1, y1, width, height = detector.detect_faces(img)[0]['box']

            x2, y2 = x1 + width, y1 + height

            res = cv2.resize(img[y1:y2, x1:x2],(160,160))

            res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

            test0.append(res)

            res2 = cv2.resize(img2[y1:y2, x1:x2],(160,160))

            res2 = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)

            test1.append(res2)

        except:

            continue

#         break

        

        

    test0 = np.stack(test0)

    test1 = np.stack(test1)

    return test0,test1

def show_img(num):

    '''select the frame to show fake and real image out of total frames set to scan'''

    fig,ax = plt.subplots(1,2,figsize=(10,10))

    ax[0].imshow(x[num])

    ax[0].set_title('FAKE')

    ax[1].imshow(y[num])

    ax[1].set_title('REAL')

    

x,y = get_data_imgio(fake_original_data[30],30)

x.shape,y.shape

# Each vides mp4 file 30 frames are selected to get faces from them 
show_img(22) #select from 0 - 29 because 30 frames selected to scan the image
x,y = get_data_imgio(fake_original_data[15],30)

show_img(17) #select from 0 - 29 because 30 frames selected to scan the image
def generate_training_data(fake_original_data):

    '''generate fake and original video  with there labels'''

    trainx = []

    trainy = []

    for data in tqdm(fake_original_data):

        try:

            train,test = get_data_imgio(data,30)

            trainx.append(train)

            trainy.append(test)

        except:

            continue

#         break

    inp = np.vstack(trainx)

    real = np.vstack(trainy)

    zero = np.zeros(inp.shape[0])

    one = np.ones(inp.shape[0])

    label = np.concatenate((zero,one))

    images = np.concatenate((inp,real),axis=0)

    images = np.expand_dims(images,axis=-1)

    

    return images,label

X,y = generate_training_data(fake_original_data)
X.shape,y.shape