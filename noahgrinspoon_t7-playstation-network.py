# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install cvlib

import numpy as np

import time

import sys

import os

import random

from skimage import io

import pandas as pd

from matplotlib import pyplot as plt

from shutil import copyfile

from tqdm import tqdm

import cv2

import tensorflow as tf

from PIL import Image

#ImageFile.LOAD_TRUNCATED_IMAGES = True

import csv

import math

import copy

import cvlib as cv
dataset = '/kaggle/input' 

data_dir = 'asn10e-final-submission-coml-facial-recognition' # Write the file name of the photos

dataset_dir = os.path.join(dataset, data_dir)

annotations_df = pd.DataFrame(columns=['FileName', 'XMin', 'YMin', 'XMax', 'YMax', 'AveragePrecisions', 'ClassName'])



paths = [i for i in os.listdir(dataset_dir) if i[-3:] == "jpg"]

print(paths)

# get filenames of in dataset

img_fns = os.listdir(dataset_dir)

print('There are {} files in {}'.format(len(img_fns), dataset_dir));



image_path = os.path.join(dataset_dir, paths[0])



# Find boxes of each image and put them in dataframe

im = cv2.imread(image_path)

faces, confidences = cv.detect_face(im)

print('img one') 

for face in faces:

    (startX,startY) = face[0],face[1]

    (endX,endY) = face[2],face[3]

    print(startX)

    annotations_df = annotations_df.append({'FileName': paths[0], 

                                            'XMin': startX, 

                                            'YMin': startY, 

                                            'XMax': endX,

                                            'YMax': endY,

                                            'AveragePrecisions': confidences[0],

                                            'ClassName': "Face"},

                                           ignore_index=True)



img_path = os.path.join(dataset_dir, paths[1])    

im = plt.imread(img_path)

faces, confidences = cv.detect_face(im)

print('img two') 

for face in faces:

    (startX,startY) = face[0],face[1]

    (endX,endY) = face[2],face[3]

    print(startX)

    annotations_df = annotations_df.append({'FileName': paths[1], 

                                            'XMin': startX, 

                                            'YMin': startY, 

                                            'XMax': endX,

                                            'YMax': endY,

                                            'AveragePrecisions': confidences[0],

                                            'ClassName': "Face"},

                                           ignore_index=True)

   

        

        

# save annotations_df to file, without pandas row indexes

annotations_fn = os.path.join('../output', 'annotations.csv')

annotations_df.to_csv('annotations.csv', index=False, index_label=False)

print("Saved annotations to :", annotations_fn)
annotation_file_path = '/kaggle/working/annotations.csv'

dataset_dir_path = os.path.dirname(annotation_file_path)

with open(annotation_file_path, mode='r') as csvfile:

    annotations = csv.reader(csvfile, delimiter=',')

    header = next(annotations)

    for i, row in enumerate(annotations):

        tup1 = (filename,xmin,ymin,xmax,ymax,averagePrecisions,class_name) = row

        img_path = '/kaggle/input/asn10e-final-submission-coml-facial-recognition/1.jpg'

        ori_img = cv2.imread(img_path)            

        cv2.rectangle(ori_img, (int(tup1[1]), int(tup1[2]), int(tup1[3]) - int(tup1[1]), int(tup1[4]) - int(tup1[2])), (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(ori_img, averagePrecisions, (int(tup1[1]), int(tup1[2]) - 10), font, 0.5, (0, 255, 0), 2)

        RGB_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

        

        plt.figure()

        plt.subplot(1, 2, 1)

        plt.imshow(RGB_img)

        

        plt.subplot(1, 2, 2)

        sub_img = cv2.imread(img_path)

        

        sub_img_RGB = cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB)

        crop_img = sub_img_RGB[int(tup1[2]):int(tup1[4]), int(tup1[1]):int(tup1[3])].copy()

        plt.imshow(crop_img)

            
from keras.models import load_model

model = load_model('/kaggle/input/model1/facenet_keras.h5')
dataset = 'pm-train' 

data_dir = '/kaggle/input'

dataset_dir = os.path.join(data_dir, dataset)

annotations_df = pd.DataFrame(columns=['FileName', 'XMin', 'YMin', 'XMax', 'YMax', 'AveragePrecisions', 'ClassName'])

img_path = '/kaggle/input/asn10e-final-submission-coml-facial-recognition/1.jpg'



# get filenames of in dataset

img_fns = os.listdir(dataset_dir)

print('There are {} files in {}'.format(len(img_fns), dataset_dir));



# Find boxes of each image and put them in dataframe

for img_fn in tqdm(img_fns):

    img_path = os.path.join(dataset_dir, img_fn)

    

    

    try:

        with Image.open(img_path) as img:

            

            im = cv2.imread(img_path)

            plt.imshow(im)

            

            faces, confidences = cv.detect_face(im)

           

            for face in faces:

                (startX,startY) = face[0],face[1]

                (endX,endY) = face[2],face[3]

                annotations_df = annotations_df.append({'FileName': img_fn, 

                                                        'XMin': startX, 

                                                        'YMin': startY, 

                                                        'XMax': endX,

                                                        'YMax': endY,

                                                        'AveragePrecisions': confidences[0],

                                                        'ClassName': "Face"},

                                                    ignore_index=True)

                break;

    except:

        print("Skipped due to Image.open error: ", img_path)

        

annotations_fn = os.path.join('../output', 'annotations_train.csv')

annotations_df.to_csv('annotations_train.csv', index=False, index_label=False)

print("Saved annotations to :", annotations_fn)
annotation_file_path = '/kaggle/working/annotations_train.csv'

dataset_dir_path = '/kaggle/input/pm-train'

with open(annotation_file_path, mode='r') as csvfile:

    annotations = csv.reader(csvfile, delimiter=',')

    header = next(annotations)

    for i, row in enumerate(annotations):

        tup1 = (filename,xmin,ymin,xmax,ymax,averagePrecisions,class_name) = row

        

        img_path = os.path.join(dataset_dir_path, tup1[0])

        ori_img = cv2.imread(img_path)            

        cv2.rectangle(ori_img, (int(tup1[1]), int(tup1[2]), int(tup1[3]) - int(tup1[1]), int(tup1[4]) - int(tup1[2])), (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(ori_img, averagePrecisions, (int(tup1[1]), int(tup1[2]) - 10), font, 0.5, (0, 255, 0), 2)

        RGB_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

        

        plt.figure()

        plt.subplot(1, 2, 1)

        plt.imshow(RGB_img)

        

        plt.subplot(1, 2, 2)

        sub_img = cv2.imread(img_path)

        

        sub_img_RGB = cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB)

        crop_img = sub_img_RGB[int(tup1[2]):int(tup1[4]), int(tup1[1]):int(tup1[3])].copy()

        plt.imshow(crop_img)

            
# calculate a face embedding for each face in the dataset using facenet

import os

import cvlib as cv

import cv2

import pandas as pd

import numpy as np

from numpy import load

from numpy import expand_dims

from numpy import asarray

from numpy import savez_compressed

from keras.models import load_model

import csv

from PIL import Image

from numpy import load

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import Normalizer

from sklearn.svm import SVC



# get the face embedding for one face

def get_embedding(model, face_pixels):

    # scale pixel values

    face_pixels = face_pixels.astype('float32')

    # standardize pixel values across channels (global)

    mean, std = face_pixels.mean(), face_pixels.std()

    face_pixels = (face_pixels - mean) / std

    # transform face into one sample

    samples = expand_dims(face_pixels, axis=0)

    # make prediction to get embedding

    yhat = model.predict(samples)

    return yhat[0]



# convert each face in the train set to an embedding



annotation_file_path = '/kaggle/working/annotations_train.csv'

dataset_dir_path = '/kaggle/input/pm-train'

required_size=(160, 160)

vectors_df = pd.DataFrame(columns=['ImageDir', 'Name', 'Vector'])

vectors_train = list()



with open(annotation_file_path, mode='r') as csvfile:

    annotations = csv.reader(csvfile, delimiter=',')

    header = next(annotations)

    

    

    

    for i, row in enumerate(annotations):

        tup1 = (filename,xmin,ymin,xmax,ymax,averagePrecisions,class_name) = row

        

        img_path = os.path.join(dataset_dir_path, str(filename))

        ori_img = cv2.imread(img_path)

        sub_img_RGB = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

        face = sub_img_RGB[int(tup1[2]):int(tup1[4]), int(tup1[1]):int(tup1[3])].copy()

        image = Image.fromarray(face)

        image = image.resize(required_size)

        face_array = asarray(image)

        

        embedding = get_embedding(model, face_array)

        

        vectors_train.append(embedding)

        name = str(tup1[0])

        person_name = name[0] 

        vectors_df = vectors_df.append({'ImageDir': name,

                                        'Name': person_name, 

                                        'Vector': embedding},

                                      ignore_index=True)

        

        

vectors_fn = os.path.join('../output', 'vectors_train.csv')

vectors_df.to_csv('vectors_train.csv', index=False, index_label=False)

print("Saved vectors to :", vectors_fn)



# convert each face in the test set to an embedding



annotation_file_path = '/kaggle/working/annotations.csv'

dataset_dir_path = '/kaggle/input/asn10e-final-submission-coml-facial-recognition/1.jpg'



vectors_test = list()

file_name_list =list()

test_vectors_df = pd.DataFrame(columns=['ImageDir', 'Name', 'Vector'])



with open(annotation_file_path, mode='r') as csvfile:

    annotations = csv.reader(csvfile, delimiter=',')

    header = next(annotations)

    

    

    

    for i, row in enumerate(annotations):

        tup1 = (filename,xmin,ymin,xmax,ymax,averagePrecisions,class_name) = row

        

        img_path = dataset_dir_path

        ori_img = cv2.imread(img_path)

        #sub_img_RGB = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

        #print(img_path)

        

        face = ori_img[int(tup1[2]):int(tup1[4]), int(tup1[1]):int(tup1[3])].copy()

        image = Image.fromarray(face)

        image = image.resize(required_size)

        face_array = asarray(image)

        

        embedding = get_embedding(model, face_array)

        #arr = np.array(embedding)

        #newArr = arr.reshape(1, -1)

        # normalize input vectors

        #in_encoder = Normalizer(norm='l2')

        #testX = in_encoder.transform(newArr)

        #test = testX.reshape(-1)

        vectors_test.append(embedding)

        name = str(tup1[0])

        file_name_list.append(name)

        person_name = name[0 : len(name) - 9] 

        test_vectors_df = test_vectors_df.append({'ImageDir': name,

                                                  'Name': person_name, 

                                                  'Vector': embedding},

                                                ignore_index=True)

        #print(name)

print(file_name_list)

vectors_fn = os.path.join('../output', 'vectors_test.csv')

test_vectors_df.to_csv('vectors_test.csv', index=False, index_label=False)

print("Saved vectors to :", vectors_fn)
import os

import cvlib as cv

import cv2

import pandas as pd

import numpy as np

from numpy import load

from numpy import expand_dims

from numpy import asarray

from numpy import savez_compressed

from keras.models import load_model

import csv

from PIL import Image

from numpy import load

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import Normalizer

from sklearn.svm import SVC





vectors_file_path_train = '/kaggle/working/vectors_train.csv'

dataset_dir_path_train = '/kaggle/input/pm-train'



vector_list_train = list()

name_list_train = list()



with open(vectors_file_path_train, mode='r') as csvfile:

    annotations = csv.reader(csvfile, delimiter=',')

    header = next(annotations)

    

    for i, row in enumerate(annotations):

        tup = (FileName,Name,Vector) = row

        name_list_train.append(tup[1])

        vector_list_train.extend(tup[2])

    #print(name_list_train)

    #print(vector_list_train)

vectors_file_path_test = '/kaggle/working/vectors_test.csv'

dataset_dir_path_test = '/kaggle/input/asn10e-final-submission-coml-facial-recognition/1.jpg'



test_vectors_df = pd.DataFrame(columns=['Name', 'Vector'])

vector_list_test = list()

name_list_test = list()



with open(vectors_file_path_test, mode='r') as csvfile:

    annotations = csv.reader(csvfile, delimiter=',')

    header = next(annotations)

    

    for i, row in enumerate(annotations):

        tup = (FileName, Name, Vector) = row

        name_list_test.append(tup[1])

        vector_list_test.extend(tup[2])



in_encoder = Normalizer(norm='l2')

vectors_train = in_encoder.transform(vectors_train)

vectors_test = in_encoder.transform(vectors_test)

        

#out_encoder = LabelEncoder()

#out_encoder.fit(name_list_train)

    

#name_list_train = out_encoder.transform(name_list_train)

#name_list_test = out_encoder.transform(name_list_test)



# fit model

model = SVC(kernel='linear', probability=True)

model.fit(vectors_train, name_list_train)

# predict

yhat_train = model.predict(vectors_train)

yhat_test = model.predict(vectors_test)

# score

score_train = accuracy_score(name_list_train, yhat_train)

score_test = accuracy_score(name_list_test, yhat_test)

# summarize

print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))

import os

import cvlib as cv

import cv2

import pandas as pd

import numpy as np

from numpy import load

from numpy import expand_dims

from numpy import asarray

from numpy import savez_compressed

from keras.models import load_model

import csv

from PIL import Image

from numpy import load

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import Normalizer

from sklearn.svm import SVC

import random

from matplotlib import pyplot as plt



model = SVC(kernel='linear', probability=True)

model.fit(vectors_train, name_list_train)



result_df = pd.DataFrame(columns=['filename', 'person_id', 'id', 'xmin', 'xmax', 'ymin', 'ymax'])







annotation_file_path = '/kaggle/input/annotations-file/annotations.csv'



with open(annotation_file_path, mode='r') as csvfile:

    

    annotations = csv.reader(csvfile, delimiter=',')

    header = next(annotations)

    for i, row in enumerate(annotations):

        

        tup1 = (filename,person_id, id, xmin,xmax,ymin,ymax) = row

        # test model on a random example from the test dataset

        selection = i



        #random_face_pixels = testX_faces[selection]

        random_face_emb = vectors_test[selection]

        random_face_class = name_list_test[selection]

        random_face_name = tup1[0]



        # prediction for the face

        samples = expand_dims(random_face_emb, axis=0)

       

        yhat_class = model.predict(samples)

        yhat_prob = model.predict_proba(samples)

        # get name

        class_index = yhat_class[0]

        #predict_names = out_encoder.inverse_transform(yhat_class)

        predict_names = tup1[0]

        person_name = str(random_face_name)

        name = person_name[2 : len(person_name) - 2]

        result_df = result_df.append({'filename': tup1[0],

                                      'person_id': tup1[1],

                                      'id': tup1[2],

                                      'xmin': tup1[3],

                                      'xmax': tup1[4],

                                      'ymin': tup1[5],

                                      'ymax': tup1[6]},

                                    ignore_index=True)

        

        

        

    for i in range(47):

        if (i!=26 and i!=38 and i!=12 and i!=45 and i!=28 and i!=9 and i!=5 and i!=6 and i!=18) :

            result_df = result_df.append({'filename': 0,

                                          'person_id': i,

                                          'id': '0_' + str(i),

                                          'xmin': 0,

                                          'xmax': 0,

                                          'ymin': 0,

                                          'ymax': 0},

                                        ignore_index=True)



result_fn = os.path.join('../output', 'results.csv')

result_df.to_csv('results.csv', index=False, index_label=False)

print("Saved vectors to :", result_fn)







        



# #def parse_annotations_csv1(annotation_file_path):

# result_file_path = '/kaggle/working/results.csv'

# dataset_dir_path = os.path.dirname(result_file_path)

# original_img_path = os.path.dirname('/kaggle/input/pm-train')

# annotation_file_path_dir = os.path.dirname('/kaggle/working/annotations.csv')



# # plot for fun

# with open(result_file_path, mode='r') as csvfile:

    

#     annotations = csv.reader(csvfile, delimiter=',')

#     header = next(annotations)

    

#     for i, row in enumerate(annotations):

#         tup1 = (filename,name,prediction,expected,xmin,ymin,xmax,ymax) = row

        

#         file_name = str(tup1[0]) #+ '_0001.jpg'

#         img_path = '/kaggle/input/asn10e-final-submission-coml-facial-recognition/1.jpg'

#         print(img_path)

#         ori_img = cv2.imread(img_path)

#         cv2.rectangle(ori_img, (int(tup1[4]), int(tup1[5]), int(tup1[6]) - int(tup1[4]), int(tup1[7]) - int(tup1[5])), (0, 255, 0), 2)

#         font = cv2.FONT_HERSHEY_SIMPLEX

#         cv2.putText(ori_img, tup1[2], (int(tup1[5]), int(tup1[6]) - 10), font, 0.5, (0, 255, 0), 2)

#         RGB_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

#         #title = "Predicted: " + tup1[2]

        

#         plt.figure()

#         plt.subplot(1, 2, 1)

#         plt.imshow(RGB_img)

        

        

#         if i == 5:

#             break


