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
#install pre-trained model
!pip install face_recognition
#import libraries
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
import cv2
#hog image of photos with faces
photos = ['/kaggle/input/face-recognition-dataset/face recognition/train/Greta thunber.jpg', '/kaggle/input/face-recognition-dataset/face recognition/train/download.jpg', '/kaggle/input/face-recognition-dataset/face recognition/train/michelle-obama-gettyimages-85246899.jpg']
for photo in photos:
 image = cv2.imread(photo)
 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
 #fd is feature descriptor    
 fd, hog_image = hog(image, orientations = 8, pixels_per_cell=(16, 16), cells_per_block = (1,1), visualize = True, multichannel = True)
 #plot input image and hog image 
 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,6), sharex=True, sharey=True)
 ax1.axis("off")
 ax1.imshow(image, cmap=plt.cm.gray)
 ax1.set_title("Input image")

 #rescaling for better display
 hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0,15))

 ax2.axis("off")
 ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
 ax2.set_title("Histogram of Oriented Gradients")
 plt.show()

#compare the dimension of hog image and the original image
for photo in photos:
 image = cv2.imread(photo)
 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)     
 fd, hog_image = hog(image, orientations = 8, pixels_per_cell=(16, 16), cells_per_block = (1,1), visualize = True, multichannel = True)  
 #hog image
 print(len(fd))
for photo in photos:
 image = cv2.imread(photo)
 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
 #original image   
 print(image.shape)
import face_recognition

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
import numpy as np
import cv2
%matplotlib inline
photos = ['/kaggle/input/face-recognition-dataset/face recognition/train/Greta thunber.jpg', '/kaggle/input/face-recognition-dataset/face recognition/train/download.jpg', '/kaggle/input/face-recognition-dataset/face recognition/train/michelle-obama-gettyimages-85246899.jpg']

for photo in photos:
 image = cv2.imread(photo)
 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 plt.imshow(image)
 plt.show()
photos = ['/kaggle/input/face-recognition-dataset/face recognition/train/Greta thunber.jpg', '/kaggle/input/face-recognition-dataset/face recognition/train/download.jpg', '/kaggle/input/face-recognition-dataset/face recognition/train/michelle-obama-gettyimages-85246899.jpg']



for photo in photos:
 image = cv2.imread(photo)
 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   
 face_locations = face_recognition.face_locations(image)
 number_of_faces = len(face_locations)
 #number of faces in photos   
 print("Found {} faces(s) in input image.".format(number_of_faces))
for photo in photos:
 image = cv2.imread(photo)
 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   
 face_locations = face_recognition.face_locations(image)   
 #face locations in a photo
 print(face_locations)
#face detection by showing it in a rectangular box defined by face locations
for photo in photos:
 image = cv2.imread(photo)
 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
 face_locations = face_recognition.face_locations(image)    
 plt.figure(figsize=(30, 10))
 plt.imshow(image)
 ax = plt.gca()
 for face_location in face_locations:
    top, right, bottom, left = face_location
    rect = Rectangle((right, top), left-right, bottom-top, fill=False, color = 'green')  
    ax.add_patch(rect)

 plt.show()
#3 known photos of 3 different individuals in a pretrained model

image = cv2.imread('/kaggle/input/face-recognition-dataset/face recognition/train/Greta thunber.jpg')
greta = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


image = cv2.imread('/kaggle/input/face-recognition-dataset/face recognition/train/download.jpg')
koebe = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


image = cv2.imread('/kaggle/input/face-recognition-dataset/face recognition/train/michelle-obama-gettyimages-85246899.jpg')
michelle_obama = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#encoding the faces
greta_encoding = face_recognition.face_encodings(greta)[0]
koebe_encoding = face_recognition.face_encodings(koebe)[0]
michelle_obama_encoding = face_recognition.face_encodings(michelle_obama)[0]

#making a list of the encondings of known faces 
known_face_encodings = [greta_encoding, koebe_encoding, michelle_obama_encoding]
#reading an unknown image in the pre-trained model
photos_to_recognize = ['/kaggle/input/face-recognition-dataset/face recognition/testing/82549739.jpg', '/kaggle/input/face-recognition-dataset/face recognition/testing/Michelle_Obama_2013_official_portrait.jpg', '/kaggle/input/face-recognition-dataset/face recognition/testing/greta thunberg.webp' , '/kaggle/input/face-recognition-dataset/face recognition/testing/26kobe-hp-fader-ss-slide-YWIZ-superJumbo.jpg', '/kaggle/input/face-recognition-dataset/face recognition/testing/7s7rstpg_greta-thunberg-in-brussels-afp_625x300_04_March_20.webp']
for photo in photos_to_recognize:
 image = cv2.imread(photo)
 unknown_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 plt.imshow(unknown_image)
 face_locations = face_recognition.face_locations(unknown_image)   
 unknown_face_encodings = face_recognition.face_encodings(unknown_image)
 ax = plt.gca()
 for face_location in face_locations:
    top, right, bottom, left = face_location
    rect = Rectangle((right, top), left-right, bottom-top, fill=False, color = 'green')  
    ax.add_patch(rect)
 from scipy.spatial import distance 
 for unknown_face_encoding in unknown_face_encodings:
    results = []
    for known_face_encoding in known_face_encodings:
        d = distance.euclidean(known_face_encoding, unknown_face_encoding)
        results.append(d)
        
    #set threshold    
    threshold = 0.5
    results = np.array(results)<=threshold
    
    name = "Unknown"
    
    if results[0]:
        name = "Greta Thunberg"
    elif results[1]:
        name = "Koebe Bryant"
    elif results[2]:    
        name = "Michelle Obama"
    print("Found {} in the photo.".format(name))
 plt.show()
#face recognition
image = cv2.imread('/kaggle/input/face-landmarks/IMG_20200815_114234.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
face_landmarks_list = face_recognition.face_landmarks(image)
import matplotlib.lines as mlines
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
plt.imshow(image)
ax = plt.gca()

for face_landmarks in face_landmarks_list:
    left_eyebrow_pts = face_landmarks['left_eyebrow']
    pre_x, pre_y = left_eyebrow_pts[0]
    for (x, y) in left_eyebrow_pts[1:]:
        l = mlines.Line2D([pre_x, x], [pre_y,y], color = "green")
        ax.add_line(l)
        pre_x, pre_y = x,y
    right_eyebrow_pts = face_landmarks['right_eyebrow']
    pre_x, pre_y = right_eyebrow_pts[0]
    for (x, y) in right_eyebrow_pts[1:]:
        l = mlines.Line2D([pre_x, x], [pre_y,y], color = "green")
        ax.add_line(l)
        pre_x, pre_y = x,y
    p = Polygon(face_landmarks['top_lip'], facecolor='lightsalmon', edgecolor = 'orangered')#, fill = (120, 0, 28))
    ax.add_patch(p)
    p = Polygon(face_landmarks['bottom_lip'], facecolor='lightsalmon', edgecolor = 'orangered')#, fill = (120, 0, 28))
    ax.add_patch(p)
plt.show()
    