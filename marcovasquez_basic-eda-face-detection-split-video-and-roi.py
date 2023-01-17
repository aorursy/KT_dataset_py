



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import os




train_dir = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/'

train_video_files = [train_dir + x for x in os.listdir(train_dir)]

test_dir = '/kaggle/input/deepfake-detection-challenge/test_videos/'

test_video_files = [test_dir + x for x in os.listdir(test_dir)]



df_train = pd.read_json('/kaggle/input/deepfake-detection-challenge/train_sample_videos/metadata.json').transpose()

df_train.head()
df_train.shape # We have 400 videos 
df_train.original.nunique()  # from this aprox 209 originals create 400, would be the same video? lest check soon
df_train.label.value_counts()
df_train.label.value_counts().plot.bar()
df_train.label.value_counts(normalize=True) # Check that aprox 80% are fake
df_train['original'].value_counts()
df_train[df_train['original']=='meawmsgiti.mp4']  # Looking the same files of videos
video1= train_dir + 'akxoopqjqz.mp4'  

video2 =train_dir + 'altziddtxi.mp4'

video3 = train_dir+ 'arlmiizoob.mp4'
def display_img(video):

    cap = cv2.VideoCapture(video)  # take 1 picture

    ret, frame = cap.read()

    fig = plt.figure(figsize=(8,8))

    ax = fig.add_subplot(111)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    ax.imshow(frame)
display_img(video1)
display_img(video2)

display_img(video3)  # lets check this picture 
first_Video = train_video_files[8]

first_Video
name_video = first_Video.split('/', 5)[5] # I will use this funtion soon
df_train[df_train.index == name_video] 
count = 0

cap = cv2.VideoCapture(first_Video)

ret,frame = cap.read()



while count < 3:

    cap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))   

    ret,frame = cap.read()

    if count == 0:

        image0 = frame

    elif count == 1:

        image1 = frame

    elif count == 2:

        image2 = frame

    

    #cv2.imwrite( filepath+ "\frame%d.jpg" % count, image)     # Next I will save frame as JPEG

    count = count + 1
def display(img):

    

    fig = plt.figure(figsize=(8,8))

    ax = fig.add_subplot(111)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ax.imshow(img)
display(image0)  # frame 1
display(image1) # frame 2
display(image2) # frame 3
# You need to Download or add this file to your notebook, Check in the input files



face_cascade = cv2.CascadeClassifier('/kaggle/input/cascade/haarcascade_frontalface_default.xml' ) # Cascade for faces
def detect_face(img):

    

    face_img = img.copy()

  

    face_rects = face_cascade.detectMultiScale(face_img,scaleFactor=1.3, minNeighbors=5) 

    

    for (x,y,w,h) in face_rects: 

        cv2.rectangle(face_img, (x,y), (x+w,y+h), (0,255,0), 10) 

        

    return face_img
result = detect_face(image2)
display(result)
second_Video= train_video_files[10]

name_video2 = second_Video.split('/', 5)[5] # I will use this funtion soon

name_video2
df_train[df_train.index == name_video2] 


count = 0

cap = cv2.VideoCapture(second_Video)

ret,frame = cap.read()



while count < 5:

    cap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))   

    ret,frame = cap.read()

    if count == 0:

        image0 = frame

    elif count == 1:

        image1 = frame

    elif count == 2:

        image2 = frame

    elif count == 3:

        image3 = frame

    elif count == 4:

        image4 = frame

    

    #cv2.imwrite( pathOut + "\\frame%d.jpg" % count, image)     # Next I will save frame as JPEG

    count = count + 1
image = detect_face(image0)

display(image)
image = detect_face(image2)

display(image)
image = detect_face(image3)

display(image)
def ROI(img):

    

    face_img = img.copy()

  

    face_rects = face_cascade.detectMultiScale(face_img,scaleFactor=1.3, minNeighbors=5) 

    

    for (x,y,w,h) in face_rects: 

        roi = face_img[y:y+h,x:x+w] 

        

        

    return roi
image = ROI(image3)

display(image)
def ROI_Expand(img):

    

    offset = 50  # play around this value

    

    face_img = img.copy()

  

    face_rects = face_cascade.detectMultiScale(face_img,scaleFactor=1.3, minNeighbors=5) 

    

    for (x,y,w,h) in face_rects: 

        roi = face_img[y-offset:y+h+offset,x-offset:x+w+offset] 

        

        

    return roi
image = ROI_Expand(image3)

display(image)
submission = pd.read_csv("/kaggle/input/deepfake-detection-challenge/sample_submission.csv")

submission['label'] = 0.5 # 

submission.to_csv('submission.csv', index=False)