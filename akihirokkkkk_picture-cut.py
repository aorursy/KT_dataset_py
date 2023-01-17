import pandas as pd

train_sample_metadata = pd.read_json('../input/deepfake-detection-challenge/train_sample_videos/metadata.json').T

train_sample_metadata.head()
#正しく開けることはわかった

import cv2

from matplotlib import pyplot as plt



face_cascade_path = "/opt/conda/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml"

def detectFace(image):

    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    src_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    facerect = face_cascade.detectMultiScale(src_gray)

 

    return facerect



# 何かしらの動画ファイル。

video_path = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/akxoopqjqz.mp4'

cap = cv2.VideoCapture(video_path)

print("正しく開けているかどうか : {}".format(cap.isOpened()))

framenum = 0

faceframenum = 0

color = (255, 255, 255)

 

while(1):

    framenum += 1

    #retは,cap.read()でフレームが正しく読み込めたかどうかを教えてくれるフラグ

    ret, image = cap.read()

    if not ret:

        break

 

    if framenum%10==0:

        facerect = detectFace(image)

        if len(facerect) == 0: continue

 

        for rect in facerect:

            croped = image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]

            plt.imshow(cv2.cvtColor(croped, cv2.COLOR_BGR2RGB))

            cv2.imwrite("../output/kaggle/working/deepfake-detection-challenge/face_picture/" + str(faceframenum) + ".jpg", croped)

 

        faceframenum += 1



cap.release()