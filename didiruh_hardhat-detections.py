# impor pustaka yang diperlukan

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.layers import AveragePooling2D

from tensorflow.keras.layers import Dropout

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Input

from tensorflow.keras.models import Model

from tensorflow.keras.models import load_model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.preprocessing.image import img_to_array

from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.preprocessing import MultiLabelBinarizer

import numpy as np

import matplotlib.pyplot as plt

import os

import cv2

import time
# download imutils

!pip install imutils

from imutils import paths

from imutils.video import VideoStream
# buat argparse class

# karena pada kaggle tidak dapat menginstall argparse

class ap:

    # untuk training model 

    dataset_1 = '../input/pascal-voc-cropping'

    dataset_2 = "../input/face-mask-detection-data"

    plot = 'plot.png'

    model = 'hardhat_detector.model'

    

    # untuk implementasi model

    image = []

    face = 'face_detector'

    confidence = 0.5
# load model face detektor

print("[INFO] loading face detector model...")

prototxtPath = "../input/face-detection-ssd-model/CAFFE_DNN/deploy.prototxt.txt"

weightsPath = "../input/face-detection-ssd-model/CAFFE_DNN/res10_300x300_ssd_iter_140000.caffemodel"

faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

faceNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

faceNet.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)



# load model face-mask detector

print("[INFO] loading face mask detector model...")

maskNet = load_model("../input/face-mask-training/hardhat_detector.model")



hardHatNet = load_model("../input/hardhat-training/hardhat_detector.model")
# buat fungsi untuk mendeteksi mask

# frame : video

# faceNet : model pendeteksi face

# maskNet : model pendeteksi mask

def detect_and_predict_hardhat_mask(frame, faceNet, hardHatNet, maskNet):

    # buat blob

    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),

                                (104.0, 177.0, 123.0))

    # lewatkan blob pada face detection

    faceNet.setInput(blob)

    detections = faceNet.forward()

    

    # initialize list dari face, lokasi, dan prediksi

    faces = []

    locs = []

    preds_hardhat = []

    preds_mask = []

    

    # loop pada detections

    for i in range(0, detections.shape[2]):

        # extract confidence (probabilitas) pada detections

        confidence = detections[0, 0, i, 2]

        

        # filter weak detections yang kurang dari

        # confidence minimum

        if confidence > ap.confidence:

            # buat bounding box pada face yang terdeteksi

            box = detections[0, 0, i, 3:7]*np.array([w, h, w, h])

            (startX, startY, endX, endY) = box.astype("int")

            

            startX = startX - 15

            startY = startY - 20

            endX = endX + 15

            endY = endY + 20

            

            # pastikan bounding box berada dalam frame

            (startX, startY) = (max(0, startX), max(0, startY))

            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            

            # extract face ROI (Region of Interest)

            # convert dari BGR to RGB

            # ordering, resize (224x224) dan preprocess

            face = frame[startY:endY, startX:endX]

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            face = cv2.resize(face, (224, 224))

            face = img_to_array(face)

            face = preprocess_input(face)

            

            # append face dan bounding box pada list yg telah dibuat

            faces.append(face)

            locs.append((startX, startY, endX, endY))

    

    # run faces yang telah didapat ke face mask detector

    # buat prediksi saat ada face terdeteksi

    if len(faces)>0:

        # buat batch predictions untuk mendeteksi semua face secara bersamaan

        faces = np.array(faces, dtype="float32")

        preds_hardhat = hardHatNet.predict(faces, batch_size=32)

        preds_mask = maskNet.predict(faces, batch_size=32)

        

    # return 2-tuple, face locations dan nilai prediksi

    return (locs, preds_hardhat, preds_mask)


print("[INFO] loading video stream...")

video_path = "../input/hardhat-v2/Wearing a Mask with the Hard Hat Librarian.mp4"

vs = cv2.VideoCapture(video_path)



# ambil variable height, width, dan fps dari video

height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))

width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))

fps = int(vs.get(cv2.CAP_PROP_FPS))



# simpan video output

fourcc = cv2.VideoWriter_fourcc(*"MJPG")

output = cv2.VideoWriter("face-mask-detections.avi", 

                               fourcc, fps, (width, height), True)



# inisiasi elapsed time dan fps

t1 = cv2.getTickCount()

f = 0



# loop pada frame video

while True:

    

    (grabbed, frame) = vs.read()



    if not grabbed:

        print("[INFO] video process has been done...")

        break

    

    # deteksi face pada video

    # dan tentukan apakah memakai masker atau tidak

    (locs, preds_hardhat, preds_mask) = detect_and_predict_hardhat_mask(frame, faceNet, hardHatNet, maskNet)

    

    # loop pada lokasi face yang terdeteksi

    for (box, pred_hardhat, pred_mask) in zip(locs, preds_hardhat, preds_mask):

        # unpack bounding box dan nilai prediksi

        (startX, startY, endX, endY) = box

        (hardhat, withoutHardhat) = pred_hardhat

        (mask, withoutMask) = pred_mask

        

        # tentukan label yang digunakan -

        # dan warna untuk label tersebut (hijau, dan merah)

        if hardhat > withoutHardhat and mask > withoutMask:

            label = "Hardhat + Mask"

            color = (0, 255, 0) # green

        elif hardhat > withoutHardhat and mask < withoutMask:

            label = "Hardhat"

            color = (255, 0, 0) # blue

        elif hardhat < withoutHardhat and mask > withoutMask:

            label = "Mask"

            color = (0, 255, 255) # yellow

        elif hardhat < withoutHardhat and mask < withoutMask:

            label = "No-Hardhat-Mask"

            color = (0, 0, 255) # red

        

#         label = "Helmet" if harhat > withoutHardhat else "No-Helmet"

#         color = (0, 255, 0) if label == "Helmet" else (0,0,255)



#         # sertakan nilai probabilitas pada label

#         label = "{}: {:.2f}%".format(label, max(mask, withoutMask)*100)



        # tampilkan label dan bounding box pada frame

        cv2.putText(frame, label, (startX, startY-10),

                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    

    # increment fps

    f = f + 1

    

    # write video yang telah di proses ke output

    output.write(frame)

    

t2 = cv2.getTickCount()

time = (t2 - t1) / cv2.getTickFrequency()

fps = f / time



print("[INFO] elapsed time : {} second".format(time))

print("[INFO] fps :", fps)