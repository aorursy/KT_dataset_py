!pip install /kaggle/input/facenet-pytorch-vggface2/facenet_pytorch-2.2.7-py3-none-any.whl
import os

import numpy as np

import pandas as pd

import tensorflow as tf

import keras

import cv2 as cv

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array

from tensorflow.keras.models import Sequential, load_model

from facenet_pytorch import MTCNN

import time
test_videos = '../input/deepfake-detection-challenge/test_videos/'

test_movie_files = [test_videos + x for x in sorted(os.listdir(test_videos))]
model = load_model('../input/models/MobilenetV2_third_-13-0.1018.h5')

model.summary()
detector = MTCNN(margin=50, keep_all=False, post_process=False, device='cuda:0',thresholds=[.9,.9,.9])

vid_num = 0

scores=[]

filenames = []

startTime = time.time()/60



for vid in test_movie_files:

    predict_all=[]

    count=0

    file_name_mp4 = vid.split('/')[-1]

    file_name = file_name_mp4.split('.')[0]

    v_cap = cv.VideoCapture(vid)    

    v_len = int(v_cap.get(cv.CAP_PROP_FRAME_COUNT))

    for frm in range(v_len):    

        success = v_cap.grab()

        if frm % 7 == 0:

            success, frame = v_cap.retrieve()

            if not success:

                continue

            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            frame = detector(frame)

            if frame is not None:

                frame = np.transpose(frame, (1, 2, 0))

                frame = np.array(cv.resize(np.array(frame),(160 ,160)))

                frame = (frame.flatten() / 255.0).reshape(-1, 160, 160, 3)

                count=count+frame.shape[0]

                predict = model.predict(frame)

                predict=1-predict[0][0]

                predict_all.append(predict)

            else:

                continue

        else:

            continue



    print('성공 :', file_name_mp4)

    if (count>11):

        predict_all.sort()

        scores.append((sum(predict_all[5:-5])/(count-10)))

    else:

        scores.append(0.5)

    filenames.append(file_name_mp4)



v_cap.release()

endTime = time.time()/60 - startTime

print('소요시간 :',endTime)
predict_df = pd.DataFrame({'filename':filenames, 'label':scores}) 
predict_df.loc[predict_df['label']==1,'label'] = 0.99

predict_df.loc[predict_df['label']==0,'label'] = 0.01

predict_df
predict_df.to_csv('submission.csv', index=False)

print('치맥♥')