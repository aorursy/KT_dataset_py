!pip install '/kaggle/input/dfdc-packages/dlib-19.19.0/dlib-19.19.0' 

!pip install '/kaggle/input/dfdc-packages/face_recognition_models-0.3.0/face_recognition_models-0.3.0'

!pip install '/kaggle/input/dfdc-packages/face_recognition-1.2.3-py2.py3-none-any.whl'
#########################################################################################################################

## Starting Script

#########################################################################################################################



## Imports

import time

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import face_recognition

import dlib

import os

import cv2

import multiprocessing as mp

import gc ## important for controlling memory usage

from sklearn.model_selection import train_test_split



## Confirm GPU compilation

print(dlib.DLIB_USE_CUDA)

print(dlib.cuda.get_num_devices())



## Procedures

def read_images(vf):

    cap = cv2.VideoCapture(vf)

    print("## Reading frames")

    frames = []

    while cap.isOpened():

        success, image = cap.read()



        if not success:

            break



        image = image[:, :, ::-1]

        frames.append(image)



    return frames





if __name__ == "__main__":

    data_folder = "/kaggle/input/deepfake-detection-challenge"

    train_folder = os.path.join(data_folder, "train_sample_videos")

    meta_file = os.path.join(train_folder, "metadata.json")

    

    print("## Reading meta data")

    meta_df = pd.read_json(meta_file).T

    

    ## Take only 2 files

    file_list = meta_df.index.tolist()[:2]



    for f in file_list:

        print("## Starting on {}".format(f))

        fp = os.path.join(train_folder, f)



        t0 = time.time()

        frames = read_images(fp)

        print("## Reading {:d} frames took: {:f}".format(len(frames), time.time() - t0))



        gc.collect()

        bsz = 30

        t0 = time.time()

        all_face_locations = []

        for i in range(0,len(frames), bsz):

            batch_frames = frames[i:i+bsz]

            batch_of_face_locations = face_recognition.batch_face_locations(batch_frames, number_of_times_to_upsample=0)

            all_face_locations.extend(batch_of_face_locations)

        print("## GPU Getting faces frames {:d} from frames took: {:f}".format(len(all_face_locations), time.time() - t0))

        del all_face_locations

        del frames

        gc.collect()

        print("## Done {}".format(f))