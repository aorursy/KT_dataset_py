from __future__ import absolute_import, division, print_function, unicode_literals



import tensorflow as tf



import pandas as pd

import numpy as np

import shutil, os

import subprocess

from pathlib import Path

import glob

import cv2

from PIL import Image

from numpy import asarray

import matplotlib.pyplot as plt

import datetime
pathCascadeCL = '/kaggle/input/haar-face-detector/haarcascade_frontalface_alt.xml'

face_cascade = cv2.CascadeClassifier(pathCascadeCL)

finalCalculatedThresh = 12.0
def detect_face(img,required_size=(256,256)):

    color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detected_faces_raw = face_cascade.detectMultiScale(gray,1.3,5)

    face_images = []

    for (x,y,w,h) in detected_faces_raw:

        face_boundary = color[y:y+h, x:x+w]

        face_image = Image.fromarray(face_boundary)

        face_image = face_image.resize(required_size)

        face_array = asarray(face_image)

        face_images.append(face_array)

    return face_images
def detectVideo(test_path,file):

    video = test_path + file

    capture = cv2.VideoCapture(video)

    v_len = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = capture.get(cv2.CAP_PROP_FPS)

    frame_count = int(v_len/fps)*5

    frame_idxs = np.linspace(0,v_len,frame_count, endpoint=False, dtype=np.int)

    imgs=[]

    i=0

    for frame_idx in range(int(v_len)):

        ret = capture.grab()

        if not ret: 

            pass

        if frame_idx >= frame_idxs[i]:

            ret, frame = capture.retrieve()

            if not ret or frame is None:

                pass

            else:

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                try:

                    faces=detect_face(frame)

                except Exception as err:

                    print(err)

                    continue

                for face in faces:

                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                    imgs.append(face)

            i += 1

            if i >= len(frame_idxs):

                break

    return imgs
! tar xvf /kaggle/input/ffmpegstatic/ffmpeg-git-amd64-static.tar.xz
command = f"/kaggle/working/ffmpeg-git-20191209-amd64-static/ffmpeg -loglevel error -hide_banner -nostats"

subprocess.call(command, shell=True)
path = '/kaggle/input/deepfake-detection-challenge/test_videos/'

test_files = os.listdir(path)

finalvalues = []

audioThresh = 44300

for fileToSave in test_files:

    diffs = []

    videoFile = fileToSave

    output_dir = Path(f"/kaggle/working/wavs/")

    Path(output_dir).mkdir(exist_ok=True, parents=True)

    value = 0.0

    try:

        fileToread = '/kaggle/input/deepfake-detection-challenge/test_videos/'+ fileToSave

        cap = cv2.VideoCapture(fileToread)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fps = cap.get(cv2.CAP_PROP_FPS)

        duration = float(frame_count) / float(fps)

        duration = int(duration)

        fileToSave = fileToSave[:-4]

        file = '/kaggle/input/deepfake-detection-challenge/test_videos/'+ fileToSave +'.mp4'

        command = f"/kaggle/working/ffmpeg-git-20191209-amd64-static/ffmpeg -i {file} -ab 192000 -ac 2 -ar 44100 -vn {output_dir/fileToSave}.wav"

        subprocess.call(command, shell=True)

        pathvideo = '/kaggle/working/wavs/' + fileToSave + '.wav'

        raw_audio = tf.io.read_file(pathvideo)

        test = tf.audio.decode_wav(raw_audio, desired_channels=-1, desired_samples=-1, name=None)

        audio = test.audio

        shutil.rmtree('/kaggle/working/wavs')

        if(int(audio.shape[0]/duration)>=audioThresh):

            value = 1.0

        else:

            images = detectVideo(path,videoFile)

            imagescropped = [x[60:220,20:220] for x in images]

            for image in imagescropped:

                val1 = np.sum(image)

                dst1 = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)

                val2 = np.sum(dst1)

                if val1>val2:

                    diffs.append(val1-val2)

                else:

                    diffs.append(val2-val1)

            if((min(diffs)/max(diffs)*100)<11.999):

               value = 1.0

    except Exception as e:

        value = 0.5

        pass

    finalvalues.append(value)
shutil.rmtree('/kaggle/working/ffmpeg-git-20191209-amd64-static')
submission = pd.DataFrame({"filename":test_files, "label": finalvalues})

submission.to_csv("submission.csv", index = False)