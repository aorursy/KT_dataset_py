!pip install /kaggle/input/dfdcpackages/dlib-19.19.0-cp36-cp36m-linux_x86_64.whl
import cv2

import dlib

import glob

import json

import math

import pandas as pd

import os

import random

import statistics

import tqdm.notebook as tqdm



from IPython import display

from timeit import default_timer as timer

from datetime import timedelta



DATA_PREFIX = '/kaggle/input'

SKIP_FRAMES = 75

#DATA_PREFIX = '/work/dfdc-kaggle/input'



detector = dlib.cnn_face_detection_model_v1(os.path.join(DATA_PREFIX, 'dfdcpackages', 'mmod_human_face_detector.dat'))

sp = dlib.shape_predictor(os.path.join(DATA_PREFIX, 'dfdcpackages', 'shape_predictor_5_face_landmarks.dat'))

predictor = dlib.deep_fake_detection_model_v1(os.path.join(DATA_PREFIX, 'dfdcpackages', 'deepfake_detector.dnn'))





def align_face(frame, detection_sp):

    x_center = int((detection_sp.part(0).x + detection_sp.part(2).x + detection_sp.part(4).x) / 3)

    y_center = int((detection_sp.part(4).y + detection_sp.part(0).y + detection_sp.part(2).y) / 3)



    w = 2 * abs(detection_sp.part(0).x - detection_sp.part(2).x)

    h = w



    shape = frame.shape

    face_crop = frame[

        max(int(y_center - h), 0):min(int(y_center + h), shape[0]),

        max(int(x_center - w), 0):min(int(x_center + w), shape[1])

    ]

    return cv2.resize(face_crop, (150,150))





def align_face_dlib(frame, detection_sp):

    detections = dlib.full_object_detections()

    detections.append(detection_sp)

    return dlib.get_face_chips(frame, detections, size=150)[0]





def predict_fake(face):

    """Analyze face and return probability of FAKE i.e. 0 for REAL and 1 for FAKE"""

    return predictor.predict(face)[0]





def process_frame(frame):

    labels = []

    

    dets = detector(frame, 1)

    batch_faces = dlib.full_object_detections()

    for k,d in enumerate(dets):

        face = align_face_dlib(frame, sp(frame, d.rect))

        labels.append(predict_fake(face))

    

    return labels





def process_video(video_filename):

    """Process video and return probability of being FAKE, i.e. extremes are 0 for REAL and 1 for FAKE

    """

    frame_labels = []

    frames = []



    cap = cv2.VideoCapture(video_filename)

    frame_count = 0

    while cap.isOpened():

        ret = cap.grab()

        frame_count += 1

        if not ret:

            break



        if frame_count % SKIP_FRAMES:

            continue



        _, frame = cap.retrieve()

        frame_labels.extend(process_frame(frame))

    

    cap.release()

    fakeness = statistics.mean(frame_labels)

    # rescale values from 0...1 to 0.1...0.9 to avoid LogLoss penalties near extrems

    fakeness = 0.1 + fakeness * 0.8

    return fakeness





def single_log_loss(prediction, groundtruth):

    return groundtruth * math.log(prediction) + (1-groundtruth) * math.log(1-prediction)



def estimate_loss(predictions, all_correct=True):

    result = 0

    for p in predictions:

        if all_correct:

            result += single_log_loss(p, 1 if p > 0.5 else 0)

        else:

            result += single_log_loss(p, 0 if p > 0.5 else 1)

    return -result/len(predictions)





predictions = []

filenames = glob.glob(os.path.join(DATA_PREFIX, 'deepfake-detection-challenge/test_videos/*.mp4'))

sub = pd.read_csv(os.path.join(DATA_PREFIX, 'deepfake-detection-challenge/sample_submission.csv'))

sub = sub.set_index('filename', drop=False)



print('Initialize submission')

for filename in tqdm.tqdm(filenames):

    fn = filename.split('/')[-1]

    sub.loc[fn, 'label'] = 0.5





print('CUDA usage: {}'.format(dlib.DLIB_USE_CUDA))

for filename in tqdm.tqdm(filenames):

    fn = filename.split('/')[-1]

    sub.loc[fn, 'label'] = 0.5



    try:

        start = timer()

        prediction = process_video(filename)

        sub.loc[fn, 'label'] = prediction

        sub.to_csv('submission.csv', index=False)

        predictions.append(prediction)

        print('Processed video {}, label={}, time={}'.format(filename, prediction, timedelta(seconds=timer()-start)))

        print('Possible lost: best={}, worse={}'.format(estimate_loss(predictions), estimate_loss(predictions, False)))

    except Exception as error:

        print('Failed to process {}'.format(filename))

    

sub.to_csv('submission.csv', index=False)