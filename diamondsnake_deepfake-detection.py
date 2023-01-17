# import packages

!pip install /kaggle/input/mtcnnpackage/mtcnn-0.1.0-py3-none-any.whl

! tar xvf ../input/ffmpegpackage/ffmpeg-git-amd64-static.tar.xz

#!pip install /kaggle/input/ffmpegpackage/ffmpeg_python-0.2.0-py3-none-any.whl 

import numpy as np

import pandas as pd

import cv2

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import os

from tqdm import tqdm

from mtcnn.mtcnn import MTCNN

#import ffmpeg

import subprocess

import librosa

from pathlib import Path
#for dirname, _, filenames in os.walk('/kaggle/working/ffmpeg-git-20200305-amd64-static'):

#    for filename in filenames:

#        file.remove(os.path.join(dirname, filename))
path = '/kaggle/input/deepfake-detection-challenge/'

#os.listdir(path)
## load video metadata ##

video_metadata = pd.read_json(path+'train_sample_videos/metadata.json')

video_metadata = video_metadata.T
## functions for getting facial features ##



def calculate_features(result, timestamp):

  confidence = result[0]['confidence']

  box_size = result[0]['box'][2]*result[0]['box'][3]

  eye_width = result[0]['keypoints']['right_eye'][0] - result[0]['keypoints']['left_eye'][0] 

  eye_height = result[0]['keypoints']['right_eye'][1] - result[0]['keypoints']['left_eye'][1]

  features = pd.DataFrame(data = {'confidence': confidence, 'box_size': box_size, 'eye_width': eye_width, 'eye_height': eye_height}, index = [timestamp])

  return features



def get_face_detection_features(cap, detector):  

  face_detection_features = pd.DataFrame()

  frame_no=0

  while (cap.isOpened()):

    frame_no = frame_no+1

    if frame_no <= 300:

      if frame_no%30 == 0:

        ret, frame = cap.read()

        if ret == True:

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)

            result = detector.detect_faces(frame)

            if len(result) == 1:

              features = calculate_features(result, timestamp)

              face_detection_features = face_detection_features.append(features)

        else:

            break  

    else:

      break

  return face_detection_features

  

def get_face_features(cap, file_name, detector):

  face_detection_features = get_face_detection_features(cap, detector)

  if len(face_detection_features) > 0:

    agg = face_detection_features.aggregate({"confidence":['max', 'min', 'mean', 'std'], 

                                             "box_size":['max', 'min', 'mean', 'std'], 

                                             "eye_width":['max', 'min', 'mean', 'std'],  

                                             "eye_height":['max', 'min', 'mean', 'std']})  

    face_features = pd.DataFrame(data = {'confidence_ave': agg['confidence'][2], 

                                         'confidence_std': agg['confidence'][3], 

                                         'box_size_std': agg['box_size'][3], 

                                         'eye_width_std': agg['eye_width'][3], 

                                         'eye_height_std': agg['eye_height'][3]}, 

                                 index = [file_name])

    return face_features

  else:

    return []
## functions for getting audio features ##



output_format = 'wav'

output_dir = Path(f"{output_format}s")

Path(output_dir).mkdir(exist_ok=True, parents=True)

def create_wav(file, output_dir, output_format):

    command = f"../working/ffmpeg-git-20200305-amd64-static/ffmpeg -y -i {file} -ab 192000 -ac 2 -ar 44100 -vn {output_dir/'audio_file'}.{output_format}" # -y overwries file

    subprocess.call(command, shell=True)



def get_audio_features(audio, sampling_rate, file_name):

  #mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=20).T,axis=0)

  #chromagrams = np.mean(librosa.feature.chroma_stft(y=audio, sr=sampling_rate).T,axis=0)

  rmse = np.mean(librosa.feature.rms(y=audio)) 

  chroma_stft = np.mean(librosa.feature.chroma_stft(y=audio, sr=sampling_rate)) 

  spec_cent = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sampling_rate)) 

  spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sampling_rate)) 

  rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sampling_rate)) 

  zcr = np.mean(librosa.feature.zero_crossing_rate(audio)) 

  mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc = 20).T,axis=0) 

  audio_output = np.hstack((rmse, chroma_stft, spec_cent, spec_bw, rolloff, zcr, mfccs))

  if len(audio_output)==26:

    audio_features = pd.DataFrame([list(audio_output)], 

                                  columns = ['rmse', 'chroma_stft', 'spec_cent', 'spec_bw', 'rolloff', 'zcr',

                                            'mfccs_1', 'mfccs_2', 'mfccs_3', 'mfccs_4', 'mfccs_5', 'mfccs_6', 'mfccs_7', 'mfccs_8', 'mfccs_9', 'mfccs_10',

                                            'mfccs_11', 'mfccs_12', 'mfccs_13', 'mfccs_14', 'mfccs_15', 'mfccs_16', 'mfccs_17', 'mfccs_18', 'mfccs_19', 'mfccs_20'],

                                  index=[file_name])

  else:

    audio_features = []    

  return audio_features
## functions for getting file features ##



def get_file_features(cap, file_name):

  file_features = pd.DataFrame(data = {

    'width': cap.get(cv2.CAP_PROP_FRAME_WIDTH),

    'height': cap.get(cv2.CAP_PROP_FRAME_HEIGHT),

    'fps': cap.get(cv2.CAP_PROP_FPS),

    'no_frames': cap.get(cv2.CAP_PROP_FRAME_COUNT),

    'brightness': cap.get(cv2.CAP_PROP_BRIGHTNESS)

  }, index=[file_name])

  return file_features
## function pull together all features for capture ##



def get_all_features(path, file_name, detector):

    create_wav(path+file_name, output_dir, output_format)

    #audio, sampling_rate = librosa.load('/kaggle/working/wavs/'+file_name.replace('.mp4', '.wav'))

    audio, sampling_rate = librosa.load('/kaggle/working/wavs/audio_file.wav')

    capture = cv2.VideoCapture(path+file_name)

    if capture.isOpened():

      file_features = get_file_features(capture, file_name)

      face_features = get_face_features(capture, file_name, detector)

      audio_features = get_audio_features(audio, sampling_rate, file_name)

      all_features = file_features.join(video_metadata)

      if len(face_features) > 0:

        all_features = all_features.join(face_features)

      if len(audio_features) > 0:

        all_features = all_features.join(audio_features)

    else:

      print('ERROR: '+file_name+' would not open')

    return all_features
all_feature_names = ['rmse','chroma_stft','spec_cent','spec_bw','rolloff','zcr','mfccs_1','mfccs_2','mfccs_3','mfccs_4','mfccs_5','mfccs_6','mfccs_7','mfccs_8','mfccs_9','mfccs_1','mfccs_11','mfccs_12','mfccs_13','mfccs_14','mfccs_15','mfccs_16','mfccs_17','mfccs_18','mfccs_19','mfccs_20', 'box_size_std', 'confidence_ave','confidence_std','eye_height_std','eye_width_std']

feature_names = ['chroma_stft', 'mfccs_4', 'mfccs_5', 'mfccs_6', 'mfccs_7', 'mfccs_8', 'mfccs_9', 'mfccs_10', 'mfccs_11', 'mfccs_13', 'mfccs_14', 'mfccs_15', 'mfccs_16', 'mfccs_17', 'mfccs_18', 'mfccs_19', 'mfccs_20', 'confidence_ave', 'eye_height_std', 'eye_width_std']
## create audio files ##



#output_format = 'wav'

#output_dir = Path(f"{output_format}s")

#Path(output_dir).mkdir(exist_ok=True, parents=True)

#WAV_PATH = './wavs/'

#def create_wav(file, output_dir, output_format):

#    command = f"../working/ffmpeg-git-20200305-amd64-static/ffmpeg -i {file} -ab 192000 -ac 2 -ar 44100 -vn {output_dir/file[-14:-4]}.{output_format}"

#    subprocess.call(command, shell=True)

#    

#test_videos = sorted([x for x in os.listdir(path+'test_videos/') if x[-4:] == ".mp4"])

#for video_file in test_videos: 

#    file_name = os.path.join(path,video_file)

#    create_wav(file_name, output_dir, output_format)

#    name = video_file[:-4] + ".wav"
## process into feature dataframe ##



test_df = pd.DataFrame()

file_names = os.listdir(path+'test_videos/')

#file_names = file_names[0:3]

detector = MTCNN()

for file_name in tqdm(file_names):

  all_features = get_all_features(path+'test_videos/', file_name, detector)

  test_df = test_df.append(all_features)

test_df = test_df[all_feature_names].fillna(test_df.mean()) 

test_df.head()
# load Keras model traind on Databricks

from numpy import loadtxt

from keras.models import load_model

 

# load model

model = load_model('/kaggle/input/test-model-h5/deepfake_keras_model_01.h5')

# summarize model.

model.summary()
# run prediction

predictions = model.predict(test_df)[:,0]
# generate submission

submission = pd.DataFrame(data={'filename': file_names, 'label': predictions}).fillna(0.5)

submission = pd.DataFrame({'filename': submission['filename'], 'label': submission['label'].clip(0.4, 0.6)})

submission.to_csv("/kaggle/working/submission.csv", index=False)
submission.head()