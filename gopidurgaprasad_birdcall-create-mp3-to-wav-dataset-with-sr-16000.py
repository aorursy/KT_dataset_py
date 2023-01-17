import os
import cv2
import skimage.io
from tqdm.notebook import tqdm
import zipfile
import pandas as pd
import numpy as np
import shutil

from pydub import AudioSegment
from joblib import Parallel, delayed
!mkdir -p /root/.kaggle/
!cp ../input/mykaggleapi/kaggle.json /root/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json
!mkdir -p /tmp/birdcall_dataset
!ls /tmp
data = '''{
  "title": "birdsong_recognition_wav_16000",
  "id": "gopidurgaprasad/birdsong-recognition-wav-16000",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}
'''
text_file = open("/tmp/birdcall_dataset/dataset-metadata.json", 'w+')
n = text_file.write(data)
text_file.close()
TRAIN_CSV = "../input/birdsong-recognition/train.csv"
ROOT_PATH = "../input/birdsong-recognition/train_audio"
OUTPUT_DIR = "/tmp/birdcall_dataset/train_audio_wav_16000"
os.mkdir(OUTPUT_DIR)
def save_fn(bird_code, filename):
    
    path = f"{ROOT_PATH}/{bird_code}/{filename}"
    save_path = f"{OUTPUT_DIR}/{bird_code}"
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    if os.path.exists(path):
        try:
            sound = AudioSegment.from_mp3(path)
            sound = sound.set_frame_rate(16000)
            sound.export(f"{save_path}/{filename[:-4]}.wav", format="wav")
        except:
            print(path)
train = pd.read_csv(TRAIN_CSV)
bird_code_list = list(train.ebird_code.values)
filename_list = list(train.filename.values)
Parallel(n_jobs=8, backend="multiprocessing")(
    delayed(save_fn)(bird_code, filename) for bird_code, filename in tqdm(zip(bird_code_list, filename_list), total=len(bird_code_list))
)
!sleep 10
!zip -r "/tmp/birdcall_dataset/train_audio_wav_16000.zip" "/tmp/birdcall_dataset/train_audio_wav_16000/"
!ls -l /tmp/birdcall_dataset/
!kaggle datasets create -p /tmp/birdcall_dataset
!rm -rf /tmp/birdcall_dataset

