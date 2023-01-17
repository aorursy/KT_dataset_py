!pip install moviepy --quiet

import os

import json

from glob import glob

import numpy as np

from skimage.io import imsave

from skimage.transform import resize

from tqdm import tqdm_notebook

from moviepy.editor import VideoFileClip

from concurrent.futures import ProcessPoolExecutor

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
INPUT_DIR = '/kaggle/input/deepfake-detection-challenge'

EXTRACT_EVERY = 5 # will save every 5th frame

PROCESSED_DATA_DIR = '/kaggle/working/processed_data'

H, W = 224, 224
if not os.path.exists(PROCESSED_DATA_DIR):

    os.mkdir(PROCESSED_DATA_DIR)

    os.mkdir(os.path.join(PROCESSED_DATA_DIR, 'real'))

    os.mkdir(os.path.join(PROCESSED_DATA_DIR, 'fake'))



with open(INPUT_DIR + '/train_sample_videos/metadata.json', 'r') as f:

    metadata = json.load(f)
def process_video_file(video_filepath):

    with VideoFileClip(filename=video_filepath, audio=False) as clip:

        fname = video_filepath.split('/')[-1]

        label = metadata[fname]['label'].lower()

        output_dir = os.path.join(PROCESSED_DATA_DIR, label, fname.split('.')[0])

        os.mkdir(output_dir)

        for i, frame in enumerate(clip.iter_frames()):

            if (i+1) % EXTRACT_EVERY == 0:

                output_path = os.path.join(output_dir, '{}.jpeg'.format(i+1))

                resized_frame = np.uint8(resize(frame, [H, W]) * 255)

                imsave(output_path, resized_frame, check_contrast=False)
video_paths = glob(INPUT_DIR + '/train_sample_videos/*.mp4')[:5]

with ProcessPoolExecutor(max_workers=4) as ex:

    processed = 0

    for _ in tqdm_notebook(ex.map(process_video_file, video_paths), total=len(video_paths)):

        print('Done processing {} videos'.format(processed))

        processed += 1
glob("/kaggle/working/processed_data/real/*/")
glob("/kaggle/working/processed_data/fake/*/")
glob("/kaggle/working/processed_data/fake/*/*")[:20]