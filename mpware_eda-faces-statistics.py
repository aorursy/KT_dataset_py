# Install some packages
!tar xvf /kaggle/input/ffmpeg-static-build/ffmpeg-git-amd64-static.tar.xz
!pip install mtcnn
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/working'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import glob, shutil

import timeit, os, gc

import subprocess as sp

from tqdm import tqdm

from collections import defaultdict

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import json

from IPython.display import HTML

from base64 import b64encode

import cv2
pd.set_option('display.max_colwidth', -1)

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 4000)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(font_scale=1.0)
HOME = "./"

FFMPEG = "/kaggle/working/ffmpeg-git-20191209-amd64-static"

FFMPEG_PATH = FFMPEG

DATA_FOLDER = "/kaggle/input/deepfake-detection-challenge"

TMP_FOLDER = HOME

DATA_FOLDER_TRAIN = DATA_FOLDER

VIDEOS_FOLDER_TRAIN = DATA_FOLDER_TRAIN + "/train_sample_videos"

IMAGES_FOLDER_TRAIN = TMP_FOLDER + "/images"

AUDIOS_FOLDER_TRAIN = TMP_FOLDER + "/audios"

EXTRACT_META = True # False

EXTRACT_CONTENT = True # False

EXTRACT_FACES = True # False

FRAME_RATE = 0.5 # Frame per second to extract (max is 30.0)
def run_command(*popenargs, **kwargs):

    closeNULL = 0

    try:

        from subprocess import DEVNULL

        closeNULL = 0

    except ImportError:

        import os

        DEVNULL = open(os.devnull, 'wb')

        closeNULL = 1



    process = sp.Popen(stdout=sp.PIPE, stderr=DEVNULL, *popenargs, **kwargs)

    output, unused_err = process.communicate()

    retcode = process.poll()



    if closeNULL:

        DEVNULL.close()



    if retcode:

        cmd = kwargs.get("args")

        if cmd is None:

            cmd = popenargs[0]

        error = sp.CalledProcessError(retcode, cmd)

        error.output = output

        raise error

        

    return output



def ffprobe(filename, options = ["-show_error", "-show_format", "-show_streams", "-show_programs", "-show_chapters", "-show_private_data"]):

    ret = {}

    command = [FFMPEG_PATH + "/ffprobe", "-v", "error", *options, "-print_format", "json", filename]

    ret = run_command(command)

    if ret:

        ret = json.loads(ret)

    return ret



# ffmpeg -i input.mov -r 0.25 output_%04d.png

def ffextract_frames(filename, output_folder, rate = 0.25):

    command = [FFMPEG_PATH + "/ffmpeg", "-i", filename, "-r", str(rate), "-y", output_folder + "/output_%04d.png"]

    ret = run_command(command)

    return ret



# ffmpeg -i input-video.mp4 output-audio.mp3

def ffextract_audio(filename, output_path):

    command = [FFMPEG_PATH + "/ffmpeg", "-i", filename, "-vn", "-ac", "1", "-acodec", "copy", "-y", output_path]

    ret = run_command(command)

    return ret
%%time

js = ffprobe(VIDEOS_FOLDER_TRAIN + "/"+ "bqdjzqhcft.mp4")

print(json.dumps(js, indent=4, sort_keys=True))
# Extract some meta-data

if EXTRACT_META == True:

    results = []

    subfolder = VIDEOS_FOLDER_TRAIN

    filepaths = glob.glob(subfolder + "/*.mp4")

    for filepath in tqdm(filepaths):

        js = ffprobe(filepath)

        if js:

            results.append(

                (js.get("format", {}).get("filename")[len(subfolder) + 1:],

                js.get("format", {}).get("format_long_name"),

                # Video 

                js.get("streams", [{}, {}])[0].get("codec_name"),

                js.get("streams", [{}, {}])[0].get("height"),

                js.get("streams", [{}, {}])[0].get("width"),

                js.get("streams", [{}, {}])[0].get("nb_frames"),

                js.get("streams", [{}, {}])[0].get("bit_rate"),

                js.get("streams", [{}, {}])[0].get("duration"),

                js.get("streams", [{}, {}])[0].get("start_time"),

                js.get("streams", [{}, {}])[0].get("avg_frame_rate"),

                # Audio

                js.get("streams", [{}, {}])[1].get("codec_name"),

                js.get("streams", [{}, {}])[1].get("channels"),

                js.get("streams", [{}, {}])[1].get("sample_rate"),

                js.get("streams", [{}, {}])[1].get("nb_frames"),

                js.get("streams", [{}, {}])[1].get("bit_rate"),

                js.get("streams", [{}, {}])[1].get("duration"),

                js.get("streams", [{}, {}])[1].get("start_time")),

            )



    meta_pd = pd.DataFrame(results, columns=["filename", "format", "video_codec_name", "video_height", "video_width",

                                            "video_nb_frames", "video_bit_rate", "video_duration", "video_start_time","video_fps",

                                            "audio_codec_name", "audio_channels", "audio_sample_rate", "audio_nb_frames",

                                            "audio_bit_rate", "audio_duration", "audio_start_time"])

    meta_pd["video_fps"] = meta_pd["video_fps"].apply(lambda x: float(x.split("/")[0])/float(x.split("/")[1]) if len(x.split("/")) == 2 else None)

    meta_pd["video_duration"] = meta_pd["video_duration"].astype(np.float32)

    meta_pd["video_bit_rate"] = meta_pd["video_bit_rate"].astype(np.float32)

    meta_pd["video_start_time"] = meta_pd["video_start_time"].astype(np.float32)

    meta_pd["video_nb_frames"] = meta_pd["video_nb_frames"].astype(np.float32)

    meta_pd["video_bit_rate"] = meta_pd["video_bit_rate"].astype(np.float32)

    meta_pd["audio_sample_rate"] = meta_pd["audio_sample_rate"].astype(np.float32)

    meta_pd["audio_nb_frames"] = meta_pd["audio_nb_frames"].astype(np.float32)

    meta_pd["audio_bit_rate"] = meta_pd["audio_bit_rate"].astype(np.float32)

    meta_pd["audio_duration"] = meta_pd["audio_duration"].astype(np.float32)

    meta_pd["audio_start_time"] = meta_pd["audio_start_time"].astype(np.float32)

    meta_pd.to_pickle(HOME + "videos_meta.pkl")

else:

    meta_pd = pd.read_pickle(HOME + "videos_meta.pkl")

meta_pd.head()
fig, ax = plt.subplots(1,6, figsize=(22, 3))

d = sns.distplot(meta_pd["video_fps"], ax=ax[0])

d = sns.distplot(meta_pd["video_duration"], ax=ax[1])

d = sns.distplot(meta_pd["video_width"], ax=ax[2])

d = sns.distplot(meta_pd["video_height"], ax=ax[3])

d = sns.distplot(meta_pd["video_nb_frames"], ax=ax[4])

d = sns.distplot(meta_pd["video_bit_rate"], ax=ax[5])
fig, ax = plt.subplots(1,6, figsize=(22, 3))

d = sns.distplot(meta_pd["audio_channels"], ax=ax[0])

d = sns.distplot(meta_pd["audio_sample_rate"], ax=ax[1])

d = sns.distplot(meta_pd["audio_nb_frames"], ax=ax[2])

d = sns.distplot(meta_pd["audio_bit_rate"], ax=ax[3])

d = sns.distplot(meta_pd["audio_duration"], ax=ax[4])

d = sns.distplot(meta_pd["audio_start_time"], ax=ax[5])
train_pd = pd.read_json(VIDEOS_FOLDER_TRAIN + "/metadata.json").T.reset_index().rename(columns={"index": "filename"})

train_pd.head()
train_pd = pd.merge(train_pd, meta_pd[["filename", "video_height", "video_width", "video_nb_frames", "video_bit_rate", "audio_nb_frames"]], on="filename", how="left")

train_pd["count"] = train_pd.groupby(["original"])["original"].transform('count')

# train_pd.to_pickle(HOME + "train_meta.pkl")

train_pd.head()
# Audio extract commented out to avoid disk full.

AUDIO_FORMAT = "aac" # "wav"

videos_folder = VIDEOS_FOLDER_TRAIN

images_folder_path = IMAGES_FOLDER_TRAIN

audios_folder_path = AUDIOS_FOLDER_TRAIN

if EXTRACT_CONTENT == True:

    # 1h20min for chunk#0 (11GB)

    # Extract some images + audio track

    for idx, row in tqdm(train_pd.iterrows(), total=meta_pd.shape[0]):

        try:

            video_path = videos_folder + "/" + row["filename"]

            images_path = images_folder_path + "/" + row["filename"][:-4]

            audio_path = audios_folder_path + "/" + row["filename"][:-4]

            # Extract images

            if not os.path.exists(images_path): os.makedirs(images_path)

            ret = ffextract_frames(video_path, images_path, rate = FRAME_RATE)

            # Extract audio

            if not os.path.exists(audio_path): os.makedirs(audio_path)

            # ret = ffextract_audio(video_path, audio_path + "/audio." + AUDIO_FORMAT)

        except:

            print("Cannot extract frames/audio for:" + row["filename"])
train_pd.tail()
# Preview Fake/Real (this one is obvious)

idx = 21 # 27 # 21 # 19 # 12 # 6

fake = train_pd["filename"][idx]

real = train_pd["original"][idx]

vid_width = train_pd["video_width"][idx]

vid_real = open(VIDEOS_FOLDER_TRAIN + "/" + real, 'rb').read()

data_url_real = "data:video/mp4;base64," + b64encode(vid_real).decode()

vid_fake = open(VIDEOS_FOLDER_TRAIN + "/" + fake, 'rb').read()

data_url_fake = "data:video/mp4;base64," + b64encode(vid_fake).decode()

HTML("""

<div style='width: 100%%; display: table;'>

    <div style='display: table-row'>

        <div style='width: %dpx; display: table-cell;'><b>Real</b>: %s<br/><video width=%d controls><source src="%s" type="video/mp4"></video></div>

        <div style='display: table-cell;'><b>Fake</b>: %s<br/><video width=%d controls><source src="%s" type="video/mp4"></video></div>

    </div>

</div>

""" % ( int(vid_width/3.2) + 10, 

       real, int(vid_width/3.2), data_url_real, 

       fake, int(vid_width/3.2), data_url_fake))
# OpenCV face detector

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")



def detect_face_cv2(img):

    # Move to grayscale

    gray_img = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY)

    face_locations = []

    face_rects = face_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)     

    for (x,y,w,h) in face_rects: 

        face_location = (x,y,w,h)

        face_locations.append((face_location, 1.0))

    return face_locations
# MTCNN face detector

from mtcnn import MTCNN

detector = MTCNN()



def detect_face_mtcnn(img):

    face_locations = []

    items = detector.detect_faces(img)

    for face in items:

        face_location = tuple(face.get('box'))

        face_confidence = float(face.get('confidence'))

        face_locations.append((face_location, face_confidence))

    return face_locations
# return ((x,y,w,h, confidence))

def extract_faces(files, source, detector=detect_face_cv2):

    results = []

    # for idx, file in tqdm(enumerate(files), total=len(files)):

    for idx, file in enumerate(files):

        try:

            img = cv2.cvtColor(cv2.imread(file, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)

            face_locations = detector(img)

            results.append((source, file[file.find("output_"):], face_locations, len(face_locations)))

        except:

            print("Cannot extract faces for image: %s" % file)

    return results
file = fake

dump_folder = images_folder_path + "/" + file[:-4]

files = glob.glob(dump_folder + "/*")

DETECTORS = {

    "cv2": detect_face_cv2,

    "mtcnn": detect_face_mtcnn

}

faces_pd = None

for key, value in DETECTORS.items():

    tmp_pd = pd.DataFrame(extract_faces(files, file, detector=value), columns=["filename", "image", "boxes_" + key , "faces_" + key])

    if faces_pd is None:

        faces_pd = tmp_pd

    else:

        faces_pd = pd.merge(faces_pd, tmp_pd, on=["filename", "image"], how="left")

faces_pd.head(12)
# Plot faces extracted images

def plot_faces_boxes(df, max_cols = 2, max_rows = 6, fsize=(24, 5), max_items=12):    

    idx = 0    

    for item_idx, item in df.iterrows():

        img = cv2.cvtColor(cv2.imread(IMAGES_FOLDER_TRAIN + "/" + item["filename"][:-4] +"/" + item["image"], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)    

        face_img = img #.copy()

        # grid subplots

        row = idx // max_cols

        col = idx % max_cols

        if col == 0: fig = plt.figure(figsize=fsize)

        ax = fig.add_subplot(1, max_cols, col + 1)

        ax.axis("off")

        # display image with boxes

        cols = [c for c in df.columns if "boxes" in c]

        for i, c in enumerate(cols, 0):

            face_locations = item[c]

            face_confidence = item[c]            

            if len(face_locations) > 0:

                for face_location in face_locations:        

                    ((x,y,w,h), confidence) = face_location

                    # face_img = face_img[y:y+h, x:x+w]

                    cv2.rectangle(face_img, (x, y), (x+w, y+h), (255,i*255,0), 8)

                    cv2.putText(face_img, '%.1f' % (confidence*100.0), (x+w, y+h), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255,i*255,0), 9, cv2.LINE_AA)

                ax.imshow(face_img)

            else:

                ax.imshow(img)

            ax.set_title("%s %s / %s - Faces: %d %s %s" % (item["label"] if "label" in df.columns else "", 

                                                           item["filename"], item["image"],

                                                           item["faces_mtcnn"] if "faces_mtcnn" in df.columns else len(face_locations),

                                                           item["faces_mtcnn_median"] if "faces_mtcnn_median" in df.columns else "",

                                                           item["faces"] if "faces" in df.columns else ""))

        if (col == max_cols -1): plt.show()

        idx = idx + 1

        if (max_items > 0 and idx >=max_items): break
# Compare face boxes detected by OpenCV and MTCNN

plot_faces_boxes(faces_pd)
def run_detector_on_video(videos_filename, verbose=False):

    if verbose == True: 

        print("Starting with batch of %d videos" % len(videos_filename))

    tmp_faces_pd = None

    for file in videos_filename:

        # Find out dump folder with images

        dump_folder = images_folder_path + "/" + file[:-4]

        # List files

        files = glob.glob(dump_folder + "/*")

        DETECTORS = {

            "mtcnn": detect_face_mtcnn

        }

        for key, value in DETECTORS.items():

            tmp_pd = pd.DataFrame(extract_faces(files, file, detector=value), columns=["filename", "image", "boxes_" + key , "faces_" + key])

            if tmp_faces_pd is None:

                tmp_faces_pd = tmp_pd

            else:

                tmp_faces_pd = pd.concat([tmp_faces_pd, tmp_pd], axis=0)

    return tmp_faces_pd
%%time

import multiprocessing

cpus = multiprocessing.cpu_count()

if EXTRACT_FACES == True:

    resultfutures = []

    results = []

    tasks = np.array_split(train_pd["filename"].unique(), 20)

    print("Tasks: %d" % len(tasks))

    with ThreadPoolExecutor(max_workers=cpus) as executor:

        resultfutures = tqdm(executor.map(run_detector_on_video, tasks), total=len(tasks))

    results = [x for x in resultfutures]

    executor.shutdown()

    # Gather results

    all_faces_pd = None

    for result in results:

        if all_faces_pd is None:

            all_faces_pd = result

        else:

            all_faces_pd = pd.concat([all_faces_pd, result], axis=0)

    all_faces_pd = all_faces_pd.reset_index(drop=True)

    all_faces_pd.to_pickle(HOME + "faces.pkl")

else:

    all_faces_pd = pd.read_pickle(HOME + "faces.pkl")

print(all_faces_pd.shape)
all_faces_pd.head()
# How many faces did we detect?

all_faces_pd.groupby(["faces_mtcnn"]).count()
all_faces_pd["faces_mtcnn_avg"] = all_faces_pd.groupby("filename")["faces_mtcnn"].transform(np.nanmean)

all_faces_pd["faces_mtcnn_median"] = all_faces_pd.groupby("filename")["faces_mtcnn"].transform(np.nanmedian)

all_faces_pd.head()
fig, ax = plt.subplots(1, 2, figsize=(22, 3))

d = sns.distplot(all_faces_pd["faces_mtcnn_avg"], kde=True, ax=ax[0])

d = sns.distplot(all_faces_pd["faces_mtcnn_median"], kde=False, ax=ax[1])
all_faces_pd.groupby(["faces_mtcnn_median"]).count()
# MTCNN is not perfect. It detects face inside face and in trees.

plot_faces_boxes(all_faces_pd[all_faces_pd["faces_mtcnn"] == 3], max_items=24)
# Let see some frames with 2 faces detected by MTCNN as majority (median).

plot_faces_boxes(all_faces_pd[(all_faces_pd["faces_mtcnn"] == 2) & (all_faces_pd["faces_mtcnn_median"] == 2)], max_items=32)
clean_faces_pd = pd.merge(all_faces_pd, train_pd, on="filename", how="left")

clean_faces_pd.head()
# Find out face width/height

def faces_max_item(boxes, idx1, idx2):

    ret = 0

    if len(boxes) > 0:

        ret = max(boxes, key=lambda item: item[idx1][idx2])[idx1][idx2]

    return ret



def faces_max_confidence(boxes):

    ret = 0

    if len(boxes) > 0:

        ret = max(boxes, key=lambda item: item[1])[1]

    return ret



def faces_min_confidence(boxes):

    ret = 0

    if len(boxes) > 0:

        ret = min(boxes, key=lambda item: item[1])[1]

    return ret



clean_faces_pd["faces_max_width"] = clean_faces_pd["boxes_mtcnn"].apply(lambda x: faces_max_item(x, 0, 2)) 

clean_faces_pd["faces_max_height"] = clean_faces_pd["boxes_mtcnn"].apply(lambda x: faces_max_item(x, 0, 3))

clean_faces_pd["faces_max_conf"] = clean_faces_pd["boxes_mtcnn"].apply(lambda x: faces_max_confidence(x))

clean_faces_pd["faces_min_conf"] = clean_faces_pd["boxes_mtcnn"].apply(lambda x: faces_min_confidence(x))
# If we train a CNN, we will have to define a width/height. 256x256 or 320x320 looks good.

print("Faces stats:")

print(clean_faces_pd[["faces_max_width", "faces_max_height", "faces_min_conf", "faces_max_conf"]].describe(percentiles=[0.01,0.05, 0.1,0.25,0.5,0.75,0.9,0.95,0.99]))

fig, ax = plt.subplots(1, 2, figsize=(22, 3))

d = sns.distplot(clean_faces_pd["faces_max_width"], kde=True, ax=ax[0])

d = sns.distplot(clean_faces_pd["faces_max_height"], kde=True, ax=ax[1])

plt.show()

fig, ax = plt.subplots(1, 2, figsize=(22, 3))

d = sns.distplot(clean_faces_pd["faces_min_conf"], kde=True, ax=ax[0])

d = sns.distplot(clean_faces_pd["faces_max_conf"], kde=True, ax=ax[1])

fig, ax = plt.subplots(figsize=(22, 3))

d = clean_faces_pd.plot(kind="scatter", x="faces_max_width", y="faces_max_conf", c="red", ax=ax, label="faces_max_width", alpha=0.5)

d = clean_faces_pd.plot(kind="scatter", x="faces_max_height", y="faces_max_conf", c="blue", ax=d,  label="faces_max_height", alpha=0.5)

d = plt.legend(loc="upper right")
clean_faces_pd.to_pickle(HOME + "faces.pkl")
# Clean temporary folders

shutil.rmtree(FFMPEG)

shutil.rmtree(IMAGES_FOLDER_TRAIN)

shutil.rmtree(AUDIOS_FOLDER_TRAIN)