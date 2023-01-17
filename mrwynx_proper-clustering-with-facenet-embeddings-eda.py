# Install facenet-pytorch
!pip install /kaggle/input/facenet-pytorch-vggface2/facenet_pytorch-2.2.7-py3-none-any.whl

from facenet_pytorch.models.inception_resnet_v1 import get_torch_home
torch_home = get_torch_home()

# Copy model checkpoints to torch cache so they are loaded automatically by the package
!mkdir -p $torch_home/checkpoints/
!cp /kaggle/input/facenet-pytorch-vggface2/20180402-114759-vggface2-logits.pth $torch_home/checkpoints/vggface2_DG3kwML46X.pt
!cp /kaggle/input/facenet-pytorch-vggface2/20180402-114759-vggface2-features.pth $torch_home/checkpoints/vggface2_G5aNV2VSMn.pt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
from facenet_pytorch import MTCNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import PIL.Image
from tqdm.notebook import tqdm
from time import time
import shutil

import warnings
warnings.filterwarnings("ignore")

# https://www.kaggle.com/hmendonca/kaggle-pytorch-utility-script
from kaggle_pytorch_utility_script import *

seed_everything(42)
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
# See github.com/timesler/facenet-pytorch:
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Running on device: {device}')
from torchvision.transforms import ToTensor

# load image processor
tf_img = lambda i: ToTensor()(i).unsqueeze(0)

# Load facial recognition model
resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()

# set up embedding calculator for resnet
embeddings = lambda input_data: resnet(input_data)
DEBUG = False
DEBUG_COUNT = 1000 # Limit on original training and new videos 
REPROCESS = True # Reprocess source videos from scratch, or use embeddings already extracted
SUBMIT = False # Test run or for submission

# data locations
TMP_DIR = '/kaggle/working/datasets' # for data created during the notebook run
INPUT_FOLDER = '/kaggle/input'
TEST_DIR = '/kaggle/input/deepfake-detection-challenge/test_videos/'
TRAIN_DIR = '/kaggle/input/deepfake-detection-faces*.mp4'
AUGMENT_DIR = '/kaggle/input/additional-deepfake-training-data/'

# data processing
AUGMENT_VIDEO_COUNT = 400
N_FRAMES = None
SCALE = 0.25
NUM_SAMPLES = 1
NUM_FRAMES = 4

# train/val/test setup
TRAIN_SPLIT_FRACTION = .80
class FaceExtractor:
    def __init__(self, detector, n_frames=None, resize=None):
        """
        Parameters:
            n_frames {int} -- Total number of frames to load. These will be evenly spaced
                throughout the video. If not specified (i.e., None), all frames will be loaded.
                (default: {None})
            resize {float} -- Fraction by which to resize frames from original prior to face
                detection. A value less than 1 results in downsampling and a value greater than
                1 result in upsampling. (default: {None})
        """

        self.detector = detector
        self.n_frames = n_frames
        self.resize = resize
    
    def __call__(self, filename, save_dir):
        """Load frames from an MP4 video, detect faces and save the results.

        Parameters:
            filename {str} -- Path to video.
            save_dir {str} -- The directory where results are saved.
        """

        # Create video reader and find length
        v_cap = cv2.VideoCapture(filename)
        frame_count = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        v_len = 100 if frame_count > 100 else frame_count

        # Pick 'n_frames' evenly spaced frames to sample
        if self.n_frames is None:
            sample = np.arange(0, v_len)
        else:
            sample = np.linspace(0, v_len - 1, self.n_frames).astype(int)

        # Loop through frames
        for j in range(v_len):
            success = v_cap.grab()
            if j in sample:
                # Load frame
                success, frame = v_cap.retrieve()
                if not success:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = PIL.Image.fromarray(frame)
                
                # Resize frame to desired size
                if self.resize is not None:
                    frame = frame.resize([int(d * self.resize) for d in frame.size])

                save_path = os.path.join(save_dir, f'{j}.png')

                self.detector([frame], save_path=save_path)

        v_cap.release()
face_detector = MTCNN(margin=14, keep_all=True, factor=0.5, device=device).eval()
# Define face extractor
face_extractor = FaceExtractor(detector=face_detector, n_frames=N_FRAMES, resize=SCALE)
def extract_faces_from_videos(extract_video_paths, face_extractor, save_dir):
    """Run face extractor module and save results to TMP_DIR"""
    with torch.no_grad():
        for path in tqdm(extract_video_paths):
            file_name = path.split('/')[-1]

            save_dir = os.path.join(save_dir, file_name.split(".")[0])

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Detect all faces appear in the video and save them.
            face_extractor(path, save_dir)
            
            # reset save_dir
            save_dir = '/'.join(save_dir.split('/')[:-1])
!mkdir -p $TMP_DIR
augment_videos = glob.glob(os.path.join(AUGMENT_DIR, '*.mp4'))

if DEBUG:
    augment_videos = augment_videos[0:AUGMENT_VIDEO_COUNT]
else:
    augment_videos = augment_videos[0:AUGMENT_VIDEO_COUNT]

print(len(augment_videos))
augment_extract_dir = TMP_DIR + '/augment-video-extracts'
extract_faces_from_videos(augment_videos, face_extractor, augment_extract_dir)
print(augment_videos[0:5])
len(augment_videos)
train_videos = []
metadata_dfs = []

augment_videos = glob.glob(augment_extract_dir + '/*')
train_data_folders = glob.glob('/kaggle/input/deepfake-detection-face*')

for tdf in train_data_folders:
    videos = os.listdir(tdf)
    metadata_dfs.append(pd.read_csv(tdf + '/metadata.csv'))
    for v in videos:
        if v == 'metadata.csv':
            continue
        path = os.path.join(tdf, v)
        train_videos.append(path)

        
train_videos = train_videos + augment_videos

if DEBUG:
    train_videos = train_videos[0:DEBUG_COUNT] + augment_videos[0:DEBUG_COUNT]
    
print(f"Train samples: {len(train_videos)}")
metadata = pd.concat(metadata_dfs)
metadata['video_source'] = 'competition'

adf_count = len(augment_videos)

augment_df = pd.DataFrame.from_dict({'filename': [av.split('/')[-1] + '.mp4' for av in augment_videos],
                                      'split': ['train' for i in range(adf_count)],
                                      'original': [np.nan for i in range(adf_count)],
                                      'label': ['REAL' for i in range(adf_count)],
                                      'video_source': ['augment' for i in range(adf_count)]})

metadata = pd.concat([metadata, augment_df])
fig, ax = plt.subplots(figsize=(15,7))
metadata.groupby('original').count().sort_values(by='filename', ascending=False).plot(ax=ax)
metadata.head()
def get_indices(fname):
    """Adds leading zeros to input filename"""
    fname_list = fname.split('.')
    index = fname_list[0].split('_')[0] # don't care whether frames have > 1 face within
    if not index:
        print('issue')
    return (index.zfill(3), fname)
def sort_frames(train_videos):
    result = []
    for v in train_videos:
        video_files = os.listdir(v)
        video_indices = [get_indices(f) for f in video_files]
        video_indices = sorted(video_indices, key = lambda x: x[0])
        result.append({'name': v, 'indices': video_indices})
            
    return result
originals = metadata.groupby('filename')['original'].min().to_dict()
added_tracker = {}
filtered_train_videos = []
for tv in train_videos:
    video_name = tv.split('/')[-1] + '.mp4'
    #original = originals[video_name]
    filtered_train_videos.append(tv)
    
#     if original is np.nan:
#         filtered_train_videos.append(tv)
#         continue
        
#     elif original not in added_tracker:
#         added_tracker[original] = ''
#         filtered_train_videos.append(tv)
#     else:
#         continue
        
print(len(filtered_train_videos))
sorted_frames = sort_frames(filtered_train_videos)
def draw_sample(video, samp_idx, sample_length):
    samp_rng = list(range(samp_idx.astype(int), samp_idx.astype(int) + sample_length))
    sampled = [video['indices'][i][1] for i in samp_rng]
    return sampled
def sample_frames(sorted_frames, metadata, n_sample_sets=2, sample_length=4):
    """Randomly sample n_sample_sets of sample_length from the recorded face
    frames"""
    result = []
    for video in sorted_frames:
        num_frames = len(video['indices'])
        sample_indices = np.random.choice(range(num_frames - (sample_length + 1)), n_sample_sets)
        for samp_idx in sample_indices:
            selected_frames = draw_sample(video, samp_idx, sample_length)
            names = [video['name'] for sf in selected_frames]
            idxs = [samp_idx for sf in selected_frames]
            result.append({'name': names, 'samp_idx': idxs, 'selected_frames': selected_frames})  
    return result
np.random.seed(1)
sampled_frames = sample_frames(sorted_frames, metadata, n_sample_sets=NUM_SAMPLES, sample_length=NUM_FRAMES)
sampled_frames[0:2]
def get_embedding(path_to_data):
    """Uses pretrained model to extract embeddings from a frame at path_to_data"""
    t = tf_img(PIL.Image.open(path_to_data)).to(device)
    return embeddings(t).squeeze().cpu().tolist()
def process_sample(video):    
    result = []
    with torch.no_grad():
        video_name, samp_img_list = video['name'][0], video['selected_frames']
        for img in samp_img_list:
            path_to_data = '/'.join([video_name, img])
            emb = get_embedding(path_to_data)
            result.append(emb)
        video['embeddings'] = result
    return video
def convert_to_dataframe(processed_frames, metadata):
    """Conver embeddings dataset to a pandas dataframe"""
    dfs = []
    for pf in processed_frames:
        dfs.append(pd.DataFrame(pf))
    df =  pd.concat(dfs, ignore_index=True) # make sure to ignore the index to recreate from 0 to n!
    # explode embedding vectors to columns
    num_embeddings = len(df['embeddings'].values[0])
    emb_df = pd.DataFrame(df['embeddings'].values.tolist(), columns=['emb_{}'.format(i) for i in range(num_embeddings)])
    df = df.join(emb_df)
    df['file_path'] = df['name']
    df['name'] = df['name'].apply(lambda x: x.split('/')[-1] + '.mp4')
    return df.merge(metadata[['filename', 'split', 'label', 'original']], 
              left_on='name', 
              right_on='filename', 
              how='inner').drop(['filename', 'embeddings'], axis=1)
def output_dataframe(dataset, path, fname):
    """Check if path exists, create if it doesn't, then output file"""
    file_path = '/'.join([path, fname])
    if not os.path.exists(path):
        os.mkdir(path)
        dataset.to_csv(file_path)
    
    else:
        dataset.to_csv(file_path)
OUTPUT_PATH = '../working/processed'
OUTPUT_FILE = 'embeddings_data.csv'

if REPROCESS:
    processed_frames = [result for result in map(process_sample, sampled_frames)]
    df = convert_to_dataframe(processed_frames, metadata)
    print('shape of dataframe created: ', str(df.shape))
    output_dataframe(df, OUTPUT_PATH, OUTPUT_FILE)

else:
    df = pd.read_csv('/'.join([OUTPUT_PATH, OUTPUT_FILE]))
df['label_bin'] = df['label'].apply(lambda x: 1 if x == 'REAL' else 0)
df.shape
df.head()
import random
filenames = df.groupby('name')['name'].min().values
fname_to_original = df.groupby('name')['original'].min().to_dict()
n = len(filenames)
idx = int(n * TRAIN_SPLIT_FRACTION)

random.shuffle(filenames) #shuffle randomly to mix original and augment datasets

train_fnames = {fn: '0' for fn in filenames[0:idx]}
test_fnames = {fn: '0' for fn in filenames[idx:]}

print(len(train_fnames) + len(test_fnames) == n)
print('{} training videos || {} testing videos'.format(str(len(train_fnames)), str(len(test_fnames))))
train_X, train_Y = [], [] 
test_X, test_Y = [], []
test_names = []

for pf in processed_frames:
    video_name = pf['name'][0]
    lookup_name = video_name.split('/')[-1] + '.mp4'
    y_value = df.loc[df['name'] == lookup_name, 'label_bin'].values[0] # extract 1 for REAL and 0 for FAKE 
    
    # grab embeddings and normalize across faces
    embs = pf['embeddings']
    if lookup_name in train_fnames:
        train_X.append(embs)
        train_Y.append(y_value)
        
    elif fname_to_original[lookup_name] not in train_fnames: # make sure originals haven't been seen in testing data
        test_X.append(embs)
        test_Y.append(y_value)
        test_names.append(video_name)
        
    else:
        continue
train_X_len = len(train_X)
test_X_len = len(test_X)
train_X_arr = np.reshape(train_X, (train_X_len, NUM_FRAMES, 512))
train_X_mn = train_X_arr.mean(0)
train_X_std = train_X_arr.std(0)
train_X_arr = (train_X_arr - train_X_mn) / train_X_std

test_X_arr = np.reshape(test_X, (test_X_len, NUM_FRAMES, 512))
test_X_mn = test_X_arr.mean(0)
test_X_std = test_X_arr.std(0)
test_X_arr = (test_X_arr - test_X_mn) / test_X_std
train_Y = np.array(train_Y)

print('{} training video || {} testing videos'.format(str(train_X_len), str(test_X_len)))
#np.save('/kaggle/working/models/train_X_mn', train_X_mn)
#np.save('/kaggle/working/models/train_X_std', train_X_std)
from collections import Counter

# summarize class distribution
counter = Counter(train_Y)

# indexes of reals and fakes
real_idx = np.where(train_Y == 1)[0]
fake_idx = np.where(train_Y == 0)[0]

# upsample reals to match the fakes
# add in Xception net kernel idea with readily available faces
real_samples = np.random.choice(real_idx, size=counter[0], replace=True)

train_X_samp = np.concatenate([train_X_arr[real_samples], train_X_arr[fake_idx]], axis=0)
train_Y_samp = np.concatenate([train_Y[real_samples], train_Y[fake_idx]], axis=0)

print(train_X_samp.shape)
print(train_Y_samp.shape)
print(np.unique(train_Y_samp, return_counts=True))
batch_size = 64
epochs = 25
batch_input_shape = (None, NUM_FRAMES, 512)
model = Sequential()
model.add(LSTM(256, activation='relu', return_sequences=True, batch_input_shape=batch_input_shape, stateful=False))
model.add(Dropout(0.5))
model.add(LSTM(128, activation='relu', return_sequences=False, batch_input_shape=batch_input_shape, stateful=False))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_X_samp, train_Y_samp, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=True)
#model.reset_states()
os.mkdir('/kaggle/working/models')
model.save('/kaggle/working/models/all-augment-400.h5')
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot


# predict probabilities for test set
yhat_probs = model.predict(test_X_arr, verbose=0)
# predict crisp classes for test set
yhat_classes = model.predict_classes(test_X_arr, verbose=0)
# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]
 
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(test_Y, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(test_Y, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(test_Y, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(test_Y, yhat_classes)
print('F1 score: %f' % f1)
 
# kappa
kappa = cohen_kappa_score(test_Y, yhat_classes)
print('Cohens kappa: %f' % kappa)
# ROC AUC
auc = roc_auc_score(test_Y, yhat_probs)
print('ROC AUC: %f' % auc)

lr_precision, lr_recall, _ = precision_recall_curve(test_Y, yhat_probs)
no_skill = len(yhat_classes[yhat_classes==1]) / len(yhat_classes)

pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
pyplot.plot(lr_recall, lr_precision, marker='.', label='NN Classifier')
# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_Y, yhat_classes)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['fake', 'real']); ax.yaxis.set_ticklabels(['fake', 'real']);
plt.hist(yhat_probs)
test_names[0:3]
#from itertools import compress
#error_idx = (test_Y != yhat_classes)
#list(compress(test_names, error_idx))

eval_dict = {'tp': [], 'fp': [], 'tn': [], 'fn': []}
def make_eval_dict(eval_dict, test_Y, yhat_classes):
    for i in range(len(test_Y)):
        eval_array = [test_Y[i], yhat_classes[i]]
        fname = test_names[i]
        if eval_array == [1, 0]:
            eval_dict['fn'].append(fname)
        
        elif eval_array == [0, 0]:
            eval_dict['tn'].append(fname)
        
        elif eval_array == [0, 1]:
            eval_dict['fp'].append(fname)
        
        else:
            eval_dict['tp'].append(fname)
    return eval_dict

eval_dict = make_eval_dict(eval_dict, test_Y, yhat_classes)
# quick check
for k, v in eval_dict.items():
    print(k, len(v))
df.head()
from IPython.core.display import display, Image

def show_errors(eval_dict, error_type, limit=20):
    """Get examples of videos where model makes mistakes"""
    lookup_videos = eval_dict[error_type]
    for lv in lookup_videos[:limit]:
        print(lv)
        frames = df.loc[df['file_path'] == lv, 'selected_frames'].values
        real_fake = df.loc[df['file_path'] == lv, 'label'].values[0]
        print(real_fake)
        for frame in frames:
            frame_path = lv + '/' + frame
            display(Image(filename=frame_path))
            
show_errors(eval_dict, 'fp')
