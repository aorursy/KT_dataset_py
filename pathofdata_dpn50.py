!pip install -q /kaggle/input/pip-mtcnn/mtcnn-0.1.0-py3-none-any.whl
import numpy as np

import pandas as pd

import os

from time import time

import cv2

from tqdm.notebook import tqdm

from mtcnn import MTCNN

import matplotlib

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import tensorflow as tf

from multiprocess import Pool

from itertools import repeat

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import log_loss

import lightgbm as lgb
INPUT_PATH = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/'

metadata = pd.read_json(os.path.join(INPUT_PATH, 'metadata.json')).T

EXTRACT_NOISE = True

WINDOW = 224

FACE_CONFIDENCE = .8

FRAMES_PER_VIDEO = 1

NOISE_DEPTH = 1
def face_fft(img):

    img1 = img - cv2.GaussianBlur(img, (3,3), 0)

    imgs1 = np.sum(img1, axis=2)

    sf = np.stack([

         np.fft.fftshift(np.fft.fft2( imgs1 )),

         np.fft.fftshift(np.fft.fft2( img1[:,:,0] - img1[:,:,1] )),

         np.fft.fftshift(np.fft.fft2( img1[:,:,1] - img1[:,:,2] )),

         np.fft.fftshift(np.fft.fft2( img1[:,:,2] - img1[:,:,0] )) ], axis=-1)

    return np.abs(sf).astype(np.float16)



def pattern_ftt(img):

    if np.isnan(img).any():

        return np.nan

    else:

        img1 = img - cv2.GaussianBlur(img, (3,3), 0)

        imgs1 = np.sum(img1, axis=2)

        if NOISE_DEPTH == 1:

            sf = np.fft.fftshift(np.fft.fft2(imgs1))

            eps = np.max(sf) * 1e-2

            s1 = np.log(sf + eps) - np.log(eps) 

            sf = (s1 * 255 / np.max(s1))

            sf = np.abs(sf)

        else:

            sf = np.stack([

                 np.fft.fftshift(np.fft.fft2( imgs1 )),

                 np.fft.fftshift(np.fft.fft2( img1[:,:,0] - img1[:,:,1] )),

                 np.fft.fftshift(np.fft.fft2( img1[:,:,1] - img1[:,:,2] )),

                 np.fft.fftshift(np.fft.fft2( img1[:,:,2] - img1[:,:,0] ))],

                 axis=-1)

            sf = np.abs(sf)

            nchans = sf.shape[2]

            for c in range(nchans):

                eps = np.max(sf[:,:,c]) * 1e-2

                s1 = np.log(sf[:,:,c] + eps) - np.log(eps) 

                sf[:, :, c] = (s1 * 255 / np.max(s1))

        return sf.astype(np.float16)



def check_dims(img, W):

    d1, d2 = img.shape[:2]

    if d1 == W and d2 == W:

        return True

    else:

        return False

    

def extract_faces(fn, detector):

    KPS = 1

    video_path = os.path.join(INPUT_PATH, fn)

    try:

        vidcap = cv2.VideoCapture(video_path)

        fps = round(vidcap.get(cv2.CAP_PROP_FPS))

        hop = round(fps / KPS)

        retval, image = vidcap.read()

    except:

        return np.nan

    if not vidcap.isOpened():

        return np.nan

    count = 0

    i = 0

    W = WINDOW

    extracted_image = []

    while retval:

        if count % hop == 0:

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            detected_faces = detector.detect_faces(image)

            for this_face in range(len(detected_faces)):

                if detected_faces[this_face]['confidence'] > FACE_CONFIDENCE:

                    bbox = detected_faces[this_face]['box']

                    crop_center = [int(bbox[0] + .5*bbox[2]),

                                   int(bbox[1] + .5*bbox[3])]

                    fixed_xy = [int(crop_center[0] - .5*W),

                                int(crop_center[1] - .5*W)]

                    image = image[fixed_xy[1]:fixed_xy[1]+W,

                                  fixed_xy[0]:fixed_xy[0]+W, :]

                    image = image / 255.0

                    if check_dims(image, W) and image.max() > 0:

                        extracted_image.append(image)

                        i += 1

                        if i >= FRAMES_PER_VIDEO:

                            break

                        

        if i >= FRAMES_PER_VIDEO:

            # Break outer loop

            break

        try:

            retval, image = vidcap.read()

        except:

            return np.nan

        count += 1

        if count >= fps*60:

            break

    extracted_image = np.array(extracted_image)

    if (extracted_image.size == 0):

        return np.nan

    if FRAMES_PER_VIDEO == 1:

        extracted_image = extracted_image.reshape(W,W,3)

    return extracted_image

    

def plot_model_features(s):

    nchans = s.shape[2]

    nrows = (nchans + 3) // 4

    _, ax = plt.subplots(nrows, 4, figsize=(16, 4 * nrows))

    ax = ax.flatten()



    for c in range(nchans):

        eps = np.max(s[:,:,c]) * 1e-2

        s1 = np.log(s[:,:,c] + eps) - np.log(eps) 

        img = (s1 * 255 / np.max(s1)).astype(np.uint8)

        ax[c].imshow(cv2.equalizeHist(img))

        ax[c].grid(False)

        ax[c].xaxis.set_visible(False)

        ax[c].yaxis.set_visible(False)

    for ax1 in ax[nchans:]:

        ax1.axis('off')



def build_df(metadata_df, return_patterns=True):

    detector = MTCNN()

    df = pd.DataFrame({'filename': metadata.index.values,

                        'label': metadata.label.values,

                      })

    df['binary_label'] = 0

    df.loc[df.label == 'FAKE', 'binary_label'] = 1

    tqdm.pandas()

    patterns = list(tqdm(map(extract_faces, df.filename.values, repeat(detector)),

                         total=df.filename.shape[0], desc='Face Extraction'))

    if return_patterns:

        with Pool() as pool:

            patterns = list(tqdm(pool.imap(pattern_ftt, patterns),

                                 total=len(patterns), desc='Pattern Extraction'))

    df['pattern'] = patterns

    return df



def build_test_df(metadata_df, return_patterns=True):

    detector = MTCNN()

    df = pd.DataFrame({'filename': metadata_df.filename.values})

    tqdm.pandas()

    patterns = list(tqdm(map(extract_faces, df.filename.values, repeat(detector)),

                         total=df.filename.shape[0], desc='Face Extraction'))

    if return_patterns:

        with Pool() as pool:

            patterns = list(tqdm(pool.imap(pattern_ftt, patterns),

                                 total=len(patterns), desc='Pattern Extraction'))

    df['pattern'] = patterns

    return df



def plot_frames(fn, label):

    KPS = 1

    video_path = os.path.join(INPUT_PATH, fn)

    vidcap = cv2.VideoCapture(video_path)

    fps = round(vidcap.get(cv2.CAP_PROP_FPS))

    hop = round(fps / KPS)

    detector = MTCNN()

    retval, image = vidcap.read()

    fig, axes = plt.subplots(1, 1, figsize=(8, 5))

    count = 0

    i = 0

    W = WINDOW

    all_frames = []

    while retval:

        if count % hop == 0:

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            detected_faces = detector.detect_faces(image)

            axes.imshow(image)

            if len(detected_faces):

                for this_face in range(len(detected_faces)):

                    if detected_faces[this_face]['confidence'] > FACE_CONFIDENCE:

                        bbox = detected_faces[0]['box']

                        crop_center = [int(bbox[0] + .5*bbox[2]),

                                       int(bbox[1] + .5*bbox[3])]

                        fixed_xy = [int(crop_center[0] - .5*W),

                                    int(crop_center[1] - .5*W), W, W]

                        image = image[fixed_xy[1]:fixed_xy[1]+W,

                                      fixed_xy[0]:fixed_xy[0]+W, :]

                        image = image / 255.0

                        if check_dims(image, W) and image.max() > 0:

                            image = face_fft(image)

                            all_frames.append(image)

                            rect = patches.Rectangle((fixed_xy[0],

                                                      fixed_xy[1]),

                                                     fixed_xy[2],

                                                     fixed_xy[3],

                                                     fill=False, linewidth=3.)

                            axes.add_patch(rect)

                            axes.set_title(f'Filename: {fn} - Label: {label}',

                                             color='black')

                            axes.xaxis.set_visible(False)

                            axes.yaxis.set_visible(False)

                            i += 1

                            if i >= FRAMES_PER_VIDEO:

                                break

                            

        if i >= FRAMES_PER_VIDEO:

            break

        retval, image = vidcap.read()

        count += 1

        if count >= fps*60:

            break

    all_frames = np.array(all_frames)

    if (all_frames.size == 0):

        print(f'No Pattern found')

    else:

        all_frames = np.mean(all_frames, axis=0)

        plot_model_features(all_frames)

    plt.tight_layout()

    plt.show()
FILES = metadata.index

LABELS = metadata.label

rnd_file = np.random.randint(0, FILES.shape[0], 1)

plot_frames(FILES[rnd_file[0]], LABELS[rnd_file[0]])
noise_pattern_df = build_df(metadata, return_patterns=EXTRACT_NOISE)

s0 = noise_pattern_df.shape[0]

noise_pattern_df = noise_pattern_df[~noise_pattern_df.pattern.apply(

    lambda x: pd.isna(np.ravel(x)).any())].copy()

print(f'{s0-noise_pattern_df.shape[0]} videos have been dropped.')



X = np.stack(noise_pattern_df['pattern'].values) / 255.0

X = X.reshape(X.shape[0], -1)

y = noise_pattern_df['binary_label'].values



pos = np.where(y==1)[0].shape[0]

neg = np.where(y==0)[0].shape[0]

print(f'Number of Fake videos: {pos} - Number of Real videos: {neg}')

print('Training model ...')



# Parameters were tuned offline with Bayesian optimization

best_params = {

        'num_leaves': int(12.373597389205074),

        'max_bin': 63,

        'min_data_in_leaf': int(15.182532994098365),

        'learning_rate': 0.04770828591430053,

        'min_sum_hessian_in_leaf': 0.00266281112712854,

        'bagging_fraction': 0.28258080526211365,

        'bagging_freq': int(5.0310417355831465),

        'feature_fraction': 0.458867976391893,

        'lambda_l1': 1.4680707418683974,

        'lambda_l2': 1.4388766929317436,

        'min_gain_to_split': 0.21162811600005904,

        'max_depth': int(3.232403494443565),

        'save_binary': True, 

        'seed': 1337,

        'feature_fraction_seed': 1337,

        'bagging_seed': 1337,

        'drop_seed': 1337,

        'data_random_seed': 1337,

        'objective': 'binary',

        'boosting_type': 'gbdt',

        'verbose': 1,

        'metric': 'binary_logloss',

        'is_unbalance': True,

        'boost_from_average': False,   



    }



skf = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=0)



predictions = np.zeros(X.shape[0])

for i, (train_index, test_index) in enumerate(skf.split(X, y)):

        X_train_fold, X_test_fold = X[train_index], X[test_index]

        y_train_fold, y_test_fold = y[train_index], y[test_index]

    

        xg_train = lgb.Dataset(X_train_fold,

                               label=y_train_fold)

        xg_valid = lgb.Dataset(X_test_fold,

                               y_test_fold)

        num_round = 5000

        clf = lgb.train(best_params, xg_train,

                        num_round, valid_sets=[xg_valid],

                        verbose_eval=0, early_stopping_rounds = 50)

        predictions[test_index] = clf.predict(X_test_fold,

                                              num_iteration=clf.best_iteration)

        score = log_loss(y, predictions)

print('Model Training complete.')

print(f'CV score: {score:.6f}')
pred_df = pd.DataFrame({'predictions': predictions,

                        'labels': y})

y_pred = []

y_pred.append(pred_df[pred_df.labels == 1].predictions.values)

y_pred.append(pred_df[pred_df.labels == 0].predictions.values)

y_label = pred_df.groupby('labels').size()

with plt.style.context('seaborn'):

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].hist(y_pred, stacked=True, bins=10)

    ax[0].set_title('Distribution of predictions')

    _ = y_label.plot(kind='bar', ax=ax[1],

                     color=['#55a868', '#4c72b0'],

                     title='Distribution of Labels')

    plt.tight_layout()

    plt.show()
with plt.style.context('seaborn'):

    plt.scatter(range(len(predictions)), predictions, c=y,

                cmap=matplotlib.colors.ListedColormap(['#55a868', '#4c72b0']))

    plt.plot([], 'o', c='#55a868', label='Real Videos')

    plt.plot([], 'o', c='#4c72b0', label='Fake Videos')

    plt.axhline(.5, linestyle=':', c='tab:red')

    plt.xlabel('Sample index')

    plt.ylabel('Prediction %')

    plt.title('Train set Predictions')

    plt.tight_layout()

    plt.legend(loc='lower left')

    plt.show()
# Extract noise patterns from validation dataset

sub_path = '/kaggle/input/deepfake-detection-challenge'

sub_metadata = pd.read_csv(os.path.join(sub_path, 'sample_submission.csv'))

INPUT_PATH = os.path.join(sub_path, 'test_videos')

submission_data = build_test_df(sub_metadata,

                                return_patterns=EXTRACT_NOISE)

submission_subset = submission_data[~submission_data.pattern.apply(

    lambda x: pd.isna(np.ravel(x)).any())].copy()

print(f'{sub_metadata.shape[0]-submission_subset.shape[0]} videos have been dropped.')



# Predict the videos

test_data = np.stack(submission_subset['pattern'].values) / 255.0

test_data = test_data.reshape(test_data.shape[0], -1)

predictions = clf.predict(test_data, num_iteration=clf.best_iteration)



# Submit the predictions

submission_subset['label'] = predictions

submission_subset = submission_subset[['filename', 'label']]

submissions = pd.DataFrame({'filename': submission_data.filename.values})

submissions = submissions.merge(submission_subset, how='left', on='filename')

submissions.label.fillna(.5, inplace=True)

submissions.to_csv('submission.csv', index=False)
with plt.style.context('seaborn'):

    plt.scatter(range(len(submissions.label.values)),

                submissions.label.values,

                )

    plt.axhline(.5, linestyle=':', c='tab:red')

    plt.xlabel('Sample index')

    plt.ylabel('Prediction %')

    plt.title('Sample Predictions')

    plt.tight_layout()

    plt.legend(loc='lower left')

    plt.show()