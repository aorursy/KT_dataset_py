import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from tqdm import tqdm_notebook as tqdm



import glob

import cv2

import os



from colorama import Fore, Back, Style



# Setting color palette.

plt.rcdefaults()

plt.style.use('dark_background')



import warnings

warnings.filterwarnings("ignore")
# Assigning paths to variables

INPUT_PATH = os.path.join('..', 'input')

DATASET_PATH = os.path.join(INPUT_PATH, 'landmark-recognition-2020')

TRAIN_IMAGE_PATH = os.path.join(DATASET_PATH, 'train')

TEST_IMAGE_PATH = os.path.join(DATASET_PATH, 'test')

TRAIN_CSV_PATH = os.path.join(DATASET_PATH, 'train.csv')

SUBMISSION_CSV_PATH = os.path.join(DATASET_PATH, 'sample_submission.csv')
train = pd.read_csv(TRAIN_CSV_PATH)

print("training dataset has {} rows and {} columns".format(train.shape[0],train.shape[1]))



submission = pd.read_csv(SUBMISSION_CSV_PATH)

print("submission dataset has {} rows and {} columns \n".format(submission.shape[0],submission.shape[1]))
# understand folder structure

print(Fore.YELLOW + "If you want to access image a40d00dc4fcc3a10, you should traverse as shown below:\n",Style.RESET_ALL)



print(Fore.GREEN + f"Image name: {train['id'].iloc[9]}\n",Style.RESET_ALL)



print(Fore.BLUE + f"First folder to look inside: {train['id'][9][0]}")

print(Fore.BLUE + f"Second folder to look inside: {train['id'][9][1]}")

print(Fore.BLUE + f"Second folder to look inside: {train['id'][9][2]}",Style.RESET_ALL)
print(Fore.BLUE + f"{'---'*20} \n Mapping for Training Data \n {'---'*20}")

data_label_dict = {'image': [], 'target': []}

for i in tqdm(range(train.shape[0])):

    data_label_dict['image'].append(

        TRAIN_IMAGE_PATH + '/' +

        train['id'][i][0] + '/' + 

        train['id'][i][1]+ '/' +

        train['id'][i][2]+ '/' +

        train['id'][i] + ".jpg")

    data_label_dict['target'].append(

        train['landmark_id'][i])



#Convert to dataframe

train_pathlabel = pd.DataFrame(data_label_dict)

print(train_pathlabel.head())

    

print(Fore.BLUE + f"{'---'*20} \n Mapping for Test Data \n {'---'*20}",Style.RESET_ALL)

data_label_dict = {'image': []}

for i in tqdm(range(submission.shape[0])):

    data_label_dict['image'].append(

        TEST_IMAGE_PATH + '/' +

        submission['id'][i][0] + '/' + 

        submission['id'][i][1]+ '/' +

        submission['id'][i][2]+ '/' +

        submission['id'][i] + ".jpg")



test_pathlabel = pd.DataFrame(data_label_dict)

print(test_pathlabel.head())
# list of unique landmark ids

train.landmark_id.unique()
# count of unique landmark_ids

print("There are", train.landmark_id.nunique(), "landmarks in the training dataset")
# each class count-wise

train.landmark_id.value_counts()
files = train_pathlabel.image[:10]

print(Fore.BLUE + "Shape of files from training dataset",Style.RESET_ALL)

for i in range(10):

    im = cv2.imread(files[i])

    print(im.shape)





print("------------------------------------")    

print("------------------------------------")    

print("------------------------------------")    



files = test_pathlabel.image[:10]

print(Fore.BLUE + "Shape of files from test dataset",Style.RESET_ALL)

for i in range(10):

    im = cv2.imread(files[i])

    print(im.shape)
plt.figure(figsize = (12, 8))



sns.kdeplot(train['landmark_id'], color="yellow",shade=True)

plt.xlabel("LandMark IDs")

plt.ylabel("Probability Density")

plt.title('Class Distribution - Density plot')



plt.show()
fig = plt.figure(figsize = (12,8))



count = train.landmark_id.value_counts().sort_values(ascending=False)[:6]



sns.countplot(x=train.landmark_id,

             order = train.landmark_id.value_counts().sort_values(ascending=False).iloc[:6].index)



plt.xlabel("LandMark Id")

plt.ylabel("Frequency")

plt.title("Top 6 Classes in the Dataset")



plt.show()
top6 = train.landmark_id.value_counts().sort_values(ascending=False)[:6].index



images = []



for i in range(6):

    img=cv2.imread(train_pathlabel[train_pathlabel.target == top6[i]]['image'].values[1])   

    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    images.append(img)



f, ax = plt.subplots(3,2, figsize=(20,15))

for i, img in enumerate(images):        

        ax[i//2, i%2].imshow(img)

        ax[i//2, i%2].axis('off')

       
fig = plt.figure(figsize = (12,8))



count = train.landmark_id.value_counts().sort_values(ascending=False)[:50]



sns.countplot(x=train.landmark_id,

             order = train.landmark_id.value_counts().sort_values(ascending=False).iloc[:50].index)



plt.xticks(rotation = 90)



plt.xlabel("LandMark Id")

plt.ylabel("Frequency")

plt.title("Top 50 Classes in the Dataset")



plt.show()
top50 = train.landmark_id.value_counts().sort_values(ascending=False).index[:50]



images = []



for i in range(50):

    img=cv2.imread(train_pathlabel[train_pathlabel.target == top50[i]]['image'].values[1])   

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    images.append(img)



f, ax = plt.subplots(10,5, figsize=(20,15))

for i, img in enumerate(images):        

        ax[i//5, i%5].imshow(img)

        ax[i//5, i%5].axis('off')

       
fig = plt.figure(figsize = (12,8))



count = train.landmark_id.value_counts()[-6:]



sns.countplot(x=train.landmark_id,

             order = train_pathlabel.target.value_counts().iloc[-6:].index)



plt.xlabel("LandMark Id")

plt.ylabel("Frequency")

plt.title("Bottom 6 Classes in the Dataset")



plt.show()
bottom6 = train.landmark_id.value_counts()[-6:].index



images = []



for i in range(6):

    img=cv2.imread(train_pathlabel[train_pathlabel.target == bottom6[i]]['image'].values[1])   

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    images.append(img)



f, ax = plt.subplots(3,2, figsize=(20,15))

for i, img in enumerate(images):        

        ax[i//2, i%2].imshow(img)

        ax[i//2, i%2].axis('off')

       
top5 = train.landmark_id.value_counts().sort_values(ascending=False).index[:5]

for i in range(5):

    images = []      

    for j in range(12):

        img=cv2.imread(train_pathlabel[train_pathlabel.target == top5[i]]['image'].values[j])   

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        images.append(img)           

    f, ax = plt.subplots(3,4,figsize=(20,15))

    for k, img in enumerate(images):        

        ax[k//4, k%4].imshow(img)

        ax[k//4, k%4].axis('off')

plt.show()
files = train_pathlabel.image[11:23]



images = []



for i in range(11,23):    

    img=cv2.imread(files[i])   

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    images.append(img)

f, ax = plt.subplots(3,4, figsize=(20,15))

for i, img in enumerate(images):

        ax[i//4, i%4].imshow(img)

        ax[i//4, i%4].axis('off')
files = test_pathlabel.image[11:23]

images = []



for i in range(11,23):

    img=cv2.imread(files[i])   

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    images.append(img)

f, ax = plt.subplots(3,4, figsize=(20,15))

for i, img in enumerate(images):

        ax[i//4, i%4].imshow(img)

        ax[i//4, i%4].axis('off')
files = train_pathlabel.image[:4]



fig = plt.figure(figsize = (20,9))



for i in range(4):

    img=cv2.imread(files[i])   

    plt.subplot(2,2,i+1)

    plt.hist(img.ravel(), bins = 256,color = 'gold')

    

plt.suptitle("Histogram for Grayscale Images",fontsize = 25)    

plt.show()
fig = plt.figure(figsize = (20,9))



for i in range(4):

    img=cv2.imread(files[i])   

    plt.subplot(2,2,i+1)

    plt.hist(img.ravel(), bins = 256,color = 'magenta',cumulative = True)



plt.suptitle("Cumulative Histogram for Grayscale Images",fontsize = 25)    

plt.show()
fig = plt.figure(figsize = (20,9))



for i in range(4):

    img=cv2.imread(files[i])   

    plt.subplot(2,2,i+1)

    plt.hist(img.ravel(), bins = 8, color = "coral")



plt.suptitle("Cumulative Histogram for Grayscale Images - Bin Size = 8",fontsize = 25)    

plt.show()
fig = plt.figure(figsize = (20,9))



for i in range(4):

    img=cv2.imread(files[i])   

    plt.subplot(2,2,i+1)

    plt.hist(img.ravel(), bins = 256, color = 'orange', )

    plt.hist(img[:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.5)

    plt.hist(img[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)

    plt.hist(img[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)

    plt.xlabel('Intensity Value')

    plt.ylabel('Count')

    plt.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])



plt.suptitle("Color Histograms",fontsize = 25)    

plt.show()
import copy

import csv

import gc

import operator

import os

import pathlib

import shutil



import numpy as np

import PIL

import pydegensac

from scipy import spatial

import tensorflow as tf
# Dataset parameters:

INPUT_DIR = os.path.join('..', 'input')



DATASET_DIR = os.path.join(INPUT_DIR, 'landmark-recognition-2020')

TEST_IMAGE_DIR = os.path.join(DATASET_DIR, 'test')

TRAIN_IMAGE_DIR = os.path.join(DATASET_DIR, 'train')

TRAIN_LABELMAP_PATH = os.path.join(DATASET_DIR, 'train.csv')
# DEBUGGING PARAMS:

NUM_PUBLIC_TRAIN_IMAGES = 1580470 # Used to detect if in session or re-run.

MAX_NUM_EMBEDDINGS = -1  # Set to > 1 to subsample dataset while debugging.


# Retrieval & re-ranking parameters:

NUM_TO_RERANK = 3

TOP_K = 3 # Number of retrieved images used to make prediction for a test image.

# RANSAC parameters:

MAX_INLIER_SCORE = 35

MAX_REPROJECTION_ERROR = 7.0

MAX_RANSAC_ITERATIONS = 8500000

HOMOGRAPHY_CONFIDENCE = 0.99
# DELG model:

SAVED_MODEL_DIR = '../input/delg-saved-models/local_and_global'

DELG_MODEL = tf.saved_model.load(SAVED_MODEL_DIR)

DELG_IMAGE_SCALES_TENSOR = tf.convert_to_tensor([0.70710677, 1.0, 1.4142135])

DELG_SCORE_THRESHOLD_TENSOR = tf.constant(175.)

DELG_INPUT_TENSOR_NAMES = [

    'input_image:0', 'input_scales:0', 'input_abs_thres:0'

]
# Global feature extraction:

NUM_EMBEDDING_DIMENSIONS = 2048

GLOBAL_FEATURE_EXTRACTION_FN = DELG_MODEL.prune(DELG_INPUT_TENSOR_NAMES,

                                                ['global_descriptors:0'])



# Local feature extraction:

LOCAL_FEATURE_NUM_TENSOR = tf.constant(1000)

LOCAL_FEATURE_EXTRACTION_FN = DELG_MODEL.prune(

    DELG_INPUT_TENSOR_NAMES + ['input_max_feature_num:0'],

    ['boxes:0', 'features:0'])
def to_hex(image_id) -> str:

  return '{0:0{1}x}'.format(image_id, 16)





def get_image_path(subset, image_id):

  name = to_hex(image_id)

  return os.path.join(DATASET_DIR, subset, name[0], name[1], name[2],

                      '{}.jpg'.format(name))





def load_image_tensor(image_path):

  return tf.convert_to_tensor(

      np.array(PIL.Image.open(image_path).convert('RGB')))





def extract_global_features(image_root_dir):

  """Extracts embeddings for all the images in given `image_root_dir`."""



  image_paths = [x for x in pathlib.Path(image_root_dir).rglob('*.jpg')]



  num_embeddings = len(image_paths)

  if MAX_NUM_EMBEDDINGS > 0:

    num_embeddings = min(MAX_NUM_EMBEDDINGS, num_embeddings)



  ids = num_embeddings * [None]

  embeddings = np.empty((num_embeddings, NUM_EMBEDDING_DIMENSIONS))



  for i, image_path in enumerate(image_paths):

    if i >= num_embeddings:

      break



    ids[i] = int(image_path.name.split('.')[0], 16)

    image_tensor = load_image_tensor(image_path)

    features = GLOBAL_FEATURE_EXTRACTION_FN(image_tensor,

                                            DELG_IMAGE_SCALES_TENSOR,

                                            DELG_SCORE_THRESHOLD_TENSOR)

    embeddings[i, :] = tf.nn.l2_normalize(

        tf.reduce_sum(features[0], axis=0, name='sum_pooling'),

        axis=0,

        name='final_l2_normalization').numpy()



  return ids, embeddings





def extract_local_features(image_path):

  """Extracts local features for the given `image_path`."""



  image_tensor = load_image_tensor(image_path)



  features = LOCAL_FEATURE_EXTRACTION_FN(image_tensor, DELG_IMAGE_SCALES_TENSOR,

                                         DELG_SCORE_THRESHOLD_TENSOR,

                                         LOCAL_FEATURE_NUM_TENSOR)



  # Shape: (N, 2)

  keypoints = tf.divide(

      tf.add(

          tf.gather(features[0], [0, 1], axis=1),

          tf.gather(features[0], [2, 3], axis=1)), 2.0).numpy()



  # Shape: (N, 128)

  descriptors = tf.nn.l2_normalize(

      features[1], axis=1, name='l2_normalization').numpy()



  return keypoints, descriptors





def get_putative_matching_keypoints(test_keypoints,

                                    test_descriptors,

                                    train_keypoints,

                                    train_descriptors,

                                    max_distance=0.9):

  """Finds matches from `test_descriptors` to KD-tree of `train_descriptors`."""



  train_descriptor_tree = spatial.cKDTree(train_descriptors)

  _, matches = train_descriptor_tree.query(

      test_descriptors, distance_upper_bound=max_distance)



  test_kp_count = test_keypoints.shape[0]

  train_kp_count = train_keypoints.shape[0]



  test_matching_keypoints = np.array([

      test_keypoints[i,]

      for i in range(test_kp_count)

      if matches[i] != train_kp_count

  ])

  train_matching_keypoints = np.array([

      train_keypoints[matches[i],]

      for i in range(test_kp_count)

      if matches[i] != train_kp_count

  ])



  return test_matching_keypoints, train_matching_keypoints





def get_num_inliers(test_keypoints, test_descriptors, train_keypoints,

                    train_descriptors):

  """Returns the number of RANSAC inliers."""



  test_match_kp, train_match_kp = get_putative_matching_keypoints(

      test_keypoints, test_descriptors, train_keypoints, train_descriptors)



  if test_match_kp.shape[

      0] <= 4:  # Min keypoints supported by `pydegensac.findHomography()`

    return 0



  try:

    _, mask = pydegensac.findHomography(test_match_kp, train_match_kp,

                                        MAX_REPROJECTION_ERROR,

                                        HOMOGRAPHY_CONFIDENCE,

                                        MAX_RANSAC_ITERATIONS)

  except np.linalg.LinAlgError:  # When det(H)=0, can't invert matrix.

    return 0



  return int(copy.deepcopy(mask).astype(np.float32).sum())





def get_total_score(num_inliers, global_score):

  local_score = min(num_inliers, MAX_INLIER_SCORE) / MAX_INLIER_SCORE

  return local_score + global_score





def rescore_and_rerank_by_num_inliers(test_image_id,

                                      train_ids_labels_and_scores):

  """Returns rescored and sorted training images by local feature extraction."""



  test_image_path = get_image_path('test', test_image_id)

  test_keypoints, test_descriptors = extract_local_features(test_image_path)



  for i in range(len(train_ids_labels_and_scores)):

    train_image_id, label, global_score = train_ids_labels_and_scores[i]



    train_image_path = get_image_path('train', train_image_id)

    train_keypoints, train_descriptors = extract_local_features(

        train_image_path)



    num_inliers = get_num_inliers(test_keypoints, test_descriptors,

                                  train_keypoints, train_descriptors)

    total_score = get_total_score(num_inliers, global_score)

    train_ids_labels_and_scores[i] = (train_image_id, label, total_score)



  train_ids_labels_and_scores.sort(key=lambda x: x[2], reverse=True)



  return train_ids_labels_and_scores





def load_labelmap():

  with open(TRAIN_LABELMAP_PATH, mode='r') as csv_file:

    csv_reader = csv.DictReader(csv_file)

    labelmap = {row['id']: row['landmark_id'] for row in csv_reader}



  return labelmap





def get_prediction_map(test_ids, train_ids_labels_and_scores):

  """Makes dict from test ids and ranked training ids, labels, scores."""



  prediction_map = dict()



  for test_index, test_id in enumerate(test_ids):

    hex_test_id = to_hex(test_id)



    aggregate_scores = {}

    for _, label, score in train_ids_labels_and_scores[test_index][:TOP_K]:

      if label not in aggregate_scores:

        aggregate_scores[label] = 0

      aggregate_scores[label] += score



    label, score = max(aggregate_scores.items(), key=operator.itemgetter(1))



    prediction_map[hex_test_id] = {'score': score, 'class': label}



  return prediction_map





def get_predictions(labelmap):

  """Gets predictions using embedding similarity and local feature reranking."""



  test_ids, test_embeddings = extract_global_features(TEST_IMAGE_DIR)



  train_ids, train_embeddings = extract_global_features(TRAIN_IMAGE_DIR)



  train_ids_labels_and_scores = [None] * test_embeddings.shape[0]



  # Using (slow) for-loop, as distance matrix doesn't fit in memory.

  for test_index in range(test_embeddings.shape[0]):

    distances = spatial.distance.cdist(

        test_embeddings[np.newaxis, test_index, :], train_embeddings,

        'cosine')[0]

    partition = np.argpartition(distances, NUM_TO_RERANK)[:NUM_TO_RERANK]



    nearest = sorted([(train_ids[p], distances[p]) for p in partition],

                     key=lambda x: x[1])



    train_ids_labels_and_scores[test_index] = [

        (train_id, labelmap[to_hex(train_id)], 1. - cosine_distance)

        for train_id, cosine_distance in nearest

    ]



  del test_embeddings

  del train_embeddings

  del labelmap

  gc.collect()



  pre_verification_predictions = get_prediction_map(

      test_ids, train_ids_labels_and_scores)



#  return None, pre_verification_predictions



  for test_index, test_id in enumerate(test_ids):

    train_ids_labels_and_scores[test_index] = rescore_and_rerank_by_num_inliers(

        test_id, train_ids_labels_and_scores[test_index])



  post_verification_predictions = get_prediction_map(

      test_ids, train_ids_labels_and_scores)



  return pre_verification_predictions, post_verification_predictions





def save_submission_csv(predictions=None):

  """Saves optional `predictions` as submission.csv.



  The csv has columns {id, landmarks}. The landmarks column is a string

  containing the label and score for the id, separated by a ws delimeter.



  If `predictions` is `None` (default), submission.csv is copied from

  sample_submission.csv in `IMAGE_DIR`.



  Args:

    predictions: Optional dict of image ids to dicts with keys {class, score}.

  """



  if predictions is None:

    # Dummy submission!

    shutil.copyfile(

        os.path.join(DATASET_DIR, 'sample_submission.csv'), 'submission.csv')

    return



  with open('submission.csv', 'w') as submission_csv:

    csv_writer = csv.DictWriter(submission_csv, fieldnames=['id', 'landmarks'])

    csv_writer.writeheader()

    for image_id, prediction in predictions.items():

      label = prediction['class']

      score = prediction['score']

      csv_writer.writerow({'id': image_id, 'landmarks': f'{label} {score}'})
def main():

  labelmap = load_labelmap()

  num_training_images = len(labelmap.keys())

  print(f'Found {num_training_images} training images.')



  if num_training_images == NUM_PUBLIC_TRAIN_IMAGES:

    print(

        f'Found {NUM_PUBLIC_TRAIN_IMAGES} training images. Copying sample submission.'

    )

    save_submission_csv()

    return



  _, post_verification_predictions = get_predictions(labelmap)

  save_submission_csv(post_verification_predictions)



main()