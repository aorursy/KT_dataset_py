# import csv

# import os

# import pathlib

# import pandas as pd

# import numpy as np

# from sklearn import preprocessing

# from sklearn.model_selection import StratifiedKFold

# from tqdm import tqdm

# import tensorflow as tf

# import PIL



# def _bytes_feature(value):

#     """Returns a bytes_list from a string / byte."""

#     if isinstance(value, type(tf.constant(0))):

#         value = value.numpy() # BytesList won't unpack a string from an EagerTensor.

#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



# def _float_feature(value):

#     """Returns a float_list from a float / double."""

#     return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))



# def _int64_feature(value):

#     """Returns an int64_list from a bool / enum / int / uint."""

#     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



# def serialize_example(feature0, feature1):

#     feature = {

#         'embeddings': _bytes_feature(feature0),

#         'target': _int64_feature(feature1)

#     }

#     example_proto = tf.train.Example(features = tf.train.Features(feature = feature))

#     return example_proto.SerializeToString()



# DATASET_DIR = os.path.join(INPUT_DIR, 'landmark-recognition-2020')

# TRAIN_IMAGE_DIR = os.path.join(DATASET_DIR, 'train')

# TRAIN = os.path.join(INPUT_DIR, 'landmark-train-encoded/train_encoded.csv')



# # DELG model:

# SAVED_MODEL_DIR = '../input/delg-saved-models/local_and_global'

# DELG_MODEL = tf.saved_model.load(SAVED_MODEL_DIR)

# DELG_IMAGE_SCALES_TENSOR = tf.convert_to_tensor([0.70710677, 1.0, 1.4142135])

# DELG_SCORE_THRESHOLD_TENSOR = tf.constant(175.)

# DELG_INPUT_TENSOR_NAMES = [

#     'input_image:0', 'input_scales:0', 'input_abs_thres:0'

# ]



# # Global feature extraction:

# NUM_EMBEDDING_DIMENSIONS = 2048

# GLOBAL_FEATURE_EXTRACTION_FN = DELG_MODEL.prune(DELG_INPUT_TENSOR_NAMES,

#                                                 ['global_descriptors:0'])



# def to_hex(image_id) -> str:

#     return '{0:0{1}x}'.format(image_id, 16)



# # load an image to a tf tensor

# def load_image_tensor(image_path):

#     return tf.convert_to_tensor(

#         np.array(PIL.Image.open(image_path).convert('RGB')))



# def get_tf_records(record = 0):

#     df = pd.read_csv(TRAIN)

#     # get image paths

#     image_paths = [x for x in pathlib.Path(TRAIN_IMAGE_DIR).rglob('*.jpg')]

#     # get only one group, this is a slow process so we need to make 15 different sessions

#     df = df[df['group'] == record]

#     # reset index 

#     df.reset_index(drop = True, inplace = True)

#     # get a list of ids

#     ids_list = list(df['id'].unique())

#     # write tf records

#     with tf.io.TFRecordWriter('train_{}.tfrec'.format(record)) as writer:

#         for image_path in tqdm(image_paths):

#             image_id = int(image_path.name.split('.')[0], 16)

#             image_id = to_hex(image_id)

#             if image_id in ids_list:

#                 # target

#                 target = df[df['id'] == image_id]['landmark_id_encode']

#                 image_tensor = load_image_tensor(image_path)

#                 features = GLOBAL_FEATURE_EXTRACTION_FN(image_tensor,

#                                                         DELG_IMAGE_SCALES_TENSOR,

#                                                         DELG_SCORE_THRESHOLD_TENSOR)

#                 embedding = tf.nn.l2_normalize(

#                     tf.reduce_sum(features[0], axis = 0, name = 'sum_pooling'),

#                     axis = 0,

#                     name = 'final_l2_normalization').numpy()

#                 # transform numpy array to bytes

#                 embedding = embedding.tobytes()

#                 example = serialize_example(

#                     embedding,

#                     target.values[0]

#                 )

#                 writer.write(example)

                

# get_tf_records(record = 0)
# import re

# import os

# import numpy

# import pandas as pd

# import numpy as np

# import random

# import math

# from sklearn import metrics

# from sklearn.model_selection import KFold

# import tensorflow as tf

# from tensorflow.keras import backend as K

# !pip install gcsfs



# # For tf.dataset

# AUTO = tf.data.experimental.AUTOTUNE



# # Data access

# GCS_PATH = 'gs://kds-2048c0f014df07f1ef48ea726c68e902f57cdd083fbf9f4bbb46c2b2'

# # Dictionary acces

# DICT_PATH = 'gs://kds-13db2280942fc707e90594cc5c29d055a3fc72594eff7c85cc3ab006/train_encoded.csv'



# # Configurations

# EPOCHS = 10

# BATCH_SIZE = 32

# # Seed for deterministic results

# SEED = 123

# # Learning rate

# LR = 0.001



# # Training filenames directory

# TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train*.tfrec')



# # Seed everything for deterministic results

# def seed_everything(seed):

#     random.seed(seed)

#     np.random.seed(seed)

#     os.environ['PYTHONHASHSEED'] = str(seed)

#     tf.random.set_seed(seed)



# # Parse tf records, also decode bytes embeddings and one hot target vector

# def read_tfrecord(example):

#     tfrec_format = {

#         'embeddings': tf.io.FixedLenFeature([], tf.string),

#         'target': tf.io.FixedLenFeature([], tf.int64)

#     }

#     # Parse the data

#     example = tf.io.parse_single_example(example, tfrec_format)

#     # Decode raw bytes data to float

#     embeddings = tf.io.decode_raw(input_bytes = example['embeddings'], out_type = float)

#     # One hot encode target label, we extracted the amount of classes on another kernel

#     target = tf.cast(tf.one_hot(example['target'], 27756), tf.int32)

#     embeddings = tf.cast(embeddings, tf.float32)

#     return embeddings, target



# # Load dataset for training and evaluating model

# def load_dataset(filenames, ordered = False):

#     # Read from TFRecords. For optimal performance, reading from multiple files at once and

#     # Diregarding data order. Order does not matter since we will be shuffling the data anyway

#     ignore_order = tf.data.Options()

#     if not ordered:

#         # Disable order, increase speed

#         ignore_order.experimental_deterministic = False

#     # Automatically interleaves reads from multiple files

#     dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads = AUTO)

#     # Use data as soon as it streams in, rather than in its original order

#     dataset = dataset.with_options(ignore_order)

#     # Returns a dataset of (image, label) pairs

#     dataset = dataset.map(read_tfrecord, num_parallel_calls = AUTO)

#     return dataset



# # Training pipeline

# def get_training_dataset(filenames, ordered = False):

#     dataset = load_dataset(filenames, ordered = ordered)

#     # The training dataset must repeat for several epochs

#     dataset = dataset.repeat()

#     dataset = dataset.shuffle(2048)

#     dataset = dataset.batch(BATCH_SIZE)

#     # Prefetch next batch while trianing

#     dataset = dataset.prefetch(AUTO)

#     return dataset



# # Evaluation pipeline

# def get_validation_dataset(filenames, ordered = True):

#     dataset = load_dataset(filenames, ordered = ordered)

#     dataset = dataset.batch(BATCH_SIZE)

#     # Prefetch next batch while evaluating

#     dataset = dataset.prefetch(AUTO)

#     return dataset



# # Count the number of observations with the tabular csv

# def count_data_items(filenames):

#     records = [int(re.compile(r"_([0-9]*)\.").search(filename).group(1)) for filename in filenames]

#     df = pd.read_csv(DICT_PATH)

#     n = df[df['group'].isin(records)].shape[0]

#     return n



# NUM_TOTAL_OBSERVATIONS = count_data_items(TRAINING_FILENAMES)

# print(f'Dataset has {NUM_TOTAL_OBSERVATIONS} observations')



# def gap_vector(pred, conf, true, return_x = False):

#     '''

#     Compute Global Average Precision (aka micro AP), the metric for the

#     Google Landmark Recognition competition. 

#     This function takes predictions, labels and confidence scores as vectors.

#     In both predictions and ground-truth, use None/np.nan for "no label".



#     Args:

#         pred: vector of integer-coded predictions

#         conf: vector of probability or confidence scores for pred

#         true: vector of integer-coded labels for ground truth

#         return_x: also return the data frame used in the calculation



#     Returns:

#         GAP score

#     '''

#     x = pd.DataFrame({'pred': pred, 'conf': conf, 'true': true})

#     x.sort_values('conf', ascending=False, inplace=True, na_position='last')

#     x['correct'] = (x.true == x.pred).astype(int)

#     x['prec_k'] = x.correct.cumsum() / (np.arange(len(x)) + 1)

#     x['term'] = x.prec_k * x.correct

#     gap = x.term.sum() / x.true.count()

#     if return_x:

#         return gap, x

#     else:

#         return gap





# # Simple baseline with 27756 target classes and a softmax output

# def get_model():

  

#     inp = tf.keras.layers.Input(shape = (2048))

#     x = tf.keras.layers.Dense(1024, activation = 'relu')(inp)

#     x = tf.keras.layers.BatchNormalization()(x)

#     x = tf.keras.layers.Dropout(0.2)(x)

#     x = tf.keras.layers.Dense(512, activation = 'relu')(inp)

#     x = tf.keras.layers.BatchNormalization()(x)

#     x = tf.keras.layers.Dropout(0.2)(x)

#     output = tf.keras.layers.Dense(27756, activation = 'softmax', name = 'out')(x)



#     model = tf.keras.models.Model(inputs = [inp], outputs = [output])



#     opt = tf.keras.optimizers.Adam(learning_rate = LR)



#     model.compile(

#       optimizer = opt,

#       loss = [tf.keras.losses.CategoricalCrossentropy()],

#       metrics = [tf.keras.metrics.CategoricalAccuracy()]

#     ) 



#     return model





# def train_and_predict():



#     # Out of folds confidence list

#     oof_confidence = []

#     # Out of folds target

#     oof_target = []

#     # Ground truth target

#     target = []



#     # Seed everything

#     seed_everything(SEED)



#     print('\n')

#     print('-'*50)

#     train_dataset = get_training_dataset(TRAINING_FILENAMES[0:8], ordered = False)

#     val_dataset = get_validation_dataset(TRAINING_FILENAMES[8], ordered = True)

#     STEPS_PER_EPOCH = count_data_items(TRAINING_FILENAMES[0:8]) // BATCH_SIZE

#     K.clear_session()

#     model = get_model()

#     # using early stopping using val loss

#     checkpoint = tf.keras.callbacks.ModelCheckpoint(f'baseline_model.h5', monitor = 'val_loss', save_best_only = True, save_weights_only = False)

#     # lr scheduler

#     cb_lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', mode = 'min', factor = 0.5, patience = 1, verbose = 1, min_delta = 0.0001)

#     history = model.fit(train_dataset, 

#                       steps_per_epoch = STEPS_PER_EPOCH,

#                       epochs = EPOCHS,

#                       callbacks = [checkpoint, cb_lr_schedule],

#                       validation_data = val_dataset,

#                       verbose = 1)



#     print('Getting ground truth target')

#     # Get ground truth target for the fold

#     val_target = val_dataset.map(lambda embeddings, target: target)

#     val_target = val_target.as_numpy_iterator()

#     print('Stacking')

#     val_target = np.vstack(list(val_target))

#     val_target = np.argmax(val_target, axis = 1)

#     target.extend(list(val_target))



#     # Predictions

#     val_embeddings = val_dataset.map(lambda embeddings, target: embeddings)

#     print('Predicting validation embeddings')

#     val_embeddings = list(val_embeddings.as_numpy_iterator())

#     val_emb_len = len(val_embeddings)

#     iterations = math.ceil(val_emb_len / 300)

#     predictions = []

#     # for some reason using model.predict with the tf.data.dataset generator burn out the memory of the gpu

#     for i in range(iterations):

#     prediction = model.predict(np.vstack(val_embeddings[300 * i : 300 * (i + 1)]))

#     predictions.append(prediction)

#     predictions = np.vstack(predictions)

#     print('Get max indices')

#     target_prediction = np.argmax(predictions, axis = -1)

#     print('Get max confidence')

#     target_confidence = np.max(predictions, axis = -1)

#     print('Extend predictions')

#     oof_target.extend(list(target_prediction))

#     print('Extend confidence')

#     oof_confidence.extend(list(target_confidence))



#     # Calculate global average precision for the fold

#     gap = gap_vector(list(target_prediction), list(target_confidence), list(val_target))

#     accuracy_score = metrics.accuracy_score(list(val_target), list(target_prediction))

#     print(f'Our global average precision for is {gap}')

#     print(f'Our accuracy score for is {accuracy_score}')





# train_and_predict()
import csv

import gc

import os

import math



import shutil

import pathlib

import pandas as pd

import numpy as np

import PIL

import tensorflow as tf
DATASET_DIR = '../input/landmark-train-encoded/train_encoded.csv'

TEST_IMAGE_DIR = '../input/landmark-recognition-2020/test'



# DEBUGGING PARAMS:

NUM_PUBLIC_TEST_IMAGES = 10345 # Used to detect if in session or re-run.

MAX_NUM_EMBEDDINGS = -1  # Set to > 1 to subsample dataset while debugging.



# DNN model

DNN_MODEL = tf.keras.models.load_model('../input/landmark-baseline-model/baseline_model.h5')



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



def to_hex(image_id) -> str:

    return '{0:0{1}x}'.format(image_id, 16)



# load an image to a tf tensor

def load_image_tensor(image_path):

    return tf.convert_to_tensor(

        np.array(PIL.Image.open(image_path).convert('RGB')))



# function to extract the global features using delg_model

def extract_global_features(image_root_dir):

    """Extracts embeddings for all the images in given `image_root_dir`."""

    

    # get the path for all the training or test images

    image_paths = [x for x in pathlib.Path(image_root_dir).rglob('*.jpg')]

    num_embeddings = len(image_paths)

    if MAX_NUM_EMBEDDINGS > 0:

        num_embeddings = min(MAX_NUM_EMBEDDINGS, num_embeddings)

        

    ids = num_embeddings * [None]

    embeddings = np.empty((num_embeddings, NUM_EMBEDDING_DIMENSIONS))

    

    for i, image_path in enumerate(image_paths):

        if i >= num_embeddings:

            break

            

        ids[i] = to_hex(int(image_path.name.split('.')[0], 16))

        image_tensor = load_image_tensor(image_path)

        features = GLOBAL_FEATURE_EXTRACTION_FN(image_tensor,

                                                DELG_IMAGE_SCALES_TENSOR,

                                                DELG_SCORE_THRESHOLD_TENSOR)

        

        embeddings[i, :] = tf.nn.l2_normalize(

            tf.reduce_sum(features[0], axis=0, name='sum_pooling'),

            axis=0,

            name='final_l2_normalization').numpy()

        

    return ids, embeddings
def inference_and_save_submission_csv(test_path, train_csv):

    image_paths = [x for x in pathlib.Path(test_path).rglob('*.jpg')]

    test_len = len(image_paths)

    if test_len == NUM_PUBLIC_TEST_IMAGES:

        # Dummy submission

        shutil.copyfile('../input/landmark-recognition-2020/sample_submission.csv', 'submission.csv')

        return

    else:

        # Predict

        test_ids, test_embeddings = extract_global_features(test_path)

        embeddings_len = len(test_embeddings)

        steps = math.ceil(embeddings_len / 9600)

        predictions = []

        # Predict in batches of 9600

        for i in range(steps):

            prediction = DNN_MODEL.predict(test_embeddings[i * 9600 : (i + 1) * 9600])

            predictions.append(prediction)

        predictions = np.vstack(predictions)

        target = np.argmax(predictions, axis = -1)

        confidence = np.max(predictions, axis = -1)

        final = pd.DataFrame({'id': list(test_ids), 'target': list(target), 'confidence': list(confidence)})

        # Get target dictionary

        df = pd.read_csv(train_csv)

        df = df[['landmark_id', 'landmark_id_encode']]

        df.set_index('landmark_id_encode', inplace = True)

        df = df.to_dict()['landmark_id']

        final['landmarks'] = final['target'].map(df).astype(str) + ' ' + final['confidence'].astype(str)

        final[['id', 'landmarks']].to_csv('submission.csv', index = False)



inference_and_save_submission_csv(TEST_IMAGE_DIR, DATASET_DIR)