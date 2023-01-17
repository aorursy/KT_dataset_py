# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Here are link to the datasets on kaggle

'''

https://www.kaggle.com/subratasarkar32/the-great-indian-data-scientist-hiring-challenge

https://www.kaggle.com/subratasarkar32/googleresearchbert

https://www.kaggle.com/maxjeblick/bert-pretrained-models

'''

# Uncomment the below code if you are running this notebook on your local machine or collab

'''

!wget https://www.kaggle.com/subratasarkar32/the-great-indian-data-scientist-hiring-challenge/downloads/the-great-indian-data-scientist-hiring-challenge.zip/3

!wget https://www.kaggle.com/subratasarkar32/googleresearchbert/downloads/googleresearchbert.zip/5

!wget https://www.kaggle.com/maxjeblick/bert-pretrained-models/downloads/bert-pretrained-models.zip/1

'''

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import collections

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

from __future__ import absolute_import

from __future__ import division



from __future__ import print_function



import datetime

import pkg_resources

import seaborn as sns

import time

import scipy.stats as stats

import gc

import re

import numpy as np 

import pandas as pd

import re

import gc

import os

print(os.listdir("../input"))

import fileinput

import string

import tensorflow as tf

import zipfile

import datetime

import sys

from tqdm  import tqdm

tqdm.pandas()

from nltk.tokenize import wordpunct_tokenize

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score, roc_auc_score



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation

from keras.layers.embeddings import Embedding

from sklearn.metrics import classification_report



%matplotlib inline

from tqdm import tqdm, tqdm_notebook

import warnings

warnings.filterwarnings(action='once')

import pickle





from sklearn.model_selection import train_test_split

import tensorflow as tf

import tensorflow_hub as hub

from datetime import datetime





import shutil
!pip install bert-tensorflow

!wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip

!wget https://raw.githubusercontent.com/google-research/bert/master/modeling.py 

!wget https://raw.githubusercontent.com/google-research/bert/master/optimization.py 

!wget https://raw.githubusercontent.com/google-research/bert/master/run_classifier.py 

!wget https://raw.githubusercontent.com/google-research/bert/master/tokenization.py 
folder = 'model_folder'

with zipfile.ZipFile("uncased_L-12_H-768_A-12.zip","r") as zip_ref:

    zip_ref.extractall(folder)
import bert

from bert import run_classifier

from bert import optimization

from bert import tokenization

from bert import modeling
#import tokenization

#import modeling

BERT_MODEL = 'uncased_L-12_H-768_A-12'

BERT_PRETRAINED_DIR = f'{folder}/uncased_L-12_H-768_A-12'

OUTPUT_DIR = f'{folder}/outputs'

print(f'>> Model output directory: {OUTPUT_DIR}')

print(f'>>  BERT pretrained directory: {BERT_PRETRAINED_DIR}')

BERT_VOCAB= '../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/vocab.txt'

BERT_INIT_CHKPNT = '../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/bert_model.ckpt'

BERT_CONFIG = '../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/bert_config.json'
tokenization.validate_case_matches_checkpoint(True,BERT_INIT_CHKPNT)

tokenizer = tokenization.FullTokenizer(

      vocab_file=BERT_VOCAB, do_lower_case=True)
import numpy as np                     # For mathematical calculations

import seaborn as sns                  # For data visualization

import matplotlib.pyplot as plt 

import seaborn as sn                   # For plotting graphs

import re

import nltk

%matplotlib inline

import warnings                        # To ignore any warnings

warnings.filterwarnings("ignore")

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from pandas import DataFrame

 

le = LabelEncoder()

 

df = pd.read_csv("../input/the-great-indian-data-scientist-hiring-challenge/Train.csv")



df['Item_Description'] = df['Item_Description'].apply(lambda x : re.sub(r'\b[A-Z]+\b', '',re.sub('\s+',' ',re.sub(r'[^a-zA-Z]', ' ',x))))

df['Item_Description'] = df['Item_Description'].apply(lambda x : x.lower())

identity_columns = ['CLASS-784', 'CLASS-95', 'CLASS-51', 'CLASS-559', 'CLASS-489', 'CLASS-913', 'CLASS-368', 'CLASS-816', 'CLASS-629', 'CLASS-177', 'CLASS-123', 'CLASS-671', 'CLASS-804', 'CLASS-453', 'CLASS-1042', 'CLASS-49', 'CLASS-947', 'CLASS-110', 'CLASS-278', 'CLASS-522', 'CLASS-606', 'CLASS-651', 'CLASS-765', 'CLASS-953', 'CLASS-839', 'CLASS-668', 'CLASS-758', 'CLASS-942', 'CLASS-764', 'CLASS-50', 'CLASS-75', 'CLASS-74', 'CLASS-783', 'CLASS-323', 'CLASS-322', 'CLASS-720', 'CLASS-230', 'CLASS-571'] 

for key in identity_columns:

    df.Product_Category[df.Product_Category==key] = identity_columns.index(key)+1

print(len(identity_columns))



df2 = pd.DataFrame({'text':df['Item_Description'].replace(r'\n',' ',regex=True),

            'label':LabelEncoder().fit_transform(df['Product_Category'].replace(r' ','',regex=True)),

            })



# Creating train and val dataframes according to BERT

X_train, X_test, y_train, y_test = train_test_split(df2["text"].values, df2["label"].values, test_size=0.2, random_state=42)

X_train, y_train = df2["text"].values, df2["label"].values





#x_train, x_val = train_test_split(df_bert, test_size=0.01,random_state=3,shuffle=True)

 

# Creating test dataframe according to BERT

testpd = pd.read_csv("../input/the-great-indian-data-scientist-hiring-challenge/Test.csv")

testpd['Item_Description'] = testpd['Item_Description'].apply(lambda x : re.sub(r'\b[A-Z]+\b', '',re.sub('\s+',' ',re.sub(r'[^a-zA-Z]', ' ',x))))

testpd['Item_Description'] = testpd['Item_Description'].apply(lambda x : x.lower())

test = pd.DataFrame({'text':testpd['Item_Description'].replace(r'\n',' ',regex=True)})

test = test["text"].values



# Saving dataframes to .tsv format as required by BERT

#X_train.to_csv('train.tsv', sep='\t', index=False, header=False)

#X_test.to_csv('dev.tsv', sep='\t', index=False, header=False)

#test.to_csv('test.tsv', sep='\t', index=False, header=False)
X_train[:5]  # check training data
def create_examples(lines, set_type, labels=None):

#Generate data for the BERT model

    guid=f'{set_type}'

    examples = []

    if set_type == 'train':

        for line, label in zip(lines, labels):

            

            text_a = line

            label = str(label)

            examples.append(

              run_classifier.InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

    else:

        for line in lines:

            

            text_a = line

            label = '0'

            examples.append(

              run_classifier.InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

    return examples



# Model Hyper Parameters

TRAIN_BATCH_SIZE = 16

EVAL_BATCH_SIZE = 8

LEARNING_RATE = 1e-5

NUM_TRAIN_EPOCHS = 3.0

WARMUP_PROPORTION = 0.1

MAX_SEQ_LENGTH = 100

# Model configs

SAVE_CHECKPOINTS_STEPS = 1000 #if you wish to finetune a model on a larger dataset, use larger interval

# each checpoint weights about 1,5gb

ITERATIONS_PER_LOOP = 100

NUM_TPU_CORES = 8

VOCAB_FILE = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')

CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')

INIT_CHECKPOINT = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')

DO_LOWER_CASE = BERT_MODEL.startswith('uncased')



label_list = [str(num) for num in range(38)]

tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)

train_examples = create_examples(X_train, 'train', labels=y_train)



tpu_cluster_resolver = None #Since training will happen on GPU, we won't need a cluster resolver

#TPUEstimator also supports training on CPU and GPU. You don't need to define a separate tf.estimator.Estimator.

run_config = tf.contrib.tpu.RunConfig(

    cluster=tpu_cluster_resolver,

    model_dir=OUTPUT_DIR,

    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,

    tpu_config=tf.contrib.tpu.TPUConfig(

        iterations_per_loop=ITERATIONS_PER_LOOP,

        num_shards=NUM_TPU_CORES,

        per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))



num_train_steps = int(

    len(train_examples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)

num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)



model_fn = run_classifier.model_fn_builder(

    bert_config=modeling.BertConfig.from_json_file(CONFIG_FILE),

    num_labels=len(label_list),

    init_checkpoint=INIT_CHECKPOINT,

    learning_rate=LEARNING_RATE,

    num_train_steps=num_train_steps,

    num_warmup_steps=num_warmup_steps,

    use_tpu=False, #If False training will fall on CPU or GPU, depending on what is available  

    use_one_hot_embeddings=True)



estimator = tf.contrib.tpu.TPUEstimator(

    use_tpu=False, #If False training will fall on CPU or GPU, depending on what is available 

    model_fn=model_fn,

    config=run_config,

    train_batch_size=TRAIN_BATCH_SIZE,

    eval_batch_size=EVAL_BATCH_SIZE)
print('Please wait...')

train_features = run_classifier.convert_examples_to_features(

    train_examples, label_list, MAX_SEQ_LENGTH, tokenizer)

print('>> Started training at {} '.format(datetime.now()))

print('  Num examples = {}'.format(len(train_examples)))

print('  Batch size = {}'.format(TRAIN_BATCH_SIZE))

tf.logging.info("  Num steps = %d", num_train_steps)

train_input_fn = run_classifier.input_fn_builder(

    features=train_features,

    seq_length=MAX_SEQ_LENGTH,

    is_training=True,

    drop_remainder=True)

estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

print('>> Finished training at {}'.format(datetime.now()))
def input_fn_builder(features, seq_length, is_training, drop_remainder):

  """Creates an `input_fn` closure to be passed to TPUEstimator."""



  all_input_ids = []

  all_input_mask = []

  all_segment_ids = []

  all_label_ids = []



  for feature in features:

    all_input_ids.append(feature.input_ids)

    all_input_mask.append(feature.input_mask)

    all_segment_ids.append(feature.segment_ids)

    all_label_ids.append(feature.label_id)



  def input_fn(params):

    """The actual input function."""

    print(params)

    batch_size = 500



    num_examples = len(features)



    d = tf.data.Dataset.from_tensor_slices({

        "input_ids":

            tf.constant(

                all_input_ids, shape=[num_examples, seq_length],

                dtype=tf.int32),

        "input_mask":

            tf.constant(

                all_input_mask,

                shape=[num_examples, seq_length],

                dtype=tf.int32),

        "segment_ids":

            tf.constant(

                all_segment_ids,

                shape=[num_examples, seq_length],

                dtype=tf.int32),

        "label_ids":

            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),

    })



    if is_training:

      d = d.repeat()

      d = d.shuffle(buffer_size=100)



    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)

    return d



  return input_fn
predict_examples = create_examples(X_test, 'test')

predict_features = run_classifier.convert_examples_to_features(

    predict_examples, label_list, MAX_SEQ_LENGTH, tokenizer)



predict_input_fn = input_fn_builder(

    features=predict_features,

    seq_length=MAX_SEQ_LENGTH,

    is_training=False,

    drop_remainder=False)



result = estimator.predict(input_fn=predict_input_fn)
preds = []

for prediction in result:

      preds.append(np.argmax(prediction['probabilities']))
print(preds)
from sklearn.metrics import accuracy_score
print("Accuracy of BERT is:",accuracy_score(y_test,preds))
print(classification_report(y_test,preds))
testpd = pd.read_csv("../input/the-great-indian-data-scientist-hiring-challenge/Test.csv")

testpd['Item_Description'] = testpd['Item_Description'].apply(lambda x : re.sub(r'\b[A-Z]+\b', '',re.sub('\s+',' ',re.sub(r'[^a-zA-Z]', ' ',x))))

testpd['Item_Description'] = testpd['Item_Description'].apply(lambda x : x.lower())

testln = pd.DataFrame({'guid':testpd['Inv_Id'],

    'text':testpd['Item_Description'].replace(r'\n',' ',regex=True)})

testl = testln["text"].values
predict_test = create_examples(testl, 'test')

predict_features1 = run_classifier.convert_examples_to_features(

    predict_test, label_list, MAX_SEQ_LENGTH, tokenizer)



predict_input_fn1 = input_fn_builder(

    features=predict_features1,

    seq_length=MAX_SEQ_LENGTH,

    is_training=False,

    drop_remainder=False)



result1 = estimator.predict(input_fn=predict_input_fn1)
preds1 = []

for prediction in result1 :

    #print(prediction)

    preds1.append(np.argmax(prediction['probabilities']))
print(preds1)
def create_output(predictions):

    probabilities = []

    for (i, prediction) in enumerate(predictions):

        preds = prediction

        probabilities.append(preds)

        #print(preds)

    dff = pd.DataFrame(probabilities)

    dff.head()

    

    #dff.columns = identity_columns

    

    return dff
#output_df = create_output(preds1)

predslab=[identity_columns[x] for x in preds1]

testln["Product_Category"]=predslab

merged_df =  testln

submission = merged_df.drop(['text'], axis=1)

submission.to_csv("sample_submission1.csv", index=False)
submission.head()
