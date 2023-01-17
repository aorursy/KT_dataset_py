import tensorflow as tf

import numpy as np # linear algebra

import pandas as pd 

import gc

import os

import json

import collections

from time import time

from tqdm import tqdm as tqdm_base

import os

import sys

def tqdm(*args, **kwargs):

    if hasattr(tqdm_base, '_instances'):

        for instance in list(tqdm_base._instances):

            tqdm_base._decr_instances(instance)

    return tqdm_base(*args, **kwargs)

import gc

print(tf.__version__)

# Any results you write to the current directory are saved as output.
# Input data files are available in the "../input/" directory.

IS_KAGGLE = True

INPUT_DIR = "/kaggle/input/"



# The original Bert Joint Baseline data.

class_model_dir = os.path.join(INPUT_DIR, "tf-3labels")



# This nq dir contains all files for publicly use.

TFNQ_DIR = os.path.join(INPUT_DIR, "tensorflow2-question-answering")



Trans_ = os.path.join(INPUT_DIR, 'tf-asnq/src/transformers')



# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#for dirname, _, filenames in os.walk(INPUT_DIR):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



sys.path.append(Trans_)

# Any results you write to the current directory are saved as output.



from transformers import TFRobertaForSequenceClassification, RobertaTokenizer,RobertaConfig

from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures
test_path = os.path.join(TFNQ_DIR,'simplified-nq-test.jsonl')
with open(test_path,'rt') as reader:

    df = [] 

    for line in tqdm(reader):

        exemple = json.loads(line)

        id = int(exemple["example_id"])

        candidates = exemple["long_answer_candidates"]

        question= exemple["question_text"]

        _text = exemple['document_text'].split()

        for candidate in candidates:

            local_df = []

            local_df.append(id)

            local_df.append(question)

            start = candidate['start_token']

            stop = candidate['end_token']

            text__ = ' '.join(_text[start:min(start+512,stop)])

            local_df.append(text__)

            local_df.append(f'{start}:{stop}')

            df.append(local_df)
df = pd.DataFrame(df,columns=['example_id','question','text','start_stop'])

gc.collect()
MAX_SEQUENCE_LENGTH = 512

max_length = MAX_SEQUENCE_LENGTH

task='asnq'

label_list=["1","2","3"]

output_mode="classification"

pad_on_left=False

pad_token=0

pad_token_segment_id=0

mask_padding_with_zero=True



label_map = {label: i for i, label in enumerate(label_list)}
def model_fn():

    return TFRobertaForSequenceClassification.from_pretrained(class_model_dir,config =RobertaConfig.from_pretrained(class_model_dir))
model = model_fn()
model.summary()
for i in df:

    print(i)

    
tokenizer = RobertaTokenizer.from_pretrained(class_model_dir)
def gen_data():

    for index, row in df.iterrows():

        inputs = tokenizer.encode_plus(row.question,row.text,add_special_tokens=True,max_length=max_length,)

        input_ids = inputs["input_ids"]

        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        padding_length = max_length - len(input_ids)

        input_ids = ([pad_token] * padding_length) + input_ids

        attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask

        yield (input_ids,attention_mask)
dataset = tf.data.Dataset.from_generator( 

     gen_data, 

     (tf.int64, tf.int64), 

     (tf.TensorShape([None]), tf.TensorShape([None])))



dataset = dataset.map(lambda x,y : {'input_ids': x,'attention_mask': y,},num_parallel_calls=32)

dataset = dataset.batch(128,drop_remainder=False)

dataset = dataset.prefetch(100)
start = time()

preds=np.array([])

for line in tqdm(dataset):

    preds= np.concatenate((preds,tf.argmax(model(line, training=False)[0],1).numpy()),0)

end=time()
len(preds)==len(df)
df['preds'] = preds.tolist()

preds=None

gc.collect()
df_true = df[df.preds==2]

df=None

gc.collect()

df_true

new_true_df=[]
for i in df_true.example_id.unique():

    local_df = df_true[df_true.example_id==i]

    def gen_data():

        for index, row in local_df .iterrows():

            inputs = tokenizer.encode_plus(row.question,row.text,add_special_tokens=True,max_length=max_length,)

            input_ids = inputs["input_ids"]

            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            padding_length = max_length - len(input_ids)

            input_ids = ([pad_token] * padding_length) + input_ids

            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask

            yield (input_ids,attention_mask)

    dataset = tf.data.Dataset.from_generator( gen_data, (tf.int64, tf.int64), (tf.TensorShape([None]), tf.TensorShape([None])))



    dataset = dataset.map(lambda x,y : {'input_ids': x,'attention_mask': y,},num_parallel_calls=32)

    dataset = dataset.batch(128,drop_remainder=False)

    dataset = dataset.prefetch(100)

    preds=np.array([])

    for line in tqdm(dataset):

        preds= np.concatenate((preds,model(line, training=False)[0].numpy()[:,2]),0)

    true_one = local_df.iloc[np.argmax(preds,0)]

    new_true_df.append([true_one.example_id,true_one.start_stop])
model=None

df_true = None

tokenizer=None

gc.collect()

new_true_df

new_true_df = pd.DataFrame(new_true_df,columns =['example_id','preds'])

new_true_df['example_id'] = new_true_df['example_id'].apply(lambda q: str(q)+"_long")
sample_submission = pd.read_csv("../input/tensorflow2-question-answering/sample_submission.csv")
sample_submission.PredictionString = sample_submission.join(new_true_df.set_index('example_id'),on='example_id').preds
sample_submission.PredictionString = sample_submission.PredictionString.apply(lambda x: '' if type(x)==float  else x)
sample_submission.to_csv('submission.csv', index=False)