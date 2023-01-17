# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import spacy

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

import re

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator





#Visualisation

import matplotlib.pyplot as plt

import seaborn as sns

from plotly import tools

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
df_train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

df_test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

df_sample = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')
df_train.head(5)
df_test.head(5)
df_train['comment_text'][0]
lengths = df_train.comment_text.str.len()

lengths.mean(), lengths.std(), lengths.min(), lengths.max()
lengths = df_test.comment_text.str.len()

lengths.mean(), lengths.std(), lengths.min(), lengths.max()
def preprocess_reviews(text):

    text = re.sub(r'<[^>]*>', ' ', text, re.UNICODE)

    text = re.sub(r'[^\w\s]', '', text, re.UNICODE)

    text = re.sub(r'[^0-9a-zA-Z]+',' ',text, re.UNICODE)

    text = " ".join(text.split())

    text = text.lower()

    return text



df_train['comment_text'] = df_train.comment_text.apply(lambda x: preprocess_reviews(x))

df_test['comment_text'] = df_test.comment_text.apply(lambda x: preprocess_reviews(x))
# force train into cola format, test is fine as it is

df_train = df_train[['id', 'target', 'comment_text']]

df_train['target'] = np.where(df_train['target']>=0.5,1,0)



#Sampling 30% to save training time

df_train = df_train.sample(frac=0.3)



# export as tab seperated

df_train.to_csv('train.tsv', sep='\t', index=False, header=False)

df_test.to_csv('test.tsv', sep='\t', index=False, header=True)
df_train.shape, df_test.shape
# import module we'll need to import our custom module

from shutil import copyfile



# copy our file into the working directory (make sure it has .py suffix)

for f in os.listdir('../input/xlnetcode/'):

    try:

        if f.split('.')[1] in ['py', 'json']:

            copyfile(src = "../input/xlnetcode/"+f, dst = "../working/"+f)

    except:

        continue

print(os.listdir('../working'))
from absl import flags

import xlnet

from run_classifier import *

import sys
remaining_args = FLAGS([sys.argv[0]] + [flag for flag in sys.argv if flag.startswith("--")])

assert(remaining_args == [sys.argv[0]])
FLAGS.spiece_model_file = '../input/xlnetcode/spiece.model'

FLAGS.model_config_path = '../input/xlnetcode/xlnet_config.json'

FLAGS.output_dir ="../"

FLAGS.model_dir = "../"

FLAGS.data_dir = "../working/"

FLAGS.do_train = False

FLAGS.train_steps = 1000

FLAGS.warmup_steps = 0

FLAGS.learning_rate = 1e-5

FLAGS.max_save = 999999

FLAGS.use_tpu = False



#Used not take any of the processors and get from the tasks

FLAGS.cls_scope = True
# Tokenization

import sentencepiece as spm

from prepro_utils import preprocess_text, encode_ids



sp = spm.SentencePieceProcessor()

sp.Load(FLAGS.spiece_model_file)

def tokenize_fn(text):

    text = preprocess_text(text, lower=FLAGS.uncased)

    return encode_ids(sp, text)
processor = GLUEProcessor()

label_list = processor.get_labels()

processor.label_column = 1

processor.text_a_column = 2

processor.test_text_a_column = 1

train_examples = processor.get_train_examples(FLAGS.data_dir)
train_examples[0].label, train_examples[0].text_a, train_examples[0].text_b 
start = time.time()

print("--------------------------------------------------------")

print("Starting to Train")

print("--------------------------------------------------------")
train_file = os.path.join(FLAGS.output_dir, "train.tf_record")

tf.logging.info("Use tfrecord file {}".format(train_file))

np.random.shuffle(train_examples)

tf.logging.info("Num of train samples: {}".format(len(train_examples)))

file_based_convert_examples_to_features(

        train_examples, label_list, FLAGS.max_seq_length, tokenize_fn,

        train_file, FLAGS.num_passes)
# RunConfig contains hyperparameters that could be different between pretraining and finetuning.

tpu_cluster_resolver = None

is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2



run_config = tf.contrib.tpu.RunConfig(

    cluster=tpu_cluster_resolver,

    master=FLAGS.master,

    model_dir=FLAGS.output_dir,

    save_checkpoints_steps=FLAGS.save_steps,

    tpu_config=tf.contrib.tpu.TPUConfig(

        iterations_per_loop=FLAGS.iterations,

        num_shards=FLAGS.num_core_per_host,

        per_host_input_for_training=is_per_host))

model_fn = get_model_fn(len(label_list) if label_list is not None else None)
tf.logging.set_verbosity(tf.logging.INFO)

estimator = tf.contrib.tpu.TPUEstimator(

        use_tpu=FLAGS.use_tpu,

        model_fn=model_fn,

        config=run_config,

        train_batch_size=FLAGS.train_batch_size,

        predict_batch_size=FLAGS.predict_batch_size,

        eval_batch_size=FLAGS.eval_batch_size)



tf.logging.info("***** Running training *****")

tf.logging.info("  Num examples = %d", len(train_examples))

tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)

tf.logging.info("  Num steps = %d", FLAGS.iterations)
train_input_fn = file_based_input_fn_builder(

        input_file=train_file,

        seq_length=FLAGS.max_seq_length,

        is_training=True,

        drop_remainder=True)
estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)
end = time.time()

print("--------------------------------------------------------")

print("Total time taken to complete training - ", end - start, " seconds")

print("--------------------------------------------------------")
test_examples = processor.get_test_examples(FLAGS.data_dir)

tf.logging.info("Num of test samples: {}".format(len(test_examples)))

eval_file = os.path.join(FLAGS.output_dir, "predict.tf_record")

file_based_convert_examples_to_features(

        test_examples, label_list, FLAGS.max_seq_length, tokenize_fn,

        eval_file)
os.path.getsize('../predict.tf_record')
pred_input_fn = file_based_input_fn_builder(

        input_file=eval_file,

        seq_length=FLAGS.max_seq_length,

        is_training=False,

        drop_remainder=False)

predict_results = []

with tf.gfile.Open("test_results.tsv", "w") as fout:

    fout.write("index\tprediction\n")



    for pred_cnt, result in enumerate(estimator.predict(

        input_fn=pred_input_fn,

        yield_single_examples=True)):

        if pred_cnt % 1000 == 0:

            tf.logging.info("Predicting submission for example: {}".format(

              pred_cnt))



        logits = [float(x) for x in result["logits"].flat]

        predict_results.append(logits)



        if len(logits) == 1:

            label_out = logits[0]

        elif len(logits) == 2:

            if logits[1] - logits[0] > FLAGS.predict_threshold:

                label_out = label_list[1]

            else:

                label_out = label_list[0]

        elif len(logits) > 2:

            max_index = np.argmax(np.array(logits, dtype=np.float32))

            label_out = label_list[max_index]

        else:

            raise NotImplementedError



        fout.write("{}\t{}\n".format(pred_cnt, label_out))
len(test_examples), len(predict_results)
df_test_out = pd.read_csv('test_results.tsv', sep='\t')
submission = pd.concat([df_sample.iloc[:,0], df_test_out.iloc[:,1]], axis=1)

submission.columns = ['id','prediction']

submission.to_csv('submission.csv', index=False, header=True)