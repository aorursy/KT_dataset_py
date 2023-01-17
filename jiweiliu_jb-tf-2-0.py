import time
%%time

!pip install ../input/tf-20/tensorflow_gpu-2.0.0-cp36-cp36m-manylinux2010_x86_64.whl
!ls ../input/tensorflow2-question-answering
!ls ../input/tf2qa-sub/tf2qa_sub
!ls ../input/tf-115-dependencies
%%time



!pip install ../input/tf-115-dependencies/natural_questions-1.0.4-py2.py3-none-any.whl --no-dependencies
PATH = '../input/tf2qa-sub/tf2qa_sub'
import os

#GPU_id = '4,5,6,7,0,1,2,3'

GPU_id = '0'

os.environ['CUDA_VISIBLE_DEVICES'] = GPU_id
import tensorflow as tf

print('tensorflow',tf.__version__)
%%time

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:

  try:

    # Currently, memory growth needs to be the same across GPUs

    for gpu in gpus:

      tf.config.experimental.set_memory_growth(gpu, True)

    logical_gpus = tf.config.experimental.list_logical_devices('GPU')

    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

  except RuntimeError as e:

    # Memory growth must be set before GPUs have been initialized

    print(e)
import sys

sys.path.append(f"{PATH}/models-2.0")

sys.path.append(f"{PATH}/jb_2.0")

import gzip

import random

import os

from official.nlp.bert import input_pipeline

from nq_lib import create_example_from_jsonl,CreateTFExampleFn,read_nq_examples

from nq_lib import FeatureWriter,convert_examples_to_features,read_candidates,compute_pred_dict

from nq_lib import FLAGS

from run_jb import get_strategy,read_metas,predict_jb_customized

import tokenization

import pandas as pd



from tqdm import tqdm

import json

import os

import pickle
INPUT_DIR = '../input/tensorflow2-question-answering'

MODEL_DIR='../input/tf2qa-sub/tf2qa_sub'

BERT_FILE='../input/tf2qa-sub/tf2qa_sub/bert_config.json'

BATCH_SIZE = 32

CKPT = 'ctl_step_5158.ckpt-12'
def create_test():

  

  

  input_jsonl = f'{INPUT_DIR}/simplified-nq-test.jsonl'

  output_tfrecord = 'simplified-nq-test.tfrecords'

  meta_path = 'simplified-nq-test.meta'

  vocab_file = f'{MODEL_DIR}/vocab-nq.txt'

  

  if os.path.exists(meta_path):

      print(meta_path,'exists.')

      return

  tokenizer = tokenization.FullTokenizer(

      vocab_file=vocab_file, do_lower_case=FLAGS.do_lower_case)



  eval_examples = read_nq_examples(

      input_file=input_jsonl, is_training=False)

  eval_writer = FeatureWriter(

      filename=output_tfrecord,

      is_training=False)



  real_examples = []

  all_examples = []

  

  def append_feature(feature,is_padding=False):

    if not is_padding:

      real_examples.append(1)

    eval_writer.process_feature(feature)

    all_examples.append(1)

    return len(real_examples)

  

  num_spans_to_ids = convert_examples_to_features(

      examples=eval_examples,

      tokenizer=tokenizer,

      is_training=False,

      output_fn=append_feature,

      predict_batch_size = BATCH_SIZE)

  eval_writer.close()

  

  num_orig_examples = len(eval_examples)

  num_real_examples = len(real_examples)

  num_all_examples = len(all_examples)

  

  del eval_examples,real_examples,all_examples

  meta_data = {

      "task_type": "jb_nq",

      "num_orig_examples": num_orig_examples,

      "num_real_examples": num_real_examples,

      "num_all_examples": num_all_examples,

      "max_query_length": FLAGS.max_query_length,

      "max_seq_length": FLAGS.max_seq_length,

      "doc_stride": FLAGS.doc_stride,

  }



  with tf.io.gfile.GFile(meta_path, "w") as writer:

    writer.write(json.dumps(meta_data, indent=4) + "\n")
%%time



create_test()
def predict_jb():

  

  bert_config_file= BERT_FILE

  model_dir=MODEL_DIR

  input_meta_data_path='simplified-nq-test.meta'

  predict_batch_size=BATCH_SIZE

  predict_file='simplified-nq-test.tfrecords'

  result_path=f'sub_{CKPT}'

  use_checkpoint=f'{MODEL_DIR}/{CKPT}' 

  strategy_type = 'mirror'

  tpu = False

  

  strategy = get_strategy(strategy_type,tpu)

  input_meta_data = read_metas(input_meta_data_path)

  num_eval_examples = input_meta_data['num_orig_examples']

  dataset_size = input_meta_data['num_all_examples']



  print("***** Running predictions *****")

  print("  Num orig examples = %d" % num_eval_examples)

  print("  Num split examples = %d" % dataset_size)

  print("  Batch size = %d" % predict_batch_size)



  all_results = predict_jb_customized(strategy, input_meta_data, 

    bert_config_file,

    predict_file,

    model_dir,

    use_tpu = tpu,

    predict_batch_size = predict_batch_size,

    use_checkpoint = use_checkpoint)



  with open(f'{result_path}.p','wb') as f:

    pickle.dump(all_results,f)

    

  del all_results
%%time



predict_jb() # BS = 32
def create_eval_nq_dataset(file_path, seq_length):

  """Creates input dataset from (tf)records files for eval."""

  name_to_features = {

      'unique_ids': tf.io.FixedLenFeature([], tf.int64),

      'input_ids': tf.io.FixedLenFeature([seq_length], tf.int64),

      'input_mask': tf.io.FixedLenFeature([seq_length], tf.int64),

      'segment_ids': tf.io.FixedLenFeature([seq_length], tf.int64),

      'token_map': tf.io.FixedLenFeature([seq_length], tf.int64), 

  }

  input_fn = input_pipeline.file_based_input_fn_builder(file_path, name_to_features)

  return input_fn()



def post_process():

  raw_prediction_file = f'sub_{CKPT}.p'

  predict_file = f'{INPUT_DIR}/simplified-nq-test.jsonl'

  predict_tfrecord = 'simplified-nq-test.tfrecords'

  

  with open(raw_prediction_file,'rb') as f:

    all_results = pickle.load(f)

  candidates_dict = read_candidates(predict_file) # json file to be predicted

  eval_features = [

      r for r in create_eval_nq_dataset(predict_tfrecord,FLAGS.max_seq_length)

  ]

  print(len(eval_features),len(all_results))

  results = [r._asdict() for c,r in enumerate(all_results) if c<len(eval_features)]

  nq_pred_dict,nbest_summary_dict = compute_pred_dict(candidates_dict, eval_features,results)

  output = {"predictions": list(nq_pred_dict.values())}



  return output
%%time



output = post_process()
def create_short_answer(entry):

    if entry["short_answers_score"] < short_thr: 

        return ""

    

    answer = []    

    for short_answer in entry["short_answers"]:

        if short_answer["start_token"] > -1:

            answer.append(str(short_answer["start_token"]) + ":" + str(short_answer["end_token"]))

    if entry["yes_no_answer"] != "NONE":

        answer.append(entry["yes_no_answer"])

    return " ".join(answer)



def create_long_answer(entry):

    if entry["long_answer_score"] < long_thr: 

        return ""



    answer = []

    if entry["long_answer"]["start_token"] > -1:

        answer.append(str(entry["long_answer"]["start_token"]) + ":" + str(entry["long_answer"]["end_token"]))

    return " ".join(answer)



def write_sub(test_answers_df):

  path = f'{INPUT_DIR}/sample_submission.csv'

  test_answers_df["long_answer_score"] = test_answers_df["predictions"].apply(lambda q: q["long_answer_score"])

  test_answers_df["short_answer_score"] = test_answers_df["predictions"].apply(lambda q: q["short_answers_score"])



  test_answers_df["long_answer"] = test_answers_df["predictions"].apply(create_long_answer)

  test_answers_df["short_answer"] = test_answers_df["predictions"].apply(create_short_answer)

  test_answers_df["example_id"] = test_answers_df["predictions"].apply(lambda q: str(q["example_id"]))



  long_answers = dict(zip(test_answers_df["example_id"], test_answers_df["long_answer"]))

  short_answers = dict(zip(test_answers_df["example_id"], test_answers_df["short_answer"]))



  sample_submission = pd.read_csv(path)



  long_prediction_strings = sample_submission[sample_submission["example_id"].str.contains("_long")].apply(lambda q: long_answers[q["example_id"].replace("_long", "")], axis=1)

  short_prediction_strings = sample_submission[sample_submission["example_id"].str.contains("_short")].apply(lambda q: short_answers[q["example_id"].replace("_short", "")], axis=1)



  sample_submission.loc[sample_submission["example_id"].str.contains("_long"), "PredictionString"] = long_prediction_strings

  sample_submission.loc[sample_submission["example_id"].str.contains("_short"), "PredictionString"] = short_prediction_strings



  sample_submission.to_csv('submission.csv',index=False)

  
%%time



df = pd.DataFrame(output)

long_thr = 2.7368

short_thr = 6.8347



write_sub(df)
pd.read_csv('submission.csv').count()
FLAGS.__dict__