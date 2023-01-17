import pandas as pd

import os

import numpy as np

import pandas as pd

import zipfile

from matplotlib import pyplot as plt

%matplotlib inline

import sys

import datetime
#downloading weights and cofiguration file for the model

!wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
repo = 'model_repo'

with zipfile.ZipFile("uncased_L-12_H-768_A-12.zip","r") as zip_ref:

    zip_ref.extractall(repo)
!ls 'model_repo/uncased_L-12_H-768_A-12'
!ls 'model_repo'
!wget https://raw.githubusercontent.com/google-research/bert/master/modeling.py 

!wget https://raw.githubusercontent.com/google-research/bert/master/optimization.py 

!wget https://raw.githubusercontent.com/google-research/bert/master/run_squad.py 

!wget https://raw.githubusercontent.com/google-research/bert/master/tokenization.py 
# Available pretrained model checkpoints:

#   uncased_L-12_H-768_A-12: uncased BERT base model

#   uncased_L-24_H-1024_A-16: uncased BERT large model

#   cased_L-12_H-768_A-12: cased BERT large model

#We will use the most basic of all of them

BERT_MODEL = 'uncased_L-12_H-768_A-12'

BERT_PRETRAINED_DIR = f'{repo}/uncased_L-12_H-768_A-12'

OUTPUT_DIR = f'{repo}/outputs'

print(f'***** Model output directory: {OUTPUT_DIR} *****')

print(f'***** BERT pretrained directory: {BERT_PRETRAINED_DIR} *****')

!wget https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-development.tsv

!ls
def is_whitespace(c):

    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:

        return True

    return False

def get_squad_attributes(row):

    paragraph_text = row['Text']

    is_impossible = row['is_impossible'] 

    doc_tokens = []

    char_to_word_offset = []

    prev_is_whitespace = True

    for c in paragraph_text:

        if is_whitespace(c):

            prev_is_whitespace = True

        else:

            if prev_is_whitespace:

                doc_tokens.append(c)

            else:

                doc_tokens[-1] += c

            prev_is_whitespace = False

        char_to_word_offset.append(len(doc_tokens) - 1)

        

    if not is_impossible:

        if row['A-coref']:

            orig_answer_text = row['A']

            answer_offset = row['A-offset']

        else:

            orig_answer_text = row['B']

            answer_offset = row['B-offset']

            

        answer_length = len(orig_answer_text)

        start_position = char_to_word_offset[answer_offset]

        end_position = char_to_word_offset[answer_offset + answer_length -

                                       1]

        # Only add answers where the text can be exactly recovered from the

        # document. If this CAN'T happen it's likely due to weird Unicode

        # stuff so we will just skip the example.

        #

        # Note that this means for training mode, every example is NOT

        # guaranteed to be preserved.

        actual_text = " ".join(

            doc_tokens[start_position:(end_position + 1)])

        cleaned_answer_text = " ".join(

            tokenization.whitespace_tokenize(orig_answer_text))



        if actual_text.find(cleaned_answer_text) == -1:

            tf.logging.warning("Could not find answer: '%s' vs. '%s'",

                                 actual_text, cleaned_answer_text)

#             continue

    else:

        start_position = -1

        end_position = -1

        orig_answer_text = ""

    return pd.Series({'doc_tokens':doc_tokens, 

                      'char_to_word_offset':char_to_word_offset,

                      'orig_answer_text':orig_answer_text,

                      'start_position': start_position,

                      'end_position': end_position,

                     })
from sklearn.model_selection import train_test_split

import run_squad

import modeling

import optimization

import tokenization

import tensorflow as tf



train_df =  pd.read_csv('gap-development.tsv', sep='\t')

train_df = train_df.sample(2000)

print(train_df.columns)

# add is_impossible

train_df['is_impossible'] = ~train_df['A-coref'] & ~train_df['B-coref']

# add doc tokens

# add start word number of answer text

# add end word number of answer text

train_df_full = train_df.merge(train_df.apply(lambda row: get_squad_attributes(row), axis=1),

                             left_index=True, right_index=True)



train, test = train_test_split(train_df_full, test_size = 0.3, random_state=42)



train.head()
import tensorflow as tf

if tf.test.gpu_device_name():

    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

else:

    print("Please install GPU version of TF")




def create_squad_example(row):

    return run_squad.SquadExample(

        qas_id=row['ID'],

        question_text="Which noun does "+ row['Pronoun'] + " refer to?",

        doc_tokens=row['doc_tokens'],

        orig_answer_text=row['orig_answer_text'],

        start_position=row['start_position'],

        end_position=row['end_position'],

        is_impossible=row['is_impossible']

    )





def create_examples(rows, set_type):

#Generate data for the BERT model

    guid = f'{set_type}'

    examples = []

    if guid == 'train':

        for row in rows:

            examples.append(run_squad.SquadExample(

                    qas_id=row['ID'], # ID

                    question_text=question_text, #TBD

                    doc_tokens=doc_tokens, #doc_tokens

                    orig_answer_text=orig_answer_text, # 

                    start_position=start_position,

                    end_position=end_position,

                    is_impossible=is_impossible)

                           )

    else:

        for line in lines:

            text_a = line

            label = '0'

            examples.append(

              run_classifier.InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

    return examples



# Model Hyper Parameters

TRAIN_BATCH_SIZE = 32

EVAL_BATCH_SIZE = 8

LEARNING_RATE = 2e-5

NUM_TRAIN_EPOCHS = 3.0

WARMUP_PROPORTION = 0.1

MAX_SEQ_LENGTH = 128

# Model configs

SAVE_CHECKPOINTS_STEPS = 1000 #if you wish to finetune a model on a larger dataset, use larger interval

# each checpoint weights about 1,5gb

ITERATIONS_PER_LOOP = 1000

NUM_TPU_CORES = 8

VOCAB_FILE = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')

CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')

INIT_CHECKPOINT = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')

DO_LOWER_CASE = BERT_MODEL.startswith('uncased')



tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)

train_examples = train.apply(lambda row:create_squad_example(row), axis=1)



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



model_fn = run_squad.model_fn_builder(

    bert_config=modeling.BertConfig.from_json_file(CONFIG_FILE),

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

    predict_batch_size=EVAL_BATCH_SIZE,

    eval_batch_size=EVAL_BATCH_SIZE)
"""

Note: You might see a message 'Running train on CPU'. 

This really just means that it's running on something other than a Cloud TPU, which includes a GPU.

"""

!mkdir model_repo/outputs



# Train the model.

print('Please wait...')

train_writer = run_squad.FeatureWriter(

        filename=os.path.join(OUTPUT_DIR, "train.tf_record"),

        is_training=True)

train_features = run_squad.convert_examples_to_features(

    train_examples, tokenizer, MAX_SEQ_LENGTH, 128, 64, True, train_writer.process_feature)

train_writer.close()

print('***** Started training at {} *****'.format(datetime.datetime.now()))

print('  Num examples = {}'.format(len(train_examples)))

print('  Batch size = {}'.format(TRAIN_BATCH_SIZE))

tf.logging.info("  Num steps = %d", num_train_steps)

train_input_fn = run_squad.input_fn_builder(

    input_file=train_writer.filename,

    seq_length=MAX_SEQ_LENGTH,

    is_training=True,

    drop_remainder=True)

estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

print('***** Finished training at {} *****'.format(datetime.datetime.now()))
def create_test_squad_example(row):

    return run_squad.SquadExample(

        qas_id=row['ID'],

        question_text="Which noun does "+ row['Pronoun'] + " refer to?",

        doc_tokens=row['doc_tokens'],

        orig_answer_text="",

        start_position=-1,

        end_position=-1,

        is_impossible=row['is_impossible']

    )



eval_examples = test.apply(lambda row:create_test_squad_example(row), axis=1)



eval_writer = run_squad.FeatureWriter(

    filename=os.path.join(OUTPUT_DIR, "eval.tf_record"),

    is_training=False)



eval_features = []



def append_feature(feature):

    eval_features.append(feature)

    eval_writer.process_feature(feature)

    

run_squad.convert_examples_to_features(

        examples=predict_examples,

        tokenizer=tokenizer,

        max_seq_length=MAX_SEQ_LENGTH,

        doc_stride=128,

        max_query_length=64,

        is_training=False,

        output_fn=append_feature)



eval_writer.close()



tf.logging.info("***** Running predictions *****")

tf.logging.info("  Num orig examples = %d", len(predict_examples))

tf.logging.info("  Num features = %d", len(eval_features))



predict_input_fn = run_squad.input_fn_builder(

    input_file=eval_writer.filename,

    seq_length=MAX_SEQ_LENGTH,

    is_training=False,

    drop_remainder=False)
all_results = []

for result in estimator.predict(predict_input_fn, yield_single_examples=True):

    if len(all_results) % 1000 == 0:

        tf.logging.info("Processing example: %d" % (len(all_results)))

    unique_id = int(result["unique_ids"])

    start_logits = [float(x) for x in result["start_logits"].flat]

    end_logits = [float(x) for x in result["end_logits"].flat]

    all_results.append(

          run_squad.RawResult(

              unique_id=unique_id,

              start_logits=start_logits,

              end_logits=end_logits))



output_prediction_file = os.path.join(OUTPUT_DIR, "predictions.json")

output_nbest_file = os.path.join(OUTPUT_DIR, "nbest_predictions.json")

output_null_log_odds_file = os.path.join(OUTPUT_DIR, "null_odds.json")



def write_predictions(all_examples, all_features, all_results, n_best_size,

                      max_answer_length, do_lower_case, output_prediction_file,

                      output_nbest_file, output_null_log_odds_file):

  """Write final predictions to the json file and log-odds of null if needed."""

  tf.logging.info("Writing predictions to: %s" % (output_prediction_file))

  tf.logging.info("Writing nbest to: %s" % (output_nbest_file))



  example_index_to_features = collections.defaultdict(list)

  for feature in all_features:

    example_index_to_features[feature.example_index].append(feature)



  unique_id_to_result = {}

  for result in all_results:

    unique_id_to_result[result.unique_id] = result



  _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name

      "PrelimPrediction",

      ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])



  all_predictions = collections.OrderedDict()

  all_nbest_json = collections.OrderedDict()

  scores_diff_json = collections.OrderedDict()



  for (example_index, example) in enumerate(all_examples):

    features = example_index_to_features[example_index]



    prelim_predictions = []

    # keep track of the minimum score of null start+end of position 0

    score_null = 1000000  # large and positive

    min_null_feature_index = 0  # the paragraph slice with min mull score

    null_start_logit = 0  # the start logit at the slice with min null score

    null_end_logit = 0  # the end logit at the slice with min null score

    for (feature_index, feature) in enumerate(features):

      result = unique_id_to_result[feature.unique_id]

      start_indexes = _get_best_indexes(result.start_logits, n_best_size)

      end_indexes = _get_best_indexes(result.end_logits, n_best_size)

      # if we could have irrelevant answers, get the min score of irrelevant

      for start_index in start_indexes:

        for end_index in end_indexes:

          # We could hypothetically create invalid predictions, e.g., predict

          # that the start of the span is in the question. We throw out all

          # invalid predictions.

          if start_index >= len(feature.tokens):

            continue

          if end_index >= len(feature.tokens):

            continue

          if start_index not in feature.token_to_orig_map:

            continue

          if end_index not in feature.token_to_orig_map:

            continue

          if not feature.token_is_max_context.get(start_index, False):

            continue

          if end_index < start_index:

            continue

          length = end_index - start_index + 1

          if length > max_answer_length:

            continue

          prelim_predictions.append(

              _PrelimPrediction(

                  feature_index=feature_index,

                  start_index=start_index,

                  end_index=end_index,

                  start_logit=result.start_logits[start_index],

                  end_logit=result.end_logits[end_index]))



    prelim_predictions = sorted(

        prelim_predictions,

        key=lambda x: (x.start_logit + x.end_logit),

        reverse=True)



    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name

        "NbestPrediction", ["text", "start_logit", "end_logit"])



    seen_predictions = {}

    nbest = []

    for pred in prelim_predictions:

      if len(nbest) >= n_best_size:

        break

      feature = features[pred.feature_index]

      if pred.start_index > 0:  # this is a non-null prediction

        tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]

        orig_doc_start = feature.token_to_orig_map[pred.start_index]

        orig_doc_end = feature.token_to_orig_map[pred.end_index]

        orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]

        tok_text = " ".join(tok_tokens)



        # De-tokenize WordPieces that have been split off.

        tok_text = tok_text.replace(" ##", "")

        tok_text = tok_text.replace("##", "")



        # Clean whitespace

        tok_text = tok_text.strip()

        tok_text = " ".join(tok_text.split())

        orig_text = " ".join(orig_tokens)



        final_text = run_squad.get_final_text(tok_text, orig_text, do_lower_case)

        if final_text in seen_predictions:

          continue



        seen_predictions[final_text] = True

      else:

        final_text = ""

        seen_predictions[final_text] = True



      nbest.append(

          _NbestPrediction(

              text=final_text,

              start_logit=pred.start_logit,

              end_logit=pred.end_logit))



    # In very rare edge cases we could have no valid predictions. So we

    # just create a nonce prediction in this case to avoid failure.

    if not nbest:

      nbest.append(

          _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))



    assert len(nbest) >= 1



    total_scores = []

    best_non_null_entry = None

    for entry in nbest:

      total_scores.append(entry.start_logit + entry.end_logit)

      if not best_non_null_entry:

        if entry.text:

          best_non_null_entry = entry



    probs = run_squad._compute_softmax(total_scores)



    nbest_json = []

    for (i, entry) in enumerate(nbest):

      output = collections.OrderedDict()

      output["text"] = entry.text

      output["probability"] = probs[i]

      output["start_logit"] = entry.start_logit

      output["end_logit"] = entry.end_logit

      nbest_json.append(output)



    assert len(nbest_json) >= 1



    all_predictions[example.qas_id] = nbest_json[0]["text"]



    all_nbest_json[example.qas_id] = nbest_json



  with tf.gfile.GFile(output_prediction_file, "w") as writer:

    writer.write(json.dumps(all_predictions, indent=4) + "\n")



  with tf.gfile.GFile(output_nbest_file, "w") as writer:

    writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    



def _get_best_indexes(logits, n_best_size):

  """Get the n-best logits from a list."""

  index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)



  best_indexes = []

  for i in range(len(index_and_score)):

    if i >= n_best_size:

      break

    best_indexes.append(index_and_score[i][0])

  return best_indexes



import collections

import json



write_predictions(eval_examples, eval_features, all_results,

                      20, 30,

                      True, output_prediction_file,

                      output_nbest_file, output_null_log_odds_file)



train_df_full[train_df_full.ID == "development-1446"]
!head model_repo/outputs/predictions.json
from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score



print(accuracy_score(np.array(results), test_labels))

print(f1_score(np.array(results), test_labels))