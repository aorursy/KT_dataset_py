# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import collections

print(os.listdir("../working/"))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split

import pandas as pd

import tensorflow as tf

import tensorflow_hub as hub

from datetime import datetime
!pip install bert-tensorflow
import bert

from bert import run_classifier

from bert import optimization

from bert import tokenization

from bert import modeling
train_df = pd.read_csv('/kaggle/input/av-junta-hackathon-3/train_E52nqFa/train.csv')

test_df = pd.read_csv('/kaggle/input/av-junta-hackathon-3/test_BppAoe0/test.csv')

game_df = pd.read_csv('/kaggle/input/av-junta-hackathon-3/train_E52nqFa/game_overview.csv')

print(train_df.shape, test_df.shape, game_df.shape)

#train_df.head()
train_df.iloc[100]['user_review']
tags = game_df.tags

tags = [i.strip('[') for i in tags]

tags = [i.strip(']') for i in tags]

tags = [i.strip() for i in tags]

tags = [i.split(',') for i in tags]

for i in range(len(tags)):

    tags[i] = [j.strip() for j in tags[i]]

    tags[i] = [i.strip("'") for i in tags[i]]

    tags[i] = [i.strip("'") for i in tags[i]]

    tags[i] = " ".join(tags[i])

game_df.tags = tags

game_df['tags'] = game_df['tags'].astype('str')

game_df['game'] = [str(game_df.iloc[x]['overview'])+ str(" ") +  str(game_df.iloc[x]['tags']) for x in game_df.index]
game_df.iloc[10]['tags']
def clean(string, sstring):

    for i in sstring:

        if string.startswith(i):

            return string[(len(i)):]

    return string

junk = ['Early Access Review', 'TL;DR', 'Access Review', 'Product received for freeEarly Access Review', 'Product received for free']

train_df['user_review'] = [clean(train_df.iloc[x]['user_review'], junk) for x in train_df.index]

test_df['user_review'] = [clean(test_df.iloc[x]['user_review'], junk) for x in test_df.index]
train_df.iloc[100]['user_review']
train_df = pd.merge(train_df, game_df, how = 'left', on = 'title')

test_df = pd.merge(test_df, game_df, how = 'left', on = 'title')

print(train_df.shape, test_df.shape)
train_df['user_suggestion'] = train_df['user_suggestion'].astype('int')

train_df['data'] = [train_df.iloc[x]['user_review'] + " " + train_df.iloc[x]['tags'] for x in train_df.index]

test_df['data'] = [str(test_df.iloc[x]['user_review'])+ str(" ") + str(test_df.iloc[x]['tags']) for x in test_df.index]
train_df.iloc[100]['data']
from sklearn.model_selection import train_test_split

train, test = train_test_split(train_df, test_size=0.2, random_state=42, stratify = train_df.user_suggestion)

print(test.shape, train.shape)
DATA_COLUMN = 'data'

LABEL_COLUMN = 'user_suggestion'

# label_list is the list of labels, i.e. True, False or 0, 1 or 'dog', 'cat'

label_list = [0, 1]
# Use the InputExample class from BERT's run_classifier code to create examples from the data

train_InputExamples = train.apply(lambda x: bert.run_classifier.InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this example

                                                                   text_a = x[DATA_COLUMN], 

                                                                   text_b = None, 

                                                                   label = x[LABEL_COLUMN]), axis = 1)
test_InputExamples = test.apply(lambda x: bert.run_classifier.InputExample(guid=None, 

                                                                   text_a = x[DATA_COLUMN], 

                                                                   text_b = None, 

                                                                   label = x[LABEL_COLUMN]), axis = 1)
# This is a path to an uncased (all lowercase) version of BERT

BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

#BERT_MODEL_HUB = "https://tfhub.dev/google/albert_base/3"



def create_tokenizer_from_hub_module():

  """Get the vocab file and casing info from the Hub module."""

  with tf.Graph().as_default():

    bert_module = hub.Module(BERT_MODEL_HUB)

    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)

    with tf.Session() as sess:

      vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],

                                            tokenization_info["do_lower_case"]])

      

  return bert.tokenization.FullTokenizer(

      vocab_file=vocab_file, do_lower_case=do_lower_case)



tokenizer = create_tokenizer_from_hub_module()
tokenizer.tokenize("This here's an example of using the BERT tokenizer")
MAX_SEQ_LENGTH = 400

# Convert our train and test features to InputFeatures that BERT understands.

train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)

test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,

                 num_labels):

  """Creates a classification model."""



  bert_module = hub.Module(

      BERT_MODEL_HUB,

      trainable=True)

  bert_inputs = dict(

      input_ids=input_ids,

      input_mask=input_mask,

      segment_ids=segment_ids)

  bert_outputs = bert_module(

      inputs=bert_inputs,

      signature="tokens",

      as_dict=True)



  # Use "pooled_output" for classification tasks on an entire sentence.

  # Use "sequence_outputs" for token-level output.

  output_layer = bert_outputs["pooled_output"]



  hidden_size = output_layer.shape[-1].value



  # Create our own layer to tune for politeness data.

  output_weights = tf.get_variable(

      "output_weights", [num_labels, hidden_size],

      initializer=tf.truncated_normal_initializer(stddev=0.02))



  output_bias = tf.get_variable(

      "output_bias", [num_labels], initializer=tf.zeros_initializer())



  with tf.variable_scope("loss"):



    # Dropout helps prevent overfitting

    output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)



    logits = tf.matmul(output_layer, output_weights, transpose_b=True)

    logits = tf.nn.bias_add(logits, output_bias)

    log_probs = tf.nn.log_softmax(logits, axis=-1)



    # Convert labels into one-hot encoding

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)



    predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))

    # If we're predicting, we want predicted labels and the probabiltiies.

    if is_predicting:

      return (predicted_labels, log_probs)



    # If we're train/eval, compute loss between predicted and actual label

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)

    loss = tf.reduce_mean(per_example_loss)

    return (loss, predicted_labels, log_probs)
# model_fn_builder actually creates our model function

# using the passed parameters for num_labels, learning_rate, etc.

def model_fn_builder(num_labels, learning_rate, num_train_steps,

                     num_warmup_steps):

  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument

    """The `model_fn` for TPUEstimator."""



    input_ids = features["input_ids"]

    input_mask = features["input_mask"]

    segment_ids = features["segment_ids"]

    label_ids = features["label_ids"]



    is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

    

    # TRAIN and EVAL

    if not is_predicting:



      (loss, predicted_labels, log_probs) = create_model(

        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)



      train_op = bert.optimization.create_optimizer(

          loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)



      # Calculate evaluation metrics. 

      def metric_fn(label_ids, predicted_labels):

        accuracy = tf.metrics.accuracy(label_ids, predicted_labels)

        f1_score = tf.contrib.metrics.f1_score(

            label_ids,

            predicted_labels)

        auc = tf.metrics.auc(

            label_ids,

            predicted_labels)

        recall = tf.metrics.recall(

            label_ids,

            predicted_labels)

        precision = tf.metrics.precision(

            label_ids,

            predicted_labels) 

        true_pos = tf.metrics.true_positives(

            label_ids,

            predicted_labels)

        true_neg = tf.metrics.true_negatives(

            label_ids,

            predicted_labels)   

        false_pos = tf.metrics.false_positives(

            label_ids,

            predicted_labels)  

        false_neg = tf.metrics.false_negatives(

            label_ids,

            predicted_labels)

        return {

            "eval_accuracy": accuracy,

            "f1_score": f1_score,

            "auc": auc,

            "precision": precision,

            "recall": recall,

            "true_positives": true_pos,

            "true_negatives": true_neg,

            "false_positives": false_pos,

            "false_negatives": false_neg

        }



      eval_metrics = metric_fn(label_ids, predicted_labels)



      if mode == tf.estimator.ModeKeys.TRAIN:

        return tf.estimator.EstimatorSpec(mode=mode,

          loss=loss,

          train_op=train_op)

      else:

          return tf.estimator.EstimatorSpec(mode=mode,

            loss=loss,

            eval_metric_ops=eval_metrics)

    else:

      (predicted_labels, log_probs) = create_model(

        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)



      predictions = {

          'probabilities': log_probs,

          'labels': predicted_labels

      }

      return tf.estimator.EstimatorSpec(mode, predictions=predictions)



  # Return the actual model function in the closure

  return model_fn
def get_features(features, num_labels):

  """Creates a output layer."""

  input_ids = features["input_ids"]

  input_mask = features["input_mask"]

  segment_ids = features["segment_ids"]

  label_ids = features["label_ids"]



  bert_module = hub.Module(

      BERT_MODEL_HUB,

      trainable=True)

  bert_inputs = dict(

      input_ids=input_ids,

      input_mask=input_mask,

      segment_ids=segment_ids)

  bert_outputs = bert_module(

      inputs=bert_inputs,

      signature="tokens",

      as_dict=True)



  # Use "pooled_output" for classification tasks on an entire sentence.

  # Use "sequence_outputs" for token-level output.

  output_layer = bert_outputs["pooled_output"]

  return output_layer
import shutil

#shutil.rmtree("/kaggle/working/kaggle/output/")
# Compute train and warmup steps from batch size

# These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)

BATCH_SIZE = 32

LEARNING_RATE = 8e-6

NUM_TRAIN_EPOCHS = 5.0

# Warmup is a period of time where hte learning rate 

# is small and gradually increases--usually helps training.

WARMUP_PROPORTION = 0.1

# Model configs

SAVE_CHECKPOINTS_STEPS = 10000

SAVE_SUMMARY_STEPS = 5000
# Compute # train and warmup steps from batch size

num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)

num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
print(num_train_steps, num_warmup_steps)
model_fn = model_fn_builder(

  num_labels=len(label_list),

  learning_rate=LEARNING_RATE,

  num_train_steps=num_train_steps,

  num_warmup_steps=num_warmup_steps)
# Specify outpit directory and number of checkpoint steps to save

run_config = tf.estimator.RunConfig(

    model_dir='kaggle/output/',

    save_summary_steps=SAVE_SUMMARY_STEPS,

    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)
model_fn = model_fn_builder(

  num_labels=len(label_list),

  learning_rate=LEARNING_RATE,

  num_train_steps=num_train_steps,

  num_warmup_steps=num_warmup_steps)



estimator = tf.estimator.Estimator(

  model_fn=model_fn,

  config=run_config,

  params={"batch_size": BATCH_SIZE})
train_input_fn = bert.run_classifier.input_fn_builder(

    features=train_features,

    seq_length=MAX_SEQ_LENGTH,

    is_training=True,

    drop_remainder=False)
print(f'Beginning Training!')

current_time = datetime.now()

estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

print("Training took time ", datetime.now() - current_time)
test_input_fn = run_classifier.input_fn_builder(

    features=test_features,

    seq_length=MAX_SEQ_LENGTH,

    is_training=False,

    drop_remainder=False)
estimator.evaluate(input_fn=test_input_fn, steps=None)
def getPrediction(in_sentences):

  labels = ["Negative", "Positive"]

  input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = 0) for x in in_sentences] # here, "" is just a dummy label

  input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)

  predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)

  predictions = estimator.predict(predict_input_fn)

  return [(sentence, prediction['probabilities'], labels[prediction['labels']]) for sentence, prediction in zip(in_sentences, predictions)]
test_reviews = list(test_df.data)

predictions = getPrediction(test_reviews)
sub = pd.DataFrame(test_df.review_id)

preds = [i[2] for i in predictions]

preds = [1 if x=='Positive' else 0 for x in preds]
sub['user_suggestion'] = preds

sub.tail()
sub.to_csv("bert_clean_both_over_tag_re.csv", index=False)
train_reviews = list(train.data)

predictions = getPrediction(train_reviews)
new_train = pd.DataFrame(train.review_id)

preds = [i[1][0] for i in predictions]

new_train['positive'] = preds

preds = [i[1][1] for i in predictions]

new_train['negative'] = preds

new_train['user_suggestion'] = train.user_suggestion

new_train.head()
test_review = list(test.data)

predictions = getPrediction(test_review)
new_test = pd.DataFrame(test.review_id)

preds = [i[1][0] for i in predictions]

new_test['positive'] = preds

preds = [i[1][1] for i in predictions]

new_test['negative'] = preds

new_test['user_suggestion'] = test.user_suggestion
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth = 2, n_estimators = 50)

clf.fit(new_train[['positive', 'negative']], new_train.user_suggestion)
from sklearn.metrics import f1_score, confusion_matrix

print(f1_score(clf.predict(new_test[['positive', 'negative']]), new_test.user_suggestion))

confusion_matrix(clf.predict(new_test[['positive', 'negative']]), new_test.user_suggestion)
test_reviews = list(test_df.data)

predictions = getPrediction(test_reviews)
sub_test = pd.DataFrame(test_df.review_id)

preds = [i[1][0] for i in predictions]

sub_test['positive'] = preds

preds = [i[1][1] for i in predictions]

sub_test['negative'] = preds
sub['user'] = clf.predict(sub_test[['positive', 'negative']])
sub['user_review'] = test_df.user_review
sub.tail()
sub.to_csv("bert_clean_ensemble.csv", index=False)