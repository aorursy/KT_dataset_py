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
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

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
#import tokenization

#import modeling



#https://towardsdatascience.com/building-a-multi-label-text-classifier-using-bert-and-tensorflow-f188e0ecdc5d

#BERT_VOCAB= '../input/uncased-l12-h768-a12/vocab.txt'

#BERT_INIT_CHKPNT = '../input/uncased-l12-h768-a12/bert_model.ckpt'

#BERT_CONFIG = '../input/uncased-l12-h768-a12/bert_config.json'
#tokenization.validate_case_matches_checkpoint(True,BERT_INIT_CHKPNT)

#tokenizer = tokenization.FullTokenizer(

#      vocab_file=BERT_VOCAB, do_lower_case=True)
train_data_path='../input/jigsaw-toxic-comment-classification-challenge/train.csv'

train = pd.read_csv(train_data_path)

test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')
train.head()
ID = 'id'

DATA_COLUMN = 'comment_text'

LABEL_COLUMNS = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
class InputExample(object):

    """A single training/test example for simple sequence classification."""



    def __init__(self, guid, text_a, text_b=None, labels=None):

        """Constructs a InputExample.



        Args:

            guid: Unique id for the example.

            text_a: string. The untokenized text of the first sequence. For single

            sequence tasks, only this sequence must be specified.

            text_b: (Optional) string. The untokenized text of the second sequence.

            Only must be specified for sequence pair tasks.

            labels: (Optional) [string]. The label of the example. This should be

            specified for train and dev examples, but not for test examples.

        """

        self.guid = guid

        self.text_a = text_a

        self.text_b = text_b

        self.labels = labels





class InputFeatures(object):

    """A single set of features of data."""



    def __init__(self, input_ids, input_mask, segment_ids, label_ids, is_real_example=True):

        self.input_ids = input_ids

        self.input_mask = input_mask

        self.segment_ids = segment_ids

        self.label_ids = label_ids,

        self.is_real_example=is_real_example
def create_examples(df, labels_available=True):

    """Creates examples for the training and dev sets."""

    examples = []

    for (i, row) in enumerate(df.values):

        guid = row[0]

        text_a = row[1]

        if labels_available:

            labels = row[2:]

        else:

            labels = [0,0,0,0,0,0]

        examples.append(

            InputExample(guid=guid, text_a=text_a, labels=labels))

    return examples
TRAIN_VAL_RATIO = 0.9

LEN = train.shape[0]

SIZE_TRAIN = int(TRAIN_VAL_RATIO*LEN)



x_train = train[:SIZE_TRAIN]

x_val = train[SIZE_TRAIN:]



# Use the InputExample class from BERT's run_classifier code to create examples from the data

train_examples = create_examples(x_train)
train.shape, x_train.shape, x_val.shape
import pandas



def convert_examples_to_features(examples,  max_seq_length, tokenizer):

    """Loads a data file into a list of `InputBatch`s."""



    features = []

    for (ex_index, example) in enumerate(examples):

        print(example.text_a)

        tokens_a = tokenizer.tokenize(example.text_a)



        tokens_b = None

        if example.text_b:

            tokens_b = tokenizer.tokenize(example.text_b)

            # Modifies `tokens_a` and `tokens_b` in place so that the total

            # length is less than the specified length.

            # Account for [CLS], [SEP], [SEP] with "- 3"

            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

        else:

            # Account for [CLS] and [SEP] with "- 2"

            if len(tokens_a) > max_seq_length - 2:

                tokens_a = tokens_a[:(max_seq_length - 2)]



        # The convention in BERT is:

        # (a) For sequence pairs:

        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]

        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1

        # (b) For single sequences:

        #  tokens:   [CLS] the dog is hairy . [SEP]

        #  type_ids: 0   0   0   0  0     0 0

        #

        # Where "type_ids" are used to indicate whether this is the first

        # sequence or the second sequence. The embedding vectors for `type=0` and

        # `type=1` were learned during pre-training and are added to the wordpiece

        # embedding vector (and position vector). This is not *strictly* necessary

        # since the [SEP] token unambigiously separates the sequences, but it makes

        # it easier for the model to learn the concept of sequences.

        #

        # For classification tasks, the first vector (corresponding to [CLS]) is

        # used as as the "sentence vector". Note that this only makes sense because

        # the entire model is fine-tuned.

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]

        segment_ids = [0] * len(tokens)



        if tokens_b:

            tokens += tokens_b + ["[SEP]"]

            segment_ids += [1] * (len(tokens_b) + 1)



        input_ids = tokenizer.convert_tokens_to_ids(tokens)



        # The mask has 1 for real tokens and 0 for padding tokens. Only real

        # tokens are attended to.

        input_mask = [1] * len(input_ids)



        # Zero-pad up to the sequence length.

        padding = [0] * (max_seq_length - len(input_ids))

        input_ids += padding

        input_mask += padding

        segment_ids += padding



        assert len(input_ids) == max_seq_length

        assert len(input_mask) == max_seq_length

        assert len(segment_ids) == max_seq_length

        

        labels_ids = []

        for label in example.labels:

            labels_ids.append(int(label))



        if ex_index < 0:

            logger.info("*** Example ***")

            logger.info("guid: %s" % (example.guid))

            logger.info("tokens: %s" % " ".join(

                    [str(x) for x in tokens]))

            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))

            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))

            logger.info(

                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

            logger.info("label: %s (id = %s)" % (example.labels, labels_ids))



        features.append(

                InputFeatures(input_ids=input_ids,

                              input_mask=input_mask,

                              segment_ids=segment_ids,

                              label_ids=labels_ids))

    return features
# We'll set sequences to be at most 128 tokens long.

MAX_SEQ_LENGTH = 128
def create_model(is_training, input_ids, input_mask, segment_ids,labels, num_labels, use_one_hot_embeddings):

    bert_module = hub.Module( BERT_MODEL_HUB, trainable=True)

    bert_inputs = dict(input_ids=input_ids,input_mask=input_mask,segment_ids=segment_ids)

    bert_outputs = bert_module(inputs=bert_inputs, signature="tokens", as_dict=True)



    # Use "pooled_output" for classification tasks on an entire sentence.

    # Use "sequence_outputs" for token-level output.

    output_layer = bert_outputs["pooled_output"]



    hidden_size = output_layer.shape[-1].value





    output_weights = tf.get_variable( "output_weights", [num_labels, hidden_size], initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable("output_bias", [num_labels], initializer=tf.zeros_initializer())







    with tf.variable_scope("loss"):

        if is_training:

            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)

        logits = tf.nn.bias_add(logits, output_bias)

        probabilities = tf.nn.sigmoid(logits)

        

        labels = tf.cast(labels, tf.float32)

        tf.logging.info("num_labels:{};logits:{};labels:{}".format(num_labels, logits, labels))

        per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

        loss = tf.reduce_mean(per_example_loss)

        

        return (loss, per_example_loss, logits, probabilities)




def model_fn_builder(num_labels, learning_rate,

                     num_train_steps, num_warmup_steps, use_tpu,

                     use_one_hot_embeddings):

    """Returns `model_fn` closure for TPUEstimator."""



    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument

        """The `model_fn` for TPUEstimator."""



        #tf.logging.info("*** Features ***")

        #for name in sorted(features.keys()):

        #    tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))



        input_ids = features["input_ids"]

        input_mask = features["input_mask"]

        segment_ids = features["segment_ids"]

        label_ids = features["label_ids"]

        is_real_example = None

        if "is_real_example" in features:

             is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)

        else:

             is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)



        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

        is_eval = (mode == tf.estimator.ModeKeys.EVAL)

        

        (total_loss, per_example_loss, logits, probabilities) = create_model(is_training, input_ids, input_mask, segment_ids, label_ids, num_labels, use_one_hot_embeddings)

        

        output_spec = None

        

        if is_training:

            train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu = False)

            output_spec = tf.estimator.EstimatorSpec(

                    mode=mode,

                    loss=total_loss,

                    train_op=train_op,

                    )

            

        elif is_eval:

            

            def metric_fn(per_example_loss, label_ids, probabilities, is_real_example):



                logits_split = tf.split(probabilities, num_labels, axis=-1)

                label_ids_split = tf.split(label_ids, num_labels, axis=-1)

                # metrics change to auc of every class

                eval_dict = {}

                for j, logits in enumerate(logits_split):

                    label_id_ = tf.cast(label_ids_split[j], dtype=tf.int32)

                    current_auc, update_op_auc = tf.metrics.auc(label_id_, logits)

                    eval_dict[str(j)] = (current_auc, update_op_auc)

                eval_dict['eval_loss'] = tf.metrics.mean(values=per_example_loss)

                return eval_dict



            eval_metrics = metric_fn(per_example_loss, label_ids, probabilities, is_real_example)

            output_spec = tf.estimator.EstimatorSpec(

                mode=mode,

                loss=total_loss,

                eval_metric_ops=eval_metrics,

                )

            

        else:

            output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions={"probabilities": probabilities})

        

        return output_spec





    return model_fn
# Compute train and warmup steps from batch size

# These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)

BATCH_SIZE = 32

LEARNING_RATE = 2e-5

NUM_TRAIN_EPOCHS = 2.0



# Warmup is a period of time where hte learning rate 

# is small and gradually increases--usually helps training.

WARMUP_PROPORTION = 0.1

# Model configs

SAVE_CHECKPOINTS_STEPS = 1000

SAVE_SUMMARY_STEPS = 500
OUTPUT_DIR = "../working/output"

# Specify outpit directory and number of checkpoint steps to save

run_config = tf.estimator.RunConfig(

    model_dir=OUTPUT_DIR,

    save_summary_steps=SAVE_SUMMARY_STEPS,

    keep_checkpoint_max=1,

    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)
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

    all_label_ids.append(feature.label_ids)



  def input_fn(params):

    """The actual input function."""

    batch_size = params["batch_size"]



    num_examples = len(features)



    # This is for demo purposes and does NOT scale to large data sets. We do

    # not use Dataset.from_generator() because that uses tf.py_func which is

    # not TPU compatible. The right way to load data is with TFRecordReader.

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

            tf.constant(all_label_ids, shape=[num_examples, len(LABEL_COLUMNS)], dtype=tf.int32),

    })



    if is_training:

      d = d.repeat()

      d = d.shuffle(buffer_size=100)



    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)

    return d



  return input_fn

class PaddingInputExample(object):

    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples

    to be a multiple of the batch size, because the TPU requires a fixed batch

    size. The alternative is to drop the last batch, which is bad because it means

    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding

    battches could cause silent errors.

    """
def convert_single_example(ex_index, example, max_seq_length,

                           tokenizer):

    """Converts a single `InputExample` into a single `InputFeatures`."""



    if isinstance(example, PaddingInputExample):

        return InputFeatures(

            input_ids=[0] * max_seq_length,

            input_mask=[0] * max_seq_length,

            segment_ids=[0] * max_seq_length,

            label_ids=0,

            is_real_example=False)



    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None

    if example.text_b:

        tokens_b = tokenizer.tokenize(example.text_b)



    if tokens_b:

        # Modifies `tokens_a` and `tokens_b` in place so that the total

        # length is less than the specified length.

        # Account for [CLS], [SEP], [SEP] with "- 3"

        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

    else:

        # Account for [CLS] and [SEP] with "- 2"

        if len(tokens_a) > max_seq_length - 2:

            tokens_a = tokens_a[0:(max_seq_length - 2)]



    # The convention in BERT is:

    # (a) For sequence pairs:

    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]

    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1

    # (b) For single sequences:

    #  tokens:   [CLS] the dog is hairy . [SEP]

    #  type_ids: 0     0   0   0  0     0 0

    #

    # Where "type_ids" are used to indicate whether this is the first

    # sequence or the second sequence. The embedding vectors for `type=0` and

    # `type=1` were learned during pre-training and are added to the wordpiece

    # embedding vector (and position vector). This is not *strictly* necessary

    # since the [SEP] token unambiguously separates the sequences, but it makes

    # it easier for the model to learn the concept of sequences.

    #

    # For classification tasks, the first vector (corresponding to [CLS]) is

    # used as the "sentence vector". Note that this only makes sense because

    # the entire model is fine-tuned.

    tokens = []

    segment_ids = []

    tokens.append("[CLS]")

    segment_ids.append(0)

    for token in tokens_a:

        tokens.append(token)

        segment_ids.append(0)

    tokens.append("[SEP]")

    segment_ids.append(0)



    if tokens_b:

        for token in tokens_b:

            tokens.append(token)

            segment_ids.append(1)

        tokens.append("[SEP]")

        segment_ids.append(1)



    input_ids = tokenizer.convert_tokens_to_ids(tokens)



    # The mask has 1 for real tokens and 0 for padding tokens. Only real

    # tokens are attended to.

    input_mask = [1] * len(input_ids)



    # Zero-pad up to the sequence length.

    while len(input_ids) < max_seq_length:

        input_ids.append(0)

        input_mask.append(0)

        segment_ids.append(0)



    assert len(input_ids) == max_seq_length

    assert len(input_mask) == max_seq_length

    assert len(segment_ids) == max_seq_length



    labels_ids = []

    for label in example.labels:

        labels_ids.append(int(label))





    feature = InputFeatures(

        input_ids=input_ids,

        input_mask=input_mask,

        segment_ids=segment_ids,

        label_ids=labels_ids,

        is_real_example=True)

    return feature





def file_based_convert_examples_to_features(

        examples, max_seq_length, tokenizer, output_file):

    """Convert a set of `InputExample`s to a TFRecord file."""



    writer = tf.python_io.TFRecordWriter(output_file)



    for (ex_index, example) in enumerate(examples):

        #if ex_index % 10000 == 0:

            #tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))



        feature = convert_single_example(ex_index, example,

                                         max_seq_length, tokenizer)



        def create_int_feature(values):

            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

            return f



        features = collections.OrderedDict()

        features["input_ids"] = create_int_feature(feature.input_ids)

        features["input_mask"] = create_int_feature(feature.input_mask)

        features["segment_ids"] = create_int_feature(feature.segment_ids)

        features["is_real_example"] = create_int_feature(

            [int(feature.is_real_example)])

        if isinstance(feature.label_ids, list):

            label_ids = feature.label_ids

        else:

            label_ids = feature.label_ids[0]

        features["label_ids"] = create_int_feature(label_ids)



        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writer.write(tf_example.SerializeToString())

    writer.close()





def file_based_input_fn_builder(input_file, seq_length, is_training,

                                drop_remainder):

    """Creates an `input_fn` closure to be passed to TPUEstimator."""



    name_to_features = {

        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),

        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),

        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),

        "label_ids": tf.FixedLenFeature([6], tf.int64),

        "is_real_example": tf.FixedLenFeature([], tf.int64),

    }



    def _decode_record(record, name_to_features):

        """Decodes a record to a TensorFlow example."""

        example = tf.parse_single_example(record, name_to_features)



        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.

        # So cast all int64 to int32.

        for name in list(example.keys()):

            t = example[name]

            if t.dtype == tf.int64:

                t = tf.to_int32(t)

            example[name] = t



        return example



    def input_fn(params):

        """The actual input function."""

        batch_size = params["batch_size"]



        # For training, we want a lot of parallel reading and shuffling.

        # For eval, we want no shuffling and parallel reading doesn't matter.

        d = tf.data.TFRecordDataset(input_file)

        if is_training:

            d = d.repeat()

            d = d.shuffle(buffer_size=100)



        d = d.apply(

            tf.contrib.data.map_and_batch(

                lambda record: _decode_record(record, name_to_features),

                batch_size=batch_size,

                drop_remainder=drop_remainder))



        return d



    return input_fn





def _truncate_seq_pair(tokens_a, tokens_b, max_length):

    """Truncates a sequence pair in place to the maximum length."""



    # This is a simple heuristic which will always truncate the longer sequence

    # one token at a time. This makes more sense than truncating an equal percent

    # of tokens from each, since if one sequence is very short then each token

    # that's truncated likely contains more information than a longer sequence.

    while True:

        total_length = len(tokens_a) + len(tokens_b)

        if total_length <= max_length:

            break

        if len(tokens_a) > len(tokens_b):

            tokens_a.pop()

        else:

            tokens_b.pop()
#from pathlib import Path

train_file = os.path.join('../working', "train.tf_record")

#filename = Path(train_file)

if not os.path.exists(train_file):

    open(train_file, 'w').close()
# Compute # train and warmup steps from batch size

num_train_steps = int(len(train_examples) / BATCH_SIZE * NUM_TRAIN_EPOCHS)

num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
file_based_convert_examples_to_features(

            train_examples, MAX_SEQ_LENGTH, tokenizer, train_file)

tf.logging.info("***** Running training *****")

tf.logging.info("  Num examples = %d", len(train_examples))

tf.logging.info("  Batch size = %d", BATCH_SIZE)

tf.logging.info("  Num steps = %d", num_train_steps)

train_input_fn = file_based_input_fn_builder(

    input_file=train_file,

    seq_length=MAX_SEQ_LENGTH,

    is_training=True,

    drop_remainder=True)
model_fn = model_fn_builder(

    num_labels=len(LABEL_COLUMNS),

    learning_rate=LEARNING_RATE,

    num_train_steps=num_train_steps,

    num_warmup_steps=num_warmup_steps, 

    use_tpu=False,

    use_one_hot_embeddings=False)



estimator = tf.estimator.Estimator(

    model_fn=model_fn,

    config=run_config,

    params={"batch_size": BATCH_SIZE})
print(f'Beginning Training!')

current_time = datetime.now()

estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)



print("Training took time ", datetime.now() - current_time)
eval_file = os.path.join('../working', "eval.tf_record")

#filename = Path(train_file)

if not os.path.exists(eval_file):

    open(eval_file, 'w').close()



eval_examples = create_examples(x_val)

file_based_convert_examples_to_features(

    eval_examples, MAX_SEQ_LENGTH, tokenizer, eval_file)
# This tells the estimator to run through the entire set.

eval_steps = None



eval_drop_remainder = False

eval_input_fn = file_based_input_fn_builder(

    input_file=eval_file,

    seq_length=MAX_SEQ_LENGTH,

    is_training=False,

    drop_remainder=False)



result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
output_eval_file = os.path.join("../working", "eval_results.txt")

with tf.gfile.GFile(output_eval_file, "w") as writer:

    tf.logging.info("***** Eval results *****")

    for key in sorted(result.keys()):

        tf.logging.info("  %s = %s", key, str(result[key]))

        writer.write("%s = %s\n" % (key, str(result[key])))
x_test = test#[125000:140000]

x_test = x_test.reset_index(drop=True)



test_file = os.path.join('../working', "test.tf_record")

#filename = Path(train_file)

if not os.path.exists(test_file):

    open(test_file, 'w').close()



test_examples = create_examples(x_test, False)

file_based_convert_examples_to_features(test_examples, MAX_SEQ_LENGTH, tokenizer, test_file)

#test_features = convert_examples_to_features(test_examples, MAX_SEQ_LENGTH, tokenizer)
predict_input_fn = file_based_input_fn_builder(

    input_file=test_file,

    seq_length=MAX_SEQ_LENGTH,

    is_training=False,

    drop_remainder=False)
print('Begin predictions!')

current_time = datetime.now()

predictions = estimator.predict(predict_input_fn)

print("Predicting took time ", datetime.now() - current_time)
predictions
type(predictions)
def create_output(predictions):

    probabilities = []

    for (i, prediction) in enumerate(predictions):

        preds = prediction["probabilities"]

        probabilities.append(preds)

    dff = pd.DataFrame(probabilities)

    dff.columns = LABEL_COLUMNS

    

    return dff

        
output_df = create_output(predictions)


merged_df =  pd.concat([x_test, output_df], axis=1)

submission = merged_df.drop(['comment_text'], axis=1)

submission.to_csv("sample_submission.csv", index=False)
submission.tail()