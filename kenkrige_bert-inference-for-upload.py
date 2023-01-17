!cp ../input/bert_files/* .
import collections

import os

import re

import pandas as pd

import modeling

import optimization

import tokenization

import tensorflow as tf

tf.reset_default_graph()
df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

# Remove next line for inference on whole set

df = df[:1000]
class FLAGS(object):

    bert_config_file = '../input/bert_files/bert_config.json'

    vocab_file = '../input/bert_files/vocab.txt'

    model_dir = "../input/toxic1"

    do_lower_case = True,

    max_seq_length = 128



class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, label_id, is_real_example=True):

        self.input_ids = input_ids

        self.input_mask = input_mask

        self.segment_ids = segment_ids

        self.label_id = label_id
def convert_single_example(ex_index, text, label_list, max_seq_length, tokenizer):



    tokens_a = tokenizer.tokenize(text)



    if len(tokens_a) > max_seq_length - 2:

        offset = int(max_seq_length / 2) - 1

        tokens_a = tokens_a[:offset] + tokens_a[-offset:]



    tokens = ["[CLS]"] + tokens_a + ['[SEP]']

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)



    while len(input_ids) < max_seq_length:

        input_ids.append(0)

        input_mask.append(0)

    segment_ids = [0] * len(input_ids)



    if ex_index < 3:

        tf.logging.info("*** Examples ***")

        tf.logging.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))

        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))

        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))

        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))



    feature = InputFeatures(

      input_ids=input_ids,

      input_mask=input_mask,

      segment_ids=segment_ids,

      label_id=0

    #, is_real_example=True

    )

    return feature
def file_based_convert_examples_to_features(df, label_list, max_seq_length, tokenizer, output_file):



    writer = tf.python_io.TFRecordWriter(output_file)

    print(len(df))

    for ex_index in range(len(df)):

        if ex_index % 5000 == 0:

            tf.logging.info("Writing example %d of %d" % (ex_index, len(df)))

        row = df.iloc[ex_index]

        feature = convert_single_example(ex_index, row['comment_text'], label_list,

                                         max_seq_length, tokenizer)



        def create_int_feature(values):

            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

            return f



        features = collections.OrderedDict()

        features["input_ids"] = create_int_feature(feature.input_ids)

        features["input_mask"] = create_int_feature(feature.input_mask)

        features["segment_ids"] = create_int_feature(feature.segment_ids)

        features["label_ids"] = create_int_feature([feature.label_id])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writer.write(tf_example.SerializeToString())

    writer.close()
def file_based_input_fn_builder(input_file, seq_length):



    name_to_features = {

      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),

      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),

      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),

      "label_ids": tf.FixedLenFeature([], tf.int64),

    }



    def input_fn(params):

        d = tf.data.TFRecordDataset(input_file)



        d = d.apply(

            tf.contrib.data.map_and_batch(

                lambda record: tf.parse_single_example(record, name_to_features),

                batch_size=8,

                drop_remainder=False))

        return d



    return input_fn

def create_infer_model(input_ids, input_mask, segment_ids, labels, num_labels):



    model = modeling.BertModel(

      config=modeling.BertConfig.from_json_file(FLAGS.bert_config_file),

      is_training=False,

      input_ids=input_ids,

      input_mask=input_mask,

      token_type_ids=segment_ids,

      use_one_hot_embeddings=False)



    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value



    output_weights = tf.get_variable(

      "output_weights", [num_labels, hidden_size],

      initializer=tf.truncated_normal_initializer(stddev=0.02))



    output_bias = tf.get_variable(

      "output_bias", [num_labels], initializer=tf.zeros_initializer())



    logits = tf.matmul(output_layer, output_weights, transpose_b=True)

    logits = tf.nn.bias_add(logits, output_bias)

    probabilities = tf.nn.sigmoid(logits)

    return probabilities
def infer_fn_builder(num_labels):



    def model_fn(features):

        input_ids = features["input_ids"]

        input_mask = features["input_mask"]

        segment_ids = features["segment_ids"]

        label_ids = features["label_ids"]

        is_real_example = None



        probabilities = create_infer_model(

            input_ids, input_mask, segment_ids, label_ids, num_labels)



        output_spec = tf.estimator.EstimatorSpec(

            predictions={"probabilities": probabilities},

            mode='infer'

        )

        return output_spec



    return model_fn
import time

tic = time.time()

tf.logging.set_verbosity(tf.logging.INFO)

label_list = ["0", "1"]



tokenizer = tokenization.FullTokenizer(

  vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)



model_fn = infer_fn_builder(num_labels=len(label_list))



estimator = tf.estimator.Estimator(

  model_fn=model_fn,

  config=tf.estimator.RunConfig(model_dir=FLAGS.model_dir)

)



if not os.path.isfile("predict.tf_record"):

    file_based_convert_examples_to_features(df, label_list, FLAGS.max_seq_length, 

                                            tokenizer, "predict.tf_record")



predict_input_fn = file_based_input_fn_builder("predict.tf_record", FLAGS.max_seq_length)



result = estimator.predict(input_fn=predict_input_fn)



predictions = []

for pred in result:

    predictions.append(pred['probabilities'])



print(f'{len(predictions)} records done in {time.time() - tic}s')

print(predictions[:10])
out = pd.DataFrame(predictions)

out.columns = ['civil','toxic']

df['prediction'] = out['toxic']

submission = df[['id', 'prediction']]

submission.to_csv('submission.csv', index=False)

submission.head(10)