from sklearn.model_selection import train_test_split

import pandas as pd

import tensorflow as tf

import tensorflow_hub as hub

from tensorflow import keras

import os

import re

import numpy as np
!pip install bert-tensorflow
# import bert related packages

import bert

from bert import modeling

from bert import run_classifier

from bert import tokenization
# prepare dataset

data = ['he is happy because he got a new car'

        ,'He is a lovable person'

        ,'john is a cheerful guy'

        ,'he was in a merry mood'

        ,'the whole crowd was joyful'

        ,'she is a loving person'

        ,'he was delighted to see me'

        ,'he was smiling at me when i got a new mobile'

        ,'he is in a jovial mood'

        ,'he was sad because his friend died'

        ,'he was unhappy with his dog'

        ,'the company has miserable status'

        ,'he was sorrowful as he lost all his money'

        ,'The bird was sorrowful as it had no food'

        ,'the dog was glum as his master was not at home'

        ,'he was in gloomy mood'

        ,'she was depressed as she lost her job'

        ,'do not be downhearted']

label = [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0]



zip_list = list(zip(data,label))

df = pd.DataFrame(zip_list, columns = ['sentence','polarity'])

train_df, test_df = train_test_split(df, test_size=0.1)



train_df.reset_index(drop=True, inplace=True)

test_df.reset_index(drop=True, inplace=True)



DATA_COLUMN = 'sentence'

LABEL_COLUMN = 'polarity'

# label_list is the list of labels, i.e. True, False or 0, 1 or 'dog', 'cat'

label_list = [0, 1]





print('Train data shape:', train_df.shape)

print('Test data shape:', test_df.shape)
# Use the InputExample class from BERT's run_classifier code to create examples from the data

train_InputExamples = train_df.apply(lambda x: bert.run_classifier.InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this example

                                                                   text_a = x[DATA_COLUMN], 

                                                                   text_b = None, 

                                                                   label = x[LABEL_COLUMN]), axis = 1)



test_InputExamples = test_df.apply(lambda x: bert.run_classifier.InputExample(guid=None, 

                                                                   text_a = x[DATA_COLUMN], 

                                                                   text_b = None, 

                                                                   label = x[LABEL_COLUMN]), axis = 1)

print(train_InputExamples[0].text_a)

print(train_InputExamples[0].label)
# preprocess our data so that it matches the data BERT was trained on



# This is a path to an uncased (all lowercase) version of BERT

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



# see what tokenizer does

print(tokenizer.tokenize("This here's an example of using the BERT tokenizer"))

print(tokenizer.tokenize(train_df['sentence'][0]))
# Using our tokenizer, we'll call run_classifier.convert_examples_to_features on our InputExamples to convert them into features BERT understands

MAX_SEQ_LENGTH = 20 # at most these many tokens long

# Convert our train and test features to InputFeatures that BERT understands.

train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)

test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
# Lets understand the InputFeatures data structures by printing values for a single data row.

print('Text:', train_InputExamples[0].text_a)

print('Tokens:', tokenizer.tokenize(train_InputExamples[0].text_a))

print('Vocab index (input_ids):', train_features[0].input_ids)

print('Get back the tokens from vocab index:', tokenizer.convert_ids_to_tokens(train_features[0].input_ids))

print('input_mask:', train_features[0].input_mask)

print('segment_ids:', train_features[0].segment_ids)
from tensorflow.keras import backend as K





class BertLayer(tf.keras.layers.Layer):

    def __init__(

        self,

        n_fine_tune_layers=10,

        pooling="first",

        bert_path=None,

        max_len = None,

        return_sequences=False,

        **kwargs,

    ):

        self.n_fine_tune_layers = n_fine_tune_layers

        self.trainable = True

        self.output_size = 768

        self.pooling = pooling

        self.bert_path = bert_path

        self.return_sequences = return_sequences

        self.output_key = 'sequence_output' if return_sequences else 'pooled_output'

        self.max_len = max_len

        if self.pooling not in ["first", "mean"]:

            raise NameError(

                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"

            )



        super(BertLayer, self).__init__(**kwargs)



    def build(self, input_shape):



        self.trainable = self.n_fine_tune_layers > 0

        self.bert = hub.Module(

            self.bert_path, trainable=self.trainable, name=f"{self.name}_module"

        )

        # Remove unused layers

        trainable_vars = self.bert.variables



        if self.pooling == "first":

            trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]

            trainable_layers = ["pooler/dense"]



        elif self.pooling == "mean":

            trainable_vars = [

                var

                for var in trainable_vars

                if not "/cls/" in var.name and not "/pooler/" in var.name

            ]

            trainable_layers = []

        else:

            raise NameError(

                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"

            )



        # Select how many layers to fine tune

        for i in range(self.n_fine_tune_layers):

            trainable_layers.append(f"encoder/layer_{str(11 - i)}")



        # Update trainable vars to contain only the specified layers

        trainable_vars = [

            var

            for var in trainable_vars

            if any([l in var.name for l in trainable_layers])

        ]



        # Add to trainable weights

        for var in trainable_vars:

            self._trainable_weights.append(var)



        for var in self.bert.variables:

            if var not in self._trainable_weights:

                self._non_trainable_weights.append(var)



        super(BertLayer, self).build(input_shape)



    def call(self, inputs, **kwargs):

        inputs = [K.cast(x, dtype="int32") for x in inputs]

        input_ids, input_mask, segment_ids = inputs

        bert_inputs = dict(

            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids

        )



        if self.pooling == "first":

            pooled = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[

                "pooled_output"

            ]

        elif self.pooling == "mean":

            result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[

                "sequence_output"

            ]



            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)

            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (

                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)

            input_mask = tf.cast(input_mask, tf.float32)

            pooled = masked_reduce_mean(result, input_mask)

        else:

            raise NameError(f"Undefined pooling type (must be either first or mean, but is {self.pooling}")



        sequence_output = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)["sequence_output"]

        sequence_output = tf.reshape(sequence_output, (-1, self.max_len, self.output_size))

        return [sequence_output, pooled]



    def compute_output_shape(self, input_shape):

        return [(input_shape[0], self.max_len, self.output_size),(input_shape[0], self.output_size)]
def build_model(bert_path, max_seq_length):

    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")

    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")

    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")

    bert_inputs = [in_id, in_mask, in_segment]



    bert_output = BertLayer(bert_path=bert_path, pooling='first', n_fine_tune_layers=0, max_len=MAX_SEQ_LENGTH)(bert_inputs)



    dense = tf.keras.layers.Dense(128, activation="relu")(bert_output[1])

    pred = tf.keras.layers.Dense(1, activation="sigmoid")(dense)



    model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.summary()



    return model

  

def initialize_vars(allow_growth=True):

    gpu_options = tf.GPUOptions(allow_growth=allow_growth)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    sess.run(tf.local_variables_initializer())

    sess.run(tf.global_variables_initializer())

    sess.run(tf.tables_initializer())

    K.set_session(sess)
# our model needs the following inputs

train_input_ids = []

train_input_mask = []

train_segment_ids = []

train_label_ids = []

for feature in train_features:

    train_input_ids.append(feature.input_ids)

    train_input_mask.append(feature.input_mask)

    train_segment_ids.append(feature.segment_ids)

    train_label_ids.append(feature.label_id)



test_input_ids = []

test_input_mask = []

test_segment_ids = []

test_label_ids = []

for feature in test_features:

    test_input_ids.append(feature.input_ids)

    test_input_mask.append(feature.input_mask)

    test_segment_ids.append(feature.segment_ids)

    test_label_ids.append(feature.label_id)
model = build_model(BERT_MODEL_HUB, MAX_SEQ_LENGTH)



initialize_vars()



train_inputs = [train_input_ids, train_input_mask, train_segment_ids]

train_labels = train_label_ids



history = model.fit(train_inputs, train_labels,

          validation_data=None,

          epochs=10,batch_size=2,shuffle=True )
def getPredictionFromSentence(in_sentences):

  labels = ["Negative", "Positive"]

  input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = 0) for x in in_sentences] # here, "" is just a dummy label

  input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)



  test_input_ids = []

  test_input_mask = []

  test_segment_ids = []

  test_label_ids = []

  for feature in input_features:

      test_input_ids.append(feature.input_ids)

      test_input_mask.append(feature.input_mask)

      test_segment_ids.append(feature.segment_ids)

      test_label_ids.append(feature.label_id)

  

  probabilities = getPredictionFromFeatures(test_input_ids, test_input_mask, test_segment_ids)

  # print(probabilities)

  predictions = (probabilities > 0.5).astype(np.int)

  # print(predictions)

  

  return [(sentence, proba, labels[prediction[0]]) for sentence, proba, prediction in zip(in_sentences, probabilities, predictions)]



def getPredictionFromFeatures(test_input_ids, test_input_mask, test_segment_ids):

  return model.predict([test_input_ids, test_input_mask, test_segment_ids])
# get little more test data

pred_sentences = [

  "That movie was absolutely awful",

  "The acting was a bit lacking",

  "The film was creative and surprising",

  "Absolutely fantastic!"

]



predictions = getPredictionFromSentence(pred_sentences)

print(predictions)



# test with our original test data

predictions = getPredictionFromSentence(test_df['sentence'].tolist())

print(predictions)
# collect all the layers of the tfhub bert model

# layers_of_bert = [i.values() for i in tf.get_default_graph().get_operations()]

# check their names

# print(layers_of_bert[-15:]) # there are just too many layersß



# picked some random layer

some_layer_name = 'bert_layer_module/bert/encoder/layer_3/attention/output/LayerNorm/moments/SquaredDifference:0'



# this helped: https://stackoverflow.com/questions/55333558/how-to-access-bert-intermediate-layer-outputs-in-tf-hub-module

some_layer_output = K.get_session().run(tf.get_default_graph().get_tensor_by_name(some_layer_name)

      , feed_dict={'bert_layer_module/input_ids:0': test_input_ids, 'bert_layer_module/input_mask:0': test_input_mask, 'bert_layer_module/segment_ids:0': test_segment_ids})

print('Input data shape:', len(test_input_ids))

print('Feature shape: {}, remember MAX_SEQ_LENGTH value is: {}'.format(some_layer_output.shape, MAX_SEQ_LENGTH))

print('Feature matrix:')

print(some_layer_output)