!pip install bert-tensorflow

!pip install --upgrade pandas

!wget --no-clobber 'https://github.com/ameasure/autocoding-class/raw/master/msha.xlsx'
import pandas as pd



df = pd.read_excel('msha.xlsx')

df['ACCIDENT_YEAR'] = df['ACCIDENT_DT'].apply(lambda x: x.year)

df['ACCIDENT_YEAR'].value_counts()

df_train = df[df['ACCIDENT_YEAR'].isin([2010, 2011])].copy()

df_valid = df[df['ACCIDENT_YEAR'] == 2012].copy()

print('training rows:', len(df_train))

print('validation rows:', len(df_valid))
import tensorflow as tf

import keras

from keras import backend as K

import tensorflow_hub as hub

import bert

from bert import run_classifier

from bert import optimization

from bert import tokenization



# preprocess data, convert each row into a bert "InputExample" object

def bert_preprocess(row, axis=None):

  return bert.run_classifier.InputExample(guid=None,

                                          text_a=row['NARRATIVE'],

                                          text_b=None,

                                          label=row['INJ_BODY_PART'])



processed_train = df_train.apply(bert_preprocess, axis=1)

processed_valid = df_valid.apply(bert_preprocess, axis=1)
# Get the tokenizer for our BERT MODEL

# path to BERT MODEL

BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"



with tf.Graph().as_default():

  # load the model from tensorflow hub

  bert_module = hub.Module(BERT_MODEL_HUB)

  # get the vocab file and do_lower_case function from the module

  with tf.Session() as sess:

    tokenization_info = bert_module(signature='tokenization_info', as_dict=True)

    vocab_file = sess.run(tokenization_info['vocab_file'])

    do_lower_case = sess.run(tokenization_info['do_lower_case'])
tokenizer = bert.tokenization.FullTokenizer(vocab_file=vocab_file,

                                            do_lower_case=do_lower_case)



tokenizer.tokenize('EE was loading a Gabion Grizzly when he was struck by falling debris')
# Now we'll convert our inputs to the numeric representation that BERT expects,

# a list of feature objects. Each feature object has 4 attributes:

# input_ids = a list of numbers representing words in our narrative

# input_mask = a list of 1/0s indicating which words should be masked (if the sequence is less than MAX_SEQ_LENGTH we mask these)

# segment_ids = a list of 1/0s indicating which sequence each token belongs to (for multi-segment tasks)

# label_id = id indicating the code for this example

MAX_SEQ_LENGTH = 128

LABELS = df['INJ_BODY_PART'].unique()

train_features = bert.run_classifier.convert_examples_to_features(processed_train, 

                                                                  LABELS, 

                                                                  MAX_SEQ_LENGTH,

                                                                  tokenizer)

valid_features = bert.run_classifier.convert_examples_to_features(processed_valid, 

                                                                  LABELS, 

                                                                  MAX_SEQ_LENGTH,

                                                                  tokenizer)



train_input_ids, train_input_mask, train_segment_ids, train_label_id = [], [], [], []

for f in train_features:

  train_input_ids.append(f.input_ids)

  train_input_mask.append(f.input_mask)

  train_segment_ids.append(f.segment_ids)

  train_label_id.append(f.label_id)

  

valid_input_ids, valid_input_mask, valid_segment_ids, valid_label_id = [], [], [], []

for f in valid_features:

  valid_input_ids.append(f.input_ids)

  valid_input_mask.append(f.input_mask)

  valid_segment_ids.append(f.segment_ids)

  valid_label_id.append(f.label_id)
# Create the Keras layer which will hold our BERT model

class BertLayer(keras.layers.Layer):

    def __init__(self, n_fine_tune_layers=10, **kwargs):

        self.n_fine_tune_layers = n_fine_tune_layers

        self.trainable = True

        self.output_size = 768

        super(BertLayer, self).__init__(**kwargs)



    def update_trainable_weights(self, n_fine_tune_layers):

        self.n_fine_tune_layers = n_fine_tune_layers

        

        # Select how many layers to fine tune

        self.trainable_var_strings = ['pooler/dense']

        for i in range(self.n_fine_tune_layers):

            self.trainable_var_strings.append(f'encoder/layer_{str(11 - i)}')

        

        bert_vars = self.bert.variables

        # Remove unused layers

        trainable_vars = []

        for var in bert_vars:

            if any(s in var.name for s in self.trainable_var_strings):

                trainable_vars.append(var)

                

        # Add to trainable weights

        print('trainable weights:')

        for var in trainable_vars:

            self._trainable_weights.append(var)

        

        # Add non-trainable weights

        for var in self.bert.variables:

            if var not in self._trainable_weights:

                self._non_trainable_weights.append(var)        

        

    def build(self, input_shape):

        self.bert = hub.Module(

            BERT_MODEL_HUB,

            trainable=self.trainable,

            name="{}_module".format(self.name)

        )

        self.update_trainable_weights(self.n_fine_tune_layers)

        super(BertLayer, self).build(input_shape)



    def call(self, inputs):

        inputs = [K.cast(x, dtype="int32") for x in inputs]

        input_ids, input_mask, segment_ids = inputs

        bert_inputs = dict(

            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids

        )

        result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[

            "pooled_output"

        ]

        return result



    def compute_output_shape(self, input_shape):

        return (input_shape[0], self.output_size)
def create_model(n_fine_tune_layers):

  # Specify layers of Keras model

  in_id = keras.layers.Input(shape=(MAX_SEQ_LENGTH,), name="input_ids")

  in_mask = keras.layers.Input(shape=(MAX_SEQ_LENGTH,), name="input_masks")

  in_segment = keras.layers.Input(shape=(MAX_SEQ_LENGTH,), name="segment_ids")

  bert_inputs = [in_id, in_mask, in_segment]



  # Load the pretrained BERT Layer

  bert_output = BertLayer(n_fine_tune_layers, name='bert')(bert_inputs)



  # Build the rest of the classifier 

  #dense = keras.layers.Dense(256, activation='relu')(bert_output)

  drop = keras.layers.Dropout(0.5)(bert_output)

  pred = keras.layers.Dense(len(LABELS), activation='softmax')(drop)

  model = keras.models.Model(inputs=bert_inputs, outputs=pred) 

  return model
from keras.optimizers import Adam



model = create_model(n_fine_tune_layers=10)

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=2e-5),metrics=['accuracy'])

model.summary()
bert_layer = model.get_layer('bert')

bert_layer.bert.variables
model.fit(

      [train_input_ids, train_input_mask, train_segment_ids], 

      train_label_id,

      validation_data=([valid_input_ids, valid_input_mask, valid_segment_ids], valid_label_id),

      epochs=5,

      batch_size=32

  )