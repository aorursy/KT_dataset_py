from pathlib import Path    # platform independent paths

from IPython.display import Markdown, display

import pandas as pd

from nltk.probability import FreqDist    # frequency dictionary

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns
# set figure size

plt.rcParams['figure.figsize'] = [20, 10]
def printmd(string, color=None):

    ''' NOT MINE

    Markdown printing from a code cell

    Ex. printmd("**bold and blue**", color="blue")

    https://stackoverflow.com/questions/23271575/printing-bold-colored-etc-text-in-ipython-qtconsole

    '''

    colorstr = "<span style='color:{}'>{}</span>".format(color, string)

    display(Markdown(colorstr))
def freq_dist(some_list, min_freq = 1, exact_freq = 1, top = 0):

    '''

    Adding in fucntionality to FreqDist

    Need to add in raise errors if more than one optional argument used.

    '''

    temp = FreqDist(some_list)

    if (min_freq != 1):

        temp = [(k, v) for k, v in temp.items() if v > min_freq]

        temp.sort(key=lambda x: x[1], reverse = True)

        return temp

    elif (exact_freq != 1):

        temp = [(k, v) for k, v in temp.items() if v == exact_freq]

        return temp

    elif (top != 0):

        return temp.most_common(top)

    else:

        temp = [(k, v) for k, v in temp.items()]

        temp.sort(key=lambda x: x[1], reverse = True)

        return temp
# File locations

file_neg = Path('../input/comments_negative.csv')

file_pos = Path('../input/comments_positive.csv')
# Load into dataframes

df_neg = pd.read_csv(file_neg)

df_pos = pd.read_csv(file_pos)
#df_neg = df_neg.sample(n=100000, random_state=1)

#df_pos = df_pos.sample(n=100000, random_state=1)
df_neg.head()
df_pos.head()
totalDF = pd.concat([df_pos,df_neg])
totalDF = totalDF[totalDF['text'].notnull()]

totalDF = totalDF[totalDF['parent_text'].notnull()]
df_pos = 5

df_neg = 5
conditions = [

    (totalDF['score'] < -1000) ,

    (totalDF['score'] > -1000) & (totalDF['score'] < -100),

    (totalDF['score'] > -100) & (totalDF['score'] < 100),

    (totalDF['score'] > 100) & (totalDF['score'] < 1000),

    (totalDF['score'] > 1000)]

choices = [-2, -1, 0,1,2]

totalDF['category'] = np.select(conditions, choices, default=0)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(totalDF, totalDF["category"], test_size=0.5)




!pip install bert



!pip install bert-tensorflow
import tensorflow as tf

import pandas as pd

import tensorflow_hub as hub

import os

import re

import numpy as np

from bert.tokenization import FullTokenizer

from tqdm import tqdm_notebook

from tensorflow.keras import backend as K

# Initialize session

sess = tf.Session()



# Params for bert model and tokenization

bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
X_train =  X_train.sample(n=500000, random_state=1)

#X_train =  X_train.sample(n=100000, random_state=1)
X_train.head
#max_seq_length = 32

max_seq_length = 18



# Create datasets (Only take up to max_seq_length words for memory)

train_text = X_train['text'].tolist()

train_text = [' '.join(t.split()[0:max_seq_length]) for t in train_text]

train_text = np.array(train_text, dtype=object)[:, np.newaxis]





#train_label = totalDF['score'].tolist()

train_label = X_train['category'].tolist()



train2_text = X_train['parent_text'].tolist()

train2_text = [' '.join(t.split()[0:max_seq_length]) for t in train2_text]

train2_text = np.array(train2_text, dtype=object)[:, np.newaxis]



test_text = X_test['text'].tolist()

test_text = [' '.join(t.split()[0:max_seq_length]) for t in test_text]

test_text = np.array(test_text, dtype=object)[:, np.newaxis]



test2_text = X_test['parent_text'].tolist()

test2_text = [' '.join(t.split()[0:max_seq_length]) for t in test2_text]

test2_text = np.array(test2_text, dtype=object)[:, np.newaxis]



test_label =  X_test['category'].tolist()
np.unique(train_label)
np.unique(test_label)
# train_label = totalDF[.tolist()

# train_text = totalDF['text'].tolist()
class PaddingInputExample(object):

    """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples

  to be a multiple of the batch size, because the TPU requires a fixed batch

  size. The alternative is to drop the last batch, which is bad because it means

  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding

  battches could cause silent errors.

  """



class InputExample(object):

    """A single training/test example for simple sequence classification."""



    def __init__(self, guid, text_a, text_b=None, label=None):

        """Constructs a InputExample.

    Args:

      guid: Unique id for the example.

      text_a: string. The untokenized text of the first sequence. For single

        sequence tasks, only this sequence must be specified.

      text_b: (Optional) string. The untokenized text of the second sequence.

        Only must be specified for sequence pair tasks.

      label: (Optional) string. The label of the example. This should be

        specified for train and dev examples, but not for test examples.

    """

        self.guid = guid

        self.text_a = text_a

        self.text_b = text_b

        self.label = label



def create_tokenizer_from_hub_module():

    """Get the vocab file and casing info from the Hub module."""

    bert_module =  hub.Module(bert_path)

    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)

    vocab_file, do_lower_case = sess.run(

        [

            tokenization_info["vocab_file"],

            tokenization_info["do_lower_case"],

        ]

    )



    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)



def convert_single_example(tokenizer, example, max_seq_length=256):

    """Converts a single `InputExample` into a single `InputFeatures`."""



    if isinstance(example, PaddingInputExample):

        input_ids = [0] * max_seq_length

        input_mask = [0] * max_seq_length

        segment_ids = [0] * max_seq_length

        label = 0

        return input_ids, input_mask, segment_ids, label



    tokens_a = tokenizer.tokenize(example.text_a)

    if len(tokens_a) > max_seq_length - 2:

        tokens_a = tokens_a[0 : (max_seq_length - 2)]



    tokens = []

    segment_ids = []

    tokens.append("[CLS]")

    segment_ids.append(0)

    for token in tokens_a:

        tokens.append(token)

        segment_ids.append(0)

    tokens.append("[SEP]")

    segment_ids.append(0)



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



    return input_ids, input_mask, segment_ids, example.label



def convert_examples_to_features(tokenizer, examples, max_seq_length=256):

    """Convert a set of `InputExample`s to a list of `InputFeatures`."""



    input_ids, input_masks, segment_ids, labels = [], [], [], []

    for example in tqdm_notebook(examples, desc="Converting examples to features"):

        input_id, input_mask, segment_id, label = convert_single_example(

            tokenizer, example, max_seq_length

        )

        input_ids.append(input_id)

        input_masks.append(input_mask)

        segment_ids.append(segment_id)

        labels.append(label)

    return (

        np.array(input_ids),

        np.array(input_masks),

        np.array(segment_ids),

        np.array(labels).reshape(-1, 1),

    )



def convert_text_to_examples(texts, labels):

    """Create InputExamples"""

    InputExamples = []

    for text, label in zip(texts, labels):

        InputExamples.append(

            InputExample(guid=None, text_a=" ".join(text), text_b=None, label=label)

        )

    return InputExamples



# Instantiate tokenizer

tokenizer = create_tokenizer_from_hub_module()



# Convert data to InputExample format

train_examples = convert_text_to_examples(train_text, train_label)

train2_examples = convert_text_to_examples(train2_text, train_label)

test_examples = convert_text_to_examples(test_text, test_label)

test2_examples = convert_text_to_examples(test2_text, test_label)



# Convert to features

(train_input_ids, train_input_masks, train_segment_ids, train_labels 

) = convert_examples_to_features(tokenizer, train_examples, max_seq_length=max_seq_length)



(train2_input_ids, train2_input_masks, train2_segment_ids, train_labels 

) = convert_examples_to_features(tokenizer, train2_examples, max_seq_length=max_seq_length)



train_label[4]
import sklearn

label_binarizer = sklearn.preprocessing.LabelBinarizer()

label_binarizer.fit([-2,-1,0,1,2])

train_label_bak = train_labels

train_labels = label_binarizer.transform(train_labels)

#test_labels = label_binarizer.transform(test_labels)
plt.hist(train_label_bak,10, facecolor='blue', alpha=0.5)

plt.show()
train_labels[1]
#train_labels = pd.get_dummies(train_labels)
class BertLayer(tf.keras.layers.Layer):

    def __init__(

        self,

        n_fine_tune_layers=10,

        pooling="first",

        bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",

        **kwargs,

    ):

        self.n_fine_tune_layers = n_fine_tune_layers

        self.trainable = True

        self.output_size = 768

        self.pooling = pooling

        self.bert_path = bert_path

        if self.pooling not in ["first", "mean"]:

            raise NameError(

                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"

            )



        super(BertLayer, self).__init__(**kwargs)



    def build(self, input_shape):

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



    def call(self, inputs):

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



        return pooled



    def compute_output_shape(self, input_shape):

        return (input_shape[0], self.output_size)
# define metrics to measure during runtime of keras

def f1(y_true, y_pred):

    def recall(y_true, y_pred):

        """Recall metric.



        Only computes a batch-wise average of recall.



        Computes the recall, a metric for multi-label classification of

        how many relevant items are selected.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



    def precision(y_true, y_pred):

        """Precision metric.



        Only computes a batch-wise average of precision.



        Computes the precision, a metric for multi-label classification of

        how many selected items are relevant.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision

    precision = precision(y_true, y_pred)

    recall = recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# Build model

def build_model(max_seq_length): 

    # add width to acommodate combined parent_text and text features

    in_id = tf.keras.layers.Input(shape=(max_seq_length*2,), name="input_ids")

    in_mask = tf.keras.layers.Input(shape=(max_seq_length*2,), name="input_masks")

    in_segment = tf.keras.layers.Input(shape=(max_seq_length*2,), name="segment_ids")

    bert_inputs = [in_id, in_mask, in_segment]

    

    bert_output = BertLayer(n_fine_tune_layers=3, pooling="first")(bert_inputs)

    #dense = tf.keras.layers.Dense(256, activation='relu')(bert_output)

    dense = tf.keras.layers.Dense(42, activation='relu')(bert_output)

    #pred = tf.keras.layers.Dense(1, activation='sigmoid')(dense)

    pred = tf.keras.layers.Dense(5, activation='softmax')(dense)

    

    model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)

    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[f1])

    model.summary()

    

    return model



def initialize_vars(sess):

    sess.run(tf.local_variables_initializer())

    sess.run(tf.global_variables_initializer())

    sess.run(tf.tables_initializer())

    K.set_session(sess)



train3_input_ids = np.hstack((train_input_ids, train2_input_ids))

train3_input_masks = np.hstack((train_input_masks, train2_input_masks))

train3_segment_ids = np.hstack((train_segment_ids, train2_segment_ids))
train_labels.shape
model = build_model(max_seq_length)



# Instantiate variables

initialize_vars(sess)



model.fit(

    [train3_input_ids, train3_input_masks, train3_segment_ids], 

    train_labels,

   # validation_data=([test_input_ids, test_input_masks, test_segment_ids], test_labels),

    epochs=1,

    batch_size=32

)
model.save('BertModel.h5')

# pre_save_preds = model.predict([test3_input_ids[0:100], 

#                                 test3_input_masks[0:100], 

#                                 test3_segment_ids[0:100]]

#                               ) # predictions before we clear and reload model



# # Clear and load model

# model = None

# model = build_model(max_seq_length)

# initialize_vars(sess)

# model.load_weights('BertModel.h5')



#post_save_preds = model.predict([test3_input_ids[0:100], 

                             #   test3_input_masks[0:100], 

                           #     test3_segment_ids[0:100]]

                         #     ) # predictions after we clear and reload model

#all(pre_save_preds == post_save_preds) # Are they the same?
scores = model.evaluate( [train3_input_ids, train3_input_masks, train3_segment_ids], 

    train_labels, 

                        verbose=1)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# check class performance for very negative posts

VNeg = train_labels[:,0] == 1









scores = model.evaluate( [train3_input_ids[VNeg], train3_input_masks[VNeg], train3_segment_ids[VNeg]], 

    train_labels[VNeg], 

                        verbose=1)

print("Very Negative Class performance %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# check class performance for very postive posts

VNeg = train_labels[:,4] == 1







scores = model.evaluate( [train3_input_ids[VNeg], train3_input_masks[VNeg], train3_segment_ids[VNeg]], 

    train_labels[VNeg], 

                        verbose=1)

print("Very Positive Class performance %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
from keras.layers import Input, Conv1D, Dense, concatenate
def build_model2(max_seq_length): 

    in_id = tf.keras.layers.Input(shape=(max_seq_length*2,), name="input_ids")

    in_mask = tf.keras.layers.Input(shape=(max_seq_length*2,), name="input_masks")

    in_segment = tf.keras.layers.Input(shape=(max_seq_length*2,), name="segment_ids")



        

    in_non_bert = tf.keras.layers.Input(shape=(1,), name="parent_score")

    

    all_inputs = [in_id, in_mask, in_segment,in_non_bert]

    bert_inputs = [in_id, in_mask, in_segment]



    

    bert_output = BertLayer(n_fine_tune_layers=3, pooling="first")(bert_inputs)

    #dense = tf.keras.layers.Dense(256, activation='relu')(bert_output)

    merged = tf.keras.layers.concatenate([bert_output, in_non_bert])

    dense = tf.keras.layers.Dense(42, activation='relu')( merged)

    #pred = tf.keras.layers.Dense(1, activation='sigmoid')(dense)

    pred = tf.keras.layers.Dense(5, activation='softmax')(dense)

    

    model = tf.keras.models.Model(inputs=all_inputs , outputs=pred)

    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[f1])

    model.summary()

    

    return model
train_parent_scores = X_train['parent_score']
train_parent_scores = train_parent_scores.as_matrix()
model2 = build_model2(max_seq_length)



# Instantiate variables

initialize_vars(sess)



model2.fit(

    [train3_input_ids, train3_input_masks, train3_segment_ids,train_parent_scores], 

    train_labels,

   # validation_data=([test_input_ids, test_input_masks, test_segment_ids], test_labels),

    epochs=1,

    batch_size=32

)
scores = model2.evaluate( [train3_input_ids, train3_input_masks, train3_segment_ids,train_parent_scores], 

    train_labels, 

                        verbose=1)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# check class performance for very postive posts

VNeg = train_labels[:,4] == 1







scores = model2.evaluate( [train3_input_ids[VNeg], train3_input_masks[VNeg], train3_segment_ids[VNeg]], 

    train_labels[VNeg], 

                        verbose=1)

print("Very Positive Class performance %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# check class performance for very postive posts

VNeg = train_labels[:,0] == 1







scores = model2.evaluate( [train3_input_ids[VNeg], train3_input_masks[VNeg], train3_segment_ids[VNeg]], 

    train_labels[VNeg], 

                        verbose=1)

print("Very Positive Class performance %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
(test_input_ids, test_input_masks, test_segment_ids, test_labels 

) = convert_examples_to_features(tokenizer, test_examples, max_seq_length=max_seq_length)



(test2_input_ids, test2_input_masks, test2_segment_ids, test_labels 

) = convert_examples_to_features(tokenizer, test2_examples, max_seq_length=max_seq_length)
test3_input_ids = np.hstack((test_input_ids, test2_input_ids))

test3_input_masks = np.hstack((test_input_masks, test2_input_masks))

test3_segment_ids = np.hstack((test_segment_ids, test2_segment_ids))
#test2_examples

test_parent_scores = X_test['parent_score']

test_parent_scores = test_parent_scores.as_matrix()
print(test3_input_ids.shape)

print(test3_input_masks.shape)

print(test3_segment_ids.shape)

print(test_labels.shape)
scores = model.evaluate([test3_input_ids, 

                                test3_input_masks, 

                                test3_segment_ids],

                        test_labels, 

                        verbose=1)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
scores2 = model2.evaluate([test3_input_ids, 

                                test3_input_masks, 

                                test3_segment_ids,test_parent_scores],

                        test_labels, 

                        verbose=1)

print("%s: %.2f%%" % (model2.metrics_names[1], scores2[1]*100))