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
import tensorflow as tf

import pandas as pd

import tensorflow_hub as hub

import os

import re

import numpy as np

from tqdm import tqdm

from tensorflow.keras import backend as K
BERT_TFHUB_DIR = '../input/tfhub-bert-uncased-l12-h768-a12/tfhub-bert/tfhub-bert'

max_seq_length = 300

MAX_RECORDS = 100000

SEED = 7514

sess = tf.Session()
def get_balanced_set(df_in, FOCUS_COLUMN, size = 0):

    df = df_in.sample(frac=1)

    if (size == 0):

        size = df.loc[df[FOCUS_COLUMN] == True].shape[0] * 2

    true_df = df.loc[df[FOCUS_COLUMN] == True][:round(size/2)]

    false_df = df.loc[df[FOCUS_COLUMN] == False][:round(size/2)]

    joined = pd.concat([true_df, false_df])

    return joined  

# coding=utf-8

# Copyright 2018 The Google AI Language Team Authors.

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

#     http://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.

"""Tokenization classes."""



from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



import collections

import re

import unicodedata

import six

import tensorflow as tf





def convert_to_unicode(text):

  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""

  if six.PY3:

    if isinstance(text, str):

      return text

    elif isinstance(text, bytes):

      return text.decode("utf-8", "ignore")

    else:

      raise ValueError("Unsupported string type: %s" % (type(text)))

  elif six.PY2:

    if isinstance(text, str):

      return text.decode("utf-8", "ignore")

    elif isinstance(text, unicode):

      return text

    else:

      raise ValueError("Unsupported string type: %s" % (type(text)))

  else:

    raise ValueError("Not running on Python2 or Python 3?")





def printable_text(text):

  """Returns text encoded in a way suitable for print or `tf.logging`."""



  # These functions want `str` for both Python2 and Python3, but in one case

  # it's a Unicode string and in the other it's a byte string.

  if six.PY3:

    if isinstance(text, str):

      return text

    elif isinstance(text, bytes):

      return text.decode("utf-8", "ignore")

    else:

      raise ValueError("Unsupported string type: %s" % (type(text)))

  elif six.PY2:

    if isinstance(text, str):

      return text

    elif isinstance(text, unicode):

      return text.encode("utf-8")

    else:

      raise ValueError("Unsupported string type: %s" % (type(text)))

  else:

    raise ValueError("Not running on Python2 or Python 3?")





def load_vocab(vocab_file):

  """Loads a vocabulary file into a dictionary."""

  vocab = collections.OrderedDict()

  index = 0

  with tf.gfile.GFile(vocab_file, "r") as reader:

    while True:

      token = convert_to_unicode(reader.readline())

      if not token:

        break

      token = token.strip()

      vocab[token] = index

      index += 1

  return vocab





def convert_by_vocab(vocab, items):

  """Converts a sequence of [tokens|ids] using the vocab."""

  output = []

  for item in items:

    output.append(vocab[item])

  return output





def convert_tokens_to_ids(vocab, tokens):

  return convert_by_vocab(vocab, tokens)





def convert_ids_to_tokens(inv_vocab, ids):

  return convert_by_vocab(inv_vocab, ids)





def whitespace_tokenize(text):

  """Runs basic whitespace cleaning and splitting on a piece of text."""

  text = text.strip()

  if not text:

    return []

  tokens = text.split()

  return tokens





class FullTokenizer(object):

  """Runs end-to-end tokenziation."""



  def __init__(self, vocab_file, do_lower_case=True):

    self.vocab = load_vocab(vocab_file)

    self.inv_vocab = {v: k for k, v in self.vocab.items()}

    self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)



  def tokenize(self, text):

    split_tokens = []

    for token in self.basic_tokenizer.tokenize(text):

      for sub_token in self.wordpiece_tokenizer.tokenize(token):

        split_tokens.append(sub_token)



    return split_tokens



  def convert_tokens_to_ids(self, tokens):

    return convert_by_vocab(self.vocab, tokens)



  def convert_ids_to_tokens(self, ids):

    return convert_by_vocab(self.inv_vocab, ids)





class BasicTokenizer(object):

  """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""



  def __init__(self, do_lower_case=True):

    """Constructs a BasicTokenizer.



    Args:

      do_lower_case: Whether to lower case the input.

    """

    self.do_lower_case = do_lower_case



  def tokenize(self, text):

    """Tokenizes a piece of text."""

    text = convert_to_unicode(text)

    text = self._clean_text(text)



    # This was added on November 1st, 2018 for the multilingual and Chinese

    # models. This is also applied to the English models now, but it doesn't

    # matter since the English models were not trained on any Chinese data

    # and generally don't have any Chinese data in them (there are Chinese

    # characters in the vocabulary because Wikipedia does have some Chinese

    # words in the English Wikipedia.).

    text = self._tokenize_chinese_chars(text)



    orig_tokens = whitespace_tokenize(text)

    split_tokens = []

    for token in orig_tokens:

      if self.do_lower_case:

        token = token.lower()

        token = self._run_strip_accents(token)

      split_tokens.extend(self._run_split_on_punc(token))



    output_tokens = whitespace_tokenize(" ".join(split_tokens))

    return output_tokens



  def _run_strip_accents(self, text):

    """Strips accents from a piece of text."""

    text = unicodedata.normalize("NFD", text)

    output = []

    for char in text:

      cat = unicodedata.category(char)

      if cat == "Mn":

        continue

      output.append(char)

    return "".join(output)



  def _run_split_on_punc(self, text):

    """Splits punctuation on a piece of text."""

    chars = list(text)

    i = 0

    start_new_word = True

    output = []

    while i < len(chars):

      char = chars[i]

      if _is_punctuation(char):

        output.append([char])

        start_new_word = True

      else:

        if start_new_word:

          output.append([])

        start_new_word = False

        output[-1].append(char)

      i += 1



    return ["".join(x) for x in output]



  def _tokenize_chinese_chars(self, text):

    """Adds whitespace around any CJK character."""

    output = []

    for char in text:

      cp = ord(char)

      if self._is_chinese_char(cp):

        output.append(" ")

        output.append(char)

        output.append(" ")

      else:

        output.append(char)

    return "".join(output)



  def _is_chinese_char(self, cp):

    """Checks whether CP is the codepoint of a CJK character."""

    # This defines a "chinese character" as anything in the CJK Unicode block:

    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)

    #

    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,

    # despite its name. The modern Korean Hangul alphabet is a different block,

    # as is Japanese Hiragana and Katakana. Those alphabets are used to write

    # space-separated words, so they are not treated specially and handled

    # like the all of the other languages.

    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #

        (cp >= 0x3400 and cp <= 0x4DBF) or  #

        (cp >= 0x20000 and cp <= 0x2A6DF) or  #

        (cp >= 0x2A700 and cp <= 0x2B73F) or  #

        (cp >= 0x2B740 and cp <= 0x2B81F) or  #

        (cp >= 0x2B820 and cp <= 0x2CEAF) or

        (cp >= 0xF900 and cp <= 0xFAFF) or  #

        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #

      return True



    return False



  def _clean_text(self, text):

    """Performs invalid character removal and whitespace cleanup on text."""

    output = []

    for char in text:

      cp = ord(char)

      if cp == 0 or cp == 0xfffd or _is_control(char):

        continue

      if _is_whitespace(char):

        output.append(" ")

      else:

        output.append(char)

    return "".join(output)





class WordpieceTokenizer(object):

  """Runs WordPiece tokenziation."""



  def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):

    self.vocab = vocab

    self.unk_token = unk_token

    self.max_input_chars_per_word = max_input_chars_per_word



  def tokenize(self, text):

    """Tokenizes a piece of text into its word pieces.



    This uses a greedy longest-match-first algorithm to perform tokenization

    using the given vocabulary.



    For example:

      input = "unaffable"

      output = ["un", "##aff", "##able"]



    Args:

      text: A single token or whitespace separated tokens. This should have

        already been passed through `BasicTokenizer.



    Returns:

      A list of wordpiece tokens.

    """



    text = convert_to_unicode(text)



    output_tokens = []

    for token in whitespace_tokenize(text):

      chars = list(token)

      if len(chars) > self.max_input_chars_per_word:

        output_tokens.append(self.unk_token)

        continue



      is_bad = False

      start = 0

      sub_tokens = []

      while start < len(chars):

        end = len(chars)

        cur_substr = None

        while start < end:

          substr = "".join(chars[start:end])

          if start > 0:

            substr = "##" + substr

          if substr in self.vocab:

            cur_substr = substr

            break

          end -= 1

        if cur_substr is None:

          is_bad = True

          break

        sub_tokens.append(cur_substr)

        start = end



      if is_bad:

        output_tokens.append(self.unk_token)

      else:

        output_tokens.extend(sub_tokens)

    return output_tokens





def _is_whitespace(char):

  """Checks whether `chars` is a whitespace character."""

  # \t, \n, and \r are technically contorl characters but we treat them

  # as whitespace since they are generally considered as such.

  if char == " " or char == "\t" or char == "\n" or char == "\r":

    return True

  cat = unicodedata.category(char)

  if cat == "Zs":

    return True

  return False





def _is_control(char):

  """Checks whether `chars` is a control character."""

  # These are technically control characters but we count them as whitespace

  # characters.

  if char == "\t" or char == "\n" or char == "\r":

    return False

  cat = unicodedata.category(char)

  if cat in ("Cc", "Cf"):

    return True

  return False





def _is_punctuation(char):

  """Checks whether `chars` is a punctuation character."""

  cp = ord(char)

  # We treat all non-letter/number ASCII as punctuation.

  # Characters such as "^", "$", and "`" are not in the Unicode

  # Punctuation class but we treat them as punctuation anyways, for

  # consistency.

  if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or

      (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):

    return True

  cat = unicodedata.category(char)

  if cat.startswith("P"):

    return True

  return False
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





def create_tokenizer_from_hub_module(bert_path):

    """Get the vocab file and casing info from the Hub module."""

    bert_module = hub.Module(bert_path)

    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)

    vocab_file, do_lower_case = sess.run(

        [tokenization_info["vocab_file"], tokenization_info["do_lower_case"]]

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

    for example in tqdm(examples, desc="Converting examples to features"):

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

        np.array(labels),

    )





def convert_text_to_examples(texts, labels):

    """Create InputExamples"""

    InputExamples = []

    for text, label in zip(texts, labels):

        InputExamples.append(

            InputExample(guid=None, text_a=" ".join(text), text_b=None, label=label)

        )

    return InputExamples
class BertLayer(tf.layers.Layer):

    def __init__(

        self,

        n_fine_tune_layers=10,

        pooling="mean",

        bert_path=BERT_TFHUB_DIR,

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
%%time

complete_train_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

test_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')



IDENTITY_COLUMNS = [

    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',

    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'

]



train_df = complete_train_df.sample(round(MAX_RECORDS / 2), random_state = SEED)

for id in IDENTITY_COLUMNS:

    group_df = get_balanced_set(complete_train_df, id, size = round(MAX_RECORDS / 18))

    train_df = pd.concat([train_df, group_df])



print('dropping duplicates')

train_df = train_df.drop_duplicates(subset=['id']).reset_index(drop=True)

train_df.head()
len(train_df)
%%time

train_size = round(len(train_df)*.95)



train_text = train_df['comment_text'][:train_size].tolist()

train_text = [' '.join(t.split()[0:max_seq_length]) for t in train_text]

train_text = np.array(train_text, dtype=object)[:, np.newaxis]

train_label = train_df['target'].tolist()



val_text = train_df['comment_text'][train_size:].tolist()

val_text = [' '.join(t.split()[0:max_seq_length]) for t in val_text]

val_text = np.array(val_text, dtype=object)[:, np.newaxis]

val_label = train_df['target'][train_size:].tolist()



test_text = test_df['comment_text'].tolist()

test_text = [' '.join(t.split()[0:max_seq_length]) for t in test_text]

test_text = np.array(test_text, dtype=object)[:, np.newaxis]
%%time

# Instantiate tokenizer

tokenizer = create_tokenizer_from_hub_module(BERT_TFHUB_DIR)

# tokenizer = FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)



# Convert data to InputExample format

train_examples = convert_text_to_examples(train_text, train_label)

val_examples = convert_text_to_examples(val_text, val_label)

test_examples = convert_text_to_examples(test_text, ['']*len(test_text))



# Convert to features

(train_input_ids, train_input_masks, train_segment_ids, train_labels 

) = convert_examples_to_features(tokenizer, train_examples, max_seq_length=max_seq_length)

(val_input_ids, val_input_masks, val_segment_ids, val_labels

) = convert_examples_to_features(tokenizer, val_examples, max_seq_length=max_seq_length)

(test_input_ids, test_input_masks, test_segment_ids, test_labels

) = convert_examples_to_features(tokenizer, test_examples, max_seq_length=max_seq_length)
def build_model(max_seq_length):

    model = None

    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")

    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")

    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")

    bert_inputs = [in_id, in_mask, in_segment]



    bert_output = BertLayer(n_fine_tune_layers=3)(bert_inputs)

    dense = tf.keras.layers.Dense(256, activation="relu")(bert_output) 

    dense = tf.keras.layers.Dense(128, activation="relu")(dense) 

    dense = tf.keras.layers.Dense(64, activation="relu")(dense) 

    pred = tf.keras.layers.Dense(1, activation="sigmoid")(dense)



    model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)

    model.compile(loss="binary_crossentropy", optimizer="adam")

    model.summary()



    return model





def initialize_vars(sess):

    sess.run(tf.local_variables_initializer())

    sess.run(tf.global_variables_initializer())

    sess.run(tf.tables_initializer())

    K.set_session(sess)

model = build_model(max_seq_length)



# Instantiate variables

initialize_vars(sess)
%%time

model.fit(

    [train_input_ids, train_input_masks, train_segment_ids],

    train_labels,

    validation_data=(

        [val_input_ids, val_input_masks, val_segment_ids],

        val_labels,

    ),

    epochs=3,

    batch_size=64,

)
%%time

predictions = model.predict([test_input_ids, test_input_masks, test_segment_ids], batch_size=512).flatten()

# predictions = model.predict([test_input_ids[:512], test_input_masks[:512], test_segment_ids[:512]], batch_size=512).flatten()
submission = pd.DataFrame.from_dict({

    'id': test_df.id,

    'prediction': predictions

})

submission.to_csv('submission.csv', index=False)
submission.head(25)