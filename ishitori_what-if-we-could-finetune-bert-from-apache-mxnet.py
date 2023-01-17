# bert_adam.py - BERT authors did a change to Adam algorightm for better convergence
# taken from: https://github.com/dmlc/gluon-nlp/blob/master/src/gluonnlp/optimizer/bert_adam.py

# coding: utf-8
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Weight updating functions."""
from mxnet.optimizer import Optimizer, register
from mxnet.ndarray import zeros, NDArray

__all__ = ['BERTAdam']

@register
class BERTAdam(Optimizer):
    """The Adam optimizer with weight decay regularization for BERT.

    Updates are applied by::

        rescaled_grad = clip(grad * rescale_grad, clip_gradient)
        m = beta1 * m + (1 - beta1) * rescaled_grad
        v = beta2 * v + (1 - beta2) * (rescaled_grad**2)
        w = w - learning_rate * (m / (sqrt(v) + epsilon) + wd * w)

    Note that this is different from `mxnet.optimizer.Adam`, where L2 loss is added and
    accumulated in m and v. In BERTAdam, the weight decay term decoupled from gradient
    based update.

    This is also slightly different from the AdamW optimizer described in
    *Fixing Weight Decay Regularization in Adam*, where the schedule multiplier and
    learning rate is decoupled, and the bias-correction terms are removed.
    The BERTAdam optimizer uses the same learning rate to apply gradients
    w.r.t. the loss and weight decay.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`mxnet.optimizer.Optimizer`.

    Parameters
    ----------
    beta1 : float, optional
        Exponential decay rate for the first moment estimates.
    beta2 : float, optional
        Exponential decay rate for the second moment estimates.
    epsilon : float, optional
        Small value to avoid division by 0.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 **kwargs):
        super(BERTAdam, self).__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def create_state(self, index, weight): # pylint: disable=unused-argument
        """Initialization for mean and var."""
        return (zeros(weight.shape, weight.context, dtype=weight.dtype), #mean
                zeros(weight.shape, weight.context, dtype=weight.dtype)) #variance

    def update(self, index, weight, grad, state):
        """Update method."""
        try:
            from mxnet.ndarray.contrib import adamw_update
        except ImportError:
            raise ImportError('Failed to import nd.contrib.adamw_update from MXNet. '
                              'BERTAdam optimizer requires mxnet>=1.5.0b20181228. '
                              'Please upgrade your MXNet version.')
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)

        kwargs = {'beta1': self.beta1, 'beta2': self.beta2, 'epsilon': self.epsilon,
                  'rescale_grad': self.rescale_grad}
        if self.clip_gradient:
            kwargs['clip_gradient'] = self.clip_gradient

        mean, var = state
        adamw_update(weight, grad, mean, var, out=weight, lr=1, wd=wd, eta=lr, **kwargs)

# bert_tokenizer.py - bunch of classes need to run tokenization for BERT
# Taken from https://github.com/dmlc/gluon-nlp/blob/master/scripts/bert/tokenizer.py

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
import unicodedata
import io
import six

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode('utf-8', 'ignore')
        else:
            raise ValueError('Unsupported string type: %s' % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode('utf-8', 'ignore')
        elif isinstance(text, unicode):  # noqa: F821
            return text
        else:
            raise ValueError('Unsupported string type: %s' % (type(text)))
    else:
        raise ValueError('Not running on Python2 or Python 3?')


def printable_text(text):
    """Returns text encoded in a way suitable for print."""
    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode('utf-8', 'ignore')
        else:
            raise ValueError('Unsupported string type: %s' % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):  # noqa: F821
            return text.encode('utf-8')
        else:
            raise ValueError('Unsupported string type: %s' % (type(text)))
    else:
        raise ValueError('Not running on Python2 or Python 3?')


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with io.open(vocab_file, 'r') as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def convert_tokens_to_ids(vocab, tokens):
    """Converts a sequence of tokens into ids using the vocab."""
    ids = []
    for token in tokens:
        ids.append(vocab[token])
    return ids


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class FullTokenizer(object):
    """Runs end-to-end tokenziation.

    Parameters
    ----------
    vocab : Vocab
        Vocabulary for the corpus.
    do_lower_case : bool, default True
        Convert tokens to lower cases.
    """

    def __init__(self, vocab, do_lower_case=True):
        self.vocab = vocab
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_tokens_to_ids(self.vocab, tokens)


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

        output_tokens = whitespace_tokenize(' '.join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize('NFD', text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == 'Mn':
                continue
            output.append(char)
        return ''.join(output)

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

        return [''.join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(' ')
                output.append(char)
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)

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
        if ((cp >= 0x4E00 and cp <= 0x9FFF)
                or (cp >= 0x3400 and cp <= 0x4DBF)
                or (cp >= 0x20000 and cp <= 0x2A6DF)
                or (cp >= 0x2A700 and cp <= 0x2B73F)
                or (cp >= 0x2B740 and cp <= 0x2B81F)
                or (cp >= 0x2B820 and cp <= 0x2CEAF)
                or (cp >= 0xF900 and cp <= 0xFAFF)
                or (cp >= 0x2F800 and cp <= 0x2FA1F)):
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
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenziation."""

    def __init__(self, vocab, unk_token='[UNK]', max_input_chars_per_word=100):
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
                    substr = ''.join(chars[start:end])
                    if start > 0:
                        substr = '##' + substr
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
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char in [' ', '\t', '\n', '\r']:
        return True
    cat = unicodedata.category(char)
    if cat == 'Zs':
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char in ['\t', '\n', '\r']:
        return False
    cat = unicodedata.category(char)
    if cat.startswith('C'):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    group0 = cp >= 33 and cp <= 47
    group1 = cp >= 58 and cp <= 64
    group2 = cp >= 91 and cp <= 96
    group3 = cp >= 123 and cp <= 126
    if (group0 or group1 or group2 or group3):
        return True
    cat = unicodedata.category(char)
    if cat.startswith('P'):
        return True
    return False

# bert_transform.py - Class incapsulates data transformation logic for Classification.
# Taken from: https://github.com/dmlc/gluon-nlp/blob/master/scripts/bert/dataset.py

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and DMLC.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six


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


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode('utf-8', 'ignore')
        else:
            raise ValueError('Unsupported string type: %s' % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode('utf-8', 'ignore')
        elif isinstance(text, unicode):  # noqa: F821
            return text
        else:
            raise ValueError('Unsupported string type: %s' % (type(text)))
    else:
        raise ValueError('Not running on Python2 or Python 3?')


class BERTTransform(object):
    """BERT style data transformation.

    Parameters
    ----------
    tokenizer : BasicTokenizer or FullTokensizer.
        Tokenizer for the sentences.
    max_seq_length : int.
        Maximum sequence length of the sentences.
    pad : bool, default True
        Whether to pad the sentences to maximum length.
    pair : bool, default True
        Whether to transform sentences or sentence pairs.
    """
    def __init__(self, tokenizer, max_seq_length, pad=True, pair=True):
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._pad = pad
        self._pair = pair

    def __call__(self, line):
        """Perform transformation for sequence pairs or single sequences.

        The transformation is processed in the following steps:
        - tokenize the input sequences
        - insert [CLS], [SEP] as necessary
        - generate type ids to indicate whether a token belongs to the first
          sequence or the second sequence.
        - generate valid length

        For sequence pairs, the input is a tuple of 2 strings:
        text_a, text_b.

        Inputs:
            text_a: 'is this jacksonville ?'
            text_b: 'no it is not'
        Tokenization:
            text_a: 'is this jack ##son ##ville ?'
            text_b: 'no it is not .'
        Processed:
            tokens:  '[CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]'
            type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
            valid_length: 14

        For single sequences, the input is a tuple of single string: text_a.
        Inputs:
            text_a: 'the dog is hairy .'
        Tokenization:
            text_a: 'the dog is hairy .'
        Processed:
            text_a:  '[CLS] the dog is hairy . [SEP]'
            type_ids: 0     0   0   0  0     0 0
            valid_length: 7

        Parameters
        ----------
        line: tuple of str
            Input strings. For sequence pairs, the input is a tuple of 3 strings:
            (text_a, text_b). For single sequences, the input is a tuple of single
            string: (text_a,).

        Returns
        -------
        np.array: input token ids in 'int32', shape (batch_size, seq_length)
        np.array: valid length in 'int32', shape (batch_size,)
        np.array: input token type ids in 'int32', shape (batch_size, seq_length)
        """
        # convert to unicode
        text_a = line[0]
        text_a = convert_to_unicode(text_a)
        if self._pair:
            assert len(line) == 2
            text_b = line[1]
            text_b = convert_to_unicode(text_b)

        tokens_a = self._tokenizer.tokenize(text_a)
        tokens_b = None

        if self._pair:
            tokens_b = self._tokenizer.tokenize(text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, self._max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > self._max_seq_length - 2:
                tokens_a = tokens_a[0:(self._max_seq_length - 2)]

        # The embedding vectors for `type=0` and `type=1` were learned during
        # pre-training and are added to the wordpiece embedding vector
        # (and position vector). This is not *strictly* necessary since
        # the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.

        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append('[CLS]')
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append('[SEP]')
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append('[SEP]')
            segment_ids.append(1)

        input_ids = self._tokenizer.convert_tokens_to_ids(tokens)

        # The valid length of sentences. Only real  tokens are attended to.
        valid_length = len(input_ids)

        if self._pad:
            # Zero-pad up to the sequence length.
            padding_length = self._max_seq_length - valid_length
            # use padding tokens for the rest
            input_ids.extend([self._tokenizer.vocab['[PAD]']] * padding_length)
            segment_ids.extend([self._tokenizer.vocab['[PAD]']] * padding_length)

        return np.array(input_ids, dtype='int32'), np.array(valid_length, dtype='int32'),\
               np.array(segment_ids, dtype='int32')


class ClassificationTransform(object):
    """Dataset Transformation for BERT-style Sentence Classification.

    Parameters
    ----------
    tokenizer : BasicTokenizer or FullTokensizer.
        Tokenizer for the sentences.
    labels : list of int.
        List of all label ids for the classification task.
    max_seq_length : int.
        Maximum sequence length of the sentences.
    pad : bool, default True
        Whether to pad the sentences to maximum length.
    pair : bool, default True
        Whether to transform sentences or sentence pairs.
    """
    def __init__(self, tokenizer, labels, max_seq_length, pad=True, pair=True):
        self._label_map = {}
        for (i, label) in enumerate(labels):
            self._label_map[label] = i
        self._bert_xform = BERTTransform(tokenizer, max_seq_length, pad=pad, pair=pair)

    def __call__(self, line):
        """Perform transformation for sequence pairs or single sequences.

        The transformation is processed in the following steps:
        - tokenize the input sequences
        - insert [CLS], [SEP] as necessary
        - generate type ids to indicate whether a token belongs to the first
          sequence or the second sequence.
        - generate valid length

        For sequence pairs, the input is a tuple of 3 strings:
        text_a, text_b and label.

        Inputs:
            text_a: 'is this jacksonville ?'
            text_b: 'no it is not'
            label: '0'
        Tokenization:
            text_a: 'is this jack ##son ##ville ?'
            text_b: 'no it is not .'
        Processed:
            tokens:  '[CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]'
            type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
            valid_length: 14
            label: 0

        For single sequences, the input is a tuple of 2 strings: text_a and label.
        Inputs:
            text_a: 'the dog is hairy .'
            label: '1'
        Tokenization:
            text_a: 'the dog is hairy .'
        Processed:
            text_a:  '[CLS] the dog is hairy . [SEP]'
            type_ids: 0     0   0   0  0     0 0
            valid_length: 7
            label: 1

        Parameters
        ----------
        line: tuple of str
            Input strings. For sequence pairs, the input is a tuple of 3 strings:
            (text_a, text_b, label). For single sequences, the input is a tuple
            of 2 strings: (text_a, label).

        Returns
        -------
        np.array: input token ids in 'int32', shape (batch_size, seq_length)
        np.array: valid length in 'int32', shape (batch_size,)
        np.array: input token type ids in 'int32', shape (batch_size, seq_length)
        np.array: label id in 'int32', shape (batch_size, 1)
        """
        label = line[-1]
        label = convert_to_unicode(label)
        label_id = self._label_map[label]
        label_id = np.array([label_id], dtype='int32')
        input_ids, valid_length, segment_ids = self._bert_xform(line[:-1])
        return input_ids, valid_length, segment_ids, label_id
# data_pipeline.py - Dataset for Quora data with a few helper methods.

import os
import csv
from mxnet.gluon.data import ArrayDataset


def get_sub_segment_from_list(dataset, indices):
    return ArrayDataset([dataset[i] for i in indices])


class QuoraDataset(ArrayDataset):
    """This dataset provides access to Quora insincere data competition"""
    def __init__(self, segment, root_dir="../input/"):
        self._root_dir = root_dir
        self._segment = segment
        self._segments = {
            'train': 'train.csv',
            'test': 'test.csv'
        }

        super(QuoraDataset, self).__init__(self._read_data())

    def _read_data(self):
        file_path = os.path.join(self._root_dir, self._segments[self._segment])
        with open(file_path, mode='r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            # ignore 1st line - which is header
            data = [(i, ) + tuple(row) for i, row in enumerate(reader) if i > 0]

        return data

# model.py - Fine-tuned mode, that using BERT as the base.
from mxnet.gluon.nn import Dense, Dropout, HybridSequential, Block


class QuoraModel(Block):
    def __init__(self, bert, dropout=0.1, prefix=None, params=None):
        super(QuoraModel, self).__init__(prefix=prefix, params=params)

        self.bert = bert

        with self.name_scope():
            self.output = HybridSequential()

            with self.output.name_scope():
                if dropout:
                    self.output.add(Dropout(dropout))
                self.output.add(Dense(units=1))

    def forward(self, inputs, token_types, valid_length=None):  # pylint: disable=arguments-differ
        _, pooler_out = self.bert(inputs, token_types, valid_length)
        return self.output(pooler_out)

# train.py - main training and evaluation logic using everything on top.

import os

import argparse
import time

import pandas as pd
import numpy as np
import pickle
from gluonnlp.data import FixedBucketSampler
from gluonnlp.data.batchify import Tuple, Stack
from gluonnlp.model import get_model

import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon import Trainer
from mxnet.gluon.data import DataLoader, SimpleDataset
from mxnet.gluon.utils import clip_global_norm

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Uncomment this if copy files locally
# from src.bert_adam import BERTAdam
# from src.bert_tokenizer import FullTokenizer
# from src.bert_transform import ClassificationTransform
# from src.data_pipeline import QuoraDataset, get_sub_segment_from_list
# from src.model import QuoraModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default=None, help='Index of GPU to use')
    parser.add_argument('--processed_train_data', type=str, default=None,
                        help='Path to processed training data')
    parser.add_argument('--processed_dev_data', type=str, default=None,
                        help='Path to processed val data')
    parser.add_argument('--processed_word_vocab', type=str, default=None,
                        help='Path to processed word level vocab data')
    parser.add_argument('--logging_path', default="./log.txt",
                        help='logging file path')
    parser.add_argument('--model_path', default='./model',
                        help='saving model in model_path')
    parser.add_argument('--epochs', type=int, default=5, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--num_buckets', type=int, default=10, help='Buckets')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate for everything')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='ratio of warmup steps used in NOAM\'s stepsize schedule')
    parser.add_argument('--lr', type=float, default=5e-6, help='Initial learning rate')
    parser.add_argument('--log_interval', type=int, default=200, help='Initial learning rate')
    parser.add_argument('--threshold', type=float, default=0.4, help='Positive/negative class threshold')
    parser.add_argument('--output', type=str, default='submission.txt', help='Submission file')
    parser.add_argument('--clip', type=int, default=1, help='Gradient clipping value')
    parser.add_argument('--sw_0_class', type=float, default=0.7, help='Sample weight for 0 class')
    parser.add_argument('--sw_1_class', type=float, default=0.5, help='Sample weight for 1 class (will be added to sample weight of 0 class)')
    return parser.parse_args()


def find_best_f1(outputs, labels):
    tmp = [0, 0, 0]  # idx, cur, max
    threshold = 0

    for tmp[0] in np.arange(0.1, 0.501, 0.01):
        tmp[1] = f1_score(labels.asnumpy(), outputs.asnumpy() > tmp[0])
        if tmp[1] > tmp[2]:
            threshold = tmp[0]
            tmp[2] = tmp[1]

    return tmp[2], threshold


def filter_out_question_id(*items):
    """Method makes dataset DataLoadable"""
    # q_idx, q_id, input_ids, valid_lengths, type_ids, label
    if len(items) == 6:
        return items[0], items[2], items[3], items[4], items[5]
    else:
        return items[0], items[2], items[3], items[4]


def run_training(net, trainer, train_dataloader, val_dataloader,
                 epochs, clip, sw_0_class, sw_1_class, log_interval, threshold, context):
    loss_fn = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
    print("Start training for {} epochs: {}".format(epochs, time.ctime()))

    differentiable_params = [p for p in net.collect_params().values() if p.grad_req != 'null']

    for e in range(epochs):
        train_loss = 0.
        total_items = 0
        val_outputs = []
        val_l = []

        for step_num, (q_idx, input_ids, valid_lengths, type_ids, label) in enumerate(train_dataloader):
            items_per_iteration = q_idx.shape[0]
            total_items += items_per_iteration

            input_ids = gluon.utils.split_and_load(input_ids, context, even_split=False)
            valid_lengths = gluon.utils.split_and_load(valid_lengths.astype('float32'), context, even_split=False)
            type_ids = gluon.utils.split_and_load(type_ids, context, even_split=False)
            label = gluon.utils.split_and_load(label, context, even_split=False)

            losses = []

            for i, (id, vl, ti, l) in enumerate(zip(input_ids, valid_lengths, type_ids, label)):
                with autograd.record():
                    out = net(id, ti, vl)
                    # penalize for class 1 more than for class 0
                    sample_weights = (sw_1_class * l.astype('float32')) + sw_0_class
                    loss = loss_fn(out, l.astype('float32'), sample_weights)
                    losses.append(loss)

            for loss in losses:
                loss.backward()

            grads = [p.grad(context[0]) for p in differentiable_params]
            clip_global_norm(grads, clip)

            if len(context) > 1:
                # in multi gpu mode we propagate new gradients to the rest of gpus
                for p in differentiable_params:
                    grads = p.list_grad()
                    source = grads[0]
                    destination = grads[1:]

                    for dest in destination:
                        source.copyto(dest)

            trainer.step(1)

            for loss in losses:
                train_loss += loss.sum().asscalar()

            # Uncomment this, if you want to display info after every iteration
            # print('[Epoch {} Batch {}/{}] loss={:.4f}, lr={:.7f}'
            #       .format(e, step_num + 1, len(train_dataloader),
            #               train_loss / total_items,
            #               trainer.learning_rate))

        nd.waitall()
        # print("Epoch training finished at {}.".format(time.ctime()))

        for step_num, (q_idx, input_ids, valid_lengths, type_ids, label) in enumerate(val_dataloader):
            input_ids = gluon.utils.split_and_load(input_ids, context, even_split=False)
            valid_lengths = gluon.utils.split_and_load(valid_lengths.astype('float32'), context, even_split=False)
            type_ids = gluon.utils.split_and_load(type_ids, context, even_split=False)
            label = gluon.utils.split_and_load(label, context, even_split=False)

            for i, (id, vl, ti, l) in enumerate(zip(input_ids, valid_lengths, type_ids, label)):
                out = net(id, ti, vl)
                val_outputs.append(out.sigmoid().as_in_context(mx.cpu()))
                val_l.append(l.reshape(shape=(l.shape[0], 1)).as_in_context(mx.cpu()))

        val_outputs = mx.nd.concat(*val_outputs, dim=0)
        val_l = mx.nd.concat(*val_l, dim=0)

        val_f1, threshold = find_best_f1(val_outputs, val_l)
        nd.waitall()

        print("Epoch {}. Current Loss: {:.5f}. Val F1: {:.3f}, Threshold: {:.3f}"
              .format(e, train_loss / total_items, val_f1, threshold))
        print(time.ctime())

    return net, threshold


def run_evaluation(net, dataloader, threshold):
    print("Start predicting")
    outputs = []
    result = []

    # label here is added by BERTTransformer - it is fake and always 0
    for step_num, (q_idx, input_ids, valid_lengths, type_ids, label) in enumerate(dataloader):
        q_idx = gluon.utils.split_and_load(q_idx, context, even_split=False)
        input_ids = gluon.utils.split_and_load(input_ids, context, even_split=False)
        valid_lengths = gluon.utils.split_and_load(valid_lengths.astype('float32'), context,
                                                   even_split=False)
        type_ids = gluon.utils.split_and_load(type_ids, context, even_split=False)

        for i, (qid, id, vl, ti) in enumerate(zip(q_idx, input_ids, valid_lengths, type_ids)):
            out = net(id, ti, vl)
            outputs.append((qid.as_in_context(mx.cpu()),
                            out.sigmoid().as_in_context(mx.cpu()) > threshold))

    for batch in outputs:
        result.extend([(int(q_idx.asscalar()), int(pred.asscalar()))
                       for q_idx, pred in zip(batch[0], batch[1])])

    return result


def load_and_process_dataset(dataset, word_vocab, path=None):
    if path is not None and os.path.exists(path):
        processed_dataset = pickle.load(open(path, 'rb'))
        return processed_dataset

    tokenizer = FullTokenizer(word_vocab, do_lower_case=True)
    transformer = ClassificationTransform(tokenizer, ['0', '1'], max_seq_length=74, pair=False)
    # test data doesn't have label - use 0
    processed_dataset = SimpleDataset([(item[0], item[1],) +
                                       transformer((item[2], item[3] if len(item) >= 4 else '0'))
                                       for item in dataset])

    if path:
        pickle.dump(processed_dataset, open(path, 'wb'))

    return processed_dataset


def main():
    args = parse_args()
    context = [mx.cpu(0)] if args.gpu is None else [mx.gpu(int(i)) for i in args.gpu.split(',')]

    # Loading pretrained BERT model and Vocab
    bert, word_vocab = get_model('bert_12_768_12',
                                 dataset_name='wiki_multilingual',
                                 pretrained=True,
                                 use_pooler=True,
                                 use_decoder=False,
                                 use_classifier=False,
                                 ctx=context)

    model = QuoraModel(bert, dropout=args.dropout)
    model.output.initialize(init=mx.init.Normal(0.02), ctx=context)

    dataset = QuoraDataset('train')

    train_indices, dev_indices = train_test_split(range(len(dataset)), train_size=0.98)

    train_dataset = load_and_process_dataset(get_sub_segment_from_list(dataset, train_indices),
                                             word_vocab,
                                             args.processed_train_data)

    dev_dataset = load_and_process_dataset(get_sub_segment_from_list(dataset, dev_indices),
                                           word_vocab,
                                           args.processed_dev_data)

    batchify_fn = Tuple(Stack(),
                        Stack(),
                        Stack(),
                        Stack(),
                        Stack())

    train_sampler = FixedBucketSampler(lengths=[item[3] for item in train_dataset],
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_buckets=args.num_buckets)

    dev_sampler = FixedBucketSampler(lengths=[item[3] for item in dev_dataset],
                                     batch_size=args.batch_size,
                                     shuffle=True)

    train_dataloader = DataLoader(train_dataset.transform(filter_out_question_id),
                                  batchify_fn=batchify_fn,
                                  batch_sampler=train_sampler,
                                  num_workers=10)

    dev_dataloader = DataLoader(dev_dataset.transform(filter_out_question_id),
                                batchify_fn=batchify_fn,
                                batch_sampler=dev_sampler,
                                num_workers=5)

    optimizer = BERTAdam(learning_rate=args.lr)
    trainer = Trainer(model.collect_params(), optimizer=optimizer)

    best_model, best_threshold = run_training(model, trainer, train_dataloader, dev_dataloader,
                                              args.epochs, args.clip, args.sw_0_class,
                                              args.sw_1_class, args.log_interval, args.threshold,
                                              context)

    test_dataset = QuoraDataset('test')
    processed_test_dataset = load_and_process_dataset(test_dataset, word_vocab)

    batchify_test_fn = Tuple(Stack(),
                             Stack(),
                             Stack(),
                             Stack(),
                             Stack())

    test_sampler = FixedBucketSampler(lengths=[item[3] for item in processed_test_dataset],
                                      batch_size=args.batch_size)

    test_dataloader = DataLoader(processed_test_dataset.transform(filter_out_question_id),
                                 batchify_fn=batchify_test_fn,
                                 batch_sampler=test_sampler,
                                 num_workers=10,
                                 shuffle=False)

    predictions = run_evaluation(model, test_dataloader, best_threshold)

    submission = pd.DataFrame()
    mapping = {item[0]: item[1] for item in test_dataset}
    submission['qid'] = [mapping[p[0]] for p in predictions]
    submission['prediction'] = [p[1] for p in predictions]

    submission.to_csv(args.output, index=False)


