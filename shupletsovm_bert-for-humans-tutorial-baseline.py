!pip install "tensorflow>=2"

!pip install "tensorflow_hub>=0.7"

!pip install bert-for-tf2

!pip install sentencepiece
import tensorflow as tf

import tensorflow_hub as hub

print("TF version: ", tf.__version__)

print("Hub version: ", hub.__version__)
import tensorflow_hub as hub

import tensorflow as tf

import bert

FullTokenizer = bert.bert_tokenization.FullTokenizer

from tensorflow.keras.models import Model       # Keras is the new high level API for TensorFlow

import math

import numpy as np
max_seq_length = 128  # Your choice here.

input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,

                                       name="input_word_ids")

input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,

                                   name="input_mask")

segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,

                                    name="segment_ids")

bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",

                            trainable=True)

pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])



model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[pooled_output, sequence_output])



vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = FullTokenizer(vocab_file, do_lower_case)
# See BERT paper: https://arxiv.org/pdf/1810.04805.pdf

# And BERT implementation convert_single_example() at https://github.com/google-research/bert/blob/master/run_classifier.py



def get_masks(tokens, max_seq_length):

    """Mask for padding"""

    if len(tokens)>max_seq_length:

        raise IndexError("Token length more than max seq length!")

    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))





def get_segments(tokens, max_seq_length):

    """Segments: 0 for the first sequence, 1 for the second"""

    if len(tokens)>max_seq_length:

        raise IndexError("Token length more than max seq length!")

    segments = []

    current_segment_id = 0

    for token in tokens:

        segments.append(current_segment_id)

        if token == "[SEP]":

            current_segment_id = 1

    return segments + [0] * (max_seq_length - len(tokens))





def get_ids(tokens, tokenizer, max_seq_length):

    """Token ids from Tokenizer vocab"""

    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))

    return input_ids
def tokenize_sentence(sentence):

    stokens = tokenizer.tokenize(sentence)

    stokens = ["[CLS]"] + stokens + ["[SEP]"]

    

    input_ids = get_ids(stokens, tokenizer, max_seq_length)

    input_masks = get_masks(stokens, max_seq_length)

    input_segments = get_segments(stokens, max_seq_length)

    

    return input_ids, input_masks, input_segments



def compare_sentences(sentence_1, sentence_2, distance_metric):

    input_ids_1, input_masks_1, input_segments_1 = tokenize_sentence(sentence_1)

    input_ids_2, input_masks_2, input_segments_2 = tokenize_sentence(sentence_2)

    

    pool_embs_1, all_embs_1 = model.predict([[input_ids_1],[input_masks_1],[input_segments_1]])

    pool_embs_2, all_embs_2 = model.predict([[input_ids_2],[input_masks_2],[input_segments_2]])

    

    return distance_metric(pool_embs_1[0], pool_embs_2[0])

    

def square_rooted(x):

    return math.sqrt(sum([a*a for a in x]))



def cosine_similarity(x,y):

    numerator = sum(a*b for a,b in zip(x,y))

    denominator = square_rooted(x)*square_rooted(y)

    return numerator/float(denominator)



def dummy_metric(x,y):

    return 42
s1 = 'How are you doing?'

# s2 = '''Right after Ricky Gervais talks about how the Hollywood Foreign Press is racist and doesn't include people of color the cameraman zooms out to show just how few people of color were invited to this event.'''

s2 = 'How are we feeling?'

compare_sentences(s1, s2, cosine_similarity)