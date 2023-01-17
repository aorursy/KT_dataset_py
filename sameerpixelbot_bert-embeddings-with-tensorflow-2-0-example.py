!pip install --upgrade tensorflow
!pip install bert-tensorflow
!pip install tf-hub-nightly
import tensorflow as tf
import tensorflow_hub as hub
print("TF version: ", tf.__version__)
print("Hub version: ", hub.__version__)
import tensorflow_hub as hub
import tensorflow as tf
from bert.tokenization import FullTokenizer     # Still from bert module
from tensorflow.keras.models import Model       # Keras is the new high level API for TensorFlow
import math
max_seq_length = 1  # Your choice here.
input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,),
                                       dtype=tf.int32,
                                       name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(max_seq_length,),
                                   dtype=tf.int32,
                                   name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(max_seq_length,),
                                    dtype=tf.int32,
                                    name="segment_ids")
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=True)
pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[pooled_output, sequence_output])
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
# Google Colab don't need this. FullTokenizer is not updated to tf2.0 yet
tf.gfile = tf.io.gfile
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case)
s0 = "nice"
stokens0 = tokenizer.tokenize(s0)

input_ids0 = get_ids(stokens0, tokenizer, max_seq_length)
input_masks0 = get_masks(stokens0, max_seq_length)
input_segments0 = get_segments(stokens0, max_seq_length)

print(stokens0)
print(input_ids0)
print(input_masks0)
print(input_segments0)
s1 = "nasty"
stokens1 = tokenizer.tokenize(s1)

input_ids1 = get_ids(stokens1, tokenizer, max_seq_length)
input_masks1 = get_masks(stokens1, max_seq_length)
input_segments1 = get_segments(stokens1, max_seq_length)

print(stokens1)
print(input_ids1)
print(input_masks1)
print(input_segments1)
import numpy as np
input_ids0 = np.array(input_ids0)
input_masks0 = np.array(input_masks0)
input_segments0 = np.array(input_segments0)
input_ids1 = np.array(input_ids1)
input_masks1 = np.array(input_masks1)
input_segments1 = np.array(input_segments1)
type(input_masks0)
len(s0)
pool_embs0, all_embs0 = model.predict([[input_ids0],[input_masks0],[input_segments0]])
pool_embs1, all_embs1 = model.predict([[input_ids1],[input_masks1],[input_segments1]])
def square_rooted(x):
    return math.sqrt(sum([a*a for a in x]))


def cosine_similarity(x,y):
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return numerator/float(denominator)
cosine_similarity(pool_embs0[0], all_embs0[0][127])
cosine_similarity(pool_embs1[0], all_embs1[0][127])
cosine_similarity(pool_embs0[0],pool_embs1[0])
cosine_similarity(all_embs0[0][0],all_embs1[0][0])
all_embs0[0][0]
pool_embs0.shape
pool_embs1.shape
all_embs1.shape
all_embs0.shape
len(pool_embs[0])
pooled_output[0]
pool_embs0[0]
print(pool_embs0[0])
print(pool_embs1[0])
len(all_embs[0][0])
all_embs[0][1]
