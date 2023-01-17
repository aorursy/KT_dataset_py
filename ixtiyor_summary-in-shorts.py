# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import tensorflow_datasets as tfds

import unicodedata

import re

import time

import matplotlib.pyplot as plt

import os

from sklearn.model_selection import train_test_split



# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
input_file = "../input/inshorts-news-data/Inshorts Cleaned Data.xlsx"

df = pd.read_excel(input_file)
# Converts the unicode file to ascii

def unicode_to_ascii(s):

  return ''.join(c for c in unicodedata.normalize('NFD', s)

      if unicodedata.category(c) != 'Mn')





def preprocess_sentence(w):

  w = unicode_to_ascii(w.lower().strip())



  # creating a space between a word and the punctuation following it

  # eg: "he is a boy." => "he is a boy ."

  # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation

  w = re.sub(r"([?.!,¿])", r" \1 ", w)

  w = re.sub(r'[" "]+', " ", w)



  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")

  w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)



  w = w.strip()



  # adding a start and an end token to the sentence

  # so that the model know when to start and stop predicting.

  w = unicode_to_ascii('<go> ') + w + unicode_to_ascii(' <stop>')

  return w
document = df['Headline']

summary = df['Short']

summary = summary.apply(lambda x: preprocess_sentence(x))

summary[0]
tokenizer=  None

tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
max_l = 0

for i in range(len(document)):

    if(max_l < len(document[i].split(' '))):

        max_l = len(document[i].split(' '))
max_l
tokenizer.fit_on_texts(summary)

summary_tokens = tokenizer.texts_to_sequences(summary)

headline_tokens = tokenizer.texts_to_sequences(document)
sample_string = np.array(['tracking weak cues from the asian markets '])

tokenized_string  = tokenizer.texts_to_sequences(sample_string)

tokenized_string = tf.keras.preprocessing.sequence.pad_sequences(tokenized_string, padding='post', maxlen=89)

original_string = tokenizer.sequences_to_texts(tokenized_string)
print('original string', original_string)

print('tokenized ', tokenized_string)

print(tokenizer.word_index['<stop>'])

print(tokenizer.word_index['<go>'])

tokenizer.word_index['<go>']


"""

max summary length is 89

max headline length is  14

"""

token_sum  = tokenizer.texts_to_sequences(summary)

token_sum = tf.keras.preprocessing.sequence.pad_sequences(token_sum, padding='post', maxlen=89)



token_hd  = tokenizer.texts_to_sequences(document)

token_hd = tf.keras.preprocessing.sequence.pad_sequences(token_hd, padding='post', maxlen=14)



input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(token_sum, token_hd, test_size=0.2)



# Show length

print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))
# BUFFER_SIZE = 20000

# BATCH_SIZE = 64

BUFFER_SIZE = len(input_tensor_train)

BATCH_SIZE = 1024

steps_per_epoch = len(input_tensor_train)//BATCH_SIZE

embedding_dim = 256

units = 1024

vocab_inp_size = len(tokenizer.word_index)+1

vocab_tar_size = len(tokenizer.word_index)+1



dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)

dataset = dataset.padded_batch(BATCH_SIZE, drop_remainder=True)

dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
vocab_inp_size
example_input_batch, example_target_batch = next(iter(dataset))

example_input_batch.shape, example_target_batch.shape
def get_angles(pos, i, d_model):

  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))

  return pos * angle_rates
def positional_encoding(position, d_model):

  angle_rads = get_angles(np.arange(position)[:, np.newaxis],

                          np.arange(d_model)[np.newaxis, :],

                          d_model)

  

  # apply sin to even indices in the array; 2i

  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  

  # apply cos to odd indices in the array; 2i+1

  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    

  pos_encoding = angle_rads[np.newaxis, ...]

    

  return tf.cast(pos_encoding, dtype=tf.float32)
pos_encoding = positional_encoding(50, 512)

print (pos_encoding.shape)



plt.pcolormesh(pos_encoding[0], cmap='RdBu')

plt.xlabel('Depth')

plt.xlim((0, 512))

plt.ylabel('Position')

plt.colorbar()

plt.show()
def create_padding_mask(seq):

  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

  

  # add extra dimensions to add the padding

  # to the attention logits.

  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])

create_padding_mask(x)
def create_look_ahead_mask(size):

  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

  return mask  # (seq_len, seq_len)
x = tf.random.uniform((1, 3))

temp = create_look_ahead_mask(x.shape[1])

temp
def scaled_dot_product_attention(q, k, v, mask):

  """Calculate the attention weights.

  q, k, v must have matching leading dimensions.

  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.

  The mask has different shapes depending on its type(padding or look ahead) 

  but it must be broadcastable for addition.

  

  Args:

    q: query shape == (..., seq_len_q, depth)

    k: key shape == (..., seq_len_k, depth)

    v: value shape == (..., seq_len_v, depth_v)

    mask: Float tensor with shape broadcastable 

          to (..., seq_len_q, seq_len_k). Defaults to None.

    

  Returns:

    output, attention_weights

  """



  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  

  # scale matmul_qk

  dk = tf.cast(tf.shape(k)[-1], tf.float32)

  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)



  # add the mask to the scaled tensor.

  if mask is not None:

    scaled_attention_logits += (mask * -1e9)  



  # softmax is normalized on the last axis (seq_len_k) so that the scores

  # add up to 1.

  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)



  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)



  return output, attention_weights
def print_out(q, k, v):

  temp_out, temp_attn = scaled_dot_product_attention(

      q, k, v, None)

  print ('Attention weights are:')

  print (temp_attn)

  print ('Output is:')

  print (temp_out)
np.set_printoptions(suppress=True)



temp_k = tf.constant([[10,0,0],

                      [0,10,0],

                      [0,0,10],

                      [0,0,10]], dtype=tf.float32)  # (4, 3)



temp_v = tf.constant([[   1,0],

                      [  10,0],

                      [ 100,5],

                      [1000,6]], dtype=tf.float32)  # (4, 2)



# This `query` aligns with the second `key`,

# so the second `value` is returned.

temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)

print_out(temp_q, temp_k, temp_v)
# This query aligns with a repeated key (third and fourth), 

# so all associated values get averaged.

temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)  # (1, 3)

print_out(temp_q, temp_k, temp_v)
class MultiHeadAttention(tf.keras.layers.Layer):

  def __init__(self, d_model, num_heads):

    super(MultiHeadAttention, self).__init__()

    self.num_heads = num_heads

    self.d_model = d_model

    

    assert d_model % self.num_heads == 0

    

    self.depth = d_model // self.num_heads

    

    self.wq = tf.keras.layers.Dense(d_model)

    self.wk = tf.keras.layers.Dense(d_model)

    self.wv = tf.keras.layers.Dense(d_model)

    

    self.dense = tf.keras.layers.Dense(d_model)

        

  def split_heads(self, x, batch_size):

    """Split the last dimension into (num_heads, depth).

    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)

    """

    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))

    return tf.transpose(x, perm=[0, 2, 1, 3])

    

  def call(self, v, k, q, mask):

    batch_size = tf.shape(q)[0]

    

    q = self.wq(q)  # (batch_size, seq_len, d_model)

    k = self.wk(k)  # (batch_size, seq_len, d_model)

    v = self.wv(v)  # (batch_size, seq_len, d_model)

    

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)

    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)

    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)

    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)

    scaled_attention, attention_weights = scaled_dot_product_attention(

        q, k, v, mask)

    

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)



    concat_attention = tf.reshape(scaled_attention, 

                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)



    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        

    return output, attention_weights
temp_mha = MultiHeadAttention(d_model=512, num_heads=8)

y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)

out, attn = temp_mha(y, k=y, q=y, mask=None)

out.shape, attn.shape
def point_wise_feed_forward_network(d_model, dff):

  return tf.keras.Sequential([

      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)

      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)

  ])
sample_ffn = point_wise_feed_forward_network(512, 2048)

sample_ffn(tf.random.uniform((64, 50, 512))).shape
class EncoderLayer(tf.keras.layers.Layer):

  def __init__(self, d_model, num_heads, dff, rate=0.1):

    super(EncoderLayer, self).__init__()



    self.mha = MultiHeadAttention(d_model, num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)



    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    

    self.dropout1 = tf.keras.layers.Dropout(rate)

    self.dropout2 = tf.keras.layers.Dropout(rate)

    

  def call(self, x, training, mask):



    attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)

    attn_output = self.dropout1(attn_output, training=training)

    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

    

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)

    ffn_output = self.dropout2(ffn_output, training=training)

    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

    

    return out2
sample_encoder_layer = EncoderLayer(512, 8, 2048)



sample_encoder_layer_output = sample_encoder_layer(

    tf.random.uniform((64, 43, 512)), False, None)



sample_encoder_layer_output.shape  # (batch_size, input_seq_len, d_model)
class DecoderLayer(tf.keras.layers.Layer):

  def __init__(self, d_model, num_heads, dff, rate=0.1):

    super(DecoderLayer, self).__init__()



    self.mha1 = MultiHeadAttention(d_model, num_heads)

    self.mha2 = MultiHeadAttention(d_model, num_heads)



    self.ffn = point_wise_feed_forward_network(d_model, dff)

 

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    

    self.dropout1 = tf.keras.layers.Dropout(rate)

    self.dropout2 = tf.keras.layers.Dropout(rate)

    self.dropout3 = tf.keras.layers.Dropout(rate)

    

    

  def call(self, x, enc_output, training, 

           look_ahead_mask, padding_mask):

    # enc_output.shape == (batch_size, input_seq_len, d_model)



    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)

    attn1 = self.dropout1(attn1, training=training)

    out1 = self.layernorm1(attn1 + x)

    

    attn2, attn_weights_block2 = self.mha2(

        enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)

    attn2 = self.dropout2(attn2, training=training)

    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

    

    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)

    ffn_output = self.dropout3(ffn_output, training=training)

    out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

    

    return out3, attn_weights_block1, attn_weights_block2
sample_decoder_layer = DecoderLayer(512, 8, 2048)



sample_decoder_layer_output, _, _ = sample_decoder_layer(

    tf.random.uniform((64, 50, 512)), sample_encoder_layer_output, 

    False, None, None)



sample_decoder_layer_output.shape  # (batch_size, target_seq_len, d_model)
class Encoder(tf.keras.layers.Layer):

  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,

               maximum_position_encoding, rate=0.1):

    super(Encoder, self).__init__()



    self.d_model = d_model

    self.num_layers = num_layers

    

    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)

    self.pos_encoding = positional_encoding(maximum_position_encoding, 

                                            self.d_model)

    

    

    self.enc_layers =[EncoderLayer(d_model, num_heads, dff, rate) 

                       for _ in range(num_layers)]

  

    self.dropout = tf.keras.layers.Dropout(rate)

        

  def call(self, x, training, mask):



    seq_len = tf.shape(x)[1]

    

    # adding embedding and position encoding.

    x = self.embedding(x)  # (batch_size, input_seq_len, d_model)

    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

    x += self.pos_encoding[:, :seq_len, :]



    x = self.dropout(x, training=training)

    

    for i in range(self.num_layers):

        x = self.enc_layers[i](x, training, mask)

    

    return x  # (batch_size, input_seq_len, d_model)

sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8, 

                         dff=2048, input_vocab_size=8500,

                         maximum_position_encoding=10000)

temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)



sample_encoder_output = sample_encoder(temp_input, training=False, mask=None)



print (sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)
class Decoder(tf.keras.layers.Layer):

  def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,

               maximum_position_encoding, rate=0.1):

    super(Decoder, self).__init__()



    self.d_model = d_model

    self.num_layers = num_layers

    

    self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)

    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

    

    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)# [

                       for _ in range(num_layers)]

    self.dropout = tf.keras.layers.Dropout(rate)

    

  def call(self, x, enc_output, training, 

           look_ahead_mask, padding_mask):



    seq_len = tf.shape(x)[1]

    attention_weights = {}

    

    x = self.embedding(x)  # (batch_size, target_seq_len, d_model)

    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

    x += self.pos_encoding[:, :seq_len, :]

    

    x = self.dropout(x, training=training)



    for i in range(self.num_layers):

    

        x, block1, block2 = self.dec_layers[i](x, enc_output, training,

                                         look_ahead_mask, padding_mask)



        attention_weights['decoder_layer{}_block1'.format(i+1)] = block1

        attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

    

    # x.shape == (batch_size, target_seq_len, d_model)

    return x, attention_weights
sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8, 

                         dff=2048, target_vocab_size=8000,

                         maximum_position_encoding=5000)

temp_input = tf.random.uniform((64, 26), dtype=tf.int64, minval=0, maxval=200)



output, attn = sample_decoder(temp_input, 

                              enc_output=sample_encoder_output, 

                              training=False,

                              look_ahead_mask=None, 

                              padding_mask=None)



print(output.shape, attn['decoder_layer2_block2'].shape)
class Transformer(tf.keras.Model):

  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 

               target_vocab_size, pe_input, pe_target, rate=0.1):

    super(Transformer, self).__init__()



    self.encoder = Encoder(num_layers, d_model, num_heads, dff, 

                           input_vocab_size, pe_input, rate)



    self.decoder = Decoder(num_layers, d_model, num_heads, dff, 

                           target_vocab_size, pe_target, rate)



    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    

  def call(self, inp, tar, training, enc_padding_mask, 

           look_ahead_mask, dec_padding_mask):



    enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

    

    # dec_output.shape == (batch_size, tar_seq_len, d_model)

    dec_output, attention_weights = self.decoder(

        tar, enc_output, training, look_ahead_mask, dec_padding_mask)

    

    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

    

    return final_output, attention_weights
sample_transformer = Transformer(

    num_layers=2, d_model=512, num_heads=8, dff=2048, 

    input_vocab_size=tokenizer.document_count, target_vocab_size=tokenizer.document_count, 

    pe_input=10000, pe_target=6000)



temp_input = tf.random.uniform((64, 89), dtype=tf.int64, minval=0, maxval=200)

temp_target = tf.random.uniform((64, 14), dtype=tf.int64, minval=0, maxval=200)



fn_out, _ = sample_transformer(temp_input, temp_target, training=False, 

                               enc_padding_mask=None, 

                               look_ahead_mask=None,

                               dec_padding_mask=None)



fn_out.shape  # (batch_size, tar_seq_len, target_vocab_size
num_layers = 4

d_model = 128

dff = 512

num_heads = 8



input_vocab_size = 65262 + 2

target_vocab_size = 65262 + 2

dropout_rate = 0.1
tokenizer.word_index['robotarium']
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, d_model, warmup_steps=4000):

    super(CustomSchedule, self).__init__()

    

    self.d_model = d_model

    self.d_model = tf.cast(self.d_model, tf.float32)



    self.warmup_steps = warmup_steps

    

  def __call__(self, step):

    arg1 = tf.math.rsqrt(step)

    arg2 = step * (self.warmup_steps ** -1.5)

    

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
learning_rate = CustomSchedule(d_model)



optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 

                                     epsilon=1e-9)
temp_learning_rate_schedule = CustomSchedule(d_model)



plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))

plt.ylabel("Learning Rate")

plt.xlabel("Train Step")
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(

    from_logits=True, reduction='none')
def loss_function(real, pred):

  mask = tf.math.logical_not(tf.math.equal(real, 0))

  loss_ = loss_object(real, pred)



  mask = tf.cast(mask, dtype=loss_.dtype)

  loss_ *= mask

  

  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)
train_loss = tf.keras.metrics.Mean(name='train_loss')

train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(

    name='train_accuracy')
transformer = Transformer(num_layers, d_model, num_heads, dff,

                          input_vocab_size, target_vocab_size, 

                          pe_input=input_vocab_size, 

                          pe_target=target_vocab_size,

                          rate=dropout_rate)
def create_masks(inp, tar):

  # Encoder padding mask

  enc_padding_mask = create_padding_mask(inp)

  

  # Used in the 2nd attention block in the decoder.

  # This padding mask is used to mask the encoder outputs.

  dec_padding_mask = create_padding_mask(inp)

  

  # Used in the 1st attention block in the decoder.

  # It is used to pad and mask future tokens in the input received by 

  # the decoder.

  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])

  dec_target_padding_mask = create_padding_mask(tar)

  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

  

  return enc_padding_mask, combined_mask, dec_padding_mask
checkpoint_path = "./checkpoints/train"



ckpt = tf.train.Checkpoint(transformer=transformer,

                           optimizer=optimizer)



ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)



# if a checkpoint exists, restore the latest checkpoint.

if ckpt_manager.latest_checkpoint:

  ckpt.restore(ckpt_manager.latest_checkpoint)

  print ('Latest checkpoint restored!!')
EPOCHS = 100
# The @tf.function trace-compiles train_step into a TF graph for faster

# execution. The function specializes to the precise shape of the argument

# tensors. To avoid re-tracing due to the variable sequence lengths or variable

# batch sizes (the last batch is smaller), use input_signature to specify

# more generic shapes.



train_step_signature = [

    tf.TensorSpec(shape=(None, None), dtype=tf.int32),

    tf.TensorSpec(shape=(None, None), dtype=tf.int32),

]



@tf.function(input_signature=train_step_signature)

def train_step(inp, tar):

  tar_inp = tar[:, :-1]

  tar_real = tar[:, 1:]

  

  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

#   print(np.shape(enc_padding_mask), np.shape(combined_mask), np.shape(dec_padding_mask))

  with tf.GradientTape() as tape:

    predictions, _ = transformer(inp, tar_inp, 

                                 True, 

                                 enc_padding_mask, 

                                 combined_mask, 

                                 dec_padding_mask)

    loss = loss_function(tar_real, predictions)



  gradients = tape.gradient(loss, transformer.trainable_variables)    

  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  

  train_loss(loss)

  train_accuracy(tar_real, predictions)
for epoch in range(EPOCHS):

  start = time.time()

  

  train_loss.reset_states()

  train_accuracy.reset_states()

  

  # inp -> portuguese, tar -> english

  for (batch, (inp, tar)) in enumerate(dataset):

    train_step(inp, tar)

    

    if batch % 50 == 0:

      print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(

          epoch + 1, batch, train_loss.result(), train_accuracy.result()))

      

  if (epoch + 1) % 5 == 0:

    ckpt_save_path = ckpt_manager.save()

    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,

                                                         ckpt_save_path))

    

  print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 

                                                train_loss.result(), 

                                                train_accuracy.result()))



  print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
kk = next(dataset.as_numpy_iterator())
kk[0].shape
kk[1].shape
tf.test.is_gpu_available(

    cuda_only=True

)