import os, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

df = pd.read_json("../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json", lines=True)
df.head()
df = df.drop(['article_link'], axis=1)
max_features = 10000
maxlen = 25
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(df['headline']))
X = tokenizer.texts_to_sequences(df['headline'])
X = pad_sequences(X, maxlen=maxlen)
Y = df['is_sarcastic']
print(f'Shape of data: {X.shape}')
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, train_size=0.9)
BUFFER_SIZE = 10000
BATCH_SIZE = 256
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train.values.reshape(-1, 1)))
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test.values.reshape(-1, 1)))
test_dataset = test_dataset.cache()
test_dataset = test_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.prefetch(AUTOTUNE)
x_batch, y_batch = next(iter(train_dataset))
def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
  pos_encoding = angle_rads[np.newaxis, ...]
  return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  return seq[:, tf.newaxis, tf.newaxis, :] 
def scaled_dot_product_attention(q, k, v, mask):
  matmul_qk = tf.matmul(q, k, transpose_b=True)
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
  output = tf.matmul(attention_weights, v)
  return output, attention_weights
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
    
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]
    
    q = self.wq(q)  
    k = self.wk(k) 
    v = self.wv(v)  
    
    q = self.split_heads(q, batch_size)  
    k = self.split_heads(k, batch_size)  
    v = self.split_heads(v, batch_size) 
    
    
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)
    
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3]) 

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  
    output = self.dense(concat_attention) 
        
    return output, attention_weights
def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  
      tf.keras.layers.Dense(d_model)  
  ])
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

    attn_output, _ = self.mha(x, x, x, mask)  
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  
    
    ffn_output = self.ffn(out1)  
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  
    
    return out2
class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, embedding_weights=None, rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    
    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model, weights=embedding_weights)
    self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                            self.d_model)
    
    
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
  
    self.dropout = tf.keras.layers.Dropout(rate)
        
  def call(self, x, training, mask):

    seq_len = tf.shape(x)[1]
    
   
    x = self.embedding(x)  
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)
    
    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)
    
    return x  
class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
               pe_input, fl_dff, embedding_weights=None, rate=0.1):
    super(Transformer, self).__init__()

    self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                           input_vocab_size, pe_input, embedding_weights, rate)

    self.final_layer = tf.keras.layers.Dense(fl_dff)
    
  def call(self, inp, training, enc_padding_mask):

    enc_output = self.encoder(inp, training, enc_padding_mask) 
    enc_output_shape = enc_output.shape
    enc_output = tf.reshape(enc_output, shape=(enc_output_shape[0],
                                               enc_output_shape[1] * enc_output_shape[2]))
    
    final_output = self.final_layer(enc_output) 
    return final_output
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
def train_step(inp, tar, transformer):
  with tf.GradientTape() as tape:
    predictions = transformer(inp, True, None)
    loss = loss_object_train(tar, predictions)
  gradients = tape.gradient(loss, transformer.trainable_variables)
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
  train_loss(loss)
  train_accuracy(tar, predictions)
def custom_train_loop(transformer, save_checkpoint=False):

  history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

  for epoch in range(EPOCHS):
    start = time.time()
    train_loss.reset_states()
    train_accuracy.reset_states()

    for (batch, (inp, tar)) in enumerate(train_dataset):
      train_step(inp, tar, transformer)

      if batch % 50 == 0:
        print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

    if save_checkpoint and (epoch + 1) % 5 == 0:
      ckpt_save_path = ckpt_manager.save()
      print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')

    print(f'Epoch {epoch + 1} Train_Loss {train_loss.result():.4f} Train_Accuracy {train_accuracy.result():.4f}')

    history['loss'].append(train_loss.result().numpy())
    history['accuracy'].append(train_accuracy.result().numpy())

    val_loss.reset_states()
    val_accuracy.reset_states()
    for (inp, tar) in test_dataset:
      predictions = transformer(inp, True, None)
      loss_val = loss_object_test(tar, predictions)
      val_loss(loss_val)
      val_accuracy(tar, predictions)

    print(f'Epoch {epoch + 1} Val_loss {val_loss.result():.4f} Val_Accuracy {val_accuracy.result():.4f}')
    history['val_loss'].append(val_loss.result().numpy())
    history['val_accuracy'].append(val_accuracy.result().numpy())
    print(f'Time taken for 1 epoch: {time.time() - start:.4f} secs\n')
    print('-------------------------------------\n')
  
  return history
num_layers = 4
d_model = 200
dff = 512
num_heads = 10
input_vocab_size = max_features
dropout_rate = 0.1
fl_dff = 1
EPOCHS = 20
learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
loss_object_train = tf.keras.losses.BinaryCrossentropy(
    from_logits=True, reduction='none')
loss_object_test = tf.keras.losses.BinaryCrossentropy(
    from_logits=True, reduction='none')

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')
transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size,  
                          pe_input=input_vocab_size, 
                          fl_dff=fl_dff,
                          embedding_weights=None,
                          rate=dropout_rate)
checkpoint_path = './checkpoints/train'
ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
history_transformer = custom_train_loop(transformer, 
                                        save_checkpoint=False)
def plot(train, val, label, title):
  plt.plot(range(EPOCHS), train, label='Train_' + label)
  plt.plot(range(EPOCHS), val, label='Val_' + label, color='g')
  if label == 'Accuracy':
    vline_cut = np.argmax(val)
  if label == 'Loss':
    vline_cut = np.argmin(val)
  plt.axvline(vline_cut, color='k', ls='--')
  plt.ylabel(label)
  plt.xlabel('Epoch')
  plt.legend()
  plt.grid(True)
  plt.title(title)
  plt.show()
plot(history_transformer['accuracy'], 
     history_transformer['val_accuracy'], 
     'Accuracy', 'Transformer Accuracy')
plot(history_transformer['loss'], 
     history_transformer['val_loss'], 
     'Loss', 'Transformer Loss')
print('Result:\n')
print(f"Transformer Accuracy: {np.max(history_transformer['val_accuracy']):.3f}")
