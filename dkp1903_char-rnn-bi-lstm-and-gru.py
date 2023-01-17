## IT-412 - Project - Text Generation using RNN - Implemented using Tensorflow-Keras
## Implementation : Character Level RNN has been implemented as Bi-LSTM/GRU Model
## Using tensorboard to view metrics
%load_ext tensorboard
## Importing the necessary libraries
import tensorflow as tf
import numpy as np
import os
import datetime
import time
#from keras import backend as K
## Dataset used is Texts of Trade Agreements (ToTA) dataset - Courtesy of Dr Wolgang Alschner
## The dataset is a set of 450 XML files in English as well as other languages, written as Bilateral treaties.
## The data has been compiled into a single text file, split into 80:20 - Train and test
path_to_file = '/content/drive/My Drive/dataset_1.txt'
path_to_test = '/content/drive/My Drive/test.txt'
## Opening the train file
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
## Vocab is a sorted set of all chars appearing in the train data
## We have used Python set to remove duplication
vocab = sorted(set(text))
print ('Length of train text: {} characters'.format(len(text)))
## Opening the test file
test = open(path_to_test, 'rb').read().decode(encoding='utf-8')
print(type(test))
vocab_test = sorted(set(test))
print ('Length of test text: {} characters'.format(len(test)))
## Mapping the chars to indices
char2idx = {u:i for i, u in enumerate(vocab)}
## Index 0 - 0th character - Expressed as numpy array
idx2char = np.array(vocab)
## Shows the integer each character in text is mapped to
text_as_int = np.array([char2idx[c] for c in text])
# Shows the mapping
print ('{} ---- Mapped to ---- > {}'.format(repr(text[:13]), text_as_int[:13]))
## Same for test data
ctoi = {u:i for i, u in enumerate(vocab_test)}
i2c = np.array(vocab_test)
t_as_i = np.array([ctoi[c] for c in test])
print(t_as_i)
## The maximum length sentence for a single input in characters
seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)
## Convert the text vector into a stream of character indices.
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

examples_per_epoch_test = len(test)//(seq_length + 1)
char_dataset_test = tf.data.Dataset.from_tensor_slices(t_as_i)
## Batch converts individual characters into sequences of desired size
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
sequences_test = char_dataset_test.batch(seq_length + 1, drop_remainder = True)
## For each sentence(chunk), duplicate and shift by one to form main->target. 
## Use map function to map these
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)
## Similarly for test dataset
dataset_test = sequences_test.map(split_input_target)
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
dataset_test = dataset_test.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024
## Building the model
## We use
##  1. Embedding : Input layer -> Maps char to vector
##  2. LSTM/GRU : RNN 
##  3. Dense : Output layer
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
                               
    ## Comment the LSTM function, and uncomment the following 
    ## GRU function to see a GRU based implementation
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
  
    # tf.keras.layers.GRU(rnn_units,
    #                     return_sequences=True,
    #                     stateful=True,
    #                     recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model
## Building the model

model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)
for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
model.summary()
## Sampling to get actual character indices
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ])))
## We use the categorical_crossentropy function as it is applied across the last dimension of
## the prediction
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
#loss = example_batch_loss.numpy().mean()
print("scalar_loss: ", example_batch_loss.numpy().mean())
print("Perplexity: ", tf.exp(example_batch_loss.numpy().mean()))
## Compiling the model with metric as accuracy
## What accuracy in this use case means, that how accurately can the model predict the next letter.
model.compile(optimizer='adam', loss=loss, metrics='accuracy')
# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)
EPOCHS = 20
## Uncomment to view TensorBoard analytics - This might result in errors, since the tensorboard works only after fitting to the model. 
## In such a case, please restart runtime. 

## log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
## tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
## history_for_tf = model.fit(dataset, epochs=EPOCHS, callbacks=[tensorboard_callback])
## %tensorboard --logdir logs/fit
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
tf.train.latest_checkpoint(checkpoint_dir)
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))
model.summary()
def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 10000  

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

      a = (start_string + ''.join(text_generated))
  return a
output_file = open('output.txt','w+')
t = generate_text(model, start_string=u"The government: ")
output_file.write(t)
vocab_output = sorted(set(t))
output_file.close() 

from google.colab import files
files.download('output.txt') 
model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)
optimizer = tf.keras.optimizers.Adam()
## print(type(model))
## GradientTape is used to track gradients
## 1. Initialize RNN state using reset_states
## 2. Iterate by batch and calculate the predictions
## 3. Use GradientTape for predictions and loss
## 4. Calculate the gradients of the loss with respect to the model variables
## 5. Apply the gradients using Adam optimizer
@tf.function
def train_step(inp, target):
  with tf.GradientTape() as tape:
    predictions = model(inp)
    #fin_pred = predictions
    loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            target, predictions, from_logits=True))
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  return loss

# Training step
EPOCHS = 10

for epoch in range(EPOCHS):
  start = time.time()

  # initializing the hidden state at the start of every epoch
  # initally hidden is None
  hidden = model.reset_states()

  for (batch_n, (inp, target)) in enumerate(dataset):
    predictions = model(inp)
    loss = train_step(inp, target)
    ppl = tf.exp(loss)
    prediction = tf.dtypes.cast(predictions, tf.float32)
    
    if batch_n % 100 == 0:
      template = 'Epoch {} Batch {} Loss {} PPL {} '
      print(template.format(epoch+1, batch_n, loss, ppl))

  # saving (checkpoint) the model every 5 epochs
  if (epoch + 1) % 5 == 0:
    model.save_weights(checkpoint_prefix.format(epoch=epoch))

  print ('Epoch {} Loss {:.4f} PPL : {:.6f}'.format(epoch+1, loss, ppl))
  print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

model.save_weights(checkpoint_prefix.format(epoch=epoch))
EPOCHS = 10

for epoch in range(EPOCHS):
  start = time.time()

  for (batch_n, (inp, target)) in enumerate(dataset_test):
      predictions = model(inp)
      loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            target, predictions, from_logits=True))
      ppl = 2**loss
    

      if batch_n % 100 == 0:
        template = 'Epoch {} Batch {} Loss {} PPL {}'
        print(template.format(epoch+1, batch_n, loss, 2**loss))
#calculating Jaccard distance between Output and Test datas
ctoi_output = {u:i for i, u in enumerate(vocab_output)}
t_as_i_output = np.array([ctoi_output[c] for c in t])
b = set(t_as_i.tolist())
a = set(t_as_i_output.tolist())
import nltk
print("Jaccard distance between output and test:", nltk.jaccard_distance(a, b) )
