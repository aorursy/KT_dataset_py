import tensorflow as tf



import numpy as np

import os

import time



data = open('../input/597-poems-by-emily-dickinson/final-emily.csv','rb')

corpus = data.read().decode(encoding='utf-8').strip()

vocab = sorted(set(corpus))
print ('Total characters:', len(corpus))

print ('Unique characters', len(vocab))
character_to_index = {u:i for i, u in enumerate(vocab)}

index_to_character = np.array(vocab)
corpus_int = np.array([character_to_index[c] for c in corpus])
seq_length = 100

examples_per_epoch = len(corpus)//(seq_length+1)



char_dataset = tf.data.Dataset.from_tensor_slices(corpus_int)

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
def split_input_target(chunk):

  input_text = chunk[:-1]

  target_text = chunk[1:]

  return input_text, target_text



dataset = sequences.map(split_input_target)
BATCH_SIZE = 64

BUFFER_SIZE = 10000

vocab_size = len(vocab)

embedding_dim = 256

rnn_units = 1024
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim,

                              batch_input_shape=[BATCH_SIZE, None]),

    tf.keras.layers.GRU(rnn_units,

                        return_sequences=True,

                        stateful=True,

                        recurrent_initializer='glorot_uniform'),

    tf.keras.layers.Dense(len(vocab))

])

    

model.summary()
checkpoint_dir = './tmp'



checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")



checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(

    filepath=checkpoint_prefix,

    save_weights_only=True)
def loss_fn(labels, logits):

    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
model.compile(optimizer='adam', loss=loss_fn)

history = model.fit(dataset, epochs=100, callbacks=[checkpoint_callback])
def generate_text(model, start_string):

  num_generate = 15



  input_eval = [character_to_index[s] for s in start_string]

  input_eval = tf.expand_dims(input_eval, 0)



  text_generated = []



  temperature = 1.0



  model.reset_states()

    

  while(num_generate > 0):

    predictions = model(input_eval)

    predictions = tf.squeeze(predictions, 0)



    predictions = predictions / temperature

    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()



    input_eval = tf.expand_dims([predicted_id], 0)

    text_generated.append(index_to_character[predicted_id])

    if index_to_character[predicted_id]=='\n':

        num_generate -= 1



  return (start_string + ''.join(text_generated)).strip()
model = tf.keras.Sequential([

    tf.keras.layers.Embedding(len(vocab), embedding_dim,

                              batch_input_shape=[1, None]),

    tf.keras.layers.GRU(rnn_units,

                        return_sequences=True,

                        stateful=True,

                        recurrent_initializer='glorot_uniform'),

    tf.keras.layers.Dense(len(vocab))

])

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))
print(generate_text(model, start_string=u"Love "))
print(generate_text(model, start_string=u"Flower "))