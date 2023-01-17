import tensorflow as tf
import numpy as np
import os
# Read text
path_to_file = '/kaggle/input/hp1txt/hp1.txt'
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# Build a vocabulary of unique characters in the text
vocab = sorted(set(text))

# Map each unique char to a different index
char2idx = {u: i for i, u in enumerate(vocab)}
# Map the index to the respective char
idx2char = np.array(vocab)
# Convert all the text to indices
text_as_int = np.array([char2idx[c] for c in text])

# Maximum length sentence we want for a single input
seq_length = 100

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)
def split_input_target(chunk):
    ''' Creates an input and target example for each sequence'''
    input_text = chunk[:-1]  # Removes the last character
    target_text = chunk[1:]  # Removes the first character
    return input_text, target_text
# Get inputs and targets for each sequence
dataset = sequences.map(split_input_target)
for input_example, target_example in  dataset.take(1):
  print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
  print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))
# Batch size
BATCH_SIZE = 64
BUFFER_SIZE = 10000
# Suffle the dataset and get batches
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    ''' Builds a simple sequencial 3 layers model '''
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)])
    return model
model = build_model(vocab_size=vocab_size,
                    embedding_dim=embedding_dim,
                    rnn_units=rnn_units,
                    batch_size=BATCH_SIZE)
model.summary()
def loss(labels, logits):
    ''' Performs Sparce Caterorical Crossentropy Loss '''
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)
# Define checkpoint path for each batch
checkpoint_path = "/kaggle/working/training_checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True)
# Number of epochs (full training pass over the entire dataset)
EPOCHS = 10
# Train the model
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
def generate_text(model, start_string, num_generate=1000, temperature=1.0):
    '''Generates text using the learned model'''

    # Converting our start string to numbers
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our result
    text_generated = []
    # Resets the state of metrics
    model.reset_states()

    for _ in range(num_generate):
        predictions = model(input_eval)
        # Remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # Using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(
            predictions, num_samples=1)[-1, 0].numpy()

        # We pass the predicted character as the next input to the model
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))
# Build network structure
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
# Load the weights of our latest learned model
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
# Build the learned model
model.build(tf.TensorShape([1, None]))
# Make predictions
predicted_text = generate_text(
    model, start_string='Harry ', num_generate=1000, temperature=1.0)

print(predicted_text)