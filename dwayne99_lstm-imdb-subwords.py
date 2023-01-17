!pip install tensorflow-datasets
#Get the data
import tensorflow_datasets as tfds
import tensorflow as tf

imdb, info = tfds.load("imdb_reviews/subwords8k",with_info=True,as_supervised=True)
train , test = imdb['train'],imdb['test']
tokenizer = info.features['text'].encoder
tokenizer.vocab_size
BUFFER_SIZE = 10000
BATCH_SIZE = 64 #* tpu_strategy.num_replicas_in_sync

train = train.shuffle(BUFFER_SIZE)
train = train.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(train))
test = test.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(test))
# Model architecture

model = tf.keras.Sequential([

    tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
    #LSTM layer
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.summary()
# Building the model
model.compile(
    loss = 'binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
# Training the model
NUM_EPOCHS= 10
history = model.fit(
    train,
    epochs=NUM_EPOCHS,
    validation_data= test
)
import matplotlib.pyplot as plt

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend()
    plt.show()
plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
# Model architecture

model = tf.keras.Sequential([

    tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
    #LSTM layer
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True )),
    # note for more than one LSTM layers set return_sequences=True
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.summary()
# Building the model
model.compile(
    loss = 'binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
# Training the model
NUM_EPOCHS= 10
history = model.fit(
    train,
    epochs=NUM_EPOCHS,
    validation_data= test
)
import matplotlib.pyplot as plt

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend()
    plt.show()
plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
# Model architecture

model = tf.keras.Sequential([

    tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
    # 1D-Conv layer
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.summary()
# Building the model
model.compile(
    loss = 'binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
# Training the model
NUM_EPOCHS= 10
history = model.fit(
    train,
    epochs=NUM_EPOCHS,
    validation_data= test
)
import matplotlib.pyplot as plt

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend()
    plt.show()
plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')