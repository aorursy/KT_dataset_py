import tensorflow as tf
import numpy as np
import json
print(tf.__version__)
def load_data(path='imdb.npz',
              num_words=None,
              skip_top=0,
              maxlen=None,
              seed=113,
              start_char=1,
              oov_char=2,
              index_from=3,
              **kwargs):
    """Loads the IMDB dataset.
    Arguments:
      path: where to cache the data (relative to `~/.keras/dataset`).
      num_words: max number of words to include. Words are ranked
          by how often they occur (in the training set) and only
          the most frequent words are kept
      skip_top: skip the top N most frequently occurring words
          (which may not be informative).
      maxlen: sequences longer than this will be filtered out.
      seed: random seed for sample shuffling.
      start_char: The start of a sequence will be marked with this character.
          Set to 1 because 0 is usually the padding character.
      oov_char: words that were cut out because of the `num_words`
          or `skip_top` limit will be replaced with this character.
      index_from: index actual words with this index and higher.
      **kwargs: Used for backwards compatibility.
    Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    Raises:
      ValueError: in case `maxlen` is so low
          that no input sequence could be kept.
    Note that the 'out of vocabulary' character is only used for
    words that were present in the training set but are not included
    because they're not making the `num_words` cut here.
    Words that were not seen in the training set but are in the test set
    have simply been skipped.
    """
    
    
    # Legacy support
    if 'nb_words' in kwargs:
        logging.warning('The `nb_words` argument in `load_data` ''has been renamed `num_words`.')
        num_words = kwargs.pop('nb_words')
    if kwargs:
        raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))

    with np.load(path) as f:
        x_train, labels_train = f['x_train'], f['y_train']
        x_test, labels_test = f['x_test'], f['y_test']

    np.random.seed(seed)
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    labels_train = labels_train[indices]

    indices = np.arange(len(x_test))
    np.random.shuffle(indices)
    x_test = x_test[indices]
    labels_test = labels_test[indices]

    xs = np.concatenate([x_train, x_test])
    labels = np.concatenate([labels_train, labels_test])

    if start_char is not None:
        xs = [[start_char] + [w + index_from for w in x] for x in xs]
    elif index_from:
        xs = [[w + index_from for w in x] for x in xs]

    if maxlen:
        xs, labels = _remove_long_seq(maxlen, xs, labels)
    if not xs:
        raise ValueError('After filtering for sequences shorter than maxlen=' +
                       str(maxlen) + ', no sequence was kept. '
                       'Increase maxlen.')
    if not num_words:
        num_words = max([max(x) for x in xs])

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters:
    # 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        xs = [
            [w if (skip_top <= w < num_words) else oov_char for w in x] for x in xs
        ]
    else:
        xs = [[w for w in x if skip_top <= w < num_words] for x in xs]

    idx = len(x_train)
    x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
    x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])

    return (x_train, y_train), (x_test, y_test)

def get_word_index(path='imdb_word_index.json'):
    """Retrieves the dictionary mapping word indices back to words.
    Arguments:
      path: where to cache the data (relative to `~/.keras/dataset`).
    Returns:
      The word index dictionary.
    """
    with open(path) as f:
        return json.load(f)
(train_data, train_labels), (test_data, test_labels) = load_data("../input/imdb/imdb.npz",num_words=10000)
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print("Testing entries: {}, labels: {}".format(len(test_data), len(test_labels)))
print(train_data[0])
len(train_data[0]), len(train_data[1])
# A dictionary mapping words to an integer index
word_index = get_word_index(path="../input/imdb/imdb_word_index.json")

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
decode_review(train_data[1])
train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data,maxlen=256,value=word_index["<PAD>"],padding='post')
test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data,maxlen=256,value=word_index["<PAD>"],padding='post')
print(len(train_data[0]))
print(len(test_data[0]))
print(train_data[0])
vocab_size = 10000
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size,16))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(units=16,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=1,activation=tf.nn.sigmoid))

val_data = train_data[0:10000]
val_labels = train_labels[0:10000]
part_train_data = train_data[10000:]
part_train_labels = train_labels[10000:]
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(x=part_train_data,y=part_train_labels,
                    batch_size=512,epochs=40,validation_data=(val_data,val_labels))
results = model.evaluate(test_data, test_labels)
print(results)
history_dict = history.history
history_dict.keys()
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(0, len(acc))
# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.clf()
# "bo" is for "blue dot"
plt.plot(epochs, acc, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_acc, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()