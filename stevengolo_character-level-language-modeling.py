# Load packages

import matplotlib.pyplot as plt

import numpy as np



from collections import Counter



from nltk.corpus import gutenberg

from nltk.tokenize import sent_tokenize



from tensorflow.keras.layers import Dense, LSTM

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import RMSprop



from sklearn.utils import shuffle
CORPUS_PATH = '../input/poetry/eminem.txt'

text = open(CORPUS_PATH).read().lower()

print(f'The corpus contains {len(text)} characters.')
# Print the beginning of the corpus

print(text[:100], '...')
# Replace '\n' by ' '

text = text.replace('\n', ' ')
# Split the data into train and test set

split = int(0.9 * len(text))

train_text = text[:split]

test_test = text[split:]
chars = sorted(list(set(text)))

print(f'Total number of characters: {len(chars)}.')
char_indices = dict((c, i) for i, c in enumerate(chars))

indices_char = dict((i, c) for i, c in enumerate(chars))
counter = Counter(text)

chars, counts = zip(*counter.most_common())

idx = np.arange(len(counts))
plt.figure(figsize=(14, 3))

plt.bar(idx, counts, 0.8)

plt.xticks(idx, chars)

plt.show()
def make_sequence(text, max_length=40, step=3):

    sequences = []

    next_chars = []

    for i in range(0, len(text) - max_length, step):

        sequences.append(text[i:(i + max_length)])

        next_chars.append(text[i + max_length])

    return sequences, next_chars
MAX_LENGTH = 40

STEP = 3



sequences, next_chars = make_sequence(train_text, MAX_LENGTH, STEP)

sequences_test, next_chars_test = make_sequence(test_test, MAX_LENGTH, step=10)
print(f'There are {len(sequences)} train sequences and {len(sequences_test)} test sequences.')
sequences, next_chars = shuffle(sequences, next_chars, random_state=42)
print(f'The first sequence is `{sequences[0]}` and the first next character is `{next_chars[0]}`.')
n_sequences = len(sequences)

n_sequences_test = len(sequences_test)

vocab_size = len(chars)



X = np.zeros((n_sequences, MAX_LENGTH, vocab_size),

             dtype=np.float32)

X_test = np.zeros((n_sequences_test, MAX_LENGTH, vocab_size),

                  dtype=np.float32)

y = np.zeros((n_sequences, vocab_size),

             dtype=np.float32)

y_test = np.zeros((n_sequences_test, vocab_size),

                  dtype=np.float32)



# Fill the training data

for i, sequence in enumerate(sequences):

    y[i, char_indices[next_chars[i]]] = 1

    for j, char in enumerate(sequence):

        X[i, j, char_indices[char]] = 1

        

# Fill the test data

for i, sequence in enumerate(sequences_test):

    y_test[i, char_indices[next_chars_test[i]]] = 1

    for j, char in enumerate(sequence):

        X_test[i, j, char_indices[char]] = 1
print(f'Shape of the tensor X: {X.shape}, shape of the matrix y: {y.shape}.')
def perplexity(y_true, y_pred):

    """Compute the per-character perplexity of model predictions.

    

    :param y_true: One-hot encoded ground truth

    :param y_pred: Predicted likelihoods for each class

    

    :return: 2 ** -mean(log2(p))

    """

    likelihoods = np.sum(y_pred * y_true, axis=1)

    return 2 ** (-np.mean(np.log2(likelihoods)))
# Build the model

model = Sequential()

model.add(LSTM(128, input_shape=(MAX_LENGTH, vocab_size)))

model.add(Dense(vocab_size, activation='softmax'))



optimizer = RMSprop(lr=0.01)

model.compile(optimizer=optimizer, loss='categorical_crossentropy')
def model_perplexity(model, X, y):

    predictions = model(X)

    return perplexity(y, predictions)
print(f'The model perplexity on the untrained model is {model_perplexity(model, X_test, y_test)}.')
small_train = slice(0, None, 40)

history = model.fit(X[small_train], y[small_train], validation_split=0.1,

                    batch_size=128, epochs=1)
print(f'The model perplexity on the model trained one epoch is {model_perplexity(model, X_test, y_test)}.')
def sample_one(preds, temperature=1.0):

    """Sample the next character according to the network output.

    

    Use a lower temperature to force the model to output more

    confident predictions: more peaky distribution.

    

    Draw a single sample (size=1) from a multinoulli distribution

    parameterized by the ouput of the softmax layer of our network.

    A multinoulli distribution is a multinomial distribution with

    a single trial with n_classes outcomes.

    """

    preds = np.asarray(preds).astype('float64')

    preds = np.log(preds) / temperature

    exp_preds = np.exp(preds)

    preds = exp_preds / np.sum(exp_preds)

    

    probs = np.random.multinomial(1, preds, size=1)

    return np.argmax(probs)



def generate_text(model, seed_string, length=300, temperature=1.0):

    """Recursively sample a sequence of characters, one at a time.

    

    Each prediction is concatenated to the past string of predicted

    characters so as to condition the next prediction.

    

    Feed seed string as a sequence of characters to condition the

    first predictions recursively. If see_string is lower than

    MAX_LENGTH, pad the input with zeros at the beginning of the

    conditioning string.

    """

    generated = seed_string

    prefix = seed_string

    

    for i in range(length):

        # Vectorize prefix string to feed as input to the model

        x = np.zeros((1, MAX_LENGTH, vocab_size), dtype='float32')

        shift = MAX_LENGTH - len(prefix)

        for j, char in enumerate(prefix):

            x[0, j + shift, char_indices[char]] = 1

        

        preds = model(x)[0]

        next_index = sample_one(preds, temperature)

        next_char = indices_char[next_index]

        

        generated += next_char

        prefix = prefix[1:] + next_char

    return generated
print(f'Test with a temperature lower than 1: {generate_text(model, "moms spaghetti", length=10, temperature=0.1)}.')
print(f'Test with a temperature larger than 1: {generate_text(model, "moms spaghetti", length=10, temperature=1.9)}.')
NB_EPOCHS = 30

seed_strings = ['love the way you lie', 'moms spaghetti']



history = []

perplexity_test = []

for epoch in range(NB_EPOCHS):

    history_epoch = model.fit(X, y, validation_split=0.1,

                              batch_size=128, nb_epoch=1,

                              verbose=1)

    history.append(history_epoch.history)

    perplexity_test.append(model_perplexity(model, X_test, y_test))
for temperature in [0.1, 0.5 ,1]:

    print(f'Sampling text from model at {temperature}:')

    for seed_string in seed_strings:

        print(generate_text(model, seed_string, length=100, temperature=temperature), '\n')
text_with_case = open(CORPUS_PATH).read().replace('\n', ' ')
sentences = sent_tokenize(text_with_case)
plt.hist([len(s.split()) for s in sentences], bins=100)

plt.title('Distribution of sentence lengths')

plt.xlabel('Approximate number of words')

plt.show()
sorted_sentences = sorted([s for s in sentences if len(s) > 20], key=len)
for s in sorted_sentences[:5]:

    print(s)
book_selection_text = gutenberg.raw().replace('\n', ' ')
print(f'Book corpus length: {len(book_selection_text)} characters.')