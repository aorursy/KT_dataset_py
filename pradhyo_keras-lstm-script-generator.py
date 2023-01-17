# https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/8.1-text-generation-with-lstm.ipynb
import keras
import numpy as np
# Get the script
def get_text():
    
    # On kaggle
    office_script_file_url = "https://raw.githubusercontent.com/Pradhyo/the-office-us-tv-show/master/the-office-all-episodes.txt"
    path = keras.utils.get_file('script.txt', origin=office_script_file_url)
    
    # On local machine
    # path = "the-office-all-episodes.txt"
    
    text = open(path).read().lower()
    return text

text = get_text()
print('Corpus length:', len(text))
from collections import Counter
from pprint import pprint
char_counts = Counter()
for c in text:
    char_counts[c] += 1
    
pprint(char_counts.most_common())
pprint(len(char_counts))
# Get some sample strings for each character to explore the data
def sample_strings(char, string_length=20, num_samples=5):
    sample = 0
    samples = []
    for i, c in enumerate(text):
        if i < string_length:
            continue
        if char == c:
            samples.append(text[int(i-string_length/2):int(i+string_length/2)])
            sample += 1
            if sample == num_samples:
                break
    return samples

for c in char_counts:
    print(f"{c}: {sample_strings(c)}")
# See longer strings for non alphanumeric characters
for c in char_counts:
    if not c.isalnum():
        print(f"{c}: {sample_strings(c, 40)}")
# consider these as words
consider_words = ''.join(c for c in char_counts if not c.isalnum())
print(consider_words)
numbers = '0123456789'
def replace_numbers(text):
    for n in numbers:
        text = text.replace(n, "0")
    return text

text = replace_numbers(text)
consider_words += '0' # consider 0 also a word
print(consider_words)
def split_into_words(text, consider_words):
    # Split text into words - characters above are also considered words
    text = text.replace(' ', ' | ') # pick a char not in the above list
    text = text.replace('\n', ' | ') # pick a char not in the above list

    for char in consider_words:
        text = text.replace(char, f" {char} ") # to split on spaces to get char

    words_with_pipe = text.split()
    words = [word if word != '|' else ' ' for word in words_with_pipe]
    return words

words = split_into_words(text, consider_words)
print(words[:500])
# Length of extracted word sequences
maxlen = 20

# We sample a new sequence every `step` words
step = 3

def setup_inputs(words, maxlen, step):
    try:
        # This holds our extracted sequences
        sentences = []

        # This holds the targets (the follow-up characters)
        next_words = []

        for i in range(0, len(words) - maxlen, step):
            sentences.append(words[i: i + maxlen])
            next_words.append(words[i + maxlen])
        print('Number of sequences:', len(sentences))

        # List of unique characters in the corpus
        unique_words = sorted(list(set(words)))
        print('Unique words:', len(unique_words))
        # Dictionary mapping unique characters to their index in `unique_words`
        word_indices = dict((word, unique_words.index(word)) for word in unique_words)

        # Next, one-hot encode the characters into binary arrays.
        print('Vectorization...')
        print(len(sentences), len(unique_words))
        x = np.zeros((len(sentences), maxlen, len(unique_words)), dtype=np.bool)
        y = np.zeros((len(sentences), len(unique_words)), dtype=np.bool)
        print(len(x),len(y),len(sentences))
        for i, sentence in enumerate(sentences):
            for t, word in enumerate(sentence):
                x[i, t, word_indices[word]] = 1
            y[i, word_indices[next_words[i]]] = 1
        print(len(x), len(y), len(unique_words), len(word_indices))
        return x, y, unique_words, word_indices
    except MemoryError as e:
        print(e)
        pass

# Commenting out to avoid MemoryError
# Tried catching it but didn't seem to work
# x, y, unique_words, word_indices = setup_inputs(words, maxlen, step)

text = get_text()

selected_actor = "phyllis"

def get_selected_lines(text, selected_actor):
    lines = text.split("\n")
    return "\n".join(line for line in lines if line.startswith(f"{selected_actor}:"))

text = get_selected_lines(text, selected_actor)
print(text[:2000])
text = replace_numbers(text)
words = split_into_words(text, consider_words)
x, y, unique_words, word_indices = setup_inputs(words, maxlen, step)
from keras import layers

def build_model(maxlen, num_unique_words):
    model = keras.models.Sequential()
    model.add(layers.LSTM(128, input_shape=(maxlen, num_unique_words)))
    model.add(layers.Dense(num_unique_words, activation='softmax'))
    optimizer = keras.optimizers.RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)    
    return model

model = build_model(maxlen, len(unique_words))
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
import random
import sys

def train_model(text, words, unique_words, word_indices, max_epoch, script_file, model_file=""):
    with open(script_file, "wt") as f:
        f.write("") # Just to create/overwrite the file

    for epoch in range(1, max_epoch):
        with open(script_file, "at") as f:
            f.write(f'\n\nepoch {epoch}\n\n')
        # Fit the model for 1 epoch on the available training data
        model.fit(x, y,
                  batch_size=128,
                  epochs=1)

        # Select a text seed at random
        start_index = random.randint(0, len(words) - maxlen - 1)
        generated_text = words[start_index: start_index + maxlen]

        with open(script_file, "at") as f:
            f.write('--- Generating with seed: "' + ''.join(generated_text) + '"\n')

        with open(script_file, "at") as f:        
            for temperature in [0.2, 0.5, 1.0, 1.2]:
                f.write('\n--- temperature: ' + str(temperature) + "\n")
                f.write(''.join(generated_text))

                for i in range(200):
                    sampled = np.zeros((1, maxlen, len(unique_words)))
                    for t, word in enumerate(generated_text):
                        sampled[0, t, word_indices[word]] = 1.

                    preds = model.predict(sampled, verbose=0)[0]
                    next_index = sample(preds, temperature)
                    next_word = unique_words[next_index]

                    generated_text.append(next_word)
                    generated_text = generated_text[1:]

                    f.write(next_word)
        
        if model_file:
            model.save(model_file)
    
train_model(text, words, unique_words, word_indices, 100, "phyllis_script.txt")
text = get_text()
text = replace_numbers(text)
words = split_into_words(text, consider_words)
words_counter = Counter(words)
print(len(words_counter))
print(words_counter.most_common(2000))
top_words = []
for word, count in words_counter.most_common(2000):
        top_words.append(word)

print(len(top_words))

def get_lines_with_words(top_words):
    selected_lines = []
    text = get_text()
    lines = text.split("\n")
    for line in lines:
        line = replace_numbers(line)
        words_in_line = split_into_words(line, consider_words)
        excluded_words = 0
        for word_in_line in words_in_line:
            if word_in_line not in top_words:
                excluded_words += 1
                break
        if not excluded_words:
            selected_lines.append(line)
    return selected_lines
                
                
selected_lines = get_lines_with_words(top_words)
print(len(selected_lines))
print(selected_lines[:100])
selected_text = "\n".join(selected_lines)

selected_text = replace_numbers(selected_text)
selected_words = split_into_words(selected_text, consider_words)
x, y, unique_words, word_indices = setup_inputs(selected_words, maxlen, step)
model = build_model(maxlen, len(unique_words))
train_model(selected_text, selected_words, unique_words, word_indices, 100, "generated_script.txt", "top_lines.h5")        
