# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import re

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os


# Import each episode into a dictionary
transcripts = {}
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        file = os.path.join(dirname,filename)
        f = open(file)
        transcripts[os.path.basename(file[:-4])] = f.read()
# Get word count of each episode
episode_word_count = {episode: len(re.sub('\\n',' ',transcripts[episode]).split()) for episode in transcripts.keys()}
# Choose smaller episodes

filtered_episodes = {episode: transcript for episode, transcript in transcripts.items() if episode_word_count[episode] < 8000}
data = []

for episode, transcript in list(filtered_episodes.items()):
    # Remove empty lines
    line_split = transcript.split('\n')
    while('' in line_split):
        line_split.remove('')
    while(' ' in line_split):
        line_split.remove(' ')
    # Only contain episodes if all lines are less than 100 words
    if max([len(line.split(' ')) for line in line_split]) < 50:
        data += line_split
np.shape(data)
tokenizer=tf.keras.preprocessing.text.Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(data)

total_words=len(tokenizer.word_index)+1
print(total_words)

word_index=tokenizer.word_index
input_sequences=[]
for line in data:
    token_list=tokenizer.texts_to_sequences([line])[0]
    for i in range(1,len(token_list)):
        n_gram_sequences=token_list[:i+1]
        input_sequences.append(n_gram_sequences)
max_len=max([len(x) for x in input_sequences])
print(max_len)
input_sequences[:5]
padded=tf.keras.preprocessing.sequence.pad_sequences(input_sequences,maxlen=max_len,padding='pre')
padded
input_sequences=np.array(padded)
input_sequences
train=input_sequences[:,:-1]
labels=input_sequences[:,-1]
labels
label_encoding=tf.keras.utils.to_categorical(labels,num_classes=total_words)
label_encoding
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(total_words, 100, input_length=max_len-1))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150, return_sequences = True)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(100))
model.add(tf.keras.layers.Dense(total_words/2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.Dense(total_words, activation='softmax'))
print(label_encoding.shape)
print(train.shape)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history=model.fit(train,label_encoding,epochs=100)
seed_text = input('Enter a line related to story : ').lower()
next_words = int(input('Enter the number of words you need in the story : '))

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_len-1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += ' '+output_word
print(seed_text)
