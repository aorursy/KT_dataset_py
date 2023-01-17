# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
df = pd.read_csv('/kaggle/input/chess/games.csv')
df.query('winner != "draw"')
moves = np.array(df.query('winner != "draw"')['moves'])
labels = np.array(df.query('winner != "draw"')['winner'].apply(lambda x: 1 if x== 'white' else 0))
print(moves.shape==labels.shape)
moves_all = set()
for movelist in moves:
    for move in movelist.split(' '):
        if move not in moves_all:
            moves_all.add(move)

max_vocab = len(moves_all)
max_vocab
max_len = 0

for move_list in moves:
    total = 0
    for move in move_list.split(' '):
        total+=1
        if total > max_len:
            max_len = total
print(max_len==np.max(df['turns']))
print(np.max(df['turns']))
tok = Tokenizer(num_words=max_vocab)
tok.fit_on_texts(moves)
sequences = tok.texts_to_sequences(moves)

word_index_ = tok.word_index
model_inputs = pad_sequences(sequences,maxlen=max_len) # This makes every sequence the length of the biggest sequence
print(model_inputs.shape[0]==labels.shape[0])
X_train, X_test, y_train, y_test = train_test_split(model_inputs,labels,train_size=0.7, random_state=24)
embedding_dimension = 256
inputs = tf.keras.Input(shape=max_len)
embedding = tf.keras.Embedding