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
from tensorflow.keras.preprocessing.text import one_hot

### sentences
sent=[  'the glass of milk',
     'the glass of juice',
     'the cup of tea',
    'I am a good boy',
     'I am a good developer',
     'understand the meaning of words',
     'your videos are good',]
sent
voc_size=5000 ## define voc size , you can choose any number based on your dataset.
onehot_data=[one_hot(words,voc_size)for words in sent]
print(onehot_data)
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
sent_length=8
embedding_docs=pad_sequences(onehot_data,padding='pre',maxlen=sent_length)
print(embedding_docs)
dimension=10
model=Sequential() # initializing the sequential model
model.add(Embedding(voc_size,10,input_length=sent_length))# Adding a embedding layer
model.compile('adam','mse') # using adam optimiser and MSE as evaluation of performance 
model.summary()
print(model.predict(embedding_docs)) 
print((embedding_docs)[0])
print(model.predict(embedding_docs)[0]) 
