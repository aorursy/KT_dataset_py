# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
df = pd.read_csv('../input/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
df.head()
X = df['Review']
y= df['Liked']
tokenizer = Tokenizer(num_words=1500)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
data = pad_sequences(sequences, maxlen=500)
model = Sequential()
model.add(Embedding(1500, 128, input_length=500))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))    
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, np.array(y), test_size = 0.20, random_state = 0)


# Fitting and Evaluating the Model
model.fit(data,np.array(y), validation_split=0.3, epochs=15)

model.evaluate(x=X_test, y=y_test, batch_size=None, verbose=3, sample_weight=None, steps=None)



new_review = ["The FOOD was delicious! Everything from starters to main course was fresh and flavourful. The Biryani was simply mind blowing! It is not possible for one to over hype the taste of Paradise biryani! It truly feels like Paradise! and it's not even the main branch! The rice was fragrant and moist, it blends in perfectly with the spices and the meat causing an explosion of spicy flavor in one's mouth. The meat is soft and easily separable from the bone, the raita and salan were excellent! This is by far the BEST Hyderabadi biryani I had in Chennai!!!The SERVICE was good enough, like you would expect from a place that caters to hardcore biryani lovers. They're polite to everyone in spite of the huge crowd there , not loosing their cool even for a second. The dishes were presented nicely and they didn't make us Wait for too long.The AMBIENCE was good, it may feel crowded or conjusted to some people dining on the third floor, but personally, I had no problems with it. It's hygenic, well decorated and has a very inviting atmosphere.The BEST Hyderabadi biryani in Chennai I had so far! Must try for biryani lovers"]

sequences = tokenizer.texts_to_sequences(new_review)
data = pad_sequences(sequences, maxlen=500)

# get predictions for each of your new texts
predictions = model.predict(data)
print(predictions)
new_review = ["Food was bad, I would definitely not recommend this Restaurant"]
sequences = tokenizer.texts_to_sequences(new_review)
data = pad_sequences(sequences, maxlen=500)

# get predictions for each of your new texts
predictions = model.predict(data)
print(predictions)

