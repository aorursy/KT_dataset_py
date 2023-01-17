# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.preprocessing import OneHotEncoder



from tensorflow import keras

from keras.models import Sequential

from keras.layers import Embedding, Dense, Flatten

from keras.preprocessing.text import one_hot, Tokenizer

from keras.preprocessing.sequence import pad_sequences



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_json('../input/whats-cooking-kernels-only/train.json').set_index('id')

test_df = pd.read_json('../input/whats-cooking-kernels-only/test.json').set_index('id')



X_train_input = train_df.ingredients.apply(';'.join).map(lambda x: x.replace(' ', '_'))

y_train_input = train_df.cuisine

X_test_input = test_df.ingredients.apply(';'.join).map(lambda x: x.replace(' ', '_'))



num_cuisines = len(np.unique(y_train_input))
#vectorizer = CountVectorizer(binary=True, tokenizer=lambda x: [i.strip() for i in x.split(',')])

#X_train_vec = vectorizer.fit_transform(X_train)

#X_test_vec = vectorizer.transform(X_test)

#ingredients = vectorizer.get_feature_names()
dim_size = 10000

X_train = [one_hot(rec, dim_size, filters='!"#$%&()*+,-./:;<=>?@[\]^`{|}~', lower=True, split=';') for rec in X_train_input]

X_test = [one_hot(rec, dim_size, filters='!"#$%&()*+,-./:;<=>?@[\]^`{|}~', lower=True, split=';') for rec in X_test_input]



max_ingredients = 40

X_train = pad_sequences(X_train, maxlen=max_ingredients)

X_test = pad_sequences(X_test, maxlen=max_ingredients)



cuisine_encoder = OneHotEncoder()

y_train = cuisine_encoder.fit_transform(y_train_input.values.reshape(-1, 1))
model = Sequential()

model.add(Embedding(input_dim=dim_size, output_dim=32, input_length=max_ingredients))

model.add(Flatten())

model.add(Dense(num_cuisines, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

print(model.summary())

model.fit(X_train, y_train, epochs=5, verbose=2)

loss, accuracy = model.evaluate(X_train, y_train, verbose=2)

print('Accuracy: %f' % (accuracy*100))
preds = cuisine_encoder.inverse_transform(model.predict(X_test)).flatten()

submission_nn = pd.Series(preds, index=X_test_input.index).rename('cuisine')

submission_nn.to_csv('submission_nn.csv', index=True, header=True)