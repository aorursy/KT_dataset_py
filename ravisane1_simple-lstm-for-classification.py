import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder



from tensorflow.keras import layers, models



from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing import sequence, text

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import EarlyStopping



import seaborn as sns
df = pd.read_csv('../input/spam-and-ham/spam.csv', delimiter = ',', encoding = 'latin-1')

df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)

df.head()
sns.countplot(df.v1)

plt.xlabel('Label')

plt.title('Number of ham and spam messages')
X = df.v2

y = df.v1

le = LabelEncoder()

y = le.fit_transform(y)

y = y.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)
max_words = 1000

max_len = 150

tokens = text.Tokenizer(num_words = max_words)

tokens.fit_on_texts(X_train)

train_sequences = tokens.texts_to_sequences(X_train)

train_sequences_matrix = sequence.pad_sequences(train_sequences, maxlen = max_len)

test_sequences = tokens.texts_to_sequences(X_test)

test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen = max_len)
lstm_model = models.Sequential()

lstm_model.add(layers.Input(shape = [max_len]))

lstm_model.add(layers.Embedding(max_words,50,input_length=max_len))

lstm_model.add(layers.LSTM(64))

lstm_model.add(layers.Dense(256, activation = 'relu'))               

lstm_model.add(layers.Dropout(0.5))               

lstm_model.add(layers.Dense(1, activation = 'sigmoid'))

lstm_model.summary()            
lstm_model.compile(loss = 'binary_crossentropy', optimizer = RMSprop(), metrics=['accuracy'])

history = lstm_model.fit(train_sequences_matrix, y_train, batch_size = 128, epochs=10,

          validation_split = 0.2, callbacks = [EarlyStopping(monitor = 'val_loss', min_delta = 0.0001)])
# In[10]: Step 4: Evaluate the model



plt.plot(history.history['accuracy'], label = 'Training_accuracy')

plt.plot(history.history['val_accuracy'], label = 'Validation_accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend(loc = 'lower right')

accuracy = lstm_model.evaluate(test_sequences_matrix, y_test, verbose = 2)
pred = lstm_model.predict(test_sequences_matrix)

X_test = np.array(X_test)



print("Predicted Label", "\tTest Data")

for i in range(len(y_test)):

    p = "ham" if (pred[i] < 0.5) else "spam"

    print(p, "\t", X_test[i])

    # print(p, "\t", y_test[i][0], "\t", X_test[i])
# Save the model

lstm_model.save('./saved_model/')