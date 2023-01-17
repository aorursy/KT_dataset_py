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
data= pd.read_csv('../input/data-train/train.tsv', sep="\t")
data
data.info()
data.dtypes
import seaborn as sns

sns.catplot(y="Sentiment", kind="count",

            palette="pastel", edgecolor=".6",

            data=data)
data['Sentiment'].value_counts()
y_train = data['Sentiment']
first_class = y_train[y_train == 0]

second_class = y_train[y_train == 1]

third_class = y_train[y_train == 2]

forth_class = y_train[y_train == 3]

fifth_class = y_train[y_train == 4]



print('',len(first_class),'\n',len(second_class), '\n',len(third_class), '\n', len(forth_class), '\n', 

      len(fifth_class) )







second_class = second_class[0:len(first_class)]

third_class  = third_class[0:len(first_class)]

forth_class  = forth_class[0:len(first_class)]

fifth_class  = fifth_class[0:len(first_class)]
text_first_class  = data[['Phrase','Sentiment']]  [y_train==0]

text_second_class = data[['Phrase','Sentiment']]  [y_train==1]

text_third_class  = data[['Phrase','Sentiment']]  [y_train==2]

text_forth_class  = data[['Phrase','Sentiment']]  [y_train==3]

text_fifth_class  = data[['Phrase','Sentiment']]  [y_train==4]



print('',len(text_first_class),'\n',len(text_second_class), '\n',len(text_third_class), '\n', len(text_forth_class), '\n', 

      len(text_fifth_class) )
text_second_class = text_second_class[0:len(text_first_class)]

text_third_class  = text_third_class [0:len(text_fifth_class)]

text_forth_class  = text_forth_class [0:len(text_first_class)]

text_fifth_class  = text_fifth_class [0:len(text_first_class)]
frames = [text_first_class, text_second_class, text_third_class, text_forth_class, text_fifth_class]

new_train = pd.concat(frames)
sns.catplot(y="Sentiment", kind="count",

            palette="pastel", edgecolor=".6",

            data=new_train)
new_train
X = new_train['Phrase'].values

type(X)

y = new_train['Sentiment'].values
y.shape,X.shape
import tensorflow as tf

import keras
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences
vocab_size = 10000

embedding_dim = 32

max_length = 150

trunc_type='post'

padding_type='post'

oov_tok = "<OOV>"
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

tokenizer.fit_on_texts(X)



word_index = tokenizer.word_index



training_sequences = tokenizer.texts_to_sequences(X)

training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)



import numpy as np

training_padded = np.array(training_padded)

training_padded.shape

y = y.reshape((37494,1))

y.shape

y
model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),

    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.Dense(32, activation='relu'),

    tf.keras.layers.Dense(5, activation='softmax')

])

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])



model.summary()
num_epochs = 50



history = model.fit(training_padded, y, epochs=num_epochs)
%matplotlib inline

import matplotlib.pyplot as plt

acc = history.history['accuracy']

epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.title('Training accuracy')

plt.legend()

plt.figure()



loss = history.history['loss']

plt.plot(epochs, loss, 'b', label='Training Loss')

plt.title('Training loss')

plt.legend()



plt.show()
test_data = pd.read_csv('../input/datatest/test.tsv', sep="\t")
test_data
X_test = test_data['Phrase']

X_test.shape[0]
X_test = X_test.reshape(X_test.shape[0],1)

X_test.shape

type(X_test)
testing_sequences = tokenizer.texts_to_sequences(X_test)

testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
testing_padded.shape
prediction = []



predictions = model.predict(testing_padded)
predictions

for i in predictions:

    prediction.append(np.argmax(i))
submission =  pd.DataFrame({

        "PhraseId":test_data.PhraseId ,

        "Sentiment": prediction

    })



submission.to_csv('submission.csv', index=False)