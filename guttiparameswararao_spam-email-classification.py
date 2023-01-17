# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

data=pd.read_csv(r'../input/spam-or-not-spam-dataset/spam_or_not_spam.csv')

data.head()

from sklearn.utils import shuffle

data = shuffle(data)
data['label'].value_counts()
text =[] 

  

# Iterate over each row 

for index, rows in data.iterrows(): 

    # Create list for the current row 

    my_list =str(rows.email)

      

    # append the list to the final list 

    text.append(my_list) 

  

# Print the list 

len(text)
label=list(data['label'])
len(label)
from keras.preprocessing.text import Tokenizer



tokenizer = Tokenizer(num_words=1000)

tokenizer.fit_on_texts(text)

sequences = tokenizer.texts_to_sequences(text)

x_train=sequences[:2000]

y_train=label[:2000]

x_test=sequences[2000:]

y_test=label[2000:]
maxlen = 20

from keras import preprocessing

x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)

x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
from keras.models import Sequential

from keras.layers import Flatten, Dense

from keras.layers import Embedding

model = Sequential()

model.add(Embedding(2000, 8, input_length=maxlen))

model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

model.summary()

history = model.fit(x_train, y_train,

                    epochs=10,

                    batch_size=32,

                    validation_split=0.2)
result=model.evaluate(x_test,y_test)

print("test loss:{}\ntest accuracy:{}".format(result[0],result[1]))