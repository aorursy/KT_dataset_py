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
import tensorflow as tf
data=pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

data.tail(10)
data.shape
len(data[data.keyword.isna()==True])
data.location.unique()
data.drop(["location", "keyword"], axis=1, inplace=True)
labels=data["target"]
labels.head()
data.drop("target", axis=1, inplace=True)
data.head()
import nltk

#nltk.download('stopwords')



from nltk.corpus import stopwords

stoplist = stopwords.words('english')



training_data = data.text.tolist()

train_data=[]



for i in training_data:

    string=""

    for j in i.lower().split():

        if j not in stoplist:

            string= string+j+" "

            

    train_data.append(string.rstrip())



training_lables=np.array(labels.to_numpy())
train_data[0:1]
training_lables[:3]
review_length=[len(i) for i in training_data[1:2]]

print(max(review_length))
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences
vocab_size=10000

OOV_tok="<OOV>"

padding_type="post"

trunc_type="post"

max_length=25



tokenizer=Tokenizer(num_words=vocab_size, oov_token=OOV_tok)

tokenizer.fit_on_texts(training_data)

word_index=tokenizer.word_index

#print(word_index)



training_sequences=tokenizer.texts_to_sequences(training_data)

training_pad_seq=pad_sequences(training_sequences,padding=padding_type, maxlen=max_length, truncating=trunc_type)
reverse_words=dict([(j, i) for i,j in tokenizer.word_index.items()])

#print(reverse_words)
def decode_review(review):

    return " ".join([reverse_words.get(i, "?") for i in review])

    

print(decode_review(training_pad_seq[1]))

print(training_data[1])
emb_dim=16



model=tf.keras.models.Sequential([

                                tf.keras.layers.Embedding(vocab_size, emb_dim, input_length=max_length),

                                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),

                                tf.keras.layers.Dense(6, activation="relu"),

                                tf.keras.layers.Dense(1, activation="sigmoid")

])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.summary()
epochs=10

history=model.fit(training_pad_seq, training_lables, epochs=epochs)
sentence="13,000 people receive #wildfires evacuation orders in California"

sentence_sequence = tokenizer.texts_to_sequences(sentence)

sentence_padding = pad_sequences(sentence_sequence, maxlen=max_length)

prediction=model.predict_classes(sentence_padding)

predict=np.max(prediction)

print(len(prediction))

print(predict)

if predict==1:

    print("real")

elif predict==0:

    print("fake")

test_data=pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

test_data.head()
testing_data=test_data.text.tolist()

testing_data[:3]
testing_sequences=tokenizer.texts_to_sequences(testing_data)

print(testing_sequences)

testing_pad_sequence=pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
test_predictions=model.predict_classes(testing_pad_sequence)
test_predictions
pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
output=pd.DataFrame(test_data.id)
output['target']=test_predictions
output
output.to_csv("submission.csv", index=False)

output