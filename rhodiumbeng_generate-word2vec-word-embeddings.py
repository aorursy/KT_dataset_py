import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

%matplotlib inline

import re

import gensim 

from gensim.models import Word2Vec
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

print(train_df.shape, test_df.shape)
def clean_text(text):

    """

    Convert all to lowercase and remove punctuations

    """

    text = text.lower()

    text = re.sub(r'[^\w\s]', '', text) # remove everything that isn't word or space

    text = re.sub(r'\_', '', text)      # remove underscore

    return text
# clean train_df['text']

train_df['text'] = train_df['text'].map(lambda x: clean_text(x))

train_df['text'] = train_df['text'].map(lambda x: x.strip().split())

train_df.head()
# clean test_df['text']

test_df['text'] = test_df['text'].map(lambda x: clean_text(x))

test_df['text'] = test_df['text'].map(lambda x: x.strip().split())

test_df.head()
data = []  

# iterate through each row in train_df 

for i in range(len(train_df)):

    data.append(train_df['text'][i])

for j in range(len(test_df)):

    data.append(test_df['text'][j])
print(len(data))
# Create Word2Vec model using CBOW (sg=0)

# Set min_count to 1 so as to include all words

embedding = gensim.models.Word2Vec(data, size=50, window=10, min_count=1, sg=0)
print(embedding)
# train & generate the embeddings

embedding.train(data,total_examples=len(data),epochs=30)
words = list(embedding.wv.vocab)

print(len(words))
print(embedding['capered'])
embedding.most_similar('dark', topn=5)
embedding.most_similar('shocked', topn=5)
embedding.most_similar('sprang', topn=5)
embedding.most_similar('pride', topn=5)
# convert author labels into one-hot encodings

train_df['author'] = pd.Categorical(train_df['author'])

df_Dummies = pd.get_dummies(train_df['author'], prefix='author')

train_df = pd.concat([train_df, df_Dummies], axis=1)

# Check the conversion

train_df.head()
X = train_df['text'].str[:100]

Y = train_df[['author_EAP', 'author_HPL', 'author_MWS']].values

print(X.shape, X[0], Y.shape, Y[0])
X_test = test_df['text'].str[:50]

print(X_test.shape, X_test[0])
def text_to_avg(text):

    """

    Given a list of words, extract the respective word embeddings

    and average the values into a single vector encoding the text meaning.

    """

    # initialize the average word vector

    avg = np.zeros((50,))

    # average the word vector by looping over the words in text

    for w in text:

        avg += embedding[w]

    avg = avg/len(text)

    return avg
X_avg = np.zeros((X.shape[0], 50)) # initialize X_avg

for i in range(X.shape[0]):

    X_avg[i] = text_to_avg(X[i])
print(X_avg.shape)

print(X_avg[0])
X_test_avg = np.zeros((X_test.shape[0], 50)) # initialize X_test_avg

for i in range(X_test.shape[0]):

    X_test_avg[i] = text_to_avg(X_test[i])
print(X_test_avg.shape)

print(X_test_avg[0])
from sklearn.model_selection import train_test_split

X_train, X_dev, Y_train, Y_dev = train_test_split(X_avg, Y, test_size=0.2, random_state=123)

print(X_train.shape, Y_train.shape, X_dev.shape, Y_dev.shape)
from keras import models

from keras import layers



model = models.Sequential()

model.add(layers.Dense(512, activation='relu', input_shape=(50,)))

model.add(layers.Dense(3, activation='softmax'))



model.summary()
# compile the model

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# train and validate the model

epochs = 50

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=128, validation_data=(X_dev, Y_dev))
# plot and visualise the training and validation losses

loss = history.history['loss']

dev_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='training loss')

plt.plot(epochs, dev_loss, 'b', label='validation loss')

plt.title('Training and Validation Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
model = models.Sequential()

model.add(layers.Dense(512, activation='relu', input_shape=(50,)))

model.add(layers.Dense(3, activation='softmax'))

# compile the model

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# train the model

epochs = 10

model.fit(X_avg, Y, epochs=epochs, batch_size=128)
# predict on test set

preds = model.predict(X_test_avg)

print(preds.shape)

print(preds[7])
# set the predicted labels to be the one with the highest probability

pred_labels = []

for i in range(len(X_test_avg)):

    pred_label = np.argmax(preds[i])

    pred_labels.append(pred_label)
print(pred_labels[7])
result = pd.DataFrame(preds, columns=['EAP','HPL','MWS'])

result.insert(0, 'id', test_df['id'])

result.head()
# Generate submission file in csv format

result.to_csv('submission.csv', index=False, float_format='%.20f')