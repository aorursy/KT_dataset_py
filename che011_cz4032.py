#file_review = "../input/yelp-reviews/yelp_reviews_100_thousand.csv"

file_review = "../input/yelp-reviews-100-thousand/yelp_reviews_100_thousand.csv"
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dense

from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
df = pd.read_csv(file_review)

df.info()
df.shape
df.head()
df.text.tolist()[0]
df.business_id.value_counts()
df['stars'].value_counts().sort_values(ascending=False).plot(kind='bar', title='Number of reviews with each rating')


x = []

for i in range(5):

    five = df[df.stars==i+1]

    five = five.sample(random_state=1, n=8000)

    x.append(five)



new_df = pd.concat(x)

# x_train = pd.DataFrame(xtrain, columns = df.columns)

# x_train.info()
new_df['stars'].value_counts().sort_values(ascending=False).plot(kind='bar', title='Number of reviews with each rating')
def print_plot(index):

    example = df[df.index == index][['text', 'stars']].values[0]

    if len(example) > 0:

        print(example[0])

        print('Stars:', example[1])

print_plot(10)
print("Longest review's length is ", max([len(tweet) for tweet in df['text']]))
# The maximum number of words to be used. (most frequent)

MAX_NB_WORDS = 50000

# Max number of words in each tweet.

MAX_SEQUENCE_LENGTH = 180

# This is fixed.

EMBEDDING_DIM = 100



tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

tokenizer.fit_on_texts(df['text'].values)

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
df['stars']
def label (row):

    if row['stars'] > 3 :

        return 'Positive'

    if row['stars'] < 3 :

        return 'Negative'

    if row['stars'] == 3:

        return 'Neutral'





df['label'] = df.apply (lambda row: label(row), axis=1)

df['label']
Xtrain, Xtest, Ytrain, Ytest = train_test_split(df['text'],df['label'], test_size = 0.10, random_state = 42)

print(Xtrain.shape,Ytrain.shape)

print(Xtest.shape,Ytest.shape)
X_train = tokenizer.texts_to_sequences(Xtrain.values)

X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of X_train data tensor:', X_train.shape)

X_test = tokenizer.texts_to_sequences(Xtest.values)

X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of X_test data tensor:', X_test.shape)

Y_train = pd.get_dummies(Ytrain).values

print('Shape of Y_train label tensor:', Y_train.shape)

Y_test = pd.get_dummies(Ytest).values

print('Shape of Y_test label tensor:', Y_test.shape)
from keras.layers import Conv1D, MaxPooling1D

from tensorflow.keras.callbacks import ModelCheckpoint

embedding_vecor_length = 32

model = Sequential()

model.add(Embedding(MAX_NB_WORDS, embedding_vecor_length, input_length=X_train.shape[1]))

model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))

model.add(MaxPooling1D(pool_size=2))

model.add(LSTM(100))

model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

filepath="weights_best_cnn.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max',save_weights_only=True)

callbacks_list = [checkpoint]

model.fit(X_train, Y_train, epochs=2, batch_size=64,verbose = 1,callbacks = callbacks_list,validation_data=(X_test,Y_test))
from tensorflow import argmax

from tensorflow.keras.backend import get_value

results = model.predict(X_test)

#df.sentiment[argmax(x)]

sentiment_results = [get_value(argmax(x))+1 for x in results]

label = [get_value(argmax(x))+1 for x in Y_test]

print("Model output, labels, content")

Xtest = Xtest.tolist()

wrong = []

right = []

for i in range(len(sentiment_results)):

  if label[i]!=sentiment_results[i]:

    wrong.append([sentiment_results[i], label[i], Xtest[i]])

  else:

    right.append([sentiment_results[i], label[i], Xtest[i]])

    

#print(len(sentiment_results),len(Y_test))



print("WRONG")

print(wrong[0])

print(wrong[1])

print(wrong[2])

print("\n")



count_model_sad = 0

count_model_happy = 0

for i in wrong:

    if(i[0]<3 and i[1]>3):

        count_model_sad += 1

    elif(i[1]<3 and i[0]>3):

        count_model_happy +=1

print("Proportion of model classifies positive as negative out of wrong: ", count_model_sad/len(wrong))

print("Proportion of model classifies negative as positive out of wrong: ", count_model_happy/len(wrong))

print("\n")



print("RIGHT")

print(right[0])

print(right[1])

print(right[2])

print("\n")



print("accuracy: ",len(right)/(len(wrong)+len(right)))
print(Y_test)
pd.get_dummies(Ytest)