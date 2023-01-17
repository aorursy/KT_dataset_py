import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import spacy

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score



import tensorflow as tf

from keras.models import Sequential

from keras.preprocessing import sequence

from keras.layers import Dense, Embedding, Dropout, LSTM

from keras import Model

from keras.optimizers import Adam  

import keras.backend as K

from keras.callbacks import Callback

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

print("Train shape : ", train.shape)

print("Test shape : ", test.shape)

train.head()
# Look at the classes distribution.

train.describe()
train['lenght_sentence'] = train['question_text'].apply(lambda x: len(x.split()))

print('Min questions lenght:', np.min(train['lenght_sentence'] ))

print('Max questions lenght:', np.max(train['lenght_sentence'] ))

print('Mean questions lenght:', np.mean(train['lenght_sentence'] ))

print('Standard deviation questions lenght:', np.std(train['lenght_sentence'] ))





# Plot the distribution of the lenght of the questions

plt.hist(train['lenght_sentence'], 100);
test['lenght_sentence'] = test['question_text'].apply(lambda x: len(x.split()))

print('Min questions lenght:', np.min(test['lenght_sentence'] ))

print('Max questions lenght:', np.max(test['lenght_sentence'] ))

print('Mean questions lenght:', np.mean(test['lenght_sentence'] ))

print('Standard deviation questions lenght:', np.std(test['lenght_sentence'] ))



# Plot the distribution of the lenght of the questions

plt.hist(test['lenght_sentence'], 100);
# First let's lower all the words in our train and test sets

train['question_text_truncated'] = train['question_text'].apply(lambda x: " ".join([word.lower() for word in x.split()[:20]]))

test['question_text_truncated'] = test['question_text'].apply(lambda x: " ".join([word.lower() for word in x.split()[:20]]))



train.head()
# Add all the questions

list_questions = list(train['question_text_truncated']) + list(test['question_text_truncated'])



# Split the questions into words then join them all together and finally we remove duplicates

unique_words = set((" ".join(list_questions)).split())
# Give an index to each word staring from 2.

index_from = 2



# Making the vocabulary

vocabulary = {k: (v + index_from) for v, k in enumerate(unique_words)}



vocabulary["<PAD>"] = 0

vocabulary["<START>"] = 1
vocabulary
print('Vocabulary length:', len(vocabulary))
# Tokenization of all words in a sentence using our vocabulary

def sentence_tokenization(sentence, vocabulary):

    tokenized_sentence = []

    for word in sentence.split():

        tokenized_sentence.append(vocabulary[word])

    return  tokenized_sentence

    



train["question_tokenized"] = train["question_text_truncated"].apply(lambda x: sentence_tokenization(x, vocabulary))

test["question_tokenized"] = test["question_text_truncated"].apply(lambda x: sentence_tokenization(x, vocabulary))

train.head()
input_max_length = 20

# 0 padding of the tokenized questions

X = sequence.pad_sequences(train['question_tokenized'], maxlen = input_max_length, padding = "post", truncating= "post", value = 0)

X_test = sequence.pad_sequences(test['question_tokenized'], maxlen = input_max_length, padding = "post", truncating= "post", value = 0)



y = train['target']
# We prepare our data for the training and validation steps which we will make to avoid overfitting

# Train/Validate split is less time consuming than several folds cross-validation

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
embedding_vector_length = 150

total_words = len(vocabulary) 

inputs_max_length = 20



model = Sequential()

model.add(Embedding(total_words, embedding_vector_length, input_length = inputs_max_length))

model.add(LSTM(units = 256))

model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))



print(model.summary())
# We must build a custom F1 metrics to plug it into our training steps with Keras

def f1(y_true, y_pred):

    '''

    metric from here 

    https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras

    '''

    def recall(y_true, y_pred):

        """Recall metric.



        Only computes a batch-wise average of recall.



        Computes the recall, a metric for multi-label classification of

        how many relevant items are selected.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



    def precision(y_true, y_pred):

        """Precision metric.



        Only computes a batch-wise average of precision.



        Computes the precision, a metric for multi-label classification of

        how many selected items are relevant.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision

    

    precision = precision(y_true, y_pred)

    recall = recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
# Compile and fit the model on our Train/Validate datasets



#model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=[f1])

#model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_val, y_val))
# Training on the whole dataset

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=[f1])

model.fit(X, y, epochs=3, batch_size=64)
pred_test = np.where(model.predict(X_test, batch_size=1024) < 0.5, 0, 1)


predictions = pd.DataFrame({"qid":test["qid"].values})

predictions['prediction'] = pred_test

predictions.head()
predictions.to_csv('submission.csv', index=False, sep=',')