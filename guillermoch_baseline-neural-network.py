import pandas as pd

import tensorflow as tf



from collections import Counter



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from tensorflow import keras
train = pd.read_csv('../input/nlp-getting-started/train.csv')

test = pd.read_csv('../input/nlp-getting-started/test.csv')



X_train, X_val, y_train, y_val = train_test_split(train, train.target, random_state=42, stratify=train.target)
# Convert data into feature vectors

tfidf = TfidfVectorizer()

X_train_tfidf = tfidf.fit_transform(X_train.text)
from keras import backend as K



def recall_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



def precision_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision



def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
# Network and training.

EPOCHS = 15

VERBOSE = 1

CLASSES = 2

VALIDATION_SPLIT = 0.2

DROPOUT = 0.3

INPUT_SIZE = X_train_tfidf.toarray()[0].shape[0]

BATCH_SIZE = int(X_train_tfidf.shape[0]/2)



print(X_train.shape[0], 'train samples')

print(X_val.shape[0], 'test samples')



# One-hot representations for labels.

#y_train = tf.keras.utils.to_categorical(y_train, CLASSES)

#y_test = tf.keras.utils.to_categorical(y_val, CLASSES)



# Building the model.

model = tf.keras.models.Sequential()

model.add(keras.layers.Dense(512, input_shape=(INPUT_SIZE,), name='input_layer', activation='relu'))

model.add(keras.layers.Dropout(DROPOUT))

model.add(keras.layers.Dense(128, name='hidden_layer', activation='relu'))

model.add(keras.layers.Dropout(DROPOUT))

model.add(keras.layers.Dense(1, name='output_layer', activation='sigmoid'))



model.summary()



model.compile(optimizer='Adam', 

              loss='binary_crossentropy',

              metrics=[f1_m])



# Training the model.

model.fit(X_train_tfidf.toarray(), y_train,

          batch_size=BATCH_SIZE, epochs=EPOCHS,

          verbose=VERBOSE, validation_split=VALIDATION_SPLIT,)



# Evaluating the model.

X_val_tfidf = tfidf.transform(X_val.text)

test_loss, test_f1 = model.evaluate(X_val_tfidf.toarray(), y_val)

print('Test F1:', test_f1)
test_preds = model.predict_classes(tfidf.transform(test.text).toarray())

submission_preds = [p[0] for p in test_preds]

submission = pd.DataFrame({'id': test.id, 'target': submission_preds})
submission.head()
submission.to_csv('submission.csv', index=False)