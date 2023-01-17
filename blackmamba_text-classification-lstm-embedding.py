import pandas as pd
import numpy as np
df = pd.read_csv("/kaggle/input/multiclass-text-classification/Merilytics_Clean.csv")
df_clean = df[df['review_id'].notnull() & df['text'].notnull()]

# take a peek at the data
df_clean.head()

reviews = np.array(df_clean['text'])
ratings = np.array(df_clean['stars'])

# build train and test datasets

train_len = int(0.9*len(df_clean))
print(f'Training Data {train_len} Testing Data {len(reviews)-train_len}')
train_reviews = reviews[:train_len]
train_rating = ratings[:train_len]
test_reviews = reviews[train_len:]
test_rating = ratings[train_len:]
type(train_reviews)
from nltk.tokenize import ToktokTokenizer
import time
tokenizer = ToktokTokenizer()
start = time.time()

#[x for x in t.tokenize('I am good, 2 3 4') if x.isalpha()]

tokenized_train = [[x for x in tokenizer.tokenize(review) if x.isalpha()] for review in train_reviews]
tokenized_test = [[x for x in tokenizer.tokenize(review) if x.isalpha()] for review in test_reviews]
print(f'Time Taken {time.time()-start}')
del df, df_clean
from collections import Counter

# build word to index vocabulary
token_counter = Counter([token for review in tokenized_train for token in review])
vocab_map = {item[0]: index+1 for index, item in enumerate(dict(token_counter).items())}
max_index = np.max(list(vocab_map.values()))
vocab_map['PAD_INDEX'] = 0
vocab_map['NOT_FOUND_INDEX'] = max_index+1
vocab_size = len(vocab_map)
# view vocabulary size and part of the vocabulary map
print('Vocabulary Size:', vocab_size)
print('Sample slice of vocabulary map:', dict(list(vocab_map.items())[10:20]))
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# get max length of train corpus and initialize label encoder
le = LabelEncoder()
num_classes=5
max_len = np.max([len(review) for review in tokenized_train])

## Train reviews data corpus
# Convert tokenized text reviews to numeric vectors
train_X = [[vocab_map[token] for token in tokenized_review] for tokenized_review in tokenized_train]
train_X = sequence.pad_sequences(train_X, maxlen=max_len) # pad 
## Train prediction class labels
y_tr = le.fit_transform(train_rating)
y_train = to_categorical(y_tr, num_classes)
## Test reviews data corpus
# Convert tokenized text reviews to numeric vectors
test_X = [[vocab_map[token] if vocab_map.get(token) else vocab_map['NOT_FOUND_INDEX'] 
           for token in tokenized_review] 
              for tokenized_review in tokenized_test]
test_X = sequence.pad_sequences(test_X, maxlen=max_len)
## Test prediction class labels
# Convert text sentiment labels (negative\positive) to binary encodings (0/1)
y_ts = le.transform(test_rating)
y_test = to_categorical(y_ts, num_classes)

# view vector shapes
print('Max length of train review vectors:', max_len)
print('Train review vectors shape:', train_X.shape, ' Test review vectors shape:', test_X.shape)
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, Activation, TimeDistributed, Conv1D, MaxPooling1D, Flatten, SpatialDropout1D
from keras.layers import LSTM

EMBEDDING_DIM =128 # dimension for dense embeddings for each token
LSTM_DIM = 64 # total LSTM units

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_len))
model.add(SpatialDropout1D(0.2))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(LSTM_DIM, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"]) 
##Complete Later
'''
nlp_input = Input(shape=(max_len,))
meta_input = Input(shape=(3,))
# the first branch operates on the first input
x = Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_len)(nlp_input)
x = SpatialDropout1D(0.2)(x)
x = LSTM(LSTM_DIM, dropout=0.2, recurrent_dropout=0.2)(x)
x = Model(inputs=nlp_input, outputs=x)
# the second branch opreates on the second input
y = Dense(32, activation="relu")(meta_input)
y = Dense(8, activation="relu")(y)
y = Dense(1, activation="relu")(y)
y = Model(inputs=meta_input, outputs=y)
# combine the output of the two branches
combined = concatenate([x.output, y.output])
# apply a FC layer and then a regression prediction on the
# combined outputs
z = Dense(64, activation="relu")(combined)
z = Dropout(0.4)(z)
z = Dense(16, activation="relu")
z = Dropout(0.4)(z)
z = Dense(5, activation="relu")
z = Activation("softmax")(z)
# our model will accept the inputs of the two branches and
# then output a single value
model = Model(inputs=[x.input, y.input], outputs=z)
model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])
'''
print(model.summary())
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model, show_shapes=True, show_layer_names=False, 
                 rankdir='LR').create(prog='dot', format='svg'))
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

batch_size = 20
history = model.fit(train_X, y_train, epochs=3, batch_size=batch_size, 
          shuffle=True, validation_split=0.2, verbose=1)
plot_history(history)
history.history
from sklearn.metrics import classification_report
pred_test = model.predict_classes(test_X)
predictions = le.inverse_transform(pred_test.flatten())
print(classification_report(test_rating,predictions))