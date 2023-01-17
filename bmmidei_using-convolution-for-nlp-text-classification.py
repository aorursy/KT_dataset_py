import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from keras.callbacks import Callback
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Embedding, Dropout
from keras.layers import Conv1D, MaxPool1D, Flatten, Concatenate
from keras.models import Model
from keras.utils.vis_utils import plot_model

%load_ext autoreload
%autoreload 2
# Load in training and testing data
train_df = pd.read_csv('../input/train.csv')
train_df.head()
# Extract the training data and corresponding labels
text = train_df['question_text'].fillna('unk').values
labels = train_df['target'].values

# Split into training and validation sets by making use of the scikit-learn
# function train_test_split
X_train, X_val, y_train, y_val = train_test_split(text, labels,\
                                                  test_size=0.2)
embed_size = 300 # Size of each word vector
max_words = 50000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a question to use
## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(list(X_train))

# The tokenizer will assign an integer value to each word in the dictionary
# and then convert each string of words into a list of integer values
X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)

word_index = tokenizer.word_index
print('The word index consists of {} unique tokens.'.format(len(word_index)))

## Pad the sentences to the maximum length
X_train = pad_sequences(X_train, maxlen=maxlen)
X_val = pad_sequences(X_val, maxlen=maxlen)
# Create the embedding dictionary from the word embedding file
embedding_dict = {}
filename = os.path.join('../input/embeddings/', 'glove.840B.300d/glove.840B.300d.txt')
with open(filename) as f:
    for line in f:
        line = line.split()
        token = line[0]
        try:
            coefs = np.asarray(line[1:], dtype='float32')
            embedding_dict[token] = coefs
        except:
            pass
print('The embedding dictionary has {} items'.format(len(embedding_dict)))
# Create the embedding layer weight matrix
embed_mat = np.zeros(shape=[max_words, embed_size])
for word, idx in word_index.items():
    # Word index is ordered from most frequent to least frequent
    # Ignore words that occur less frequently
    if idx >= max_words: continue
    vector = embedding_dict.get(word)
    if vector is not None:
        embed_mat[idx] = vector
def create_cnn(filter_sizes, strides, num_filters, embed_train=False, dropout=0.2, plot=False):
    # The first layer will be the word embedding layer
    # by default, the embedding layer will not be trainable as it adds a great deal of complexity
    sequence_input = Input(shape=(maxlen,), dtype='int32')
    x = Embedding(max_words, embed_size, weights=[embed_mat], trainable=embed_train)(sequence_input)
    
    # Convolutional and maxpool layers for each filter size and stride size
    # Convolution is 1D and occurs at different stride lengths.
    # eg. a filter size of 3 and stride of 2 will examine 3 words at a time
    # in the order 0,1,2 - 2,3,4 - 4,5,6 - etc
    conv_layers = []
    maxpool_layers = []
    for i in range(len(filter_sizes)):
        conv_layers.append(Conv1D(num_filters, strides=strides[i], padding='same', kernel_size=(filter_sizes[i]),
                                 kernel_initializer='he_normal', activation='relu')(x))
        # pool_size calculation: (Width - (Filter_size * 2*Padding))/Stride
        pool_size = int((maxlen-(filter_sizes[i]*2))/strides[i])
        maxpool_layers.append(MaxPool1D(pool_size=pool_size, strides=strides[i])(conv_layers[i]))

    # Concatenate pooling layers outputs
    if len(maxpool_layers)==1:
        z = maxpool_layers[0]
    else:
        z = Concatenate(axis=1)(maxpool_layers)
    
    # Finish network with flattened layer, dropout, and fully connected layer
    z = Flatten()(z)
    z = Dropout(dropout)(z)
    z = Dense(64, activation='relu')(z)
    preds = Dense(1, activation='sigmoid')(z) # Sigmoid for binary classification

    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    model.summary()
    
    if plot:
        plot_model(model, to_file='./ims/ConvNet1D.png', show_shapes=True, show_layer_names=True)
    
    return model
threshold = 0.35 # Experimentally found to be the best threshold
class F1Evaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            y_pred = (y_pred > threshold).astype(int)
            score = f1_score(self.y_val, y_pred)
            print("\n F1 Score - epoch: %d - score: %.6f \n" % (epoch+1, score))
            
F1_Score = F1Evaluation(validation_data=(X_val, y_val), interval=1)
# A few parameters to define for the network. Feel free to experiment
# Note that filter_sizes and strides must have the same length
filter_sizes = [1,3,3,5]
strides = [1,1,2,3]
num_filters = 48
dropout = 0.2
embed_train = False

epochs = 3
batch_size = 1024

# Create and train network
cnn = create_cnn(filter_sizes, strides, num_filters, embed_train=embed_train, dropout=dropout)
history = cnn.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs,
                  batch_size=batch_size, callbacks=[F1_Score])
# The best results are seen with a threshold between 0.1 and 0.5
thresholds = np.arange(0.15, 0.5, 0.05)

best_thresh = None
best_score = 0.

# Make predictions and evaluate f1 score for each threshold value
for thresh in thresholds:
    y_pred = cnn.predict(X_val, verbose=0)
    y_pred = (y_pred>thresh).astype(int)
    score = f1_score(y_val, y_pred)
    print('F1 Score for threshold {:0.1f}: {:0.3f}'.format(thresh, score))
    
    # Store best threshold for later use in predictions
    if not best_thresh or score>best_score:
        best_thresh = thresh
test_df = pd.read_csv('../input/test.csv')
X_test = test_df['question_text'].values

# Perform the same preprocessing as was done on the training set
X_test = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(X_test, maxlen=maxlen)

# Make predictions, ensure that predictions are in integer form
# Use best threshold from previous section
preds = np.rint(cnn.predict([X_test], batch_size=1024, verbose=1))
y_pred = (preds>best_thresh).astype(int)
test_df['prediction'] = y_pred
n=5
sin_sample = test_df.loc[test_df['prediction'] == 0]['question_text'].head(n)
print('Sincere Samples:')
for idx, row in enumerate(sin_sample):
    print('{}'.format(idx+1), row)

print('\n')
print('Insincere Samples:')
insin_sample = test_df.loc[test_df['prediction'] == 1]['question_text'].head(n)
for idx, row in enumerate(insin_sample):
    print('{}'.format(idx+1), row)
# Drop the question text from the dataframe leaving only question ID and preds
# Then write to submission csv for competition
test_df = test_df.drop('question_text', axis=1)
test_df.to_csv('submission.csv', index=False)