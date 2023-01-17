import numpy as np # linear algebra
import pandas as pd # data processing

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Bidirectional, SpatialDropout1D
from keras.optimizers import SGD,Adam
from keras.layers.core import Dense,Activation,Dropout
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.preprocessing import sequence,text
from keras.callbacks import Callback,EarlyStopping,ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
%matplotlib inline
import zipfile

train_zip = zipfile.ZipFile('/kaggle/input/spooky-author-identification/train.zip')
train_df = pd.read_csv(train_zip.open('train.csv'))

test_zip = zipfile.ZipFile('/kaggle/input/spooky-author-identification/test.zip')
test_df = pd.read_csv(test_zip.open('test.csv'))

sample_zip = zipfile.ZipFile('/kaggle/input/spooky-author-identification/sample_submission.zip')
sample_df = pd.read_csv(sample_zip.open('sample_submission.csv'))
train_df.head(10)
#check authors
train_df.author.unique()
test_df.head(10)
#check what is the maximum and the minimum length of text
print('max: ',len(train_df.text.max()))
print('min: ',len(train_df.text.min()))
#print the max and min length text
print('max: ',train_df.text.max())
print('min',train_df.text.min())
#convert the authors/labels into one hot encoded values
lbl_enc = LabelEncoder()
y = lbl_enc.fit_transform(train_df.author.values)
y = np_utils.to_categorical(y)
#divide the data into train and validation 
x_train, x_valid, y_train, y_valid = train_test_split(train_df.text.values,
                                                      y,
                                                      stratify = y,
                                                     random_state = 2018,
                                                     test_size = 0.2)
#use Keras Tokenizer to tokenize the texts
token = text.Tokenizer(num_words = None)
max_len = 80

token.fit_on_texts(list(x_train) + list(x_valid))
xtrain_seq = token.texts_to_sequences(x_train)
xvalid_seq = token.texts_to_sequences(x_valid)

print(xtrain_seq[:1])
# zero pad the sequences
xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len,padding = 'post')
xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len,padding = 'post')
print(xtrain_pad[:1])
word_index = token.word_index
def get_model():
    model = Sequential()
    model.add(Embedding(len(word_index) + 1,
                         300,
                         input_length=max_len))
    model.add(SpatialDropout1D(0.3))
    model.add(Bidirectional(LSTM(300, dropout=0.3, recurrent_dropout=0.3,return_sequences = True)))
    model.add(Bidirectional(LSTM(300, dropout=0.3, recurrent_dropout=0.3)))

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(3))
    model.add(Activation('softmax'))
    adam = Adam(lr=0.01, decay = 0.05)
    model.compile(loss='categorical_crossentropy', optimizer=adam,
                 metrics=['accuracy'])
    return model
## Callback for loss logging per epoch
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))

history = LossHistory()
def plot_loss():
    #plot training curve
    loss = history.losses
    val_loss = history.val_losses
    acc = history.acc
    val_acc = history.val_acc

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Acc - Loss Trend')
    plt.plot(loss, 'blue', label='Training Loss')
    plt.plot(val_loss, 'green', label='Validation Loss')
    plt.plot(acc, 'black', label='Training Accuracy')
    plt.plot(val_acc, 'red', label='Validation Accuracy')
    plt.xticks(range(0,10)[0::2])
    plt.legend()
    plt.show()
earlystop = EarlyStopping(monitor='val_loss', patience=6, verbose=0, mode='auto')
reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='min', epsilon=0.0001, min_lr=0.0000001)

model = get_model()
model.fit(xtrain_pad, y=y_train, batch_size=64, epochs=70, verbose=1, 
          validation_data=(xvalid_pad, y_valid), callbacks=[earlystop,history,reduceLR])
plot_loss()
#the same operations are also carried out on the test set
xtest_seq = token.texts_to_sequences(test_df.text)
xtest_pad = sequence.pad_sequences(xtest_seq,maxlen = max_len,padding = 'post')
# Now predict
prediction = model.predict(xtest_pad)
a2c = {'EAP': 0, 'HPL' : 1, 'MWS' : 2}
result = sample_df
for a, i in a2c.items():
    result[a] = prediction[:, i]
    
result.to_csv('lstmsubmission2.csv',index = False)
result.head(10)
result.head(10)