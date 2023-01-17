import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from nltk import word_tokenize



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import auc, roc_auc_score, roc_curve, confusion_matrix, classification_report



from keras.models import Model, load_model

from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding

from keras.optimizers import RMSprop

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical, plot_model

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))

        

# for dirname, _, filenames in os.walk('/kaggle/working'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/sms-spam-collection-dataset/spam.csv", encoding='latin-1')

df.head()
df.info()
fig, ax = plt.subplots()

sns.countplot(df.v1, ax=ax)

ax.set_xlabel('Label')

ax.set_title('Number of ham and spam messages')
X = df.loc[:, 'v2']

y = df.loc[:, 'v1']



X
X_train_data, X_test_data, y_train_labels, y_test_labels = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train_data.shape)

print(X_test_data.shape)
#find length of each sentence after tokenization

sent_lens = []

for sent in X_train_data:

    sent_lens.append(len(word_tokenize(sent)))

    

print(max(sent_lens))
sns.distplot(sent_lens, bins=10, kde=True)
#check the length of 95% of review text to help in finding max. sequence length.

np.quantile(sent_lens, 0.95)
max_sequence_length = 38



tok = Tokenizer()

tok.fit_on_texts(X_train_data.values)



vocab_length = len(tok.word_index) #len(tok.word_counts) or len(tok.index_word.keys()) will also give same results

print('No. of unique tokens(vocab_size): ', vocab_length)



X_train_sequences = tok.texts_to_sequences(X_train_data.values)

X_test_sequences = tok.texts_to_sequences(X_test_data.values)

print('No of sequences:', len(X_train_sequences)) #No of sequences will be same as the number of training samples

print(X_train_sequences[:2])



#make all sequences of equal length

X_train = pad_sequences(X_train_sequences, maxlen=max_sequence_length)

X_test = pad_sequences(X_test_sequences, maxlen=max_sequence_length)

X_train[:2]
y_train_labels.values
le = LabelEncoder()

y_train = le.fit_transform(y_train_labels)

y_test = le.fit_transform(y_test_labels)

print(y_train)



# y_train_le  = y_train_le.reshape(-1, 1)

# y_test_le  = y_test_le.reshape(-1, 1)

# print(y_train_le)



# y_train = np.asarray(y_train_le).astype('float32')

# y_test = np.asarray(y_test_le).astype('float32')

# print(y_train)
def create_model(vocab_len, max_seq_len):

    inputs = Input(name='inputs', shape=[max_seq_len])   #None, 150

    layer = Embedding(vocab_length + 1, 50, input_length=max_seq_len)(inputs) #None, 150, 50

    layer = LSTM(64)(layer)  #None, 64

    layer = Dense(256,name='FC1')(layer) #None, 256

    layer = Activation('relu')(layer) #None, 256

    layer = Dropout(0.5)(layer) #None, 256

    layer = Dense(1,name='out_layer')(layer) #None, 1

    layer = Activation('sigmoid')(layer) #None, 1

    model = Model(inputs=inputs,outputs=layer)

    model.compile(loss='binary_crossentropy',optimizer=RMSprop(), metrics=['acc'])

    return model



model = create_model(vocab_length, max_sequence_length)

model.summary()
# plot_model(model, show_shapes=True)
# Load the extension and start TensorBoard

# %load_ext tensorboard

# %tensorboard --logdir logs



filepath='model_with_best_weights.h5' #fixed path to save the best model

# filepath="weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5" #file path will change based on epoch and loss

# Checkpointing is setup to save the network weights only when there is an improvement in classification accuracy on the validation dataset (monitor=’val_accuracy’ and mode=’max’). 

# The weights are stored in a file that includes the score in the filename (weights-improvement-{val_accuracy=.2f}.hdf5).

callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=1),  #EarlyStopping(monitor='val_loss',min_delta=0.0001, patience=5),

             ModelCheckpoint(filepath=filepath, monitor='val_loss', save_best_only=True, verbose=1),

#              TensorBoard(log_dir='logs', histogram_freq=1, embeddings_freq=1)             

            ]
history = model.fit(X_train, y_train, batch_size=128, epochs=20, validation_split=0.2, callbacks=callbacks)
history_dict = history.history



# list all data in history

print(history_dict.keys())



# summarize history for loss

plt.plot(history_dict['loss'])

plt.plot(history_dict['val_loss'])

plt.title('Training and Validation Loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()



# summarize history for accuracy

plt.plot(history_dict['acc'])

plt.plot(history_dict['val_acc'])

plt.title('Training and Validation Accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
loaded_model = load_model('model_with_best_weights.h5')

test_loss, test_acc = accr = loaded_model.evaluate(X_test, y_test)

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(test_loss, test_acc))
# make class predictions with the model on new data

y_pred_proba = loaded_model.predict(X_test)



# y_pred = loaded_model.predict_classes(X_test)  #we can't use it on Model object. Can be used on Sequential object

print(np.round(y_pred_proba, 3))

y_pred = y_pred_proba > 0.5

y_pred
# summarize the first few cases

for i in range(5):

    print('%s => %d (expected %d)' % (X_test[i].tolist(), y_pred[i], y_test[i]))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
#calculate the roc auc score

auc = roc_auc_score(y_test, y_pred_proba)

print('AUC: %.3f' % auc)
#plot the roc curve

fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_proba)



def plot_roc_curve(fpr,tpr): 

  import matplotlib.pyplot as plt

  plt.plot(fpr,tpr) 

  plt.axis([0,1,0,1]) 

  plt.xlabel('False Positive Rate') 

  plt.ylabel('True Positive Rate') 

  plt.show()    

  

plot_roc_curve (fpr_keras, tpr_keras)