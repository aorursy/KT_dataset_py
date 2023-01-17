import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow import keras
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import Constant
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from nltk.tokenize import word_tokenize
train_df=pd.read_csv('../input/nlp-getting-started/train.csv')
test_df=pd.read_csv('../input/nlp-getting-started/test.csv')
train_df.head()
labels = 'Disaster Tweets', 'Non-Disaster Tweets'
sizes = np.array(train_df.target.value_counts())/len(train_df)*100

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
df_correlation = train_df.dropna()
df_correlation=df_correlation.drop(columns=['id','text'])
df_correlation['keyword']=df_correlation['keyword']=df_correlation['keyword'].astype('category').cat.codes
df_correlation['location']=df_correlation['location']=df_correlation['location'].astype('category').cat.codes
corr=df_correlation.corr()
sns.heatmap(corr, vmax=0.8)
corr_values=corr['target'].sort_values(ascending=False)
corr_values=abs(corr_values).sort_values(ascending=False)
print("Correlation of keyword and location with target in ascending order")
print(abs(corr_values).sort_values(ascending=False))
train_df=train_df.drop(columns=['location','keyword','id'])
test_df=test_df.drop(columns=['location','keyword'])
import re

def remove_URL(df):
    for i in range(df.shape[0]):
        df.text[i]=re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','',df.text[i])
        
remove_URL(train_df)
remove_URL(test_df)
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

train_df['text']=train_df['text'].apply(lambda x: remove_emoji(x))
test_df['text']=test_df['text'].apply(lambda x: remove_emoji(x))
VOCAB_SIZE=4500
MAXLEN=40
tokenizer=Tokenizer(VOCAB_SIZE,oov_token='<oov>', filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')  # filtering special characters

tokenizer.fit_on_texts(train_df.text)
def df_to_padded_sequences(df,tokenizer):
    sequences=tokenizer.texts_to_sequences(df.text)                                              #text to sequence of integers
    padded_sequences=pad_sequences(sequences,maxlen=MAXLEN, padding='post', truncating='post')  #padding
    return padded_sequences

X_train=df_to_padded_sequences(train_df,tokenizer)
X_test=df_to_padded_sequences(test_df,tokenizer)
print('Original tweet: ',train_df.text[0])
print('Tokenized tweet: ',X_train[0])
print('Token to tweet: ',tokenizer.sequences_to_texts(X_train)[0])
Y_train=train_df.target
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2)

print('Training features shape: ',X_train.shape)
print('Validation features shape: ',X_val.shape)

print('Training labels shape: ', Y_train.shape)
print('Validation labels shape: ', Y_val.shape)
model1 = keras.Sequential([
    keras.layers.Embedding(VOCAB_SIZE, 32,input_length=MAXLEN),
    keras.layers.SpatialDropout1D(0.2),
    keras.layers.Bidirectional(keras.layers.LSTM(16)),
    keras.layers.Dense(1, activation="sigmoid")
])
model1.summary()
EPOCHS=30

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_acc', 
    verbose=1,
    patience=5,
    mode='max',
    restore_best_weights=True)



model1.compile(loss="binary_crossentropy",optimizer=keras.optimizers.RMSprop(1e-4), metrics=['acc'])

history1 = model1.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=32, epochs=EPOCHS, callbacks = [early_stopping])
tokenizer=Tokenizer(oov_token='<oov>', filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')  # filtering special characters

tokenizer.fit_on_texts(train_df.text)

vocab_length = len(tokenizer.word_index) + 1
print(vocab_length)
MAXLEN=40

def df_to_padded_sequences(df,tokenizer):
    sequences=tokenizer.texts_to_sequences(df.text)                                              #text to sequence of integers
    padded_sequences=pad_sequences(sequences,maxlen=MAXLEN, padding='post', truncating='post')  #padding
    return padded_sequences

X_train2=df_to_padded_sequences(train_df,tokenizer)
X_test2=df_to_padded_sequences(test_df,tokenizer)
Y_train2=train_df.target
X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train2, Y_train2, test_size=0.2)

print('Training features shape: ',X_train2.shape)
print('Validation features shape: ',X_val2.shape)

print('Training labels shape: ', Y_train2.shape)
print('Validation labels shape: ', Y_val2.shape)
embeddings_dictionary = dict()
embedding_dim = 50
glove_file = open('../input/glove-global-vectors-for-word-representation/glove.6B.50d.txt')
for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = np.asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()



embedding_matrix = np.zeros((vocab_length, embedding_dim))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector


model2 = keras.Sequential([
    keras.layers.Embedding(vocab_length, 50, embeddings_initializer=Constant(embedding_matrix), input_length=MAXLEN, trainable=False),
    keras.layers.SpatialDropout1D(0.2),
    keras.layers.Bidirectional(keras.layers.LSTM(16)),
    keras.layers.Dense(1, activation="sigmoid")
])
model2.summary()
EPOCHS=30

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_acc', 
    verbose=1,
    patience=5,
    mode='max',
    restore_best_weights=True)



model2.compile(loss="binary_crossentropy",optimizer=keras.optimizers.RMSprop(1e-4), metrics=['acc'])

history2 = model2.fit(X_train2, Y_train2, validation_data=(X_val2, Y_val2), batch_size=32, epochs=EPOCHS, callbacks = [early_stopping])
loss1, acc1 = model1.evaluate(X_val,Y_val)
loss2, acc2 = model2.evaluate(X_val2,Y_val2)
print("Val_acc model1: ",acc1)
print("Val_acc model2: ",acc2)
acc = history1.history['acc']
val_acc = history1.history['val_acc']

loss = history1.history['loss']
val_loss = history1.history['val_loss']


plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(history1.epoch, acc, label='Training Accuracy')
plt.plot(history1.epoch, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy ')

plt.subplot(1, 2, 2)
plt.plot(history1.epoch, loss, label='Training Loss')
plt.plot(history1.epoch, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

def plot_cm(labels, predictions, p=0.5): 
  cm = confusion_matrix(labels, predictions > p)
  cm_sum = np.sum(cm, axis=1, keepdims=True)
  cm_perc = cm / cm_sum.astype(float) * 100
  annot = np.empty_like(cm).astype(str)
  nrows, ncols = cm.shape
  for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            annot[i, j] = '%.1f%%\n%d' % (p, c)
  plt.figure(figsize=(5,5))
  sns.heatmap(cm, annot=annot, fmt="",cmap="YlGnBu")
  plt.title('Confusion matrix')
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')

  print('Non-Disaster Tweet Detected (True Negatives): ', cm[0][0])
  print('Disaster Tweet Incorrectly Detected (False Positives): ', cm[0][1])
  print('Disaster Tweet Missed (False Negatives): ', cm[1][0])
  print('Disaster Tweet Detected (True Positives): ', cm[1][1])
  print('Total Disaster Tweet: ', np.sum(cm[1]))
  print('Total Non-Disaster Tweet: ', np.sum(cm[0]))
val_prediction=model1.predict(X_val)
plot_cm(Y_val, val_prediction)
Y_test=model1.predict(X_test)
print(Y_test)
Y_test = [int(i>0.5) for i in Y_test]
submission_dataframe = pd.DataFrame({"id" : test_df['id'], "target" : Y_test})
submission_dataframe.to_csv("submission.csv", index = False)
submission_dataframe