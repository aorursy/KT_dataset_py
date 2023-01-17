!pip install -q keras-bert keras-rectified-adam scikit-plot
import os
os.environ['TF_KERAS'] = '1'
import codecs
import numpy as np
import pandas as pd
from tqdm import tqdm


# Import keras
import tensorflow as tf
from tensorflow.python import keras

# Keras-bert imports
from keras_radam import RAdam
from keras_bert import Tokenizer
from keras_bert import get_custom_objects
from keras_bert import load_trained_model_from_checkpoint
from sklearn.model_selection import train_test_split
print("tf.__version__: %s" % tf.__version__)
print("Num GPUs: ", len(tf.config.experimental.list_physical_devices('GPU')))
!wget -q https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
!unzip -o uncased_L-12_H-768_A-12.zip

# Bert Model Hyper-params
SEQ_LEN = 256
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5

pretrained_path = 'uncased_L-12_H-768_A-12'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

# @title Build Fine-Tuned Bert model

# Load pretrained model
model = load_trained_model_from_checkpoint(
  config_path,
  checkpoint_path,
  training=True,
  trainable=True,
  seq_len=SEQ_LEN,
)

# Add classification layer
inputs = model.inputs[:2]
dense = model.get_layer('NSP-Dense').output
outputs = keras.layers.Dense(units=2, activation='softmax')(dense)
model = keras.models.Model(inputs, outputs)

model.compile(
  RAdam(lr=LR),
  loss='sparse_categorical_crossentropy',
  metrics=['sparse_categorical_accuracy'],
)

print (model.summary())
token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
tokenizer = Tokenizer(token_dict)

def transform_data(reviews, labels, labels_to_ids):
    """
    Input:
      reviews: List of review texts.
      sentiments: Sentiment index-wise for the review.
      labels_to_ids: Dictionary with label mapped to value
    Output:
      Tuple of x and y where
      x: List having two items viz token_input and seg_input.
      y: Output labels corresponding to x.
    """
    global tokenizer
    indices, sentiments = [], []
    for x in range(len(reviews)):
        text = reviews[x]
        sentiment = labels_to_ids[labels[x]]
        ids, segments = tokenizer.encode(text, max_len=SEQ_LEN)
        indices.append(ids)
        sentiments.append(sentiment)
    items = list(zip(indices, sentiments))
    np.random.shuffle(items)
    indices, sentiments = zip(*items)
    indices = np.array(indices)
    mod = indices.shape[0] % BATCH_SIZE
    if mod > 0:
        indices, sentiments = indices[:-mod], sentiments[:-mod]
    return [indices, np.zeros_like(indices)], np.array(sentiments)


filepath = "/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv"
dataset = pd.read_csv(filepath)
train,test = train_test_split(dataset, test_size = 0.25)

labels_to_id = {'positive': 1, 'negative': 0}
id_to_labels = {0: 'negative', 1: 'positive'}

test_x, test_y = transform_data(
    test['review'].values,
    test['sentiment'].values,
    labels_to_id
)

train_x, train_y = transform_data(
    train['review'].values,
    train['sentiment'].values,
    labels_to_id
)

print("Training on: %s samples\nTesting on: %s samples" % (len(train_y), len(test_y)))
# @title Train
history = model.fit(
    train_x,
    train_y,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.20,
    shuffle=True,
)
#@title Plot model training progress
import matplotlib.pyplot as plt
import numpy
%matplotlib inline

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.plot(history.history['sparse_categorical_accuracy'])
plt.title('model sparse_categorical_accuracy')
plt.ylabel('sparse_categorical_accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.subplot(1, 2, 2)
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#@title Get test set predictions.
predicts = model.predict(test_x, verbose=True).argmax(axis=-1)
#@title Evaluate Model Accuracy and F1 Score
#! pip install -q scikit-plot
import scikitplot as skplt
from sklearn.metrics import accuracy_score, f1_score
accuracy = accuracy_score(test_y, predicts)
macro_f1 = f1_score(test_y, predicts, average='macro')
micro_f1 = f1_score(test_y, predicts, average='micro')
weighted_f1 = f1_score(test_y, predicts, average='weighted')

print("Accuracy: %s" % accuracy)
print ('macro_f1: %s\nmicro_f1:%s\nweighted_f1:%s' %(
    macro_f1, micro_f1, weighted_f1)
)

skplt.metrics.plot_confusion_matrix(
    [id_to_labels[x] for x in test_y], 
    [id_to_labels[x] for x in predicts],
    figsize=(10,10))
#@title Classifying texts
texts = [
  "It's a must watch",
  "Can't wait for it's next part!",
  'It fell short of expectations.',
  'Wish there was more to it!',
  'Just wow!',
  'Colossial waste of time',
  'Save youself from this 90 mins trauma!'
]
for text in texts:
    ids, segments = tokenizer.encode(text, max_len=SEQ_LEN)
    inpu = np.array(ids).reshape([1, SEQ_LEN])
    predicted_id = model.predict([inpu,np.zeros_like(inpu)]).argmax(axis=-1)[0]
    print ("%s: %s"% (id_to_labels[predicted_id], text))
