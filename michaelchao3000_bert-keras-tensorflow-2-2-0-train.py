# https://www.tensorflow.org/official_models/fine_tuning_bert#setup
# import sys
# import subprocess
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'bert-for-tf2'])

# from pip._internal import main as pipmain
# pipmain(['install', 'bert-for-tf2'])

import pip
pip.main(['install', 'bert-for-tf2'])
import os
import numpy as np
import pandas as pd
import datetime
import sys
# import zipfile
# import modeling
# import optimization
# import run_classifier
# import tokenization

# from tokenization import FullTokenizer
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from sklearn.model_selection import train_test_split

import tensorflow_hub as hub
from tqdm import tqdm_notebook
from tqdm import notebook # tqdm.notebook.tqdm
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

print(f'Python {sys.version}')
print(f'Tensorflow {tf.__version__}')
# https://www.kaggle.com/c/amazon-pet-product-reviews-classification

train_df = pd.read_csv('../input/amazon-pet-product-reviews-classification/train.csv', index_col='id')
val_df = pd.read_csv('../input/amazon-pet-product-reviews-classification/valid.csv', index_col='id')
test_df = pd.read_csv('../input/amazon-pet-product-reviews-classification/test.csv', index_col='id')

label_encoder = LabelEncoder().fit(pd.concat([train_df['label'], val_df['label']]))

# Train and Validation Features (text)
X_train_val, X_test = pd.concat([train_df['text'], val_df['text']]).values, test_df['text'].values

# Train and Validation Labels
y_train_val = label_encoder.fit_transform(pd.concat([train_df['label'], val_df['label']]))

# Split into new train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=0, stratify=y_train_val)
max_seq_length = 128  # Your choice here.

train_text = X_train
train_text = [' '.join(t.split()[0:max_seq_length]) for t in train_text]
train_text = np.array(train_text, dtype=object)[:, np.newaxis]
train_label = y_train

val_text = X_val
val_text = [' '.join(t.split()[0:max_seq_length]) for t in val_text]
val_text = np.array(val_text, dtype=object)[:, np.newaxis]
val_label = y_val

test_text = X_test
test_text = [' '.join(t.split()[0:max_seq_length]) for t in test_text]
test_text = np.array(test_text, dtype=object)[:, np.newaxis]
# https://stackabuse.com/text-classification-with-bert-tokenizer-and-tf-2-0-in-python/

import bert

BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2",
                            trainable=False)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

# Test tokenizer
tokenizer.tokenize("don't be so judgmental")
class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

def create_tokenizer_from_hub_module(tf_hub):
    """
    Bert tokenizer for Tensorflow 2
    https://stackabuse.com/text-classification-with-bert-tokenizer-and-tf-2-0-in-python/

    
    Get the vocab file and casing info from the Hub module.
    """
    BertTokenizer = bert.bert_tokenization.FullTokenizer
    bert_layer = hub.KerasLayer(tf_hub,
                                trainable=False)
    vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = BertTokenizer(vocabulary_file, to_lower_case)
    
    return tokenizer

def convert_single_example(tokenizer, example, max_seq_length=256):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        label = 0
        return input_ids, input_mask, segment_ids, label

    tokens_a = tokenizer.tokenize(example.text_a)
    if len(tokens_a) > max_seq_length - 2: # Take into account that we will be prepending [CLS] and appending [SEP] to the sequence
        tokens_a = tokens_a[0 : (max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    
    #print(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, example.label

def convert_examples_to_features(tokenizer, examples, max_seq_length=256):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    input_ids, input_masks, segment_ids, labels = [], [], [], []
    for example in notebook.tqdm(examples, desc="Converting examples to features"):
        input_id, input_mask, segment_id, label = convert_single_example(
            tokenizer, example, max_seq_length
        )
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        labels.append(label)
    return (
        np.array(input_ids),
        np.array(input_masks),
        np.array(segment_ids),
        np.array(labels).reshape(-1, 1),
    )

def convert_text_to_examples(texts, labels):
    """Create InputExamples"""
    InputExamples = []
    for text, label in zip(texts, labels):
        InputExamples.append(
            InputExample(guid=None, text_a=" ".join(text), text_b=None, label=label)
        )
    return InputExamples
# Instantiate tokenizer
bert_path = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"
tokenizer = create_tokenizer_from_hub_module(bert_path)

# Convert data to InputExample format
train_examples = convert_text_to_examples(train_text, train_label)
val_examples = convert_text_to_examples(val_text, val_label)

# Convert to features
(train_input_ids, train_input_masks, train_segment_ids, train_labels) = convert_examples_to_features(tokenizer, train_examples, max_seq_length=max_seq_length)
(val_input_ids, val_input_masks, val_segment_ids, val_labels) = convert_examples_to_features(tokenizer, val_examples, max_seq_length=max_seq_length)
import tensorflow as tf
import tensorflow_hub as hub
import os
import re
import numpy as np
from tqdm import tqdm_notebook
#from tensorflow.keras import backend as K
from keras import backend as K
from tensorflow.keras.layers import Layer
import keras


'''
Resources explaining how keras.layers.Layer is implemented
You can override the build() and call() methods in a subclass of Layer because Layer.__call__() calls build() and call()

https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#__call__
https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/engine/base_layer.py#L875-L994
'''

class BertLayer(Layer):
    
    '''BertLayer which support next output_representation param:
    
    pooled_output: the first CLS token after adding projection layer () with shape [batch_size, 768]. 
    sequence_output: all tokens output with shape [batch_size, max_length, 768].
    mean_pooling: mean pooling of all tokens output [batch_size, max_length, 768].
    
    For view trainable parameters call model.trainable_weights after creating model.
    
    '''
    
    def __init__(self, output_representation='pooled_output', is_trainable=False, **kwargs):
        
        self.is_trainble = is_trainable
        self.output_size = 768
        self.tf_hub = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"
        self.output_representation = output_representation
        self.supports_masking = True
        
        super(BertLayer, self).__init__(**kwargs)


    '''
    build() is called by the Layer method __call__()
    defining build() here will make the inherited Layer.__call__() call the build() method defined in the BertLayer subclass
    build() is called in Layer.__call__() before call()
    '''
    def build(self, input_shape):
#         print('BertLayer.build() is called')

        # Get pre-trained BERT module
        self.bert = hub.KerasLayer(self.tf_hub, trainable=True)
        
        # Set trainable variables
        variables = self.bert.variables
        self.bert._trainable_weights = [var for var in variables if '/layer_10/' in var.name]
        self.bert._trainable_weights.extend([var for var in variables if '/layer_11/' in var.name])
         
        # Set non-trainable variables - these weights are "frozen"
        trainable_vars_name = [var.name for var in self._trainable_weights]
        for var in variables:
            if var.name not in trainable_vars_name:
                self.bert._non_trainable_weights.append(var)
        
        super(BertLayer, self).build(input_shape)
    
    '''
    call() is called by the Layer method __call__()
    defining call() here will make the inherited Layer.__call__() call the call() method defined in the BertLayer subclass
    build() is called in Layer.__call__() before call()
    '''
    def call(self, inputs):
#         print('BertLayer.call() is called')
        
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)
        result = self.bert(inputs=inputs) # result = [pooled_output, sequence_output]
        
        if self.output_representation == "pooled_output":
            pooled = result[0] # result["pooled_output"]
            
        elif self.output_representation == "mean_pooling":
            result_tmp = result[1] # result["sequence_output"]
        
            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
            input_mask = tf.cast(input_mask, tf.float32)
            pooled = masked_reduce_mean(result_tmp, input_mask)
            
        elif self.output_representation == "sequence_output":
            pooled = result[1] # result["sequence_output"]
       
        return pooled
    
    def compute_mask(self, inputs, mask=None):
        
        if self.output_representation == 'sequence_output':
            inputs = [K.cast(x, dtype="bool") for x in inputs]
            mask = inputs[1]
            
            return mask
        else:
            return None
        
        
    def compute_output_shape(self, input_shape):
        if self.output_representation == "sequence_output":
            return (input_shape[0][0], input_shape[0][1], self.output_size)
        else:
            return (input_shape[0][0], self.output_size)
import keras

def build_model(max_seq_length, n_classes):
    
    max_seq_length = 128  # Your choice here.
    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="segment_ids")
    bert_inputs = [input_word_ids, input_mask, segment_ids]

    bert_layer = BertLayer(output_representation='mean_pooling', is_trainable=True)
#     bert_layer.trainable = False # Set entire BERT layer to non-trainable
    bert_output = bert_layer(inputs=bert_inputs)
    
    drop = keras.layers.Dropout(0.3)(bert_output)
    dense = keras.layers.Dense(256, activation='relu')(drop)
    drop = keras.layers.Dropout(0.3)(dense)
    dense = keras.layers.Dense(128, activation='relu')(drop)
    pred = keras.layers.Dense(n_classes, activation='softmax')(dense)
    
    model = keras.models.Model(inputs=bert_inputs, outputs=pred)
    Adam = tf.keras.optimizers.Adam(lr=1e-3)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam, metrics=['sparse_categorical_accuracy'])
    model.summary()

    return model

def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)
tf.keras.backend.clear_session()

n_classes = len(label_encoder.classes_)
model = build_model(max_seq_length, n_classes)
from keras.callbacks import EarlyStopping

    
EPOCHS = 5 # SAMPLES_PER_EPOCH = 62469, BATCH_SIZE = 256
BATCH_SIZE = 256
print(f'BATCH_SIZE is {BATCH_SIZE}')
e_stopping = EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=3, verbose=1, mode='max', restore_best_weights=True)
callbacks = [e_stopping]

lr = 1e-3 # initial learning rate
for epoch in range(1, EPOCHS+1):
    history = model.fit([train_input_ids, train_input_masks, train_segment_ids],
                    train_labels,
                    validation_data = ([val_input_ids, val_input_masks, val_segment_ids], val_labels),
                    epochs=1,
                    verbose=1,
                    batch_size=BATCH_SIZE,
                    callbacks=callbacks)
    
    if (epoch%2) == 0:
        # Manually decay learning rate
        lr = lr/2
        Adam = tf.keras.optimizers.Adam(lr=lr)
        model.optimizer = Adam
        model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam, metrics=['sparse_categorical_accuracy'])
'''
# Choose an optimizer and loss function for training:
# EPOCHS = 10, SAMPLES_PER_EPOCH = 62469, BATCH_SIZE = 256
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=int(2*62469/256), decay_rate=0.5, staircase=True) 
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-07)


# Select metrics to measure the loss and the accuracy of the model. These metrics accumulate the values over epochs and then print the overall result.
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


# Use tf.GradientTape to train the model:
@tf.function
def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions) 
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)
    
    
# Test the model:
@tf.function
def test_step(model, images, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)
    
    
# Use tf.data to batch and shuffle the dataset:
train_ds = tf.data.Dataset.from_tensor_slices((train_input_ids, train_input_masks, train_segment_ids, train_labels)).shuffle(62469).batch(256)
test_ds = tf.data.Dataset.from_tensor_slices((val_input_ids, val_input_masks, val_segment_ids, val_labels)).batch(256)


# Start training loop
EPOCHS = 10
for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for input_ids, input_masks, segment_ids, labels in train_ds:
        train_step(model, [input_ids, input_masks, segment_ids], labels)

    for test_input_ids, test_input_masks, test_segment_ids, test_labels in test_ds:
        test_step(model, [test_input_ids, test_input_masks, test_segment_ids], test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                        train_loss.result(),
                        train_accuracy.result() * 100,
                        test_loss.result(),
                        test_accuracy.result() * 100))
'''
# save model
model.save("BERT_Keras_TF2")

# save weights
# model.save_weights("BERT_Keras_TF2.h5")
### PREDICTIONS ON VALIDATION SET ### 
prediction = model.predict([val_input_ids, val_input_masks, val_segment_ids], verbose=1)
preds = label_encoder.classes_[np.argmax(prediction, axis=1)]

pd.set_option('display.max_colwidth', None)
results = pd.DataFrame({'Text':val_text[:,0], 'Label':label_encoder.classes_[val_labels[:,0]], 'Pred':preds})
results.head()
results[(results.Label != results.Pred)]
results[(results.Label == results.Pred)]
total_count = len(results)
error_count = len(results[(results.Label != results.Pred)])
correct_count = total_count - error_count

print(f'Validation Results: {correct_count}/{total_count} - {100*correct_count/total_count:.2f}%')
for label in label_encoder.classes_:
    print(f'{label} - {len(results[(results.Label==results.Pred) & (results.Label==label)])}')
# https://scikit-learn.org/0.18/auto_examples/model_selection/plot_confusion_matrix.html

import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
        
    print(cm)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    

# Compute confusion matrix
# labels = label_encoder.classes_
cnf_matrix = confusion_matrix(y_true=results.Label, y_pred=results.Pred)
class_names = label_encoder.classes_ # list(set(results.Label))
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize=(10,10))
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure(figsize=(10,10))
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
test_text = [['''
This fake dog stuffed animal is great because my cat loves it!
''']]
test_label = [[0]] # input arrays need to be 2-D: [batch, seq_length]
test_examples = convert_text_to_examples(test_text, test_label)
input_ids, input_masks, segment_ids, labels = convert_examples_to_features(tokenizer, test_examples, max_seq_length=max_seq_length)

prob_a = model.predict([input_ids, input_masks, segment_ids], verbose=1)
preds = label_encoder.classes_[np.argmax(prob_a, axis=1)]

print()
print(f'Raw text: {test_text[0][0]}')
print(tokenizer.convert_ids_to_tokens(input_ids[0]))
# print(tokenizer.tokenize(test_text[0][0]))

print()
for prob, class_ in zip(prob_a[0], label_encoder.classes_):
    print(f'{class_:20} - {prob:.3f}')
    
print()
print(f'Pred: {preds[0]}')
'''
There's currently no good way to get to the intermediate layers of models imported via hub.KerasLayer
https://github.com/tensorflow/hub/issues/453
https://stackoverflow.com/questions/57410282/tensorflow-2-hub-how-can-i-obtain-the-output-of-an-intermediate-layer
https://stackoverflow.com/questions/55333558/how-to-access-bert-intermediate-layer-outputs-in-tf-hub-module

https://keras.io/getting_started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer-feature-extraction

'''