# http://jalammar.github.io/illustrated-transformer/

import pip
pip.main(['install', 'bert-for-tf2'])

import os
import numpy as np
import pandas as pd
import datetime
import sys
from matplotlib import pyplot as plt
import seaborn as sns

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
# Load saved model

tf.keras.backend.clear_session()

# n_classes = len(label_encoder.classes_)
# model = build_model(max_seq_length, n_classes)
# model.load_weights("../input/bert-keras-tensorflow-2-2-0/BERT_Keras_TF2.h5")

model = tf.keras.models.load_model('../input/bert-keras-tensorflow-2-2-0-train/BERT_Keras_TF2')
model.summary()
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
model.layers
# BERT weights

# vocab size = 30,522 # len(list(tokenizer.vocab.values()))
# hidden layer size = 768
# attention heads = 12
# max_position_embeddings is defaulted to be 512

for v in model.layers[3].weights:
    print(f'{v.name:75} {np.shape(v)}')
def get_attention_scores(model, input_ids):
    
    # Convert IDs to one-hot vectors
    input_onehot = np.zeros([128,30522])
    for i, id in enumerate(input_ids):
        input_onehot[i,id] = 1.0

    # Apply embeddings 
    input_onehot = tf.Variable(input_onehot, dtype=tf.float32)
    output = tf.matmul(input_onehot, model.layers[3].weights[0])
    output += model.layers[3].weights[1][:128] # add position embedding
    output += model.layers[3].weights[2][0] # add type embedding

    # Normalization layer
    layer_norm = tf.keras.layers.LayerNormalization(axis=0) # gamma=model.layers[3].weights[3]), beta=model.layers[3].weights[4]
    output = model.layers[3].weights[3]*layer_norm(output) + model.layers[3].weights[4] # gamma*Xnorm+beta

    # Calculate queries, keys, and values
    query_out = tf.transpose(tf.matmul(output, tf.transpose(model.layers[3].weights[5], perm=[1,0,2])), perm=[1,0,2]) + model.layers[3].weights[6]
    key_out = tf.transpose(tf.matmul(output, tf.transpose(model.layers[3].weights[7], perm=[1,0,2])), perm=[1,0,2]) + model.layers[3].weights[8]
    value_out = tf.transpose(tf.matmul(output, tf.transpose(model.layers[3].weights[9], perm=[1,0,2])), perm=[1,0,2]) + model.layers[3].weights[10]

    # Calculate attention
    attention_logits = tf.matmul(tf.transpose(query_out, perm=[1,0,2]), tf.transpose(key_out, perm=[1,2,0])) / 8

    # For each head, apply softmax and gather attention scores
    attention_array = []
    for logit in attention_logits:
        attention_array.append(tf.nn.softmax(logit, axis=0))
    attention_tensor = tf.stack([arr for arr in attention_array])
    
    return attention_tensor
# Input to apply embeddings and calculate query, key, and value tensors in order to look at attention scores
input_ids = val_input_ids[1]
attention_tensor = get_attention_scores(model=model, input_ids=input_ids)

print(input_ids)
print(' '.join(tokenizer.convert_ids_to_tokens(input_ids)))
# For each attention head, look at the self-attention scores

for n_head in range(0,12):
    attention_head = attention_tensor[n_head]
    attention_df = pd.DataFrame(columns=['words','word_idx','attention'])
    for i, word in enumerate(tokenizer.convert_ids_to_tokens(input_ids)):
        data = {'words':tokenizer.convert_ids_to_tokens(input_ids)
               ,'word_idx': i
               ,'attention': np.array(attention_head[i])}
        temp_df = pd.DataFrame(data).reset_index()
        attention_df = attention_df.append(temp_df)

    # Attention heatmap
    plt.figure(figsize=(15,10))
    ax = sns.heatmap(attention_df.pivot(index='word_idx',columns=['words','index']))
# For a single word, look at the self-attention for each head

word_idx = 5
# attention_word = attention_tensor[:,:,word_idx]
attention_word = attention_tensor[:,word_idx,:]
attention_df = pd.DataFrame(columns=['words','head_idx','attention'])
for i, head in enumerate(attention_word):
    data = {'words':tokenizer.convert_ids_to_tokens(input_ids)
           ,'head_idx': i
           ,'attention': np.array(attention_word[i])}
    temp_df = pd.DataFrame(data).reset_index()
    attention_df = attention_df.append(temp_df)

print(' '.join(tokenizer.convert_ids_to_tokens(input_ids)))
print(tokenizer.convert_ids_to_tokens([input_ids[word_idx]]))
attention_df.head()

# Attention heatmap
plt.figure(figsize=(30,5))
ax = sns.heatmap(attention_df.pivot(index='head_idx',columns=['words','index']))
# For a single word, look at the self-attention for each head

word_idx = 5
attention_word = attention_tensor[:,:,word_idx]
# attention_word = attention_tensor[:,word_idx,:]
attention_df = pd.DataFrame(columns=['words','head_idx','attention'])
for i, head in enumerate(attention_word):
    data = {'words':tokenizer.convert_ids_to_tokens(input_ids)
           ,'head_idx': i
           ,'attention': np.array(attention_word[i])}
    temp_df = pd.DataFrame(data).reset_index()
    attention_df = attention_df.append(temp_df)

print(' '.join(tokenizer.convert_ids_to_tokens(input_ids)))
print(tokenizer.convert_ids_to_tokens([input_ids[word_idx]]))
attention_df.head()

# Attention heatmap
plt.figure(figsize=(30,5))
ax = sns.heatmap(attention_df.pivot(index='head_idx',columns=['words','index']))
# For a single word, loop through the self-attention scores for each head

# Get heatmap for words and attention score
word_idx = 5
print(' '.join(tokenizer.convert_ids_to_tokens(input_ids)))
print()
print(f'Target word: {tokenizer.convert_ids_to_tokens(input_ids)[word_idx]}')

for n_head in range(0,12):
    data = {'words':tokenizer.convert_ids_to_tokens(input_ids)
           ,'attention':np.array(attention_tensor[n_head,word_idx,:])
#            ,'attention':np.array(attention_tensor[n_head,:,word_idx])
           ,'col':0}
    attention_df = pd.DataFrame(data)
    attention_df = attention_df.reset_index()
    attention_df
    
    plt.figure(figsize=(30,1))
    ax = sns.heatmap(attention_df.pivot(index='col', columns=['words','index']))
# Input to apply embeddings and calculate query, key, and value tensors in order to look at attention scores
input_text = '''
A space is a virtual place for a group of people to work together. Spaces are named by the people who create them.
'''
input_tokens = tokenizer.tokenize(input_text)
input_ids_unpadded = tokenizer.convert_tokens_to_ids(input_tokens)
input_ids = [0]*128
input_ids[0] = 101
input_ids[1:len(input_ids_unpadded)+1] = input_ids_unpadded
input_ids[len(input_ids_unpadded)+1] = 102
attention_tensor = get_attention_scores(model=model, input_ids=input_ids)
# For a single word, loop through the self-attention scores for each head

# Get heatmap for words and attention score
word_idx = 5
print(' '.join(tokenizer.convert_ids_to_tokens(input_ids)))
print()
print(f'Target word: {tokenizer.convert_ids_to_tokens(input_ids)[word_idx]}')

for n_head in range(0,12):
    data = {'words':tokenizer.convert_ids_to_tokens(input_ids)
           ,'attention':np.array(attention_tensor[n_head,word_idx,:])
#            ,'attention':np.array(attention_tensor[n_head,:,word_idx])
           ,'col':0}
    attention_df = pd.DataFrame(data)
    attention_df = attention_df.reset_index()
    attention_df
    
    plt.figure(figsize=(30,1))
    ax = sns.heatmap(attention_df.pivot(index='col', columns=['words','index']))
# For each attention head, look at the self-attention scores

for n_head in range(0,12):
    attention_head = attention_tensor[n_head]
    attention_df = pd.DataFrame(columns=['words','word_idx','attention'])
    for i, word in enumerate(tokenizer.convert_ids_to_tokens(input_ids[1:len(input_tokens)])):
        data = {'words':tokenizer.convert_ids_to_tokens(input_ids)[1:len(input_tokens)]
               ,'word_idx': i
               ,'attention': np.array(attention_head[i])[1:len(input_tokens)]}
        temp_df = pd.DataFrame(data).reset_index()
        attention_df = attention_df.append(temp_df)

    # Attention heatmap
    plt.figure(figsize=(15,10))
    ax = sns.heatmap(attention_df.pivot(index='word_idx',columns=['words','index']))
