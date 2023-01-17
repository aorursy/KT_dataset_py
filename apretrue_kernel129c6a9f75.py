import tensorflow as tf
from transformers import *
import pandas as pd
from tqdm import tqdm
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
dfdict = pd.read_csv('/kaggle/input/nlp-homework/data_with_labels.csv',header=0,index_col=0).dropna(axis=0,how='any').reset_index(drop=True).to_dict("list")
input_text = dfdict['inputs']
key_text = dfdict['keywords']
output_text = dfdict['outputs']
task_class_labels = dfdict['task_class_label_2']
pd.read_csv('/kaggle/input/nlp-homework/data_with_labels.csv',header=0,index_col=0).dropna(axis=0,how='any').reset_index(drop=True).head()
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
MAX_LEN = 32

def tokenize_sentences(sentences, tokenizer, max_seq_len = 32):
    tokenized_sentences = []

    for sentence in tqdm(sentences):
        tokenized_sentence = tokenizer.encode(
                            sentence,                  # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = max_seq_len,  # Truncate all sentences.
                            truncation=True
                    )
        
        tokenized_sentences.append(tokenized_sentence)

    return tokenized_sentences

def create_attention_masks(tokenized_and_padded_sentences):
    attention_masks = []

    for sentence in tokenized_and_padded_sentences:
        att_mask = [int(token_id > 0) for token_id in sentence]
        attention_masks.append(att_mask)

    return np.asarray(attention_masks)

input_ids = tokenize_sentences(input_text, tokenizer, MAX_LEN)
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
attention_masks = create_attention_masks(input_ids)
from sklearn.model_selection import train_test_split

labels =  [x-1 for x in task_class_labels]

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=0, test_size=0.3)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels, random_state=0, test_size=0.3)

train_size = len(train_inputs)
validation_size = len(validation_inputs)
BATCH_SIZE = 32
NR_EPOCHS = 5

def create_dataset(data_tuple, epochs=1, batch_size=32, buffer_size=10000, train=True):
    dataset = tf.data.Dataset.from_tensor_slices(data_tuple)
    if train:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size)
    if train:
        dataset = dataset.prefetch(1)
    
    return dataset

train_dataset = create_dataset((train_inputs, train_masks, train_labels), epochs=NR_EPOCHS, batch_size=BATCH_SIZE)
validation_dataset = create_dataset((validation_inputs, validation_masks, validation_labels), epochs=NR_EPOCHS, batch_size=BATCH_SIZE)

#%%
from tensorflow.keras.layers import Dense, Flatten

class BertClassifier(tf.keras.Model):    
    def __init__(self, bert: TFBertModel, num_classes: int):
        super().__init__()
        self.bert = bert
        self.dropout = tf.keras.layers.Dropout(rate=0.2)
        self.classifier = Dense(num_classes, activation='softmax')
        
    @tf.function
    def call(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        outputs = self.bert(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)
        cls_output = outputs[1]
        cls_output = self.dropout(cls_output)
        cls_output = self.classifier(cls_output)
                
        return cls_output

classes_num = max(labels)+1
model = BertClassifier(TFBertModel.from_pretrained('bert-base-chinese'), classes_num)
import time

steps_per_epoch = train_size // BATCH_SIZE
validation_steps = validation_size // BATCH_SIZE

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

valid_loss = tf.keras.metrics.Mean(name='valid_loss')
valid_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

warmup_steps = steps_per_epoch // 3
total_steps = steps_per_epoch * NR_EPOCHS - warmup_steps
optimizer= create_optimizer(init_lr=2e-5, num_train_steps=total_steps, num_warmup_steps=warmup_steps)

#%%
@tf.function
def train_step(model, token_ids, masks, labels):
    labels = tf.dtypes.cast(labels, tf.float32)

    with tf.GradientTape() as tape:
        predictions = model(token_ids, attention_mask=masks)
        loss = loss_object(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss.update_state(loss)
    train_metric.update_state(labels, predictions)
        
@tf.function
def validation_step(model, token_ids, masks, labels):
    labels = tf.dtypes.cast(labels, tf.float32)

    predictions = model(token_ids, attention_mask=masks, training=False)
    v_loss = loss_object(labels, predictions)

    valid_loss.update_state(v_loss)
    valid_metric.update_state(labels, predictions)
                                              
def train(model, train_dataset, val_dataset, train_steps_per_epoch, val_steps_per_epoch, epochs):
    for epoch in range(epochs):
        tf.print('=' * 50, f"EPOCH {epoch}", '=' * 50)

        start = time.time()

        for i, (token_ids, masks, labels) in enumerate(tqdm(train_dataset, total=train_steps_per_epoch)):
            train_step(model, token_ids, masks, labels)
        
        for i, (token_ids, masks, labels) in enumerate(tqdm(val_dataset, total=val_steps_per_epoch)):
            validation_step(model, token_ids, masks, labels)

        print(f'\nEpoch {epoch+1}, Validation Loss: {valid_loss.result()}, Time: {time.time()-start}\n')

        logs = 'Epoch={},Loss:{},Accuracy:{},Valid Loss:{},Valid Accuracy:{}'
        if epoch%1 ==0:
            tf.print(tf.strings.format(logs,
            (epoch,train_loss.result(),train_metric.result(),valid_loss.result(),valid_metric.result())))
            tf.print("")
        train_loss.reset_states()
        valid_loss.reset_states()
        train_metric.reset_states()
        valid_metric.reset_states()

        
train(model, train_dataset, validation_dataset, train_steps_per_epoch=steps_per_epoch, val_steps_per_epoch=validation_steps, epochs=NR_EPOCHS)

for i in range(10):
    print(validation_labels[i],np.argmax(model(validation_inputs[i:i+1,:], attention_mask=validation_masks[i:i+1,:])))
for i in range(10):
    print(train_labels[i],np.argmax(model(train_inputs[i:i+1,:], attention_mask=train_masks[i:i+1,:])))