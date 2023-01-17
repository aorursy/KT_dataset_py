import transformers

from tqdm.notebook import tqdm

import numpy as np

import pandas as pd

import tensorflow as tf

from transformers import BertTokenizer, TFBertModel, TFBertForSequenceClassification, BertConfig

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dense, Flatten

from transformers import create_optimizer

import time
test_df = pd.read_csv('../input/preprocessed-tweets/precessed_test.csv')

train_df = pd.read_csv('../input/preprocessed-tweets/processed_train.csv')
train_df.drop_duplicates(subset='tweet', keep='first', inplace=True)
pd.set_option('display.max_colwidth',200)
train_df['word_count'] = train_df.tweet.apply(lambda x:len(x.split()))
train_df.head()
train_df.drop(4799, 0, inplace=True)
train_df.word_count.plot.hist()
from transformers import BertTokenizer

from keras.preprocessing.sequence import pad_sequences



bert_model_name = 'bert-base-uncased'



tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=True)

MAX_LEN = 30



def tokenize_sentences(sentences, tokenizer, max_seq_len = 128):

    tokenized_sentences = []



    for sentence in tqdm(sentences):

        tokenized_sentence = tokenizer.encode(

                            sentence,                  # Sentence to encode.

                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'

                            max_length = max_seq_len,  # Truncate all sentences.

                    )

        

        tokenized_sentences.append(tokenized_sentence)



    return tokenized_sentences



def create_attention_masks(tokenized_and_padded_sentences):

    attention_masks = []



    for sentence in tokenized_and_padded_sentences:

        att_mask = [int(token_id > 0) for token_id in sentence]

        attention_masks.append(att_mask)



    return np.asarray(attention_masks)



input_ids = tokenize_sentences(train_df['tweet'], tokenizer, MAX_LEN)

input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")

attention_masks = create_attention_masks(input_ids)
labels =  train_df['label'].values



train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=0, test_size=0.1)

train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels, random_state=0, test_size=0.1)



train_size = len(train_inputs)

validation_size = len(validation_inputs)
label_cols = ['label']
class BertClassifier(tf.keras.Model):    

    

    def __init__(self, bert: TFBertModel, num_classes: int):

        

        super().__init__()

        

        self.bert = bert

        

        self.classifier = Dense(num_classes, activation='sigmoid')

        

    def call(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):

        

        outputs = self.bert(input_ids,

                            attention_mask=attention_mask,

                            token_type_ids=token_type_ids,

                            position_ids=position_ids,

                            head_mask=head_mask)

        

        cls_output = outputs[1]

        

        cls_output = self.classifier(cls_output)

                

        return cls_output
def create_dataset(data_tuple, batch_size, train=True):

    

    dataset = tf.data.Dataset.from_tensor_slices(data_tuple)

    

    dataset = dataset.repeat(1)

    

    if train:

        

        dataset = dataset.shuffle(buffer_size=10000)

    

    dataset = dataset.batch(batch_size)

    

    if train:

        

        dataset = dataset.prefetch(1)

        

    if not train:

        

        dataset = dataset.cache()

    

    return dataset
def train_step(model, token_ids, masks, labels):

    

    labels = tf.dtypes.cast(labels, tf.float32)

    

    with tf.GradientTape() as tape:

        

        predictions = model(token_ids, attention_mask=masks)

        

        loss = loss_object(labels, predictions)

    

    

    gradients = tape.gradient(loss, model.trainable_variables)

    

    optimizer.apply_gradients(zip(gradients, model.trainable_variables), name = 'gradients')

    

    train_loss(loss)



    for i, auc in enumerate(train_auc_metrics):

        

        auc.update_state(labels[:,i], predictions[:,i])

        

def validation_step(model, token_ids, masks, labels):

    

    labels = tf.dtypes.cast(labels, tf.float32)



    predictions = model(token_ids, attention_mask=masks, training=False)

    

    v_loss = loss_object(labels, predictions)



    validation_loss(v_loss)

    

    for i, auc in enumerate(validation_auc_metrics):

        

        auc.update_state(labels[:,i], predictions[:,i])
BATCH_SIZE = 32



TEST_BATCH_SIZE = 64



NR_EPOCHS = 1



MAX_LEN = 30 # try diffrent lengths



threshold = 0.5
#seeds = [0 ,31 ,97,193,1001,83,42,456,21,237] # for ensembles



seeds = [0]



for seed in range(len(seeds)):

    

    print('=' * 50, f"CV {seed+1}", '=' * 50)

    

    model = BertClassifier(TFBertModel.from_pretrained(bert_model_name), len(label_cols))

    

    labels =  train_df[label_cols].values

    

    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=seed, test_size = 0.2)



    train_masks, validation_masks = train_test_split(attention_masks, random_state=seed, test_size=0.2)



    train_size = len(train_inputs)



    validation_size = len(validation_inputs)





    train_dataset = create_dataset((train_inputs, train_masks, train_labels), batch_size=BATCH_SIZE,train=True)



    validation_dataset = create_dataset((validation_inputs, validation_masks, validation_labels), batch_size=BATCH_SIZE,train=False)

    

    

    steps_per_epoch = train_size // (BATCH_SIZE)



    #  Loss Function

    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)



    train_loss = tf.keras.metrics.Mean(name='train_loss')



    validation_loss = tf.keras.metrics.Mean(name='val_loss')



    #  Optimizer (with 1-cycle-policy)

    warmup_steps = steps_per_epoch // 3



    total_steps = steps_per_epoch * NR_EPOCHS - warmup_steps



    optimizer = create_optimizer(init_lr=2e-5, num_train_steps=total_steps, num_warmup_steps=warmup_steps)



    # Gradients

    

    gradients = 0

    

    #  Metrics

    train_auc_metrics = [tf.keras.metrics.AUC() for i in range(len(label_cols))]



    validation_auc_metrics = [tf.keras.metrics.AUC() for i in range(len(label_cols))]





    for epoch in range(NR_EPOCHS):



        print('=' * 50, f"EPOCH {epoch+1}", '=' * 50)



        start = time.time()





        for batch_no, (token_ids, masks, labels) in enumerate(tqdm(train_dataset)):



            train_step(model, token_ids, masks, labels)



            if batch_no % 100 == 0:



                    print(f'\nTrain Step: {batch_no}, Loss: {train_loss.result()}')



                    for i, label_name in enumerate(label_cols):



                        print(f"{label_name} roc_auc {train_auc_metrics[i].result()}")



                        train_auc_metrics[i].reset_states()



        for batch_no, (token_ids, masks, labels) in enumerate(tqdm(validation_dataset)):



            validation_step(model, token_ids, masks, labels)



        print(f'\nEpoch {epoch+1}, Validation Loss: {validation_loss.result()}, Time: {time.time()-start}\n')



        for i, label_name in enumerate(label_cols):



            print(f"{label_name} roc_auc {validation_auc_metrics[i].result()}")



            validation_auc_metrics[i].reset_states()



        print('\n')

        

#     probs = generate_class_probablities(model,test_dataset,test_steps)

        

#     submission.loc[:, label_cols] = probs

        

#     submission.to_csv('probs'+str(seed)+'.csv')

        

#     class_probs += probs