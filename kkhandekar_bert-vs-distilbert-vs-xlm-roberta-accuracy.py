# Generic

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, warnings

warnings.filterwarnings('ignore')



# TensorFlow

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam



# Transformer Model

from transformers import BertTokenizer, TFBertModel               #BERT

from transformers import DistilBertTokenizer, TFDistilBertModel    #DistilBERT

from transformers import XLMRobertaTokenizer, TFXLMRobertaModel    #XLM-RoBERTa





# SKLearn Library

from sklearn.model_selection import train_test_split



# Garbage Collector

import gc



# Tabulate

from tabulate import tabulate



os.environ["WANDB_API_KEY"] = "0"
# Initialize TPU



def Init_TPU():  



    try:

        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()

        tf.config.experimental_connect_to_cluster(resolver)

        tf.tpu.experimental.initialize_tpu_system(resolver)

        strategy = tf.distribute.experimental.TPUStrategy(resolver)

        REPLICAS = strategy.num_replicas_in_sync

        print("Connected to TPU Successfully:\n TPUs Initialised with Replicas:",REPLICAS)

        

        return strategy

    

    except ValueError:

        

        print("Connection to TPU Falied")

        print("Using default strategy for CPU and single GPU")

        strategy = tf.distribute.get_strategy()

        

        return strategy

    

strategy=Init_TPU()
# Define Dataset Path

path = '../input/contradictory-my-dear-watson/'
# Load Training Data

train_url = os.path.join(path,'train.csv')

train_data = pd.read_csv(train_url, header='infer')
# Garbage Collection

gc.collect()
# Transformer Model Name

Bert_model = 'bert-base-multilingual-cased'

distilBert_model = 'distilbert-base-multilingual-cased'

xlmRoberta_model = 'jplu/tf-xlm-roberta-base'



# Define Tokenizer for each

Bert_toknzr = BertTokenizer.from_pretrained(Bert_model)

distilBert_toknzr = DistilBertTokenizer.from_pretrained(distilBert_model)

xlmRoberta_toknzr = XLMRobertaTokenizer.from_pretrained(xlmRoberta_model)
# Checking the output of tokenizer

sentence = 'Elementary, My Dear Watson!'



print("BERT Model Tokenizer Output:",Bert_toknzr.convert_tokens_to_ids(list(Bert_toknzr.tokenize(sentence))))

print("DistilBERT Model Tokenizer Output:",distilBert_toknzr.convert_tokens_to_ids(list(distilBert_toknzr.tokenize(sentence))))

print("XLM-RoBERTa Model Tokenizer Output:",xlmRoberta_toknzr.convert_tokens_to_ids(list(xlmRoberta_toknzr.tokenize(sentence))))
# Create seperate list from Train & Test Dataframes with only Premise & Hypothesis

train = train_data[['premise','hypothesis']].values.tolist()
# Define Max Length

max_len = 80   # << change if you wish



# Encode the training & test data - BERT

train_encode_Bert = Bert_toknzr.batch_encode_plus(train, pad_to_max_length=True, max_length=max_len)



# Encode the training & test data - DistilBERT

train_encode_DistilBert = distilBert_toknzr.batch_encode_plus(train, pad_to_max_length=True, max_length=max_len)



# Encode the training & test data - XLM-RoBERTa

train_encode_XlmRoberta = xlmRoberta_toknzr.batch_encode_plus(train, pad_to_max_length=True, max_length=max_len)

# Split the Training Data into Training (90%) & Validation (10%)



test_size = 0.1  # << change if you wish



# BERT

x_tr_bert, x_val_bert, y_tr_bert, y_val_bert = train_test_split(train_encode_Bert['input_ids'], train_data.label.values, test_size=test_size)



# DistilBERT

x_tr_Dbert, x_val_Dbert, y_tr_Dbert, y_val_Dbert = train_test_split(train_encode_DistilBert['input_ids'], train_data.label.values, test_size=test_size)



# XLM-RoBERTa

x_tr_XR, x_val_XR, y_tr_XR, y_val_XR = train_test_split(train_encode_XlmRoberta['input_ids'], train_data.label.values, test_size=test_size)

#garbage collect

gc.collect()
# Loading Data Into TensorFlow Dataset

AUTO = tf.data.experimental.AUTOTUNE

batch_size = 16 * strategy.num_replicas_in_sync



#BERT

tr_ds_bert = (tf.data.Dataset.from_tensor_slices((x_tr_bert, y_tr_bert)).repeat().shuffle(2048).batch(batch_size).prefetch(AUTO))

val_ds_bert = (tf.data.Dataset.from_tensor_slices((x_val_bert, y_val_bert)).batch(batch_size).prefetch(AUTO))



#DistilBERT

tr_ds_Dbert = (tf.data.Dataset.from_tensor_slices((x_tr_Dbert, y_tr_Dbert)).repeat().shuffle(2048).batch(batch_size).prefetch(AUTO))

val_ds_Dbert = (tf.data.Dataset.from_tensor_slices((x_val_Dbert, y_val_Dbert)).batch(batch_size).prefetch(AUTO))



#XLM-RoBERTa

tr_ds_XR = (tf.data.Dataset.from_tensor_slices((x_tr_XR, y_tr_XR)).repeat().shuffle(2048).batch(batch_size).prefetch(AUTO))

val_ds_XR = (tf.data.Dataset.from_tensor_slices((x_val_XR, y_val_XR)).batch(batch_size).prefetch(AUTO))
# Garbage Collection

gc.collect()
def build_model(strategy):

    with strategy.scope():

        bert_encoder = TFBertModel.from_pretrained(Bert_model)  #BERT

        DistilBert_encoder = TFDistilBertModel.from_pretrained(distilBert_model)  #DistilBERT

        XLMRoberta_encoder = TFXLMRobertaModel.from_pretrained(xlmRoberta_model)  #XLM-RoBERTa

        

        input_layer = Input(shape=(max_len,), dtype=tf.int32, name="input_layer")

        

        sequence_output_bert = bert_encoder(input_layer)[0]

        sequence_output_Dbert = DistilBert_encoder(input_layer)[0]

        sequence_output_XR = XLMRoberta_encoder(input_layer)[0]

        

        cls_token_bert = sequence_output_bert[:, 0, :]

        cls_token_Dbert = sequence_output_Dbert[:, 0, :]

        cls_token_XR = sequence_output_XR[:, 0, :]

                

        output_layer_bert = Dense(3, activation='softmax')(cls_token_bert)

        output_layer_Dbert = Dense(3, activation='softmax')(cls_token_Dbert)

        output_layer_XR = Dense(3, activation='softmax')(cls_token_XR)

        

        model1 = Model(inputs=input_layer, outputs=output_layer_bert)

        model2 = Model(inputs=input_layer, outputs=output_layer_Dbert)

        model3 = Model(inputs=input_layer, outputs=output_layer_XR)

        

        

        model1.compile(

            Adam(lr=1e-5), 

            loss='sparse_categorical_crossentropy', 

            metrics=['accuracy']

        )

        

        model2.compile(

            Adam(lr=1e-5), 

            loss='sparse_categorical_crossentropy', 

            metrics=['accuracy']

        )

            

        model3.compile(

            Adam(lr=1e-5), 

            loss='sparse_categorical_crossentropy', 

            metrics=['accuracy']

        )

        

        

        return model1, model2, model3

    



# Applying the build model function

model_bert, model_Dbert, model_XLMRoberta = build_model(strategy)
# Train the Model



epochs = 30  # < change if you wish

n_steps = len(train_data) // batch_size 
# Train BERT Model



model_bert.fit(tr_ds_bert, 

          steps_per_epoch = n_steps, 

          validation_data = val_ds_bert,

          epochs = epochs)
# Garbage Collection

gc.collect()
# Train DistilBERT Model



model_Dbert.fit(tr_ds_Dbert, 

          steps_per_epoch = n_steps, 

          validation_data = val_ds_Dbert,

          epochs = epochs)
# Garbage Collection

gc.collect()
# Train XLM-RobERTa Model



model_XLMRoberta.fit(tr_ds_XR, 

          steps_per_epoch = n_steps, 

          validation_data = val_ds_XR,

          epochs = epochs)
# Garbage Collection

gc.collect()
# Evaluate BERT

res_bert = model_bert.evaluate(val_ds_bert, verbose=0)



# Evaluate DistilBERT

res_Dbert = model_Dbert.evaluate(val_ds_Dbert, verbose=0)



# Evaluate XLM-RoBERTa

res_XlmRoberta = model_XLMRoberta.evaluate(val_ds_XR, verbose=0)



#Tabulate Data

tab_data = [["BERT","30","Adam","128","1e-5",'{:.2%}'.format(res_bert[1])],

            ["DistilBERT","30","Adam","128","1e-5",'{:.2%}'.format(res_Dbert[1])],

            ["XLM-RoBERTa","30","Adam","128","1e-5",'{:.2%}'.format(res_XlmRoberta[1])]]   

    

print(tabulate(tab_data, headers=['Models','Epochs','Optimizer','Batch Size','Learning Rate','Accuracy'], tablefmt='pretty'))