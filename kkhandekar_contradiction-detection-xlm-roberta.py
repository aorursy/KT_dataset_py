# Generic

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, warnings,json

warnings.filterwarnings('ignore')



# Plot

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go



# TensorFlow

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam



# Transformer Model

from transformers import XLMRobertaTokenizer, TFXLMRobertaModel



# SKLearn Library

from sklearn.model_selection import train_test_split



# Garbage Collector

import gc



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



# Sample Submission

sample_sub_url = os.path.join(path,'sample_submission.csv')

sample_sub = pd.read_csv(sample_sub_url, header='infer')



# Load Test Data

test_url = os.path.join(path,'test.csv')

test_data = pd.read_csv(test_url, header='infer')
print("Total Records: ", train_data.shape[0])
#Inspect

train_data.head()
# Records per Label

print("Records per Label: \n", train_data.groupby('label').size())
# Records per Languages

print("Records per Language: \n", train_data.groupby('language').size())
# Pie Chart

df = pd.DataFrame({"count": train_data.language.value_counts() })

fig = px.pie(df, values='count', names=df.index, title='Language Count %',

             labels={'index':'lang'}, color_discrete_sequence=px.colors.diverging.Earth)

fig.update_traces(textinfo='percent')

fig.show()
# Bar Plot - Label Count per Language 

fig, ax = plt.subplots(figsize=(20,10))

train_data.groupby(['language','label']).count()['premise'].unstack().plot(ax=ax,kind='bar', cmap='cividis')

plt.grid(color='gray',linestyle='--',linewidth=0.2)

ax.set_facecolor('#d8dcd6')

plt.title("Label Count per Language", fontsize='18')

plt.xticks(rotation=45)
# Garbage Collection

gc.collect()
# Transformer Model Name

transformer_model = 'jplu/tf-xlm-roberta-large'



# Define Tokenizer

tokenizer = XLMRobertaTokenizer.from_pretrained(transformer_model)
# Checking the output of tokenizer

tokenizer.convert_tokens_to_ids(list(tokenizer.tokenize("Elementary, My Dear Watson!")))
# Create seperate list from Train & Test Dataframes with only Premise & Hypothesis

train = train_data[['premise','hypothesis']].values.tolist()

test = test_data[['premise','hypothesis']].values.tolist()
# Define Max Length

max_len = 80   # << change if you wish



# Encode the training & test data 

train_encode = tokenizer.batch_encode_plus(train, pad_to_max_length=True, max_length=max_len)

test_encode = tokenizer.batch_encode_plus(test, pad_to_max_length=True, max_length=max_len)
# Split the Training Data into Training (90%) & Validation (10%)



test_size = 0.1  # << change if you wish

x_train, x_val, y_train, y_val = train_test_split(train_encode['input_ids'], train_data.label.values, test_size=test_size)





# Split Test Data

x_test = test_encode['input_ids']
#garbage collect

gc.collect()
# Loading Data Into TensorFlow Dataset

AUTO = tf.data.experimental.AUTOTUNE

batch_size = 16 * strategy.num_replicas_in_sync



train_ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train)).repeat().shuffle(2048).batch(batch_size).prefetch(AUTO))

val_ds = (tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size).prefetch(AUTO))



test_ds = (tf.data.Dataset.from_tensor_slices(x_test).batch(batch_size))
# Garbage Collection

gc.collect()
def build_model(strategy,transformer):

    with strategy.scope():

        transformer_encoder = TFXLMRobertaModel.from_pretrained(transformer)  #Pretrained BERT Transformer Model

        

        input_layer = Input(shape=(max_len,), dtype=tf.int32, name="input_layer")

        

        sequence_output = transformer_encoder(input_layer)[0]

        

        cls_token = sequence_output[:, 0, :]

        

        output_layer = Dense(3, activation='softmax')(cls_token)

        

        model = Model(inputs=input_layer, outputs=output_layer)

        

        model.compile(

            Adam(lr=1e-5), 

            loss='sparse_categorical_crossentropy', 

            metrics=['accuracy']

        )

        

        return model

    



# Applying the build model function

model = build_model(strategy,transformer_model)
# Model Summary

model.summary()
# Train the Model



epochs = 30  # < change if you wish

n_steps = len(train_data) // batch_size 



model.fit(train_ds, 

          steps_per_epoch = n_steps, 

          validation_data = val_ds,

          epochs = epochs)
# Garbage Collection

gc.collect()
prediction = model.predict(test_ds, verbose=0)

sample_sub['prediction'] = prediction.argmax(axis=1)
sample_sub.to_csv("submission.csv", index=False)