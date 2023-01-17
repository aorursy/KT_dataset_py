# Generic

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, warnings,json

warnings.filterwarnings('ignore')



# Plot

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go



# TensorFlow & Transformers

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam, Adamax



# Transformer Model

from transformers import BertTokenizer, TFBertModel

from transformers import TFAutoModel, AutoTokenizer



#SK Learn

from sklearn.model_selection import train_test_split



# Garbage Collector

import gc
train_data = pd.read_csv("../input/contradictory-my-dear-watson/train.csv")

test_data = pd.read_csv("../input/contradictory-my-dear-watson/test.csv")

sample_sub = pd.read_csv("../input/contradictory-my-dear-watson/sample_submission.csv")
def Utilize_TPUs():  

    """

    Initialize training strategy using TPU if available else using default strategy for CPU and  single GPU

    

    After the TPU is initialized, you can also use manual device placement to place the computation on a single TPU device.



    """

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

    

strategy=Utilize_TPUs()
the_chosen_one="jplu/tf-xlm-roberta-base"

max_len =80

batch_size = 16 * strategy.num_replicas_in_sync



AUTO = tf.data.experimental.AUTOTUNE

epochs = 30

n_steps = len(train_data) // batch_size
def model_baseline(strategy,transformer):

    with strategy.scope():

        transformer_encoder = TFAutoModel.from_pretrained(transformer)

        input_layer = Input(shape=(max_len,), dtype=tf.int32, name="input_layer")

        sequence_output = transformer_encoder(input_layer)[0]

        cls_token = sequence_output[:, 0, :]

        output_layer = Dense(3, activation='softmax')(cls_token)

        model = Model(inputs=input_layer, outputs=output_layer)

        model.compile(

            Adamax(lr=1e-5), 

            loss='sparse_categorical_crossentropy', 

            metrics=['accuracy']

        )

        return model

    

model=model_baseline(strategy,the_chosen_one)
model.summary()
train_data.head()
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
tokenizer = AutoTokenizer.from_pretrained(the_chosen_one)
train_df = train_data[['premise', 'hypothesis']].values.tolist()
test_df = test_data[['premise', 'hypothesis']].values.tolist()
train_encoded=tokenizer.batch_encode_plus(train_df,pad_to_max_length=True,max_length=max_len)
test_encoded=tokenizer.batch_encode_plus(test_df,pad_to_max_length=True,max_length=max_len)
x_train, x_valid, y_train, y_valid = train_test_split(train_encoded['input_ids'], train_data.label.values, test_size=0.1)



x_test = test_encoded['input_ids']
train_dataset = (tf.data.Dataset.from_tensor_slices((x_train, y_train)).repeat().shuffle(2048).batch(batch_size).prefetch(AUTO))



valid_dataset = (tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(batch_size).cache().prefetch(AUTO))



test_dataset = (tf.data.Dataset.from_tensor_slices(x_test).batch(batch_size))
model.fit(train_dataset,steps_per_epoch=n_steps,validation_data=valid_dataset,epochs=epochs)
predictions = model.predict(test_dataset, verbose=1)

sample_sub['prediction'] = predictions.argmax(axis=1)
sample_sub.to_csv("submission.csv",index= False)