import os,time,tqdm,random,gc

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

import plotly.express as px

init_notebook_mode(connected=True)

import plotly.graph_objs as go

from sklearn.utils import shuffle

from IPython.display import clear_output



# !pip uninstall -y transformers

# !pip install transformers

!pip install nlp



import transformers

import tokenizers

import nlp

import tensorflow as tf





os.environ["WANDB_API_KEY"] = "0"



def seed_all(seed=2001):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

except ValueError:

    strategy = tf.distribute.get_strategy() # for CPU and single GPU

    print('Number of replicas:', strategy.num_replicas_in_sync)

  

    

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
model_name = 'bert-base-multilingual-cased'

max_len = 50
original = pd.read_csv('../input/contradictory-my-dear-watson/train.csv')

mnli = nlp.load_dataset(path='glue', name='mnli')



clear_output(wait=True)
mnli_train = pd.DataFrame(mnli['train'])

mnli_valid1 = pd.DataFrame(mnli['validation_matched'])

mnli_valid2 = pd.DataFrame(mnli['validation_mismatched'])



mnli = pd.concat([mnli_train,mnli_valid1,mnli_valid2])
mnli = mnli[['premise','hypothesis','label']]

mnli = mnli.rename(columns = {0 : 'premise', 1: 'hypothesis',2: 'label' })
original = original[['premise','hypothesis','label']].sample(len(original)//(8*strategy.num_replicas_in_sync)*8*strategy.num_replicas_in_sync)

len(original)
train = mnli

train
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
%%time

train_encoded = tokenizer.batch_encode_plus(train[['premise','hypothesis']].values.tolist(),pad_to_max_length=True,max_length=max_len,return_tensors='tf')

valid_encoded = tokenizer.batch_encode_plus(original[['premise','hypothesis']].values.tolist(),pad_to_max_length=True,max_length=max_len,return_tensors='tf')
def build_model(l):

    bert_encoder = transformers.TFBertModel.from_pretrained(model_name)

    

    input_words_ids = tf.keras.layers.Input(shape=(max_len),dtype=tf.int32,name='input_ids')

    input_mask = tf.keras.layers.Input(shape=(max_len,),dtype=tf.int32,name='attention_mask')

    input_type_ids = tf.keras.layers.Input(shape=(max_len,),dtype=tf.int32,name='token_type_ids')

    

    embedding = bert_encoder([input_words_ids,input_mask,input_type_ids])[0]



    

    output = tf.keras.layers.Dense(3,activation='softmax')(embedding[:,0,:])

    

    

    model = tf.keras.models.Model(inputs=[input_words_ids,input_mask,input_type_ids],outputs=output)

    



    model.compile(

        optimizer = tf.keras.optimizers.Adam(lr=l),

        loss='sparse_categorical_crossentropy',

        metrics=['accuracy']

    )

    

    return model



with strategy.scope():

    model = build_model(l=1e-5)

    model.summary()
hist = model.fit(dict(train_encoded),train.label.values,epochs=10,batch_size=128*strategy.num_replicas_in_sync,verbose=1,validation_data=(dict(valid_encoded),original.label.values),validation_batch_size=8*strategy.num_replicas_in_sync)
test = pd.read_csv('../input/contradictory-my-dear-watson/test.csv')



test_encoded = tokenizer.batch_encode_plus(test[['premise','hypothesis']].values.tolist(),pad_to_max_length=True,max_length=max_len,return_tensors='tf')
preds = [np.argmax(i) for i in model.predict(dict(test_encoded))]



submission = test.id.copy().to_frame()

submission['prediction'] = preds

submission.to_csv('submission.csv',index=False)
hist_df = pd.DataFrame(hist.history)

hist_df['epoch'] = np.arange(1,len(hist_df)+1)
hist_df
py.offline.init_notebook_mode()

train_acc =go.Scatter(x=hist_df['epoch'],y=hist_df['accuracy'],mode = "lines+markers",name='train_acc')

val_acc =go.Scatter(x=hist_df['epoch'],y=hist_df['val_accuracy'],mode = "lines+markers",name='valid_acc')



data = [train_acc, val_acc]

layout = dict(title = 'Accuracy',

              xaxis= dict(title= 'epoch',ticklen= 1,zeroline= False)

             )



fig = dict(data = data, layout = layout)

iplot(fig)
train_acc =go.Scatter(x=hist_df['epoch'],y=hist_df['loss'],mode = "lines+markers",name='loss')

val_acc =go.Scatter(x=hist_df['epoch'],y=hist_df['val_loss'],mode = "lines+markers",name='valid_loss')



data = [train_acc, val_acc]

layout = dict(title = 'Loss',

              xaxis= dict(title= 'epoch',ticklen= 1,zeroline= False)

             )



fig = dict(data = data, layout = layout)

iplot(fig)