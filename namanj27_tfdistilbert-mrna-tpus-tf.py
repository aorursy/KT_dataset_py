# pip install --upgrade transformers
import warnings

warnings.filterwarnings("ignore")



import os

import random

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import copy

import tensorflow.keras.layers as L



import tensorflow as tf

import tensorflow_addons as tfa

from tensorflow.keras import regularizers



from sklearn.model_selection import train_test_split, KFold, RepeatedStratifiedKFold, StratifiedKFold

from transformers import BertTokenizer, TFBertModel, BertConfig, BertModel, TFDistilBertModel, DistilBertConfig

def seed_everything(seed = 34):

    os.environ['PYTHONHASHSEED']=str(seed)

    tf.random.set_seed(seed)

    np.random.seed(seed)

    random.seed(seed)

    

seed_everything()
train = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)

test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)

sub = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')



#target columns

target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
def MCRMSE(y_true, y_pred):

    columnwise_mse = tf.reduce_mean(tf.square(y_true-y_pred), axis=1)

    return tf.reduce_mean(tf.sqrt(columnwise_mse), axis=1)
AUTO = tf.data.experimental.AUTOTUNE



try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() 



print("REPLICAS: ", strategy.num_replicas_in_sync)
config = DistilBertConfig() 
config
config.vocab_size = 10

config.dim = 128

config.hidden_dim = 128

config.max_position_embeddings = 128

config.n_layers = 2

config.n_heads = 128

# config.sinusoidal_pos_embds = True

def build_model(transformer, seq_len=107, pred_len=68, dropout=0.5, embed_dim=100, hidden_dim=128):

    ids = L.Input(shape=(seq_len,3),  dtype=tf.int32, name="input_word_ids")

    flat = L.Flatten()(ids)

    sequence_output = transformer(flat)[0]

    truncated = sequence_output[:,:pred_len, :]

    

    out = L.Dense(5, activation='linear')(truncated)

    model = tf.keras.Model(inputs=ids, outputs=out)



    model.compile(tf.keras.optimizers.Adam(), loss=MCRMSE)

    

    return model
with strategy.scope():

    transformer_layer = (

        TFDistilBertModel(config=config)

    )

    model = build_model(transformer_layer)

model.summary()
tokentoint = {x:i for i, x in enumerate('().ACGUBEHIMSX')}

def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):

    return np.transpose(

        np.array(

            df[cols]

            .applymap(lambda seq: [tokentoint[x] for x in seq])

            .values

            .tolist()

        ),

        (0, 2, 1)

    )

preprocess_inputs(train).shape
train_inputs = preprocess_inputs(train[train['signal_to_noise'] >= 1])

train_labels = np.array(train[train['signal_to_noise'] >= 1][target_cols].values.tolist()).transpose(0, 2, 1)
print(train_inputs.shape)

print(train_labels.shape)
public_df = test[test['seq_length']==107].copy()

private_df = test[test['seq_length']==130].copy()



public_inputs = preprocess_inputs(public_df)

private_inputs = preprocess_inputs(private_df)
#basic training configuration

FOLDS = 5

EPOCHS = 100

REPEATS = 1

BATCH_SIZE = 64

VERBOSE = 2

SEED = 34
# pip install livelossplot
from sklearn.model_selection import KFold

lr_callback = tf.keras.callbacks.ReduceLROnPlateau()

bert_histories = []

bert_private_preds = np.zeros((private_df.shape[0], 130, 5))

bert_public_preds = np.zeros((public_df.shape[0], 107, 5))



kf = KFold(n_splits=FOLDS,shuffle=True,random_state=42)





with strategy.scope():

   

    

    for fold, (train_index, val_index) in enumerate(kf.split(train_inputs, train_labels)):

        print(f"FOLD {fold}")

        

        model = build_model(transformer_layer)

        

        history = model.fit(

            train_inputs[train_index,:,:], train_labels[train_index,:,:], 

            batch_size=BATCH_SIZE,

            epochs=EPOCHS,

            validation_split=0.1,

                callbacks=[

                            lr_callback,

                            tf.keras.callbacks.ModelCheckpoint('model'+str(fold)+'.h5',save_weights_only=True,save_best_only=True)

                            ])



        model_short = build_model(transformer_layer,seq_len=107, pred_len=107)

        model_long = build_model(transformer_layer,seq_len=130, pred_len=130)



        model_short.load_weights('model'+str(fold)+'.h5')

        model_long.load_weights('model'+str(fold)+'.h5')

        

        bert_histories.append(history)



        bert_public_pred = model_short.predict(public_inputs) / FOLDS



        bert_private_pred = model_long.predict(private_inputs) / FOLDS



        bert_public_preds += bert_public_pred

        bert_private_preds += bert_private_pred

        

preds_bert = []



for df, preds in [(public_df, bert_public_preds), (private_df, bert_private_preds)]:

    for i, uid in enumerate(df.id):

        single_pred = preds[i]



        single_df = pd.DataFrame(single_pred, columns=target_cols)

        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]



        preds_bert.append(single_df)



preds_bert_df = pd.concat(preds_bert)

preds_bert_df.head()
submission = sub[['id_seqpos']].merge(preds_bert_df, on=['id_seqpos'])

submission.to_csv('submission.csv', index=False)