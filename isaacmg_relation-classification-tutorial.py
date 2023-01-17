import os

import pickle

import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

from kaggle_datasets import KaggleDatasets

import transformers

from transformers import TFAutoModel, AutoTokenizer

from tqdm.notebook import tqdm

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
!mkdir model_dir

#!gsutil cp gs://coronaviruspublicdata/re_final_best/new/config.json model_dir/config.json

#!gsutil cp gs://coronaviruspublicdata/re_final_best/new/sigmoid.pickle model_dir/sigmoid.pickle

#!gsutil cp gs://coronaviruspublicdata/re_final_best/new/tf_model.h5 model_dir/tf_model.h5

# Snapshot weights below uncomment if you want to use

!gsutil cp gs://coronaviruspublicdata/re_snapshot/4_13_2020/config.json model_dir/config.json

!gsutil cp gs://coronaviruspublicdata/re_snapshot/4_13_2020/sigmoid.pickle model_dir/sigmoid.pickle

!gsutil cp gs://coronaviruspublicdata/re_snapshot/4_13_2020/tf_model.h5 model_dir/tf_model.h5
def build_model(transformer, max_len=256):

    """

    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras

    """

    input_ids = Input(shape=(max_len, ), dtype=tf.int32)



    x = transformer(input_ids)[0]

    x = x[:, 0, :]

    x = Dense(1, activation='sigmoid', name='sigmoid')(x)



    # BUILD AND COMPILE MODEL

    model = Model(inputs=input_ids, outputs=x)

    model.compile(

        loss='binary_crossentropy', 

        metrics=['accuracy'], 

        optimizer=Adam(lr=1e-5)

    )

    return model
def save_model(model, transformer_dir='transformer'):

    """

    Special function to load a keras model that uses a transformer layer

    """

    transformer = model.layers[1]

    touch_dir(transformer_dir)

    transformer.save_pretrained(transformer_dir)

    sigmoid = model.get_layer('sigmoid').get_weights()

    pickle.dump(sigmoid, open('sigmoid.pickle', 'wb'))



def load_model(pickle_path, transformer_dir='transformer', max_len=512):

    """

    Special function to load a keras model that uses a transformer layer

    """

    transformer = TFAutoModel.from_pretrained(transformer_dir)

    model = build_model(transformer, max_len=max_len)

    sigmoid = pickle.load(open(pickle_path, 'rb'))

    model.get_layer('sigmoid').set_weights(sigmoid)

    

    return model
model = load_model("model_dir/sigmoid.pickle", "model_dir")
def regular_encode(texts, tokenizer, maxlen=512):

    enc_di = tokenizer.batch_encode_plus(

        texts, 

        return_attention_masks=False, 

        return_token_type_ids=False,

        pad_to_max_length=True,

        max_length=maxlen

    )

    

    return np.array(enc_di['input_ids'])
tokenizer = AutoTokenizer.from_pretrained("allenai/biomed_roberta_base")
sent0 = "Briefly, sections were transferred to microscope slides treated with poly-L-lysine, fixed with precooled acetone for 10 min, air-dried for 15 min, and then blocked with 3% BSA overnight at 4 • C"

sent2 = "Mild or classic dengue is treated with antipyretic agents such as acetaminophen, bed rest, and fluid replacement (usually administered orally and only rarely parenterally"

encode_text = regular_encode([sent0, sent2], tokenizer)
preds = model.predict(encode_text)

print("Original sentence was: " + sent0 + " prediction was " + str(preds[0]))

print("Real label is 0")
print("Original sentence was: " + sent2 + " prediction was " + str(preds[1]))

print("Real label is 1")
sent0 = "An alternative approach is to seal the two components with a hot-melt adhesive joint with ethylene vinyl acetate, or with UV cure epoxy"

sent1 = "The acetone extract of Bupleurum scorzonerifolium could inhibit the proliferation of A549 human lung cancer cells in a dose-dependent manner via causing cell cycle arrest in the G2/M phase, increasing microtubule stabilization, suppressing telomerase activity, activating ERK 1/2 and caspase-3/9 in A549 cells [77] [78] [79] "

encode_text = regular_encode([sent0, sent1], tokenizer)

preds = model.predict(encode_text)
print("Original sentence was: " + sent0 + " prediction was " + str(preds[0]))

print("Real label is 0")
print("Original sentence was: " + sent1 + " prediction was " + str(preds[1]))

print("Real label is 0")
sent0 = "Moreover, arginine uptake was increased after TGEV-infected cells were treated with AG1478."

sent2 = "A 1993 parallel-group study found evidence of suppressed short-term lower-leg growth, as measured by knemometry, with BUD (200 μg BID) or intramuscular methylprednisolone acetate (60 mg QD) when compared to terfenadine tablets (60 mg QD) in 44 children (aged 6-15 years) with AR"

encode_text = regular_encode([sent0, sent2], tokenizer)

preds = model.predict(encode_text)
print("Original sentence was: " + sent0 + " prediction was " + str(preds[0]))

print("Real label is 0")
print("Original sentence was: " + sent2 + " prediction was " + str(preds[0]))

print("Real label is 1")