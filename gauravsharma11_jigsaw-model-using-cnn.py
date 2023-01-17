import os
import pandas as pd
import numpy as np
import tensorflow as tf
DATA_PATH = "/kaggle/input/jigsaw-multilingual-toxic-comment-classification/"
os.listdir(DATA_PATH)
TEST_PATH = DATA_PATH + "test.csv"
VAL_PATH = DATA_PATH + "validation.csv"
TRAIN_PATH = DATA_PATH + "jigsaw-toxic-comment-train.csv"
val_data = pd.read_csv(VAL_PATH)
test_data = pd.read_csv(TEST_PATH)
train_data = pd.read_csv(TRAIN_PATH)

val = val_data
train = train_data
def fast_encode(texts, tokenizer, chunk_size=240, maxlen=512):
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(max_length=maxlen)
    all_ids = []
    
    for i in range(0, len(texts), chunk_size):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])
    
    return np.array(all_ids)
import transformers
from tokenizers import BertWordPieceTokenizer

tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')

save_path = '/kaggle/working/distilbert_base_uncased/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
tokenizer.save_pretrained(save_path)

fast_tokenizer = BertWordPieceTokenizer('distilbert_base_uncased/vocab.txt', 
                                        lowercase=True)
from kaggle_datasets import KaggleDatasets

AUTO = tf.data.experimental.AUTOTUNE

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.experimental.TPUStrategy(tpu)

GCS_DS_PATH = KaggleDatasets().get_gcs_path('jigsaw-multilingual-toxic-comment-classification')
x_train = fast_encode(train.comment_text.astype(str), 
                      fast_tokenizer, maxlen=512)
x_valid = fast_encode(val_data.comment_text.astype(str).values, 
                      fast_tokenizer, maxlen=512)
x_test = fast_encode(test_data.content.astype(str).values, 
                     fast_tokenizer, maxlen=512)

y_valid = val.toxic.values
y_train = train.toxic.values
BATCH_SIZE = 32 * strategy.num_replicas_in_sync

train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_train, y_train))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_valid, y_valid))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(x_test)
    .batch(BATCH_SIZE)
)
from tensorflow.keras.layers import Dense, Input, Dropout, Embedding, concatenate
from tensorflow.keras.layers import LSTM, Conv1D, SpatialDropout1D, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

def cnn_model(transformer, max_len):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    
    embed = transformer.weights[0].numpy()
    embedding = Embedding(np.shape(embed)[0], np.shape(embed)[1],
                          input_length=max_len, weights=[embed],
                          trainable=False)(input_word_ids)
    
    embedding = SpatialDropout1D(0.3)(embedding)
    conv_1 = Conv1D(64, 2)(embedding)
    conv_1 = SpatialDropout1D(0.4)(conv_1)
    conv_2 = Conv1D(128, 3)(embedding)
    conv_3 = Conv1D(256, 4)(embedding)
    conv_3 = SpatialDropout1D(0.4)(conv_3)
    conv_4 = Conv1D(64, 5)(embedding)
    conv_5 = Conv1D(64, 5)(conv_1)
    conv_6 = Conv1D(64, 5)(conv_2)
    conv_7 = Conv1D(64, 5)(conv_3)
    conv_8 = Conv1D(64, 5)(conv_4)
    
    maxpool_1 = GlobalAveragePooling1D()(conv_5)
    maxpool_2 = GlobalAveragePooling1D()(conv_6)
    maxpool_3 = GlobalAveragePooling1D()(conv_7)
    maxpool_4 = GlobalAveragePooling1D()(conv_8)
    conc = concatenate([maxpool_1, maxpool_2, maxpool_3, maxpool_4], axis=1)

    conc = Dense(64, activation='relu')(conc)
    conc = Dense(32, activation='relu')(conc)
    conc = Dense(1, activation='sigmoid')(conc)
    
    model = Model(inputs=input_word_ids, outputs=conc)
    
    model.compile(Adam(lr=0.0001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    return model
with strategy.scope():
    transformer_layer = transformers.TFDistilBertModel.from_pretrained('distilbert-base-multilingual-cased')
    model = cnn_model(transformer_layer, max_len=512)

#model.summary()
N_STEPS = x_train.shape[0] // BATCH_SIZE
EPOCHS = 15

train_history = model.fit(
    train_dataset,
    steps_per_epoch=N_STEPS,
    validation_data=valid_dataset,
    epochs=EPOCHS
)
train_history_rep = model.fit(
    valid_dataset.repeat(),
    steps_per_epoch=N_STEPS,
    epochs=5
)
sub = pd.read_csv(DATA_PATH + 'sample_submission.csv')
sub['toxic'] = model.predict(test_dataset, verbose=1)
sub.to_csv('submission_cnn.csv', index=False)