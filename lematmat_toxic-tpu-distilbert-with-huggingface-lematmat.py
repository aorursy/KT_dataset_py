import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from kaggle_datasets import KaggleDatasets
import transformers
from tqdm.notebook import tqdm
from tokenizers import BertWordPieceTokenizer
def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(max_length=maxlen)
    all_ids = []
    
    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])
    
    return np.array(all_ids)
def build_model(transformer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(cls_token)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)
AUTO = tf.data.experimental.AUTOTUNE

# Data access
GCS_DS_PATH = KaggleDatasets().get_gcs_path()

# Configuration
EPOCHS = 2
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
# First load the real tokenizer
tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')

# Save the loaded tokenizer locally
save_path = '/kaggle/working/distilbert_base_uncased/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
tokenizer.save_pretrained(save_path)

# Reload it with the huggingface tokenizers library
fast_tokenizer = BertWordPieceTokenizer('distilbert_base_uncased/vocab.txt', lowercase=True)
fast_tokenizer
!ls -al /kaggle/input/jigsaw-multilingual-toxic-comment-classification
# Engl comments from Wikipedia’s talk page
train1 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
# Civil Comments dataset with a range of additionnal labels.
train2 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")
# comments from Wikipedia talk pages in different non-English languages.                     
valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv') 
test_no_labels = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
train1.iloc[:4]
#test_no_labels : not used in this notebook because only used for Kaggle model evaluation only 
test_no_labels[0:4]
valid.iloc[:4]
print ('valid shape:',valid.shape)
# Evenly split valid data to build new valid ndtest labeled data
valid_with_labels = valid[0:3999]
test_with_labels = valid[4000:7999]
x_train = fast_encode(train1.comment_text.astype(str), fast_tokenizer, maxlen=512)
x_valid = fast_encode(valid_with_labels.comment_text.astype(str), fast_tokenizer, maxlen=512)
x_test = fast_encode(test_with_labels.comment_text.astype(str), fast_tokenizer, maxlen=512)

y_train = train1.toxic.values
y_valid = valid_with_labels.toxic.values
y_test = test_with_labels.toxic.values
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
    .from_tensor_slices((x_test, y_test))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)
   

%%time
with strategy.scope():
    transformer_layer = transformers.TFDistilBertModel.from_pretrained('distilbert-base-multilingual-cased')
    model = build_model(transformer_layer, max_len=512)
model.summary()
!ls /kaggle/working
import os
os.chdir(r'/kaggle/working')
Working_Out = '/kaggle/working'

#Save best weight model
ModelPath = Working_Out + '/weights_best_inception3_pool_over1.hdf5'
checkpointer = ModelCheckpoint(filepath = ModelPath, verbose=1, save_best_only=True)

history = model.fit(
    train_dataset,
    steps_per_epoch=N_STEPS,
    validation_data=valid_dataset,
    epochs=5,
    callbacks=[checkpointer]
)
# Plot the training process 
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy'] 

train_loss = history.history['loss']
val_loss = history.history['val_loss']
 
plt.plot(train_loss, label=['loss'])
plt.plot(val_loss, label=['val_loss'])
plt.title('Loss evolution at each epochs')
plt.legend()
plt.show()

plt.plot(train_acc , label=['accuracy'])
plt.plot(val_acc , label=['val_accuracy'])
plt.title('Accuracy evolution at each epochs')
plt.legend()
plt.show()
print('\n# Evaluate')
result = model.evaluate(test_dataset)
dict(zip(model.metrics_names, result))