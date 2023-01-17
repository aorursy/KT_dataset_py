import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import transformers
from transformers import TFAutoModel, AutoTokenizer
from sklearn.model_selection import StratifiedKFold
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
import os
path = "../input/contradictory-my-dear-watson"
os.listdir(path)
df_train = pd.read_csv(os.path.join(path,"train.csv"))
df_test = pd.read_csv(os.path.join(path,"test.csv"))
df_train['origin'] = 'ori'
df_test['origin'] = 'ori'
train_trans = pd.read_csv('../input/contradictorydatasettranslate/train_translate_all.csv')
train_trans['origin'] = 'trans' 
df_train = pd.concat([df_train, train_trans], axis=0).reset_index()
df_train.head()
# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    print('Running on TPU ', tpu.master())
except ValueError:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)
MIXED_PRECISION = False
XLA_ACCELERATE = False

if MIXED_PRECISION:
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    if tpu: policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    else: policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    print('Mixed precision enabled')

if XLA_ACCELERATE:
    tf.config.optimizer.set_jit(True)
    print('Accelerated Linear Algebra enabled')
MODEL = 'jplu/tf-xlm-roberta-base'
EPOCHS = 10
MAX_LEN = 96

# Our batch size will depend on number of replic
BATCH_SIZE= 16 * strategy.num_replicas_in_sync
AUTO = tf.data.experimental.AUTOTUNE
tokenizer = AutoTokenizer.from_pretrained(MODEL)
def lang_embding(lang, trans):
    langc = ['English', 'French', 'Thai', 'Turkish', 'Urdu', 'Russian',
           'Bulgarian', 'German', 'Arabic', 'Chinese', 'Hindi', 'Swahili',
           'Vietnamese', 'Spanish', 'Greek']
    lang_code = ['0000', '0001', '0010', '0011', '0100', '0101', '0110', '0111',
                '1000', '1001', '1010', '1011', '1100', '1101', '1110', '1111']
    lang_code = dict(zip(langc, lang_code))
    trans_code = {'ori':'0', 'trans':'1'}

    enc = lang_code[lang] + trans_code[trans]

    vec = [int(i) for i in enc]
    return vec
lang_embding('English', 'ori')
def quick_encode(df,maxlen=100):
    
    values = df[['premise','hypothesis']].values.tolist()
    lang_emb = [lang_embding(row['language'], row['origin']) for index,row in df.iterrows()]
    tokens=tokenizer.batch_encode_plus(values,max_length=maxlen,pad_to_max_length=True)
    return np.array(tokens['input_ids']), np.array(lang_emb)

x_train, x_lang = quick_encode(df_train)
x_test, x_test_lang = quick_encode(df_test)
y_train = df_train.label.values
def create_dataset(X,y ,val=False, batch_size= BATCH_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices((X,y)).shuffle(len(X))
    if not val:
        dataset = dataset.repeat().batch(batch_size).prefetch(AUTO)
    else:
        dataset = dataset.batch(batch_size).prefetch(AUTO)
    return dataset
test_dataset = (tf.data.Dataset.from_tensor_slices(((x_test, x_test_lang)))).batch(BATCH_SIZE)
        
from tensorflow.keras.layers import Concatenate

def build_model(transformer, max_len):
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
    input_lang = Input(shape=(5,), dtype=tf.float32, name='language_tag')
    sequence_output = transformer(input_ids)[0]
    cls_token = sequence_output[:,0,:]
    cls_token = Concatenate()([cls_token, input_lang])
    cls_token = Dense(32, activation='relu')(cls_token)
    out = Dense(3, activation='softmax')(cls_token)
    
    model = Model(inputs = [input_ids,input_lang], outputs = out)
    model.compile(Adam(lr=1e-5),
                  loss = 'sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model
pred_test = np.zeros((df_test.shape[0],3))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
val_score = []
history = []
for fold, (train_ind, valid_ind) in enumerate(skf.split(x_train, y_train)):
        print("fold",fold+1)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        train_data = create_dataset((x_train[train_ind], x_lang[train_ind]), y_train[train_ind],val=False)
        valid_data = create_dataset((x_train[valid_ind], x_lang[valid_ind]), y_train[valid_ind],val=True)
    
        Checkpoint=tf.keras.callbacks.ModelCheckpoint(f"roberta_base.h5", monitor='val_loss', verbose=0, save_best_only=True,
        save_weights_only=True, mode='min')
        
        with strategy.scope():
            transformer_layer = TFAutoModel.from_pretrained(MODEL)
            model = build_model(transformer_layer, max_len=MAX_LEN)
        n_steps = len(train_ind)//BATCH_SIZE
        print("training model {} ".format(fold+1))

        train_history = model.fit(
        train_data,
        steps_per_epoch=n_steps,
        validation_data=valid_data,
        epochs=EPOCHS,callbacks=[Checkpoint],verbose=1)
        
        print("Loading model...")
        model.load_weights(f"roberta_base.h5")
        print("fold {} validation accuracy {}".format(fold+1,np.mean(train_history.history['val_accuracy'])))
        print("fold {} validation loss {}".format(fold+1,np.mean(train_history.history['val_loss'])))
        
        history.append(train_history)

        val_score.append(np.mean(train_history.history['val_accuracy']))
        
        print('predict on test....')
        preds=model.predict((x_test, x_test_lang),verbose=1)
        pred_test+=preds/5
plt.figure(figsize=(15,10))

for i,hist in enumerate(history):

    plt.subplot(3,2,i+1)
    plt.plot(np.arange(EPOCHS),hist.history['accuracy'],label='train accu')
    plt.plot(np.arange(EPOCHS),hist.history['val_accuracy'],label='validation acc')
    plt.gca().title.set_text(f'Fold {i+1} accuracy curve')
    plt.legend()
submission = pd.read_csv(os.path.join(path, 'sample_submission.csv'))
submission.prediction = np.argmax(pred_test, axis=1)
submission.head()
submission.to_csv('submission.csv', index=False)
submission['prob0'] = pred_test[:,0]
submission['prob1'] = pred_test[:,1]
submission['prob2'] = pred_test[:,2]
submission.to_csv('submission_prob.csv', index=False)
