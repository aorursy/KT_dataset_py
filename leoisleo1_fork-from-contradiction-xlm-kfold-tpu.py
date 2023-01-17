import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from kaggle_datasets import KaggleDatasets

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input,Dropout

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
path="../input/contradictory-my-dear-watson"

os.listdir(path)

df_train=pd.read_csv(os.path.join(path,"train.csv"))

df_test=pd.read_csv(os.path.join(path,"test.csv"))
print('there are {} rows and {} columns in the train'.format(df_train.shape[0],df_train.shape[1]))

print('there are {} rows and {} columns in the test'.format(df_test.shape[0],df_test.shape[1]))
df_train.head(3)


plt.figure(figsize=(10,5))

plt.subplot(1,2,1)

x=df_train.language.value_counts()

sns.barplot(x.index,x.values)

plt.gca().set_xticklabels(x.index,rotation='45')



plt.subplot(1,2,2)

x=df_test.language.value_counts()

sns.barplot(x.index,x.values)

plt.gca().set_xticklabels(x.index,rotation='45')

plt.show()


plt.figure(figsize=(7,5))

x=df_train.label.value_counts()

sns.barplot(x.index,x.values)

plt.gca().set_xticklabels(x.index,rotation='45')
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
MODEL = 'jplu/tf-xlm-roberta-large'

EPOCHS = 10

MAX_LEN = 96



# Our batch size will depend on number of replic

BATCH_SIZE= 16 * strategy.num_replicas_in_sync

AUTO = tf.data.experimental.AUTOTUNE

tokenizer = AutoTokenizer.from_pretrained(MODEL)
def quick_encode(df,maxlen=100):

    

    values = df[['premise','hypothesis']].values.tolist()

    tokens=tokenizer.batch_encode_plus(values,max_length=maxlen,pad_to_max_length=True)

    

    return np.array(tokens['input_ids'])



x_train = quick_encode(df_train)

x_test = quick_encode(df_test)

y_train = df_train.label.values

    




def create_dist_dataset(X, y,val,batch_size= BATCH_SIZE):

    

    

    dataset = tf.data.Dataset.from_tensor_slices((X,y)).shuffle(len(X))

          

    if not val:

        dataset = dataset.repeat().batch(batch_size).prefetch(AUTO)

    else:

        dataset = dataset.batch(batch_size).prefetch(AUTO)



    

    

    return dataset







test_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_test))

    .batch(BATCH_SIZE)

)

def build_model(transformer,max_len):

    

    input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")

    sequence_output = transformer(input_ids)[0]

    cls_token = sequence_output[:, 0, :]

    cls_token = Dropout(0.4)(cls_token)

    cls_token = Dense(32,activation='relu')(cls_token)

    cls_token = Dropout(0.4)(cls_token)

    out = Dense(3, activation='softmax')(cls_token)



    # It's time to build and compile the model

    model = Model(inputs=input_ids, outputs=out)

    model.compile(

        Adam(lr=1e-5), 

        loss='sparse_categorical_crossentropy', 

        metrics=['accuracy']

    )

    

    return model

pred_test=np.zeros((df_test.shape[0],3))

skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=777)

val_score=[]

history=[]





for fold,(train_ind,valid_ind) in enumerate(skf.split(x_train,y_train)):

    

    if fold < 4:

    

        print("fold",fold+1)

        

       

        tf.tpu.experimental.initialize_tpu_system(tpu)

        

        train_data = create_dist_dataset(x_train[train_ind],y_train[train_ind],val=False)

        valid_data = create_dist_dataset(x_train[valid_ind],y_train[valid_ind],val=True)

    

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

        preds=model.predict(test_dataset,verbose=1)



        pred_test+=preds/4

        


plt.figure(figsize=(15,10))



for i,hist in enumerate(history):



    plt.subplot(2,2,i+1)

    plt.plot(np.arange(EPOCHS),hist.history['accuracy'],label='train accu')

    plt.plot(np.arange(EPOCHS),hist.history['val_accuracy'],label='validation acc')

    plt.gca().title.set_text(f'Fold {i+1} accuracy curve')

    plt.legend()





    


plt.figure(figsize=(15,10))



for i,hist in enumerate(history):



    plt.subplot(2,2,i+1)

    plt.plot(np.arange(EPOCHS),hist.history['loss'],label='train loss')

    plt.plot(np.arange(EPOCHS),hist.history['val_loss'],label='validation loss')

    plt.gca().title.set_text(f'Fold {i+1} loss curve')

    plt.legend()



submission = pd.read_csv(os.path.join(path,'sample_submission.csv'))

submission['prediction'] = np.argmax(pred_test,axis=1)

submission.head()
submission.to_csv('submission.csv',index=False)