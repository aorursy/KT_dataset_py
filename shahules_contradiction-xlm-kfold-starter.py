import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go

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

from sklearn.model_selection import StratifiedKFold,KFold

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




langs = df_train.language.unique()



fig = go.Figure()

fig.add_trace(go.Bar(

    x=langs,

    y=df_train.language.value_counts().values,

    name='train',

    marker_color='indianred'

))

fig.add_trace(go.Bar(

    x=langs,

    y=df_test.language.value_counts().values,

    name='test',

    marker_color='lightsalmon'

))



# Here we modify the tickangle of the xaxis, resulting in rotated labels.

fig.update_layout(barmode='group', xaxis_tickangle=-45,title="language distribution in dataset")

fig.show()




langs = df_train.label.unique()



fig = go.Figure()



fig.add_trace(go.Bar(

    x=langs,

    y=df_train.label.value_counts().values,

    name='test',

    marker_color=[ 'steelblue', 'tan', 'teal']

))



# Here we modify the tickangle of the xaxis, resulting in rotated labels.

fig.update_layout(xaxis_tickangle=-45,title="Target distribution in train dataset")

fig.show()
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

    cls_token = Dropout(0.2)(cls_token)

    cls_token = Dense(32,activation='relu')(cls_token)

    out = Dense(3, activation='softmax')(cls_token)



    # It's time to build and compile the model

    model = Model(inputs=input_ids, outputs=out)

    model.compile(

        Adam(lr=1e-5), 

        loss='sparse_categorical_crossentropy', 

        metrics=['accuracy']

    )

    

    return model

def build_lrfn(lr_start=0.00001, lr_max=0.00003, 

               lr_min=0.000001, lr_rampup_epochs=3, 

               lr_sustain_epochs=0, lr_exp_decay=.6):

    lr_max = lr_max * strategy.num_replicas_in_sync



    def lrfn(epoch):

        if epoch < lr_rampup_epochs:

            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start

        elif epoch < lr_rampup_epochs + lr_sustain_epochs:

            lr = lr_max

        else:

            lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min

        return lr

    

    return lrfn


plt.figure(figsize=(10, 7))



_lrfn = build_lrfn()

plt.plot([i for i in range(10)], [_lrfn(i) for i in range(10)]);
lrfn = build_lrfn()

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
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

        

        val_score.append(train_history.history['val_accuracy'])

        history.append(train_history)



        val_score.append(np.mean(train_history.history['val_accuracy']))

        

        print('predict on test....')

        preds=model.predict(test_dataset,verbose=1)

        

        pred_test+=preds/4

        



        

print("Mean Validation accuracy : ",np.mean(val_score))


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