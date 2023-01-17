import os, time, gc



import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint



import transformers

from transformers import TFAutoModel, AutoTokenizer

from tqdm.notebook import tqdm



from sklearn.model_selection import GroupKFold



osj = os.path.join; osdir = os.listdir
%%time

# Detect hardware, return appropriate distribution strategy

def tpu_init():

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



    return strategy
def build_model(transformer, max_len=512):

    """

    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras

    """

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    sequence_output = transformer(input_word_ids)[0]

    

    cls_token = sequence_output[:, 0, :]

    

    out = Dense(1, activation='sigmoid')(cls_token)

        

    model = Model(inputs=input_word_ids, outputs=out)

    model.compile(Adam(lr=1e-5), loss='binary_crossentropy',

                                      metrics=['accuracy', tf.keras.metrics.AUC()])

    

    return model
def tpu_bs(BATCH_SIZE_MULTIPLIER):

    strategy = tpu_init()

    bs = BATCH_SIZE_MULTIPLIER * strategy.num_replicas_in_sync

    

    return strategy, bs
debug = False  # True # False

n_rows = 100_000_000 if not debug else 16*8*2



n_splits = 4

# load models of folds 0,1,4

folds_to_train = [0,1]

folds_to_save = [0,1]

seed_num = 1

seed_model = 2020



AUTO = tf.data.experimental.AUTOTUNE



# Configuration

epochs = 2



BATCH_SIZE_MULTIPLIER = 24  # 32  # 24  # 16

MAX_LEN = 192

MODEL = 'jplu/tf-xlm-roberta-base'



out_path = './'

assert os.path.exists(out_path)



datetime_str = time.strftime("%d_%m_time_%H_%M", time.localtime())



t0 = time.time()



def keras_seed_everything(seed):

    # import tensorflow as tf

    # import os

    tf.random.set_seed(seed)

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'



keras_seed_everything(seed_model)



strategy, bs = tpu_bs(BATCH_SIZE_MULTIPLIER)



tokenizer = AutoTokenizer.from_pretrained(MODEL)
train_trans = pd.read_csv(

    '../input/jig-ds-noneng-raw-begin-data-maxl-192/train_trans_raw_begin_enc_rows_1326360_maxl_192.csv', nrows=n_rows)



train_trans = train_trans.sample(n=60_000 if not debug else 60, random_state=seed_num)



train_trans = train_trans[train_trans['df_name']=='trans']



val_8k = pd.read_csv(

    '../input/jig-ds-noneng-raw-begin-data-maxl-192/val_8k_raw_begin_enc_nrows_63812_maxl_192.csv', nrows=n_rows)



test = pd.read_csv(

    '../input/jig-ds-noneng-raw-begin-data-maxl-192/test_raw_begin_enc_nrows_63812_maxl_192.csv', nrows=n_rows)



train2_trans = pd.read_csv('../input/jig-train2-trans-enc-raw-similar-to-test-preds/train2_trans_similar_test_nrows_838658.csv')

train2_trans = train2_trans.sample(n=176_000 if not debug else 70, random_state=seed_num)

train2_trans.drop('val_fold', axis=1, inplace=True)

train2_trans['toxic'] = (train2_trans['toxic']>0.5).astype(int)



train2_trans['df_name'] = 'tr2_trans'

val_8k['df_name']='val_8k'

test['df_name'] = 'test'



enc_cols = [col for col in test.columns if col.startswith('enc_')]

cols_select = ['id','lang','df_name','toxic'] + enc_cols

train_trans = pd.concat([train_trans[cols_select], train2_trans[cols_select]])

train_trans.head(2)
def print_df_stats(df, df_name='df', text_col = 'comment_text'):

    print("="*30)

    print(f"\n{df_name}.shape:", df.shape)

    if 'toxic' in df.columns:

        print(f"\n{df_name}['toxic'].value_counts:\n", df['toxic'].value_counts())

    if 'lang' in df.columns:

        print(f"\n{df_name}['lang'].value_counts:\n", df['lang'].value_counts())

    if 'target' in df.columns:

        print(f"\n{df_name}['target'].value_counts:\n", df['target'].value_counts())



#print_df_stats(osub, 'osub');

print_df_stats(train_trans, 'train_trans'); print_df_stats(val_8k,'val_8k'); print_df_stats(test,'test', text_col = 'content') 
enc_cols = [col for col in test.columns if col.startswith('enc_')]



train_trans['target']=0

val_8k['target']=0



test['target']=1



sel_cols = ['id','lang','df_name','target']+enc_cols

train = pd.concat([train_trans[sel_cols], val_8k[sel_cols], test[sel_cols]])



train = train.sample(frac=1, replace=False, random_state=seed_num)

print_df_stats(train, 'train')

del train_trans, test, val_8k; _=gc.collect()
%%time

x_train = train[enc_cols].values.astype('int')[:n_rows]

y_train = train['target'].values.astype('int')[:n_rows]



train.drop(enc_cols, axis=1, inplace=True)

train = train[:n_rows]

#x_test = test[enc_cols].values.astype('int')



print("x_train.shape", x_train.shape)



train['lang'] = train['lang'].astype('category')

train['target'] = train['target'].astype('int')

train.dtypes



t_preproc = time.time()

print(f"Finished preprocessing in {(t_preproc-t0)/60:.2f} min.")
def tpu_bs(BATCH_SIZE_MULTIPLIER):

    strategy = tpu_init()

    bs = BATCH_SIZE_MULTIPLIER * strategy.num_replicas_in_sync

    

    return strategy, bs



strategy, bs = tpu_bs(BATCH_SIZE_MULTIPLIER)
id_to_group = {k: v for (k,v) in zip(train['id'].unique(), range(len(train['id'].unique())))}

groups = train['id'].map(id_to_group).values

del id_to_group; _=gc.collect()

train['id'].value_counts()
train['preds'] = 999.99

hist1_ls, hist2_ls = [], []

t0 = time.time()



gkf = GroupKFold(n_splits=n_splits)



for fold, (train_idx, valid_idx) in enumerate(gkf.split(x_train, y_train, groups)):

    

    if not (fold in folds_to_train):

        continue

        

    t1 = time.time()

    

    train_dataset = ( tf.data.Dataset.from_tensor_slices((x_train[train_idx], y_train[train_idx]))

                                            .repeat().shuffle(2048).batch(bs, drop_remainder=True).prefetch(AUTO)

                    )

    valid_dataset = (  tf.data.Dataset.from_tensor_slices((x_train[valid_idx],

                                                           y_train[valid_idx]))

                            .batch(bs).prefetch(AUTO) )

    

    n_steps = int( max(1, x_train[train_idx].shape[0] // bs) )

    

    print(f"1: Num train samples {len(x_train[train_idx])}, num valid samples = {len(x_train[valid_idx])}")

    

    with strategy.scope():

        transformer_layer = TFAutoModel.from_pretrained(MODEL)

        model = build_model(transformer_layer, max_len=MAX_LEN)

        

    train_history_1 = model.fit(train_dataset,

                            steps_per_epoch=n_steps,

                            validation_data=valid_dataset,

                            epochs=epochs)



    

    t2 = time.time()

    print(f"\nTrained fold {fold}, in {(t2-t1)/60:.2f} min.")

   

    

    hist1_df = pd.DataFrame(train_history_1.history)

    hist1_ls.append(hist1_df)

    

    # save model

            

    new_model_filename = f'model_fl{fold}_auc_{hist1_df.iloc[-1,-1]:.5f}_eps_{epochs}.h5'

    checkpoint_path_fn = os.path.join(out_path, new_model_filename)

    if fold in folds_to_save:

        model.save_weights(checkpoint_path_fn)

        print(f"Saved fold {fold} weights")

    

    t3 = time.time()

    

    train['preds'].iloc[valid_idx]  =  model.predict(valid_dataset, verbose=1).squeeze()

    train[['id','lang','preds']].to_csv(f'train_preds_after_fold_{fold}.csv', index=False)

    t4 = time.time()

    print(f"Predicted fold {fold} valid_idx - English in {(t4-t3)/60:.2f} min.")

    

    t5 = time.time()

    print(f"Predicted fold {fold} valid_idx - English in {(t5-t4)/60:.2f} min.")

    

    print(f"\nFOLD {fold}, TOTAL TIME: {(time.time()-t1)/60:.2f} min. \n =================== END of fold {fold} ====================\n")

    

    if fold != folds_to_train[-1]:

        #print(model.summary())

        del model;  _=gc.collect()



        strategy, bs = tpu_bs(BATCH_SIZE_MULTIPLIER)

        tf.keras.backend.clear_session()
hist_df = pd.DataFrame( np.concatenate([h.values for h in hist1_ls]), columns = hist1_ls[0].columns ,

                      )

hist_df.to_csv('hist_df.csv', index=False)

hist_df
# leave train with predictions and remove test

train_valid = train.loc[(train['preds']<=1)&(train['target']==0), 

                            ['id', 'lang','df_name','target','preds']]

print_df_stats(train_valid, df_name='train_valid')

train_valid.head(10)
train_valid['df_name'].value_counts()
train[train['df_name']=='test'].sort_values(by='preds').head(10)
train_valid['preds'].hist(bins=100)
thresholds = [1e-7, 1e-5, 1e-4, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7]



for prob_thresh in thresholds:

    train_valid_thresh = train_valid[train_valid['preds']>prob_thresh]

    #train_valid_thresh.to_csv(f"train_valid_thresh_{str(prob_thresh).replace('.','_')}_nrows_{train_valid_thresh.shape[0]}.csv")

    print(f"thresh={prob_thresh}: \tnum samples with preds > thresh : = {train_valid_thresh.shape[0]:,d}")

    #      .describe()

    #print(train_valid_thresh['lang'].train_validue_counts())

    print(train_valid_thresh['df_name'].value_counts())

    #print("\nLanguages train_validue_counts in 'train' train_valid_thresh:")

    #print(train_valid_thresh.loc[train_valid_thresh['df_name']=='train', 'lang'].train_validue_counts())

    print("="*30)
# save train_valid

train_valid[['id','lang','df_name','target','preds']].to_csv(f'train_valid_folds_{folds_to_train}.csv', index=False)
!du -ha ./