import os,gc , warnings, transformers

import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt

from tokenizers import BertWordPieceTokenizer

import tensorflow as tf

from tensorflow.keras.models import Model

from tensorflow.keras import optimizers, metrics, losses

from tensorflow.keras.layers import Dense, Input, Dropout, GlobalAveragePooling1D

from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, classification_report



def seed_everything(seed=0):

    np.random.seed(seed)

    tf.random.set_seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'



seed = 0

seed_everything(seed)

warnings.filterwarnings("ignore")
# Auxiliary functions

def plot_metrics(history, metric_list):

    fig, axes = plt.subplots(len(metric_list), 1, sharex='col', figsize=(20, 18))

    axes = axes.flatten()

    

    for index, metric in enumerate(metric_list):

        axes[index].plot(history[metric], label='Train %s' % metric)

        axes[index].plot(history['val_%s' % metric], label='Validation %s' % metric)

        axes[index].legend(loc='best', fontsize=16)

        axes[index].set_title(metric)



    plt.xlabel('Epochs', fontsize=16)

    sns.despine()

    plt.show()

    

def plot_aur_curve(y_train, train_pred, y_valid, valid_pred):

    fpr_train, tpr_train, _ = roc_curve(y_train, train_pred)

    roc_auc_train = auc(fpr_train, tpr_train)

    fpr_valid, tpr_valid, _ = roc_curve(y_valid, valid_pred)

    roc_auc_valid = auc(fpr_valid, tpr_valid)



    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    plt.title('Receiver Operating Characteristic')

    plt.plot(fpr_train, tpr_train, color='blue', label='Train AUC = %0.2f' % roc_auc_train)

    plt.plot(fpr_valid, tpr_valid, color='purple', label='ValidationAUC = %0.2f' % roc_auc_valid)

    plt.legend(loc = 'lower right')

    plt.plot([0, 1], [0, 1],'r--')

    plt.xlim([0, 1])

    plt.ylim([0, 1])

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.show()

    

def plot_confusion_matrix(y_train, train_pred, y_valid, valid_pred, labels=[0, 1]):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    train_cnf_matrix = confusion_matrix(y_train, train_pred)

    validation_cnf_matrix = confusion_matrix(y_valid, valid_pred)



    train_cnf_matrix_norm = train_cnf_matrix.astype('float') / train_cnf_matrix.sum(axis=1)[:, np.newaxis]

    validation_cnf_matrix_norm = validation_cnf_matrix.astype('float') / validation_cnf_matrix.sum(axis=1)[:, np.newaxis]



    train_df_cm = pd.DataFrame(train_cnf_matrix_norm, index=labels, columns=labels)

    validation_df_cm = pd.DataFrame(validation_cnf_matrix_norm, index=labels, columns=labels)



    sns.heatmap(train_df_cm, annot=True, fmt='.2f', cmap="Blues",ax=ax1).set_title('Train')

    sns.heatmap(validation_df_cm, annot=True, fmt='.2f', cmap=sns.cubehelix_palette(8),ax=ax2).set_title('Validation')

    plt.show()
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
train_toxic = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv", 

                          usecols=['id', 'comment_text', 'toxic'])

train_bias = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv", 

                         usecols=['id', 'comment_text', 'toxic'])

valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv', 

                    usecols=['id', 'comment_text', 'toxic'])

test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv', 

                   usecols=['id', 'content'])



print('Jigsaw toxic comment samples %d' % len(train_toxic))

display(train_toxic.head())

print('Jigsaw unintended bias samples %d' % len(train_bias))

display(train_bias.head())
MAX_LEN = 512

BATCH_SIZE = 64 * strategy.num_replicas_in_sync

EPOCHS = 12

LEARNING_RATE = 1e-5 # * strategy.num_replicas_in_sync

ES_PATIENCE = 3



base_model_name = 'distilbert-base-multilingual-cased'

model_path = 'model.h5'

vocab_path = '/kaggle/working'
tokenizer = transformers.DistilBertTokenizer.from_pretrained(base_model_name)

tokenizer.save_pretrained(vocab_path)



tokenizer = BertWordPieceTokenizer(vocab_path + '/vocab.txt', lowercase=False)

tokenizer.enable_truncation(max_length=MAX_LEN)

tokenizer.enable_padding(max_length=MAX_LEN)
x_train = [x.ids for x in tokenizer.encode_batch(train_toxic['comment_text'].apply(lambda x : x).tolist())]

x_valid = [x.ids for x in tokenizer.encode_batch(valid['comment_text'].apply(lambda x : x).tolist())]

x_test = [x.ids for x in tokenizer.encode_batch(test['content'].apply(lambda x : x).tolist())]



y_train = train_toxic['toxic'].values

y_valid = valid['toxic'].values
AUTO = tf.data.experimental.AUTOTUNE



def get_training_dataset():

    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    dataset = dataset.repeat()

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    dataset = dataset.prefetch(AUTO)

    return dataset



def get_validation_dataset():

    dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.cache()

    dataset = dataset.prefetch(AUTO)

    return dataset



def get_test_dataset():

    dataset = tf.data.Dataset.from_tensor_slices(x_test)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO)

    return dataset
del train_bias

gc.collect()



LR_START = 1e-7

LR_MIN = 1e-6

LR_MAX = LEARNING_RATE

LR_RAMPUP_EPOCHS = 2

LR_SUSTAIN_EPOCHS = 0

LR_EXP_DECAY = .6



def lrfn(epoch):

    if epoch < LR_RAMPUP_EPOCHS:

        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

        lr = LR_MAX

    else:

        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

    return lr

    

rng = [i for i in range(EPOCHS)]

y = [lrfn(x) for x in rng]



fig, ax = plt.subplots(figsize=(20, 8))

plt.plot(rng, y)

print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
def model_fn():

    base_model = transformers.TFDistilBertModel.from_pretrained(base_model_name)

    input_word_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_word_ids')

    sequence_output = base_model(input_word_ids)[0]

    

    x = GlobalAveragePooling1D()(sequence_output)

    x = Dropout(0.25)(x)

    output = Dense(1, activation='sigmoid', name='output')(x)

    

    model = Model(inputs=input_word_ids, outputs=output)

    model.compile(optimizers.Adam(lr=LEARNING_RATE), 

                  loss=losses.BinaryCrossentropy(), 

                  metrics=['accuracy', metrics.AUC()])

    

    return model



with strategy.scope():

    model = model_fn()

    

model.summary()
STEPS_PER_EPOCH = len(x_train) // BATCH_SIZE



es = EarlyStopping(monitor='val_loss', mode='min', patience=ES_PATIENCE, 

                   restore_best_weights=True, verbose=1)

lr_callback = LearningRateScheduler(lrfn, verbose=1)





history = model.fit(x=get_training_dataset(),

                    steps_per_epoch=STEPS_PER_EPOCH,

                    validation_data=get_validation_dataset(),

                    callbacks=[es, lr_callback],

                    epochs=EPOCHS, 

                    verbose=1).history
sns.set(style="whitegrid")

plot_metrics(history, metric_list=['loss', 'accuracy', 'auc'])
train_pred = model.predict(get_training_dataset(), steps=STEPS_PER_EPOCH)

valid_pred = model.predict(get_validation_dataset())

train_toxic = train_toxic[:len(train_pred)]

train_toxic['pred'] = train_pred

valid['pred'] = valid_pred



print('Train set ROC AUC %.4f' % roc_auc_score(train_toxic['toxic'], train_toxic['pred']))

print(classification_report(train_toxic['toxic'],  np.round(train_toxic['pred'])))

print('Validation set ROC AUC %.4f' % roc_auc_score(valid['toxic'], valid['pred']))

print(classification_report(valid['toxic'],  np.round(valid['pred'])))
plot_aur_curve(train_toxic['toxic'], train_toxic['pred'], valid['toxic'], valid['pred'])
plot_confusion_matrix(train_toxic['toxic'], np.round(train_toxic['pred']), 

                      valid['toxic'], np.round(valid['pred']))
print('Train set')

display(train_toxic[['comment_text', 'toxic', 'pred']].head(10))

print('Validation set')

display(valid[['comment_text', 'toxic', 'pred']].head(10))
Y_test = model.predict(get_test_dataset())

submission = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')

submission['toxic'] = Y_test

submission.to_csv('submission.csv', index=False)

submission.head(10)