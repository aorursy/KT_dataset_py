import os

os.environ["WANDB_API_KEY"] = "0"

import numpy as np

import pandas as pd

from pathlib import Path

import gc

import random



import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score



import tensorflow as tf

from tensorflow.keras.callbacks import LearningRateScheduler

from tensorflow.math import softplus, tanh

from tensorflow.keras.utils import get_custom_objects

from tensorflow.keras.layers import Dense, Input, Activation, Dropout

from tensorflow.keras.models import Model

from tensorflow_addons.optimizers import RectifiedAdam



import transformers

from transformers import AutoTokenizer, TFXLMRobertaModel
sns.set_style("darkgrid")
SEED = 42

EPOCHS = 10

MAX_LEN = 96

NUM_SPLITS = 3

LR = 3e-5

BATCH_SIZE = 16
os.environ['PYTHONHASHSEED']=str(SEED)

random.seed(SEED)

np.random.seed(SEED)

tf.random.set_seed(SEED)
path = Path("../input/contradictory-my-dear-watson-translated-dataset/")
train = pd.read_csv(path/"train_augmented.csv", index_col=["id"])

test = pd.read_csv(path/"test_augmented.csv", index_col=["id"])
df = pd.concat([train, test])

df.loc[df["label"]!=-1, "type"] = "train"

df.loc[df["label"]==-1, "type"] = "test"
plt.figure(figsize=(12, 9))

x = sns.countplot(x="language", hue="type", data=df)

_ = plt.title("Language Distribution")

_ = x.set_xticklabels(x.get_xticklabels(), rotation='45')
plt.figure(figsize=(6, 4))

_ = sns.countplot(x="label", data=train)

_ = plt.title("Label Distribution")
del df

gc.collect()
# detect and init the TPU

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)
# instantiate a distribution strategy

strategy = tf.distribute.experimental.TPUStrategy(tpu)
BATCH_SIZE = BATCH_SIZE*strategy.num_replicas_in_sync
MODEL = 'jplu/tf-xlm-roberta-large'

TOKENIZER = AutoTokenizer.from_pretrained(MODEL)
def fast_encode(df):

    text = df[['premise', 'hypothesis']].values.tolist()

    encoded = TOKENIZER.batch_encode_plus(

        text,

        pad_to_max_length=True,

        max_length=MAX_LEN

    )

    return np.array(encoded["input_ids"])
test_encoded = fast_encode(test)



test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(test_encoded)

    .batch(BATCH_SIZE)

)
# softplus - log(exp(x)+1)

def mish(x):

    return x*tanh(softplus(x))

get_custom_objects()["mish"] = Activation(mish)
def create_model(transformer):

    """

    Fine-Tuning XLM-Roberta by adding a couple of dense layers & dropout.

    Adds a dense layer at the end for 3 labels

    """

    input_ids = Input(shape=(MAX_LEN,), dtype=tf.int32)

    sequence_output = transformer(input_ids)[0]

    cls_token = sequence_output[:, 0, :]

    cls_token = Dropout(0.3)(cls_token)

    cls_token = Dense(32, activation='mish')(cls_token)

    cls_token = Dense(16, activation='mish')(cls_token)

    out = Dense(3, activation='softmax')(cls_token)



    optimizer = RectifiedAdam(lr=LR)

    model = Model(inputs=input_ids, outputs=out)

    model.compile(

        optimizer, 

        loss='sparse_categorical_crossentropy', 

        metrics=['accuracy']

    )

    

    return model
kfold = StratifiedKFold(n_splits=NUM_SPLITS, shuffle=True, random_state=SEED)
# learning rate scheduler

def build_lrfn(lr_start=0.00001, lr_max=0.00004, 

               lr_min=0.000001, lr_rampup_epochs=3, 

               lr_sustain_epochs=0, lr_exp_decay=.5):

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
lrfn = build_lrfn()

lrs = LearningRateScheduler(lrfn, verbose=1)
plt.figure(figsize=(9, 6))

epochs = list(range(EPOCHS))

learning_rates = [lrfn(i) for i in epochs]

plt.plot(epochs, learning_rates)

_ = plt.title("Learning Rate Schedule")

_ = plt.xlabel("# Epcohs")

_ = plt.ylabel("Learning Rate")
# storing labels only

oof_preds = np.zeros((len(train)))

# storing probabilities per class

test_preds = np.zeros((len(test), 3))
for fold, (train_index, valid_index) in enumerate(kfold.split(train, train['label'])):

    tf.tpu.experimental.initialize_tpu_system(tpu)



    print("*"*60)

    print("*"+" "*26+f"FOLD {fold+1}"+" "*26+"*")

    print("*"*60, end="\n\n")

    

    X_train = train.iloc[train_index, :].reset_index(drop=True)

    X_valid = train.iloc[valid_index, :].reset_index(drop=True)

    

    y_train = X_train['label'].values

    y_valid = X_valid['label'].values

    

    train_encoded = fast_encode(X_train)

    valid_encoded = fast_encode(X_valid)



    train_dataset = tf.data.Dataset.from_tensor_slices((train_encoded, y_train))

    train_dataset = train_dataset.repeat()

    train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)

    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)



    valid_dataset = tf.data.Dataset.from_tensor_slices((valid_encoded, y_valid))

    valid_dataset = valid_dataset.batch(BATCH_SIZE)

    valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)



    num_steps = len(X_train)//BATCH_SIZE

    

    # instantiating the model in the strategy scope creates the model on the TPU

    with strategy.scope():

        transformer_layer = TFXLMRobertaModel.from_pretrained(MODEL)

        model = create_model(transformer_layer)

    

    history = model.fit(

        train_dataset,

        steps_per_epoch=num_steps,

        validation_data=valid_dataset,

        epochs=EPOCHS,

        callbacks=[lrs]

    ) 

    

    # stores validation data prediction at respective indices

    valid_preds = model.predict(valid_dataset)

    oof_preds[valid_index] = valid_preds.argmax(axis=1)

    

    # adds up test prediction per fold

    preds = model.predict(test_dataset)

    test_preds += preds/NUM_SPLITS
print(f"Accuracy: {accuracy_score(train['label'], oof_preds)}")
print(f"Prediction Shape: {test_preds.shape}")

print(f"Predictions:\n{test_preds[:5]}")
test["prediction"] = test_preds.tolist()

# groupby "id" original & translated version

submission = test.groupby(by='id')['prediction'].apply(lambda x: [sum(y) for y in zip(*x)]).reset_index()
# assigning index from sample_submission file

sample_submission = pd.read_csv("../input/contradictory-my-dear-watson/sample_submission.csv")

submission = submission.set_index("id")

submission = submission.reindex(index=sample_submission["id"])

submission = submission.reset_index()
submission['prediction'] = submission["prediction"].apply(lambda x: np.argmax(x))

submission.to_csv("submission.csv", index=False)
submission["prediction"].value_counts()