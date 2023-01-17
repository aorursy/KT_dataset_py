# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



# additional imports

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

import tensorflow_hub as tf_hub

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score, f1_score

print(tf.__version__)

assert tf.__version__ >= '2.0'
# load data

tv_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')



# split out validation set for testing

seed = 54321

valid_df = tv_df.sample(frac=.2, random_state=seed)

train_df = tv_df[~tv_df.index.isin(valid_df.index)]



# take a look at the fields

train_df.info()
cols = [c for c in train_df.columns if train_df[c].dtype in (np.int64, np.float64)]

num_plots = len(cols)

fig, axs = plt.subplots((num_plots+1)//2, 2, figsize=(10, num_plots*1.2))

for i, ax in enumerate(axs.flatten()):

    if i >= num_plots:

        ax.axis('off')

        continue

    sns.distplot(train_df[cols[i]], kde=False, ax=ax)

    ax.set_title(cols[i] + ' Distribution', fontsize=14)

    ax.set_xlabel('')

fig.tight_layout(rect=[0,0,.9,1])
# function prep work

emb_mode = train_df.Embarked.mode()[0]

one_hot_cols = [

    'Pclass', 'Sex', 'age_binned', 'SibSp', 'Parch',

    'Embarked', 'name_title'

]

# scale fares

fare_scaler = MinMaxScaler()

fare_scaler.fit(train_df.Fare.values.reshape(-1, 1))

# explicitely declare fields (since pd.get_dummies output can vary)

dummy_fields = [

#     'Embarked_C',

    'Embarked_Q', 'Embarked_S', 'Embarked_nan',

#     'Parch_0.0',

    'Parch_1.0', 'Parch_2.0', 'Parch_3.0', 'Parch_4.0',

    'Parch_5.0', 'Parch_6.0', 'Parch_nan',

#     'Pclass_1.0',

    'Pclass_2.0', 'Pclass_3.0', 'Pclass_nan',

#     'Sex_female',

    'Sex_male', 'Sex_nan',

#     'SibSp_0.0',

    'SibSp_1.0', 'SibSp_2.0', 'SibSp_3.0', 'SibSp_4.0',

    'SibSp_5.0', 'SibSp_nan',

#     'name_title_Mr',

    'name_title_Mrs', 'name_title_Miss',

    'name_title_Master', 'name_title_Other',

#     'age_binned_(-0.001, 1.0]',

    'age_binned_(1.0, 2.0]', 'age_binned_(10.0, 12.0]', 'age_binned_(12.0, 14.0]',

    'age_binned_(14.0, 16.0]', 'age_binned_(16.0, 20.0]', 'age_binned_(2.0, 3.0]',

    'age_binned_(20.0, 24.0]', 'age_binned_(24.0, 28.0]' 'age_binned_(28.0, 32.0]',

    'age_binned_(3.0, 4.0]', 'age_binned_(32.0, 36.0]', 'age_binned_(36.0, 40.0]',

    'age_binned_(4.0, 6.0]', 'age_binned_(40.0, 44.0]', 'age_binned_(44.0, 48.0]',

    'age_binned_(48.0, 52.0]', 'age_binned_(52.0, 56.0]', 'age_binned_(56.0, 60.0]',

    'age_binned_(6.0, 8.0]', 'age_binned_(60.0, 64.0]', 'age_binned_(64.0, 68.0]',

    'age_binned_(68.0, 72.0]', 'age_binned_(72.0, 76.0]', 'age_binned_(76.0, 80.0]',

    'age_binned_(8.0, 10.0]', 'age_binned_(80.0, 84.0]', 'age_binned_(84.0, 88.0]',

    'age_binned_(88.0, 92.0]', 'age_binned_nan'

]

ordered_cols = ['Survived', 'Name', 'Cabin', 'Ticket', 'Fare'] + dummy_fields



# name_title helper function

def title_map(title):

    if title.lower() in ('mr', 'miss', 'mrs', 'master'):

        return title

    else: return 'Other'



def preprocess(data, data_type='train'):

    df = data.copy()

    # create additional fields

    df['cabin_bool'] = (~df.Cabin.isna()).astype(np.int32)

    df['name_title'] = df.Name.str.extract(

        pat='^.+, +([A-z ]+)\.'

    )[0].astype('str').apply(lambda t: title_map(t))

    # categorical field prep

    df.Embarked = df.Embarked.fillna(emb_mode)

    df['age_binned'] = pd.cut(

        df.Age, include_lowest=True,

        bins=list(range(4))+list(range(4,17,2))+list(range(20,96,4))

    )

    df.Parch = df.Parch.map(lambda x: 6 if x > 6 else x)

    df.SibSp = df.SibSp.map(lambda x: 5 if x > 5 else x)

    # one-hot encode categorical fields

    df = pd.get_dummies(df, columns = one_hot_cols, dummy_na=True, drop_first=False, dtype=np.int32)

    # order columns and create columns that may have been missed when creating dummies

    # leave out unnecessary fields

    df = df.reindex(columns = ordered_cols)

    df.loc[:, dummy_fields] = df.loc[:, dummy_fields].fillna(0)

    # normalize Fare field and fill nulls

    df.Fare = fare_scaler.transform(df.Fare.values.reshape(-1,1))

    df.Fare = df.Fare.fillna(df.Fare.median())

    # make remaining NaNs official values

    df.Cabin = df.Cabin.fillna('NAN')

    # fix dtypes (for Tensorflow models)

    if data_type=='train':

        df.Survived = df.Survived.astype(np.int32)

    df.Fare = df.Fare.astype(np.float32)

    df.Name = df.Name.astype(np.str)

    df.Cabin = df.Cabin.astype(np.str)

    df.Ticket = df.Ticket.astype(np.str)

    return df    
train_prepped = preprocess(train_df)

valid_prepped = preprocess(valid_df)
# inspect preprocessed columns

train_prepped.columns
# numerical cols only

X_train = train_prepped.loc[

    :, train_prepped.columns.difference(['Survived', 'Name', 'Cabin', 'Ticket'])

].values

y_train = train_prepped.Survived

X_valid = valid_prepped.loc[

    :, valid_prepped.columns.difference(['Survived', 'Name', 'Cabin', 'Ticket'])

].values

y_valid = valid_prepped.Survived
# a look at `Cabin`

train_prepped.loc[:, ['Survived', 'Cabin', 'Ticket']].head(10)
# prepare cabin and ticket tokenizers

cbn_vocab_sz = 23+10+1 

tkt_vocab_sz = 23+10+10+1  #(letters+numbers+special_chars+oov_char)

cbn_tokenizer = tf.keras.preprocessing.text.Tokenizer(

    num_words=cbn_vocab_sz, filters='', char_level=True, oov_token='<oov>'

)

tkt_tokenizer = tf.keras.preprocessing.text.Tokenizer(

    num_words=tkt_vocab_sz, filters='', char_level=True, oov_token='<oov>'

)

# get max record lengths from training data

max_nmlen = train_prepped.Name.apply(lambda x: len(x)).max()

max_cbnlen = train_prepped.Cabin.apply(lambda x: len(x)).max()

max_tktlen = train_prepped.Ticket.apply(lambda x: len(x)).max()



# fit character tokenizers to training data

cbn_tokenizer.fit_on_texts(train_prepped.Cabin)

tkt_tokenizer.fit_on_texts(train_prepped.Ticket)

# convert training texts to numerical sequences

cbn_seqs = cbn_tokenizer.texts_to_sequences(train_prepped.Cabin)

tkt_seqs = tkt_tokenizer.texts_to_sequences(train_prepped.Ticket)

cbn_valid_seqs = cbn_tokenizer.texts_to_sequences(valid_prepped.Cabin)

tkt_valid_seqs = tkt_tokenizer.texts_to_sequences(valid_prepped.Ticket)

# pad sequences

cbn_padded = tf.keras.preprocessing.sequence.pad_sequences(

    cbn_seqs, maxlen=max_cbnlen

)

tkt_padded = tf.keras.preprocessing.sequence.pad_sequences(

    tkt_seqs, maxlen=max_tktlen

)

cbn_valid_padded = tf.keras.preprocessing.sequence.pad_sequences(

    cbn_valid_seqs, maxlen=max_cbnlen

)

tkt_valid_padded = tf.keras.preprocessing.sequence.pad_sequences(

    tkt_valid_seqs, maxlen=max_tktlen

)
# prepare data in datasets

train_name_pre = tf.data.Dataset.from_tensor_slices(

    (train_prepped.Name.values, y_train.values.reshape(-1,1))

)

train_cabin_pre = tf.data.Dataset.from_tensor_slices(

    (cbn_padded, y_train.values.reshape(-1,1))

)

train_ticket_pre = tf.data.Dataset.from_tensor_slices(

    (tkt_padded,  y_train.values.reshape(-1,1))

)

train_num_pre = tf.data.Dataset.from_tensor_slices(

    (X_train,  y_train.values.reshape(-1,1))

)

valid_name_pre = tf.data.Dataset.from_tensor_slices(

    (valid_prepped.Name.values, y_valid.values.reshape(-1,1))

)

valid_cabin_pre = tf.data.Dataset.from_tensor_slices(

    (cbn_valid_padded, y_valid.values.reshape(-1,1))

)

valid_ticket_pre = tf.data.Dataset.from_tensor_slices(

    (tkt_valid_padded,  y_valid.values.reshape(-1,1))

)

valid_num_pre = tf.data.Dataset.from_tensor_slices(

    (X_valid,  y_valid.values.reshape(-1,1))

)



# zip the independent vars for combined model training

ind_zipped = tf.data.Dataset.zip((

    train_name_pre.map(lambda x,y: x),

    train_cabin_pre.map(lambda x,y: x),

    train_ticket_pre.map(lambda x,y: x),

    train_num_pre.map(lambda x,y: x)

))

ind_zipped_valid = tf.data.Dataset.zip((

    valid_name_pre.map(lambda x,y: x),

    valid_cabin_pre.map(lambda x,y: x),

    valid_ticket_pre.map(lambda x,y: x),

    valid_num_pre.map(lambda x,y: x)

))



# shuffle and batch

train_name = train_name_pre.shuffle(1000).batch(32)

train_cabin = train_cabin_pre.shuffle(1000).batch(32)

train_ticket = train_ticket_pre.shuffle(1000).batch(32)

train_num = train_num_pre.shuffle(1000).batch(32)

valid_name = valid_name_pre.batch(32)

valid_cabin = valid_cabin_pre.batch(32)

valid_ticket = valid_ticket_pre.batch(32)

valid_num = valid_num_pre.batch(32)



# zip combined ind vars with dependent variable (for combined model)

train_comb = tf.data.Dataset.zip(

    (ind_zipped, train_name_pre.map(lambda x,y: y))

).shuffle(1000).batch(32)

valid_comb = tf.data.Dataset.zip(

    (ind_zipped_valid, valid_name_pre.map(lambda x,y: y))

).batch(32)
# inspect shape

print(train_comb)
# plot loss over learning rate

def plot_loss_over_lr(history, loss_bound=(0,2)):

    loss = history.history['loss']

    lr = history.history['lr']

    fig, ax = plt.subplots(figsize=(8, 3))

    ax.semilogx(lr, loss, '-b')

    ax.legend()

    ax.set_title('Loss over Learning Rate', fontsize=18)

    ax.set_xlabel('Learning Rate')

    ax.set_ylabel('Loss')

    ax.set_ylim(loss_bound)

    plt.show()



# plot model accuracy and loss    

def plot_model(history):

    acc = history.history['acc']

    loss = history.history['loss']

    val_acc = history.history.get('val_acc')

    val_loss = history.history.get('val_loss')



    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(range(len(acc)), acc, '-b', label='acc')

    if val_acc:

        ax[0].plot(range(len(acc)), val_acc, 'orange', label='val_acc')

    ax[0].legend()

    ax[0].set_title('Accuracy', fontsize=18)

    ax[0].set_xlabel('epochs')

    ax[1].plot(range(len(loss)), loss, '-b', label='loss')

    if val_loss:

        ax[1].semilogx(range(len(loss)), val_loss, 'orange', label='val_loss')

    ax[1].legend()

    ax[1].set_title('Loss', fontsize=18)

    plt.show()



# display results after model fitting

def fit_results(model, history, train_data, valid_data):

    print(f'\nModel trained for {len(history.history["lr"])} epochs\n')

    model.evaluate(train_data)

    model.evaluate(valid_data)



# Callbacks (early stopping and learning rate schedule)

patience = 32

stp_callback = tf.keras.callbacks.EarlyStopping(

    patience=patience, restore_best_weights=True

)

def lr_sched(initial_lr):

    lr_callback = tf.keras.callbacks.LearningRateScheduler(

        lambda epoch: initial_lr if epoch < 10 else initial_lr * tf.math.exp(0.02 * (10 - epoch))

    )

    return lr_callback
# Name branch build

name_model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=[], dtype=tf.string, name='name_input'),

    tf_hub.KerasLayer(

        "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1",

        dtype=tf.string, input_shape=[], output_shape=[128]

    ),

    tf.keras.layers.Dense(16, activation='elu', kernel_regularizer='l2'),

    tf.keras.layers.Dropout(.5),

    tf.keras.layers.Dense(16, activation='elu', kernel_regularizer='l2'),

    tf.keras.layers.Dropout(.5, name='name_last_hidden'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])

name_model.compile(

    loss='binary_crossentropy', optimizer='Adam', metrics=['acc']

)

name_model.save_weights('name_model_initial_weights.h5')
# search for optimal learning rate

lr_search = tf.keras.callbacks.LearningRateScheduler(

    lambda epoch: 1e-8*10**(epoch/5.)

)

name_metrics = name_model.fit(

    train_name, epochs=42, verbose=0, callbacks=[lr_search]

)



plot_loss_over_lr(name_metrics)
# set initial learning rate

lr = 8e-3

lr_sched_name = lr_sched(lr)

# set the weights back to their initial state

name_model.load_weights('name_model_initial_weights.h5')

# fit model

history = name_model.fit(

    train_name, epochs=400, 

    validation_data=valid_name, 

    verbose=0, 

    callbacks=[stp_callback, lr_sched_name]

)

name_model.save_weights('name_model_weights.h5')

name_last_lr = history.history['lr'][-patience]

fit_results(name_model, history, train_name, valid_name)

plot_model(history)
cabin_model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(15,), name='cabin_inp'),

    tf.keras.layers.Embedding(

        input_dim=cbn_vocab_sz, output_dim=10, input_length=max_cbnlen

    ),

    tf.keras.layers.Dense(4, activation='elu', kernel_regularizer='l2'),

    tf.keras.layers.GlobalAveragePooling1D(name='cabin_last_hidden'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])



cabin_model.compile(

    loss='binary_crossentropy', optimizer='Adam', metrics=['acc']

)

cabin_model.save_weights('cabin_model_initial_weights.h5')
# search for optimal learning rate

cabin_metrics = cabin_model.fit(

    train_cabin, epochs=42, verbose=0, callbacks=[lr_search]

)

plot_loss_over_lr(cabin_metrics, loss_bound=(.6,1))
# set initial learning rate

lr = 1e-2

lr_sched_cabin = lr_sched(lr)

# set the weights back to their initial state

cabin_model.load_weights('cabin_model_initial_weights.h5')

# fit model

history = cabin_model.fit(

    train_cabin, epochs=400, 

    validation_data=valid_cabin, 

    verbose=0, 

    callbacks=[stp_callback, lr_sched_cabin]

)

cabin_model.save_weights('cabin_model_weights.h5')

cabin_last_lr = history.history['lr'][-patience]

fit_results(cabin_model, history, train_cabin, valid_cabin)

plot_model(history)
ticket_model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(18,), name='ticket_inp'),

    tf.keras.layers.Embedding(

        input_dim=tkt_vocab_sz, output_dim=10, input_length=max_tktlen

    ),

    tf.keras.layers.Dense(4, activation='elu', kernel_regularizer='l2'),

    tf.keras.layers.GlobalAveragePooling1D(name='ticket_last_hidden'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])



ticket_model.compile(

    loss='binary_crossentropy', optimizer='Adam', metrics=['acc']

)

ticket_model.save_weights('ticket_model_initial_weights.h5')
# search for optimal learning rate

ticket_metrics = ticket_model.fit(

    train_ticket, epochs=42, verbose=0, callbacks=[lr_search]

)

plot_loss_over_lr(ticket_metrics, loss_bound=(.4, 1))
# set initial learning rate

lr = 1e-2

lr_sched_ticket = lr_sched(lr)

# set the weights back to their initial state

ticket_model.load_weights('ticket_model_initial_weights.h5')

# fit model

history = ticket_model.fit(

    train_ticket, epochs=400, 

    validation_data=valid_ticket, 

    verbose=0, 

    callbacks=[stp_callback, lr_sched_ticket]

)

ticket_model.save_weights('ticket_model_weights.h5')

ticket_last_lr = history.history['lr'][-patience]

fit_results(ticket_model, history, train_ticket, valid_ticket)

plot_model(history)
numerical_model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(54,), name='numerical_inp'),

    tf.keras.layers.Dense(64, activation='elu', kernel_regularizer='l2'),

    tf.keras.layers.Dropout(.5),

    tf.keras.layers.Dense(64, activation='elu', kernel_regularizer='l2'),

    tf.keras.layers.Dropout(.5, name='numerical_last_hidden'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])



numerical_model.compile(

    loss='binary_crossentropy', optimizer='Adam', metrics=['acc']

)

numerical_model.save_weights('numerical_model_initial_weights.h5')
# search for optimal learning rate

numerical_metrics = numerical_model.fit(

    train_num, epochs=42, verbose=0, callbacks=[lr_search]

)

plot_loss_over_lr(numerical_metrics, loss_bound=(0,3))
# set initial learning rate

lr = 1e-2

lr_sched_num = lr_sched(lr)

# set the weights back to their initial state

numerical_model.load_weights('numerical_model_initial_weights.h5')

# fit model

history = numerical_model.fit(

    train_num, epochs=400, 

    validation_data=valid_num, 

    verbose=0, 

    callbacks=[stp_callback, lr_sched_num]

)

numerical_model.save_weights('numerical_model_weights.h5')

numerical_last_lr = history.history['lr'][-patience]

fit_results(numerical_model, history, train_num, valid_num)

plot_model(history)
# combined model

# get last hidden layer for each branch

name = name_model.get_layer('name_last_hidden')

cabin = cabin_model.get_layer('cabin_last_hidden')

ticket = ticket_model.get_layer('ticket_last_hidden')

numerical = numerical_model.get_layer('numerical_last_hidden')



# freeze branches for now

branches = [name_model, cabin_model, ticket_model, numerical_model]

for branch in branches:

    for layer in branch.layers:

        layer.trainable = False

        

# build combined model

x = tf.keras.layers.concatenate([

        name.output, cabin.output, ticket.output, numerical.output],

        axis=1

)

x = tf.keras.layers.Dense(64, activation='elu', kernel_regularizer='l2')(x)

x = tf.keras.layers.Dropout(.5)(x)

x = tf.keras.layers.Dense(64, activation='elu', kernel_regularizer='l2')(x)

x = tf.keras.layers.Dense(1, activation='sigmoid')(x)



combined_model = tf.keras.Model(

    [

        name_model.input, cabin_model.input,

        ticket_model.input, numerical_model.input

    ],

    x

)



combined_model.compile(

    loss='binary_crossentropy', optimizer='Adam', metrics=['acc']

)

combined_model.save_weights('combined_model_initial_weights.h5')
# search for optimal learning rate (training top only)

combined_metrics = combined_model.fit(

    train_comb, epochs=42, verbose=0, callbacks=[lr_search]

)

plot_loss_over_lr(combined_metrics, loss_bound=(0,3))
# set initial learning rate

lr = 1e-2

lr_sched_comb = lr_sched(lr)

# set the weights back to their initial state

combined_model.load_weights('combined_model_initial_weights.h5')

# fit model

history = combined_model.fit(

    train_comb, epochs=400, 

    validation_data=valid_comb, 

    verbose=0, 

    callbacks=[stp_callback, lr_sched_comb]

)

combined_model.save_weights('combined_model_weights.h5')

comb_last_lr = history.history['lr'][-patience]

fit_results(combined_model, history, train_comb, valid_comb)

plot_model(history)
# train whole combined model (all layers trainable)

# get last learning rates for components

lr = np.min([

    name_last_lr, cabin_last_lr, ticket_last_lr,

    numerical_last_lr, comb_last_lr

])

lr_sched_comb1 = lr_sched(lr)



# unfreeze branch layers for fine tuning

branches = [name_model, cabin_model, ticket_model, numerical_model]

for branch in branches:

    for layer in branch.layers:

        layer.trainable = True



# fit model

history = combined_model.fit(

    train_comb, epochs=400, 

    validation_data=valid_comb, 

    verbose=0, 

    callbacks=[stp_callback, lr_sched_comb1]

)

combined_model.save_weights('combined_model_weights.h5')

fit_results(combined_model, history, train_comb, valid_comb)

plot_model(history)
# combine training and validation data

train_p_valid = train_comb.concatenate(valid_comb).shuffle(1000)

# get last learning rate for fine tuning

last_lr = history.history['lr'][-patience]
# train model one last time including on validation data

# setup final learning rate scheduler

lr_cb = tf.keras.callbacks.LearningRateScheduler(

    lambda epoch: comb_last_lr if epoch < 4 else comb_last_lr * tf.math.exp(0.02 * (4 - epoch))

)

# Have stop callback monitor loss instead of val_loss

# (Since validation data is included in training, val_loss won't be tracked.)

stp_cb = tf.keras.callbacks.EarlyStopping(patience=patience, monitor='loss')

history = combined_model.fit(

    train_p_valid, epochs=400,

    callbacks=[lr_cb, stp_cb],

    verbose=0

)

print(f'\nModel trained for {len(history.history["lr"])} epochs')

print(f'Final learning rate: {history.history["lr"][-patience]}')

combined_model.evaluate(train_comb)

combined_model.evaluate(valid_comb)

plot_model(history)
# preprocess test data

test_prepped = preprocess(test_df, data_type='test')
# convert training texts to numerical sequences

cbn_test_seqs = cbn_tokenizer.texts_to_sequences(test_prepped.Cabin)

tkt_test_seqs = tkt_tokenizer.texts_to_sequences(test_prepped.Ticket)

# pad sequences

cbn_test_padded = tf.keras.preprocessing.sequence.pad_sequences(

    cbn_test_seqs, maxlen=max_cbnlen

)

tkt_test_padded = tf.keras.preprocessing.sequence.pad_sequences(

    tkt_test_seqs, maxlen=max_tktlen

)



# numerical values

X_test = test_prepped.loc[

    :, test_prepped.columns.difference(['Survived', 'Name', 'Cabin', 'Ticket'])

].values



test_comb = tf.data.Dataset.from_tensor_slices((

    (test_prepped.Name, cbn_test_padded, tkt_test_padded, X_test),

)).batch(32).prefetch(1)
# get test predictions

test_preds = combined_model.predict(test_comb)

# convert float predictions to integer format (required for Kaggle submission)

test_df['Survived'] = (test_preds > .5).astype('int')

# Save csv of test predictions

test_df.loc[

    :, ['PassengerId', 'Survived']

].to_csv('test_preds.csv', index=False)

# quick check that file saved correctly

!head test_preds.csv