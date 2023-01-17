from __future__ import absolute_import, division, print_function, unicode_literals



import numpy as np

import pandas as pd



import tensorflow as tf

from sklearn.model_selection import train_test_split
is_local = False

INPUT_DIR = "/kaggle/input/cat-in-the-dat-ii/"



import tensorflow as tf; 

print(tf.__version__)



if(is_local):

    INPUT_DIR = "../input/"



import os

for dirname, _, filenames in os.walk(INPUT_DIR):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv(INPUT_DIR + "train.csv")

test_df = pd.read_csv(INPUT_DIR + "test.csv")

submission_df = pd.read_csv(INPUT_DIR + "sample_submission.csv")

print("Shape of the train data is ", train_df.shape)

print("Shape of the test data is ", test_df.shape)



train_df.head()
EMBEDDING_DIMENSIONS=9

BATCH_SIZE = 1024

EPOCHS = 25

TRAIN_VAL_SPLIT_RATIO = 0.3



METRICS = [

    tf.keras.metrics.TruePositives(name='tp'),

    tf.keras.metrics.FalsePositives(name='fp'),

    tf.keras.metrics.TrueNegatives(name='tn'),

    tf.keras.metrics.FalseNegatives(name='fn'),

    tf.keras.metrics.BinaryAccuracy(name='accuracy'),

    tf.keras.metrics.Precision(name='precision'),

    tf.keras.metrics.Recall(name='recall'),

    tf.keras.metrics.AUC(name='auc'),

]

COLUMN_TYPES = {

    'id': 'index',

    'bin_0': 'binary', 'bin_1': 'binary', 'bin_2': 'binary', 'bin_3': 'binary', 

    'bin_4': 'binary', 'nom_0': 'categorical', 'nom_1': 'categorical',

    'nom_2': 'categorical', 'nom_3': 'categorical', 'nom_4': 'categorical', 

    'nom_5': 'categorical', 'nom_6': 'categorical', 'nom_7': 'categorical', 

    'nom_8': 'categorical', 'nom_9': 'categorical',

    'ord_0': 'ordinal', 'ord_1': 'ordinal', 'ord_2': 'ordinal', 

    'ord_3': 'ordinal', 'ord_4': 'ordinal', 'ord_5': 'ordinal', 

    'day': 'cyclic', 'month': 'cyclic',

    'target': 'target'

}
def fill_missing_values(dataframe, ignore_cols=['id', 'target']):

    feature_cols = [column for column in dataframe.columns if column not in ignore_cols]

    for a_column in feature_cols:

        typee = COLUMN_TYPES[a_column]

        if(typee == 'binary'):

            dataframe.loc[:, a_column] = dataframe.loc[:, a_column].astype(str).fillna(-9999999)

        elif(typee == 'numeric'):

            pass

        elif(typee == 'categorical'):

            dataframe.loc[:, a_column] = dataframe.loc[:, a_column].astype(str).fillna(-9999999)

        elif(typee == 'ordinal'):

            dataframe.loc[:, a_column] = dataframe.loc[:, a_column].astype(str).fillna(-9999999)

        elif(typee == 'cyclic'):

            median_val = np.median(dataframe[a_column].values)

            if(np.isnan(median_val)):

                median_val = np.median(dataframe[~np.isnan(dataframe[a_column])][a_column].values)

            print(a_column, median_val)

            dataframe.loc[:, a_column] = dataframe.loc[:, a_column].astype(float).fillna(median_val)

            

    return dataframe.copy(deep=True)



train_df = fill_missing_values(train_df, ignore_cols=['id', 'target'])

test_df = fill_missing_values(test_df, ignore_cols=['id'])
def get_initial_bias(df, col_name='target'):

    neg, pos = np.bincount(df[col_name])

    total = neg + pos

    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(

        total, pos, 100 * pos / total))



    initial_bias = np.log([pos/neg])

    

    return initial_bias







initial_bias = get_initial_bias(train_df)

def get_class_weights(df, col_name='target'):

    neg, pos = np.bincount(df[col_name])

    weight_for_0 = (1 / neg) * (neg + pos) / 2.0

    weight_for_1 = (1 / pos) * (neg + pos) / 2.0



    class_weight = {

        0: weight_for_0,

        1: weight_for_1

    }



    print("Class 0: ", weight_for_0, "Weightage")

    print("Class 1: ", weight_for_1, "Weightage")

    

    return class_weight



class_weight = get_class_weights(train_df)
### Stratified Split



from sklearn.model_selection import StratifiedShuffleSplit



def split_train_validation_data(df, col_name='target', stratify=True, test_size=0.3):

    train = None

    val = None

    

    if(stratify):

        



        sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=21)

        sss.get_n_splits(df, df.target)



        splits = sss.split(df, df.target) 

        

        indices = []

        for train_index, test_index in splits:

            indices.append({

                'train': train_index,

                'test': test_index

            })



        train = df.iloc[indices[0]['train']]

        val = df.iloc[indices[0]['test']]

        

    else:

        train, val = train_test_split(train_df, test_size=test_size)

    return train, val



train, val = split_train_validation_data(train_df, test_size=TRAIN_VAL_SPLIT_RATIO)
get_initial_bias(train)

get_initial_bias(val)

def handle_feature_columns(df, columns_to_remove=['id', 'target'], all_categorical_as_ohe=True):

    

    def demo(feature_column):

        feature_layer = tf.keras.layers.DenseFeatures(feature_column)

    

    def one_hot_encode(col_name, unique_values):

        from_vocab = tf.feature_column.categorical_column_with_vocabulary_list(

            col_name, unique_values

        )

        ohe = tf.feature_column.indicator_column(from_vocab)

        data.append(ohe)

        demo(ohe)

    

    def embedd(col_name, unique_values):

        from_vocab = tf.feature_column.categorical_column_with_vocabulary_list(

            col_name, unique_values

        )

        embeddings = tf.feature_column.embedding_column(from_vocab, dimension=EMBEDDING_DIMENSIONS)

        data.append(embeddings)

        demo(embeddings)

        

    def numeric(col_name, unique_values):

        from_numeric = tf.feature_column.numeric_column(

            col_name, dtype=tf.float32

        )

        data.append(from_numeric)

        demo(from_numeric)

    

    dataframe = df.copy()

    for pop_col in columns_to_remove:

        dataframe.pop(pop_col)

    data = []

    

    for a_column in dataframe.columns:

        typee = COLUMN_TYPES[a_column]

        nunique = dataframe[a_column].nunique()

        unique_values = dataframe[a_column].unique()

        print('Column :', a_column, nunique, unique_values[:10])                

        if(typee == 'binary'):

            one_hot_encode(a_column, unique_values)

        elif(typee == 'cyclic'):

            numeric(a_column, unique_values)

            

        else:

            if(all_categorical_as_ohe):

                one_hot_encode(a_column, unique_values)

            else:

                if(typee == 'categorical'):

                    if(nunique < 100):

                        one_hot_encode(a_column, unique_values)

                    else:

                        embedd(a_column, unique_values)

                elif(typee == 'ordinal'):

                    embedd(a_column, unique_values)

            

    return data
feature_columns = handle_feature_columns(train, all_categorical_as_ohe=False)
y_train = train.pop('target')

y_val = val.pop('target')
def df_to_dataset(dataframe, y, shuffle=True, batch_size=32, is_test_data=False):

    dataframe = dataframe.copy()

    ds = None

    if(is_test_data):

        ds = tf.data.Dataset.from_tensor_slices(dict(dataframe))

    else:

        

        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), y))

        if(shuffle):

            ds = ds.shuffle(buffer_size=len(dataframe))

    ds = ds.batch(batch_size)

    return ds



train_ds = df_to_dataset(train, y_train, shuffle=False, batch_size=BATCH_SIZE)

val_ds = df_to_dataset(val, y_val, shuffle=False, batch_size=BATCH_SIZE)

test_ds = df_to_dataset(test_df, None, shuffle=False, batch_size=BATCH_SIZE, is_test_data=True)
def create_silly_model_2(feature_layer, initial_bias=None):

    bias = None

    if(initial_bias):

        bias = tf.keras.initializers.Constant(initial_bias)

        

    model = tf.keras.Sequential([

        feature_layer,

        tf.keras.layers.Dense(512, activation='relu'),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(256, activation='relu'),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(128, activation='relu'),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=bias)

    ])

    model.compile(

        optimizer='adam',

        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),

        metrics=METRICS

    )

    return model


def run(

    train_data, val_data, feature_columns, 

    epochs=EPOCHS, es=False, rlr=False, 

    class_weights=None, initial_bias=None

):

    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

    model = create_silly_model_2(feature_layer, initial_bias)



    callbacks = []

    if(es):

        callbacks.append(

            tf.keras.callbacks.EarlyStopping(

                monitor='val_auc', min_delta=0.00001, patience=5, 

                mode='auto', verbose=1, baseline=None, restore_best_weights=True

            )

        )

    if(rlr):

        callbacks.append(

            tf.keras.callbacks.ReduceLROnPlateau(

                monitor='val_auc', factor=0.5, patience=3, 

                min_lr=3e-6, mode='auto', verbose=1

            )

        )



    history = model.fit(

        train_ds, 

        validation_data=val_ds, 

        epochs=epochs, 

        callbacks=callbacks,

        class_weight=class_weights

    )

    

    return model, history
model, history = run(

    train_ds, val_ds, feature_columns, 

    epochs=EPOCHS, es=False, rlr=False, 

    class_weights=None, initial_bias=None

)
predictions = model.predict(test_ds)

submit = pd.DataFrame()

submit["id"] = test_df["id"]

submit['target'] = predictions

submit.to_csv('submission_dl_stratify.csv', index=False)
import matplotlib as mpl

import matplotlib.pyplot as plt

mpl.rcParams['figure.figsize'] = (15, 10)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
def plot_metrics(history):

    metrics = ['loss', 'auc', 'precision', 'recall']

    

    for n, metric in enumerate(metrics):

        name = metric.replace("_", " ").capitalize()

        plt.subplot(2, 2, n+1)

        plt.plot(

            history.epoch, 

            history.history[metric], 

            color=colors[0], 

            label='Train'

        )

        plt.plot(

            history.epoch, 

            history.history['val_' + metric], 

            color=colors[0], 

            linestyle="--", 

            label='val'

        )

        plt.title(metric.upper())

        plt.xlabel('Epoch')

        plt.ylabel(name)

        if(metric == 'loss'):

            plt.ylim([0, plt.ylim()[1]])

        elif(metric == 'auc'):

            plt.ylim([0, 1])

        else:

            plt.ylim([0, 1])

        plt.legend()

        

plot_metrics(history)
train_predictions = model.predict(train_ds)

val_predictions = model.predict(val_ds)
from sklearn import metrics



def roc(name, labels, predictions, **kwargs):

    fp, tp, _ = metrics.roc_curve(labels, predictions)

    

    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)

    plt.xlabel('False Positives [%]')

    plt.ylabel('True Positives [%]')

    plt.xlim([-0.5, 110])

    plt.ylim([1, 110])

    plt.grid(True)

    ax = plt.gca()

    ax.set_aspect('equal')

    plt.legend()



roc('Train', y_train, train_predictions, color=colors[0])

roc('Validate', y_val, val_predictions, color=colors[0], linestyle='--')
model, history = run(

    train_ds, val_ds, feature_columns, 

    epochs=EPOCHS, es=True, rlr=True, 

    class_weights=None, initial_bias=None

)
plot_metrics(history)
train_predictions_es_rlr = model.predict(train_ds)

val_predictions_es_rlr = model.predict(val_ds)


roc('Train Baseline', y_train, train_predictions, color=colors[0])

roc('Validate Baseline', y_val, val_predictions, color=colors[0], linestyle='--')

roc('Train [ES, RLR]', y_train, train_predictions_es_rlr, color=colors[1])

roc('Validate [ES, RLR]', y_val, val_predictions_es_rlr, color=colors[1], linestyle='--')
model, history = run(

    train_ds, val_ds, feature_columns, 

    epochs=EPOCHS, es=True, rlr=True, 

    class_weights=class_weight, initial_bias=initial_bias

)
plot_metrics(history)
train_predictions_bias_cws = model.predict(train_ds)

val_predictions_bias_cws = model.predict(val_ds)
roc('Train Baseline', y_train, train_predictions, color=colors[0])

roc('Validate Baseline', y_val, val_predictions, color=colors[0], linestyle='--')

roc('Train [ES, RLR]', y_train, train_predictions_es_rlr, color=colors[1])

roc('Validate [ES, RLR]', y_val, val_predictions_es_rlr, color=colors[1], linestyle='--')

roc('Train [ES, RLR, BIAS, IniWts]', y_train, train_predictions_bias_cws, color=colors[2])

roc('Validate [ES, RLR, BIAS, IniWts]', y_val, val_predictions_bias_cws, color=colors[2], linestyle='--')
predictions = model.predict(test_ds)

submit = pd.DataFrame()

submit["id"] = test_df["id"]

submit['target'] = predictions

submit.to_csv('submission_dl_final.csv', index=False)