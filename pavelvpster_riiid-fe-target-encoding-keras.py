import numpy as np

import pandas as pd



import warnings

warnings.simplefilter('ignore')



import gc, sys

gc.enable()



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
chunksize = 10**6
train = None



load_count = 0



for load_train in pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv', chunksize=chunksize, iterator=True):

    

    load_train = load_train[load_train['answered_correctly'] != -1]

    

    load_train['user_id__content_id'] = load_train['user_id'].astype(str) + '__' + load_train['content_id'].astype(str)

    

    load_train = load_train.drop_duplicates('user_id__content_id', keep = 'last')

    

    if train is None:

        train = load_train

    else:

        train = pd.concat([train, load_train]).drop_duplicates('user_id__content_id', keep = 'last')

    

    load_count += chunksize

    print('Rows processed:', load_count, 'Train set size:', train.shape[0])

    

    if load_count >= 15 * chunksize:

        break
lectures = pd.read_csv("/kaggle/input/riiid-test-answer-prediction/lectures.csv")
questions = pd.read_csv("/kaggle/input/riiid-test-answer-prediction/questions.csv")
question_columns = ['question_id', 'bundle_id', 'part']





def merge_question_columns(data):

    return data.merge(questions[question_columns], right_on='question_id', left_on='content_id', how='left')
train = merge_question_columns(train)
lectures['type_of'] = lectures['type_of'].replace('solving question', 'solving_question')



lectures = pd.get_dummies(lectures, columns=['part', 'type_of'])



part_lectures_columns = [column for column in lectures.columns if column.startswith('part')]



types_of_lectures_columns = [column for column in lectures.columns if column.startswith('type_of_')]
lectures.head(10).T
user_lecture_stats = None





def collect_user_lecture_stats(train):

    

    # merge lecture features to train dataset

    train_lectures = train[train['content_type_id'] == 1].merge(lectures, right_on='lecture_id', left_on='content_id', how='left')

    

    # collect per user stats

    user_lecture_stats_part = train_lectures.groupby('user_id')[part_lectures_columns + types_of_lectures_columns].sum()

    

    # add boolean features

    for column in user_lecture_stats_part.columns:

        bool_column = column + '_boolean'

        user_lecture_stats_part[bool_column] = (user_lecture_stats_part[column] > 0).astype(int)

    

    return user_lecture_stats_part



def update_user_lecture_stats(user_lecture_stats_part):

    global user_lecture_stats

    if user_lecture_stats is None:

        user_lecture_stats = user_lecture_stats_part

    else:

        user_lecture_stats = user_lecture_stats.add(user_lecture_stats_part, fill_value=0.)



def merge_user_lecture_stats(data):

    return data.merge(user_lecture_stats, left_on='user_id', right_index=True, how='left').fillna(0)
from collections import Counter





per_value_counts = {}

global_counts = Counter()





columns_target_encode = ['user_id', 'content_id', 'task_container_id',

                         'bundle_id', 'part']





def add_or_update(column, value, count, answered_correctly):

    # column

    column_data = per_value_counts.get(column, {})

    per_value_counts[column] = column_data

    # value

    value_data = column_data.get(value, Counter())

    column_data[value] = value_data

    # counters

    value_data += Counter({'count': count, 'answered_correctly': answered_correctly})



def update_counts(data, column):

    agg = data.groupby(column)['answered_correctly'].agg(['count', 'mean'])

    agg['answered_correctly'] = agg['count'] * agg['mean']

    for idx,row in agg.iterrows():

        add_or_update(column, idx, row['count'], row['answered_correctly'])



def update_global_counts(data):

    global global_counts

    count = data.shape[0]

    clicks = data[data['answered_correctly'] == 1].shape[0]

    global_counts += Counter({'count': count, 'answered_correctly': clicks})



def update_all_counts(data, columns):

    for column in columns:

        update_counts(data, column)

    update_global_counts(data)



def target_encode_value(column, value):

    counts = per_value_counts.get(column, {}).get(value, Counter())

    if 'answered_correctly' in counts:

        return counts['answered_correctly'] / counts['count']

    else:

        return global_counts['answered_correctly'] / global_counts['count']



def target_encode(data, columns):

    out = pd.DataFrame(index=data.index)

    for column in columns:

        out[column] = data[column].apply(lambda value: target_encode_value(column, value))

    return out
from sklearn.preprocessing import StandardScaler





columns_std = ['prior_question_elapsed_time']





scaler = StandardScaler()
warmup_count = 0



for warmup_train in pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv', chunksize=chunksize, iterator=True):

    

    update_user_lecture_stats(collect_user_lecture_stats(warmup_train))

    

    warmup_train = warmup_train[warmup_train['answered_correctly'] != -1]

    

    warmup_train = merge_question_columns(warmup_train)

    

    update_all_counts(warmup_train, columns_target_encode)

    

    scaler.partial_fit(warmup_train[columns_std])

    

    warmup_count += chunksize

    # print('Rows processed:', warmup_count)
user_lecture_stats.head(10).T
user_lecture_stats_boolean_columns = [column for column in user_lecture_stats.columns if column.endswith('_boolean')]



user_lecture_stats_boolean_columns
columns_copy = ['prior_question_had_explanation']





def make_x(data):

    

    # copy without changes

    x = data[columns_copy + ['user_id']].fillna(0)

    

    # convert Bool to Int

    x['prior_question_had_explanation'] = x['prior_question_had_explanation'].astype(int)

    

    # merge per user lecture stats

    x_lecture_stats = merge_user_lecture_stats(x)[user_lecture_stats_boolean_columns]

    

    x = x.drop('user_id', axis=1)

    

    # target encode

    x_target_encode = target_encode(data, columns_target_encode)

    

    # std

    x_std = pd.DataFrame(scaler.transform(data[columns_std]), index=data.index, columns=columns_std)

    for i,column in enumerate(x_std.columns):

        x_std[column].fillna(scaler.mean_[i], inplace=True)

    

    return pd.concat([x, x_lecture_stats, x_target_encode, x_std], axis=1)
x_train = make_x(train)



y_train = train['answered_correctly']



del train

gc.collect()
x_train.head(10).T
import tensorflow as tf

import tensorflow_addons as tfa



import tensorflow.keras.backend as K
def make_layer(x, units, dropout_rate):

    t = tfa.layers.WeightNormalization(tf.keras.layers.Dense(units))(x)

    t = tf.keras.layers.BatchNormalization()(t)

    t = tf.keras.layers.Activation('relu')(t)

    t = tf.keras.layers.Dropout(dropout_rate)(t)

    return t





def make_model(columns, units, dropout_rates):

    

    inputs = tf.keras.layers.Input(shape=(columns,))

    x = tf.keras.layers.BatchNormalization()(inputs)



    for i in range(len(units)):

        u = units[i]

        d = dropout_rates[i]

        x = make_layer(x, u, d)

       

    y = tf.keras.layers.Dense(1, activation='sigmoid', name='dense_output')(x)

    

    model = tf.keras.Model(inputs=inputs, outputs=y)

    model.compile(loss='binary_crossentropy',

                  optimizer=tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-5),

                  metrics=['accuracy'])

    return model
from sklearn.model_selection import KFold





def fit_validate(n_splits, x_train, y_train, units, dropout_rates, epochs, verbose, random_state):



    estimators = []

    histories = []

    

    scores = []



    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_idx, valid_idx in cv.split(x_train, y_train):



        x_train_train = x_train.iloc[train_idx]

        y_train_train = y_train.iloc[train_idx]

        x_train_valid = x_train.iloc[valid_idx]

        y_train_valid = y_train.iloc[valid_idx]



        K.clear_session()

        

        es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=10,

                                              verbose=verbose, mode='max', restore_best_weights=True)



        rl = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=3, min_lr=1e-5,

                                                  mode='max', verbose=verbose)



        estimator = make_model(x_train.shape[1], units, dropout_rates)



        history = estimator.fit(x_train_train, y_train_train,

                                batch_size=128, epochs=epochs, callbacks=[es, rl],

                                validation_data=(x_train_valid, y_train_valid),

                                verbose=verbose)

        

        estimators.append(estimator)

        histories.append(history)

        

        scores.append(history.history['val_accuracy'][-1])

    

    score = np.mean(scores)

    

    return estimators, histories, score
import optuna



from logging import CRITICAL

optuna.logging.set_verbosity(CRITICAL)





def objective(trial):

    

    n_layers = trial.suggest_int('n_layers', 1, 5)

    

    units = []

    dropout_rates = []

    for i in range(n_layers):

        u = trial.suggest_categorical('units_{}'.format(i+1), [16, 32, 64, 128])

        units.append(u)

        r = trial.suggest_loguniform('dropout_rate_{}'.format(i+1), 0.1, 0.5)

        dropout_rates.append(r)

    

    print('Units:', units, "Dropout rates:", dropout_rates)

    

    _, _, score = fit_validate(3, x_train, y_train, units, dropout_rates, 10, 0, 42)

    print('Score:', score)

    

    return score





# study = optuna.create_study(direction='maximize')

# study.optimize(objective, n_trials=100)
# params = study.best_trial.params

# params
params = {

    'n_layers': 2,

    'units_1': 128,

    'dropout_rate_1': 0.1576269633262961,

    'units_2': 64,

    'dropout_rate_2': 0.26394371768001645

}

params
K.clear_session()





n_layers = params['n_layers']

units = []

dropout_rates = []

for i in range(n_layers):

    u = params['units_{}'.format(i+1)]

    units.append(u)

    d = params['dropout_rate_{}'.format(i+1)]

    dropout_rates.append(d)





estimators, histories, score = fit_validate(3, x_train, y_train, units, dropout_rates, 50, 2, 42)
print('Validation score:', score)
del x_train

del y_train

gc.collect()
import matplotlib.pyplot as plt





fig, axs = plt.subplots(2, 2, figsize=(18,18))



# accuracy

for h in histories:

    axs[0,0].plot(h.history['accuracy'], color='g')

axs[0,0].set_title('Model accuracy - Train')

axs[0,0].set_ylabel('Accuracy')

axs[0,0].set_xlabel('Epoch')



for h in histories:

    axs[0,1].plot(h.history['val_accuracy'], color='b')

axs[0,1].set_title('Model accuracy - Test')

axs[0,1].set_ylabel('Accuracy')

axs[0,1].set_xlabel('Epoch')



# loss

for h in histories:

    axs[1,0].plot(h.history['loss'], color='g')

axs[1,0].set_title('Model loss - Train')

axs[1,0].set_ylabel('Loss')

axs[1,0].set_xlabel('Epoch')



for h in histories:

    axs[1,1].plot(h.history['val_loss'], color='b')

axs[1,1].set_title('Model loss - Test')

axs[1,1].set_ylabel('Loss')

axs[1,1].set_xlabel('Epoch')



fig.show()
import riiideducation



env = riiideducation.make_env()



iter_test = env.iter_test()
for (test, sample_prediction) in iter_test:

    

    test = merge_question_columns(test)

    

    x_test = make_x(test)

    

    y_preds = []

    for estimator in estimators:

        y_pred = estimator.predict(x_test)

        y_preds.append(y_pred)

    

    test['answered_correctly'] = np.mean(y_preds, axis=0)

    

    env.predict(test.loc[test['content_type_id'] == 0, ['row_id', 'answered_correctly']])