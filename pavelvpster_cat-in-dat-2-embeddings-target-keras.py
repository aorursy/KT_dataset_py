import numpy as np

import pandas as pd



import warnings

warnings.simplefilter('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('../input/cat-in-the-dat-ii/train.csv', index_col='id')

test = pd.read_csv('../input/cat-in-the-dat-ii/test.csv', index_col='id')
train.head(3).T
def summary(df):

    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name', 'dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.loc[0].values

    summary['Second Value'] = df.loc[1].values

    summary['Third Value'] = df.loc[2].values

    return summary





summary(train)
train['missing_count'] = train.isnull().sum(axis=1)

test['missing_count'] = test.isnull().sum(axis=1)
missing_number = -99999

missing_string = 'MISSING_STRING'
numerical_features = [

    'bin_0', 'bin_1', 'bin_2',

    'ord_0',

    'day', 'month'

]



string_features = [

    'bin_3', 'bin_4',

    'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5',

    'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'

]
def impute(train, test, columns, value):

    for column in columns:

        train[column] = train[column].fillna(value)

        test[column] = test[column].fillna(value)
impute(train, test, numerical_features, missing_number)

impute(train, test, string_features, missing_string)
train['ord_5_1'] = train['ord_5'].str[0]

train['ord_5_2'] = train['ord_5'].str[1]



train.loc[train['ord_5'] == missing_string, 'ord_5_1'] = missing_string

train.loc[train['ord_5'] == missing_string, 'ord_5_2'] = missing_string



train = train.drop('ord_5', axis=1)





test['ord_5_1'] = test['ord_5'].str[0]

test['ord_5_2'] = test['ord_5'].str[1]



test.loc[test['ord_5'] == missing_string, 'ord_5_1'] = missing_string

test.loc[test['ord_5'] == missing_string, 'ord_5_2'] = missing_string



test = test.drop('ord_5', axis=1)
simple_features = [

    'missing_count'

]



oe_features = [

    'bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4',

    'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4',

    'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5_1', 'ord_5_2',

    'day', 'month'

]



ohe_features = oe_features



target_features = [

    'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'

]
y_train = train['target'].copy()

x_train = train.drop('target', axis=1)

del train



x_test = test.copy()

del test
from sklearn.preprocessing import StandardScaler





scaler = StandardScaler()

simple_x_train = scaler.fit_transform(x_train[simple_features])

simple_x_test = scaler.transform(x_test[simple_features])
from sklearn.preprocessing import OneHotEncoder





ohe = OneHotEncoder(dtype='uint16', handle_unknown="ignore")

ohe_x_train = ohe.fit_transform(x_train[ohe_features])

ohe_x_test = ohe.transform(x_test[ohe_features])
from sklearn.preprocessing import OrdinalEncoder





oe = OrdinalEncoder()

oe_x_train = oe.fit_transform(x_train[oe_features])

oe_x_test = oe.transform(x_test[oe_features])
from category_encoders import TargetEncoder

from sklearn.model_selection import StratifiedKFold
def transform(transformer, x_train, y_train, cv):

    oof = pd.DataFrame(index=x_train.index, columns=x_train.columns)

    for train_idx, valid_idx in cv.split(x_train, y_train):

        x_train_train = x_train.loc[train_idx]

        y_train_train = y_train.loc[train_idx]

        x_train_valid = x_train.loc[valid_idx]

        transformer.fit(x_train_train, y_train_train)

        oof_part = transformer.transform(x_train_valid)

        oof.loc[valid_idx] = oof_part

    return oof
target = TargetEncoder(drop_invariant=True, smoothing=0.2)



cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

target_x_train = transform(target, x_train[target_features], y_train, cv).astype('float')



target.fit(x_train[target_features], y_train)

target_x_test = target.transform(x_test[target_features]).astype('float')
import scipy





x_train = scipy.sparse.hstack([ohe_x_train, simple_x_train, target_x_train]).tocsr()

x_test = scipy.sparse.hstack([ohe_x_test, simple_x_test, target_x_test]).tocsr()
from sklearn.linear_model import LogisticRegression





logit = LogisticRegression(C=0.54321, solver='lbfgs', max_iter=10000)

logit.fit(x_train, y_train)

y_pred_logit = logit.predict_proba(x_test)[:, 1]
x_train = np.concatenate((oe_x_train, simple_x_train, target_x_train), axis=1)

x_test = np.concatenate((oe_x_test, simple_x_test, target_x_test), axis=1)
categorial_part = oe_x_train.shape[1]
import tensorflow as tf
from sklearn.metrics import roc_auc_score





def auc(y_true, y_pred):

    def fallback_auc(y_true, y_pred):

        try:

            return roc_auc_score(y_true, y_pred)

        except:

            return 0.5

    return tf.py_function(fallback_auc, (y_true, y_pred), tf.double)
def make_model(data, categorial_part):

    

    inputs = []

    

    categorial_outputs = []

    for idx in range(categorial_part):

        n_unique = np.unique(data[:,idx]).shape[0]

        n_embeddings = int(min(np.ceil(n_unique / 2), 50))

        inp = tf.keras.layers.Input(shape=(1,))

        inputs.append(inp)

        x = tf.keras.layers.Embedding(n_unique + 1, n_embeddings)(inp)

        x = tf.keras.layers.SpatialDropout1D(0.3)(x)

        x = tf.keras.layers.Reshape((n_embeddings,))(x)

        categorial_outputs.append(x)

    

    x1 = tf.keras.layers.Concatenate()(categorial_outputs)

    x1 = tf.keras.layers.BatchNormalization()(x1)

    

    inp = tf.keras.layers.Input(shape=(data.shape[1] - categorial_part,))

    inputs.append(inp)

    x2 = tf.keras.layers.BatchNormalization()(inp)

    

    x = tf.keras.layers.Concatenate()([x1, x2])

    x = tf.keras.layers.Dense(128, activation='relu')(x)

    x = tf.keras.layers.Dropout(0.3)(x)

    

    y = tf.keras.layers.Dense(1, activation='sigmoid', name='dense_output')(x)

    

    print('Expected number of inputs:', len(inputs))

    model = tf.keras.Model(inputs=inputs, outputs=y)

    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5), metrics=['accuracy', auc])

    return model
def make_inputs(data, categorial_part):

    inputs = []

    for idx in range(categorial_part):

        inputs.append(data[:, idx])

    inputs.append(data[:, categorial_part:])

    return inputs
from sklearn.model_selection import KFold

import tensorflow.keras.backend as K





n_splits = 50



trained_estimators = []

histories = []

scores = []



cv = KFold(n_splits=n_splits, random_state=42)

for train_idx, valid_idx in cv.split(x_train, y_train):

    

    x_train_train = x_train[train_idx]

    y_train_train = y_train[train_idx]

    x_train_valid = x_train[valid_idx]

    y_train_valid = y_train[valid_idx]

    

    K.clear_session()

    

    estimator = make_model(x_train, categorial_part)

    

    es = tf.keras.callbacks.EarlyStopping(monitor='val_auc', min_delta=0.001, patience=10,

                                          verbose=1, mode='max', restore_best_weights=True)

    

    rl = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=3, min_lr=1e-6, mode='max', verbose=1)

    

    history = estimator.fit(make_inputs(x_train_train, categorial_part), y_train_train, batch_size=1024, epochs=100, callbacks=[es, rl],

                            validation_data=(make_inputs(x_train_valid, categorial_part), y_train_valid))

    trained_estimators.append(estimator)

    histories.append(history)

    

    oof_part = estimator.predict(make_inputs(x_train_valid, categorial_part))

    score = roc_auc_score(y_train_valid, oof_part)

    print('Fold score:', score)

    scores.append(score)
print('Mean score:', np.mean(scores))
import matplotlib.pyplot as plt





fig, axs = plt.subplots(3, 2, figsize=(18,18))



# AUC

for h in histories:

    axs[0,0].plot(h.history['auc'], color='g')

axs[0,0].set_title('Model AUC - Train')

axs[0,0].set_ylabel('AUC')

axs[0,0].set_xlabel('Epoch')



for h in histories:

    axs[0,1].plot(h.history['val_auc'], color='b')

axs[0,1].set_title('Model AUC - Test')

axs[0,1].set_ylabel('AUC')

axs[0,1].set_xlabel('Epoch')



# accuracy

for h in histories:

    axs[1,0].plot(h.history['accuracy'], color='g')

axs[1,0].set_title('Model accuracy - Train')

axs[1,0].set_ylabel('Accuracy')

axs[1,0].set_xlabel('Epoch')



for h in histories:

    axs[1,1].plot(h.history['val_accuracy'], color='b')

axs[1,1].set_title('Model accuracy - Test')

axs[1,1].set_ylabel('Accuracy')

axs[1,1].set_xlabel('Epoch')



# loss

for h in histories:

    axs[2,0].plot(h.history['loss'], color='g')

axs[2,0].set_title('Model loss - Train')

axs[2,0].set_ylabel('Loss')

axs[2,0].set_xlabel('Epoch')



for h in histories:

    axs[2,1].plot(h.history['val_loss'], color='b')

axs[2,1].set_title('Model loss - Test')

axs[2,1].set_ylabel('Loss')

axs[2,1].set_xlabel('Epoch')



fig.show()
len(trained_estimators)
y_pred = np.zeros(x_test.shape[0])

x_test_inputs = make_inputs(x_test, categorial_part)

for estimator in trained_estimators:

    y_pred += estimator.predict(x_test_inputs).reshape(-1) / len(trained_estimators)
y_pred_keras = y_pred
y_pred = np.add(y_pred_logit, y_pred_keras) / 2
submission = pd.read_csv('../input/cat-in-the-dat-ii/sample_submission.csv', index_col='id')

submission['target'] = y_pred

submission.to_csv('logit_keras.csv')
submission.head()