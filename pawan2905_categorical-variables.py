import warnings

warnings.filterwarnings("ignore")



import os

import gc

import joblib

import pandas as pd

import numpy as np

import tensorflow as tf

from sklearn.model_selection import StratifiedKFold

from sklearn import metrics, preprocessing

from tensorflow.keras import layers

from tensorflow.keras import optimizers

from tensorflow.keras.models import Model, load_model

from tensorflow.keras import callbacks

from tensorflow.keras import backend as K

from tensorflow.keras import utils
train = pd.read_csv("../input/cat-in-the-dat-ii/train.csv")

test = pd.read_csv("../input/cat-in-the-dat-ii/test.csv")

sample  = pd.read_csv("../input/cat-in-the-dat-ii/sample_submission.csv")
train.head()
test.loc[:,"target"]=-1
data  = pd.concat([train,test]).reset_index(drop=True)
data.shape,train.shape,test.shape
features = [f for f in train.columns if f not in ["id","target"]]
features
for feat in features:

    lbl_enc = preprocessing.LabelEncoder()

    data.loc[:,feat] = lbl_enc.fit_transform(data[feat].astype(str).fillna("-1").values)
data.head()
train = data[data.target != -1 ].reset_index(drop=True)

test = data[data.target == -1 ].reset_index(drop=True)

test_data = [test.loc[:, features].values[:, k] for k in range(test.loc[:, features].values.shape[1])]
len(test_data)
train.shape,test.shape
def get_model(df,categorical_columns):

    inputs = []

    outputs = []

    for c in categorical_columns:

        num_unique_vals = int(df[c].nunique())

        embed_dim = int(min(np.ceil(num_unique_vals/2),50))

        inp = layers.Input(shape=(1,))

        out = layers.Embedding(num_unique_vals + 1, embed_dim,name =c)(inp)

        # apply dropout here

        out = layers.Reshape(target_shape = (embed_dim,))(out)

        inputs.append(inp)

        outputs.append(out)

        

    x = layers.Concatenate()(outputs)

    x = layers.Dense(300,activation='relu')(x)

    x = layers.Dropout(0.3)(x)

    y = layers.Dense(1,activation='sigmoid')(x)

    model = Model(inputs=inputs,outputs=y)

    return model
get_model(train,features).summary()
model = get_model(train,features)

model.compile(loss = "binary_crossentropy",optimizer="adam")
model.fit([train.loc[:,f].values for f in features],train.target.values)
def auc(y_true, y_pred):

    def fallback_auc(y_true, y_pred):

        try:

            return metrics.roc_auc_score(y_true, y_pred)

        except:

            return 0.5

    return tf.py_function(fallback_auc, (y_true, y_pred), tf.double)
def create_model(data, catcols):    

    inputs = []

    outputs = []

    for c in catcols:

        num_unique_values = int(data[c].nunique())

        embed_dim = int(min(np.ceil((num_unique_values)/2), 50))

        inp = layers.Input(shape=(1,))

        out = layers.Embedding(num_unique_values + 1, embed_dim, name=c)(inp)

        out = layers.SpatialDropout1D(0.3)(out)

        out = layers.Reshape(target_shape=(embed_dim, ))(out)

        inputs.append(inp)

        outputs.append(out)

    

    x = layers.Concatenate()(outputs)

    x = layers.BatchNormalization()(x)

    

    x = layers.Dense(300, activation="relu")(x)

    x = layers.Dropout(0.3)(x)

    x = layers.BatchNormalization()(x)

    

    x = layers.Dense(300, activation="relu")(x)

    x = layers.Dropout(0.3)(x)

    x = layers.BatchNormalization()(x)

    

    y = layers.Dense(2, activation="softmax")(x)



    model = Model(inputs=inputs, outputs=y)

    return model
oof_preds = np.zeros((len(train)))

test_preds = np.zeros((len(test)))



skf = StratifiedKFold(n_splits=50)

for train_index, test_index in skf.split(train, train.target.values):

    X_train, X_test = train.iloc[train_index, :], train.iloc[test_index, :]

    X_train = X_train.reset_index(drop=True)

    X_test = X_test.reset_index(drop=True)

    y_train, y_test = X_train.target.values, X_test.target.values

    model = create_model(data, features)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auc])

    X_train = [X_train.loc[:, features].values[:, k] for k in range(X_train.loc[:, features].values.shape[1])]

    X_test = [X_test.loc[:, features].values[:, k] for k in range(X_test.loc[:, features].values.shape[1])]

    

    es = callbacks.EarlyStopping(monitor='val_auc', min_delta=0.001, patience=5,

                                 verbose=1, mode='max', baseline=None, restore_best_weights=True)



    rlr = callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5,

                                      patience=3, min_lr=1e-6, mode='max', verbose=1)

    

    model.fit(X_train,

              utils.to_categorical(y_train),

              validation_data=(X_test, utils.to_categorical(y_test)),

              verbose=1,

              batch_size=1024,

              callbacks=[es, rlr],

              epochs=100

             )

    valid_fold_preds = model.predict(X_test)[:, 1]

    test_fold_preds = model.predict(test_data)[:, 1]

    oof_preds[test_index] = valid_fold_preds.ravel()

    test_preds += test_fold_preds.ravel()

    print(metrics.roc_auc_score(y_test, valid_fold_preds))

    K.clear_session()
print("Overall AUC={}".format(metrics.roc_auc_score(train.target.values, oof_preds)))
test_preds.shape
test_preds /= 50

test_ids = test.id.values

print("Saving submission file")

submission = pd.DataFrame.from_dict({

    'id': test_ids,

    'target': test_preds

})

submission.to_csv("submission.csv", index=False)
submission.shape