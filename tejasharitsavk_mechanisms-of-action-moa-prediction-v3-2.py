import pandas as pd

import numpy as np

import os

import time



from keras.layers import Dense, BatchNormalization, Dropout

from keras.models import Sequential, load_model

from keras.callbacks import ModelCheckpoint

from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss

import tensorflow as tf
seed = 0

np.random.seed(seed)

os.environ['PYTHONHASHSEED'] = str(seed)

tf.random.set_seed(seed)
root_path = "../input/lish-moa"
x_df = pd.read_csv(os.path.join(root_path, "train_features.csv"))

x_df.head()
y_df = pd.read_csv(os.path.join(root_path, "train_targets_scored.csv"))

y_df.head()
x_df.info()
x_df.describe()
y_df.info()
y_df.describe()
x_df.cp_type.describe()
x_df.cp_dose.describe()
def encode_labels(df):

    df.cp_type = [0 if i == "trt_cp" else 1 for i in df.cp_type]

    df.cp_dose = [0 if i == "D1" else 1 for i in df.cp_type]

    return df
x_df = encode_labels(x_df)

x_df.head()
x = x_df.iloc[:,1:]

y = y_df.iloc[:,1:]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

y_train = y_train.astype(np.float64)

y_test = y_test.astype(np.float64)
def moa_model(X,Y):

    model = Sequential()

    model.add(Dense(1000, activation="relu", input_dim=X))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(Dense(2048, activation="relu"))

    model.add(BatchNormalization())

#     model.add(Dropout(0.25))

    model.add(Dense(5000, activation="relu"))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(Dense(8000, activation="relu"))

    model.add(BatchNormalization())

#     model.add(Dropout(0.25))

    model.add(Dense(Y, activation="sigmoid"))

    

    adam = Adam(learning_rate=1e-04, epsilon=1e-08,decay=1e-14)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model
model = moa_model(x_train.shape[1], y_train.shape[1])
folds = 5

start = time.time()

log_losses = []

y_preds = []

for i in range(folds):

    print("Training Fold: 0{}".format(i+1))

    model_checkpoint = ModelCheckpoint("MOA_Model_02.{}.weights".format(i+1), monitor='val_loss', verbose=1,  save_best_only=True, save_weights_only=True)

    model.fit(x_train,y_train,batch_size=100, epochs=25,validation_data=(x_test,y_test), callbacks=[model_checkpoint])

    y_pred = model.predict(x_test)

    logloss = log_loss(y_test, y_pred) / 207

    print("log_loss:", logloss)

    log_losses.append(logloss)

    del model

    model = moa_model(x_train.shape[1], y_train.shape[1])

    model.load_weights("MOA_Model_02.{}.weights".format(i+1))

    y_pred = model.predict(x_test)

    y_preds.append(np.array(y_pred))

    

end = time.time()

print("Training Complete")

print("Total Training Time: {}s".format(end-start))

print("log_losses:", log_losses)
y_pred = model.predict(x_test)

print("log_loss:", log_loss(y_test, y_pred) / 207)
def mean_y_pred(y_preds):

    y_pred = np.array(y_preds[0])

    for i in y_preds[1:]:

        y_pred+=np.array(i)

        

    return y_pred/len(y_preds)
y_pred = mean_y_pred(y_preds)

print("log_loss:", log_loss(y_test, y_pred) / 207)
test_df = pd.read_csv(os.path.join(root_path, "test_features.csv"))

test_df.head()
test_df = encode_labels(test_df)

test_df.head()
def mean_model_predictions(test_df):

    y_preds = []

    for i in range(folds):

        model.load_weights("MOA_Model_02.{}.weights".format(i+1))

        y_pred = model.predict(test_df.iloc[:,1:])

        y_preds.append(y_pred)

        

    y_pred = mean_y_pred(y_preds)

    return y_pred
predictions = mean_model_predictions(test_df)

pred_df = pd.DataFrame(predictions)

pred_df.head()
pred_df.index = test_df.sig_id

pred_df.columns = y_df.columns[1:]

pred_df.head()
pred_df.to_csv("submission.csv")