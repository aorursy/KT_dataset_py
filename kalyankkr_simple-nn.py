# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

from keras.layers import Dense, BatchNormalization,Dropout

from keras.models import Sequential

from sklearn.model_selection import StratifiedKFold

from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

from sklearn.metrics import roc_auc_score

# Any results you write to the current directory are saved as output.
train_df=pd.read_csv("../input/train.csv")

test_df=pd.read_csv("../input/test.csv")
train=train_df.drop(["ID_code","target"],axis=1)

test=test_df.drop("ID_code",axis=1)

target=train_df.target
model=Sequential()

model.add(Dense(256,input_dim=train.shape[1],activation="relu"))

model.add(BatchNormalization())

model.add(Dropout(0.25))

model.add(Dense(128,activation="relu"))

model.add(BatchNormalization())

model.add(Dense(1,activation="sigmoid"))
model.compile(loss="binary_crossentropy",optimizer='adam',metrics=["accuracy"])
model.summary()
random_seed = 42

from sklearn.model_selection import train_test_split,StratifiedKFold

X_train, X_val, Y_train, Y_val = train_test_split(train, target,test_size = 0.2, random_state=random_seed)
output=model.fit(X_train,Y_train,epochs=20,batch_size=128,verbose=1,validation_data=(X_val,Y_val))
print("model accuracy: {}".format(np.mean(output.history["acc"])))

print("model validation accuracy: {}".format(np.mean(output.history["val_acc"])))
import matplotlib.pyplot as plt

plt.plot(output.history['acc'])

plt.plot(output.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(output.history['loss'])

plt.plot(output.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
from sklearn.metrics import confusion_matrix

predictions=model.predict(test)
prediction = (predictions > 0.33).astype("int64")

submission = test_df[['ID_code']].copy()

submission['target'] = predictions

submission.to_csv('submission_.csv', index=False)
confusion_matrix(target,prediction)
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

class roc_auc_callback(Callback):

    def __init__(self,training_data,validation_data):

        self.x = training_data[0]

        self.y = training_data[1]

        self.x_val = validation_data[0]

        self.y_val = validation_data[1]



    def on_train_begin(self, logs={}):

        return



    def on_train_end(self, logs={}):

        return



    def on_epoch_begin(self, epoch, logs={}):

        return



    def on_epoch_end(self, epoch, logs={}):

        y_pred = self.model.predict_proba(self.x, verbose=0)

        roc = roc_auc_score(self.y, y_pred)

        logs['roc_auc'] = roc_auc_score(self.y, y_pred)

        logs['norm_gini'] = ( roc_auc_score(self.y, y_pred) * 2 ) - 1



        y_pred_val = self.model.predict_proba(self.x_val, verbose=0)

        roc_val = roc_auc_score(self.y_val, y_pred_val)

        logs['roc_auc_val'] = roc_auc_score(self.y_val, y_pred_val)

        logs['norm_gini_val'] = ( roc_auc_score(self.y_val, y_pred_val) * 2 ) - 1



        print('\rroc_auc: %s - roc_auc_val: %s - norm_gini: %s - norm_gini_val: %s' % (str(round(roc,5)),str(round(roc_val,5)),str(round((roc*2-1),5)),str(round((roc_val*2-1),5))), end=10*' '+'\n')

        return



    def on_batch_begin(self, batch, logs={}):

        return



    def on_batch_end(self, batch, logs={}):

        return

from sklearn.metrics import roc_auc_score

from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

patience = 10

y_test  = np.zeros((len(test)))

y_train = np.zeros((len(train)))

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cvscores = []

for f_ ,(x, y) in enumerate(kfold.split(train, target)):

    print("fold: {}".format(f_))

    callbacks = [

        roc_auc_callback(training_data=(train.iloc[x], target.iloc[x]),validation_data=(train.iloc[y], target.iloc[y])),

        EarlyStopping(monitor='norm_gini_val', patience=patience, mode='max', verbose=1),

    ]

    model.fit(train.iloc[x], target.iloc[x], epochs=10, batch_size=150, verbose=1,callbacks=callbacks)



    y_val_preds = model.predict(train.iloc[y])

    y_train[y] = y_val_preds.reshape(y_val_preds.shape[0])

    y_test_preds = model.predict(test)

    y_test += y_test_preds.reshape(y_test_preds.shape[0])

    # evaluate the model

    scores = model.evaluate(train.iloc[y], target.iloc[y], verbose=0)

    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

y_test = y_test / 5   
roc_auc_score(target, y_train)
submission = test_df[['ID_code']].copy()

submission['target'] = y_test

submission.to_csv('submission_10.csv', index=False)