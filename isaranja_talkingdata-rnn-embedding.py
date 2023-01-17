# import required libraries
import numpy as np 
import pandas as pd 

from sklearn import metrics
from sklearn import preprocessing

from keras.layers import Input, Embedding, Dense, Dropout, concatenate, Flatten
from keras.models import Model

import os
dtypes = {
        'ip'             : 'uint32',
        'app'            : 'uint16',
        'device'         : 'uint16',
        'os'             : 'uint16',
        'channel'        : 'uint16',
        'is_attributed'  : 'uint8',
        }
train_df = pd.read_csv("../input/train.csv", nrows=1000000,usecols=['ip','app','device','os', 'channel', 'is_attributed'],dtype=dtypes)

le_ip = preprocessing.LabelEncoder()
le_os = preprocessing.LabelEncoder()
le_dev = preprocessing.LabelEncoder()
le_ch = preprocessing.LabelEncoder()
le_app = preprocessing.LabelEncoder()

max_ip  = np.max(le_ip.fit_transform(train_df.ip))+1
max_dev = np.max(le_dev.fit_transform(train_df.device))+1
max_os  = np.max(le_os.fit_transform(train_df.os))+1
max_ch  = np.max(le_ch.fit_transform(train_df.channel))+1
max_app = np.max(le_app.fit_transform(train_df.app))+1
X_train = {
        'ip': np.array(le_ip.transform(train_df.ip)),
        'os': np.array(le_os.transform(train_df.os)),
        'dev': np.array(le_dev.transform(train_df.device)),
        'ch': np.array(le_ch.transform(train_df.channel)),
        'app': np.array(le_app.transform(train_df.app))
    }
y_train = train_df.is_attributed
emb_n = 10
dense_n = 50

in_ip = Input(shape=[1], name = 'ip')
emb_ip = Embedding(max_ip, emb_n)(in_ip)
in_os = Input(shape=[1], name = 'os')
emb_os = Embedding(max_os, emb_n)(in_os)
in_dev = Input(shape=[1], name = 'dev')
emb_dev = Embedding(max_dev, emb_n)(in_dev)
in_ch = Input(shape=[1], name = 'ch')
emb_ch = Embedding(max_ch, emb_n)(in_ch)
in_app = Input(shape=[1], name = 'app')
emb_app = Embedding(max_app, emb_n)(in_app)
in_dy = Input(shape=[1], name = 'dy')
in_hr = Input(shape=[1], name = 'hr')
in_wd = Input(shape=[1], name = 'wd')
              
x = concatenate([(emb_ip),(emb_app), (emb_ch), (emb_dev), emb_os ])

x = Flatten()(x)
x = Dropout(0.2)(Dense(dense_n,activation='relu')(x))
x = Dropout(0.2)(Dense(dense_n,activation='relu')(x))
x = Dropout(0.2)(Dense(dense_n,activation='relu')(x))
outp = Dense(1,activation='sigmoid')(x)

model = Model(inputs=[in_ip,in_app,in_ch,in_dev,in_os], outputs=outp)

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
#training
batch_size = 1024
class_weight = {0: 1.,
                1: 50.}
validation_split=0.95
epochs=20
model.fit(X_train, y_train, batch_size=batch_size,validation_split=validation_split, epochs=epochs)
# evaluating
dtypes = {
        'ip'             : 'uint32',
        'app'            : 'uint16',
        'device'         : 'uint16',
        'os'             : 'uint16',
        'channel'        : 'uint16',
        'is_attributed'  : 'uint8',
        }
test_df = pd.read_csv("../input/train.csv", nrows=100000,usecols=['ip','app','device','os', 'channel', 'is_attributed'],dtype=dtypes)
X_test = {
        'ip': np.array(le_ip.transform(test_df.ip)),
        'os': np.array(le_os.transform(test_df.os)),
        'dev': np.array(le_dev.transform(test_df.device)),
        'ch': np.array(le_ch.transform(test_df.channel)),
        'app': np.array(le_app.transform(test_df.app))
    }
y_test = test_df.is_attributed

y_pred = model.predict(X_test)
# accuracy
cm = metrics.confusion_matrix(y_test, y_pred > 0.5)
print(cm)
# AUC
fpr, tpr, thresholds = metrics.roc_curve(y_test.values+1, y_pred, pos_label=2)
metrics.auc(fpr, tpr)
