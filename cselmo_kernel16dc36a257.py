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
import pandas as pd

from nltk.tokenize import TweetTokenizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
train_df=pd.read_hdf("../input/nlp-itba-2019/train_data_NLP_ITBA_2019.hdf",mode='r',key="df")

valid_df=pd.read_hdf("../input/nlp-itba-2019/valid_data_NLP_ITBA_2019.hdf",mode='r',key="df")

test_df=pd.read_hdf("../input/nlp-itba-2019/test_data_NLP_ITBA_2019.hdf",mode='r',key="df")
train_df.head()
tfidf=TfidfVectorizer(min_df=0,max_df=1.0)
X_train=tfidf.fit_transform(train_df["text_proc"].tolist())

X_val=tfidf.transform(valid_df["text_proc"].tolist())

y_train=train_df["gold_label"].tolist()

y_val=valid_df["gold_label"].tolist()
X_train.shape
import numpy as np

from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential

from keras.layers import Dense, Activation,Dropout

from keras.regularizers import l2

import keras



enc = OneHotEncoder(handle_unknown='ignore')

y_train_oh=enc.fit_transform(np.array(y_train).reshape(-1,1))

y_val_oh=enc.transform(np.array(y_val).reshape(-1,1))

reg=l2()

model = Sequential([

    Dense(100, input_shape=(X_train.shape[1],),kernel_initializer=keras.initializers.glorot_normal()),

    Dropout(0.3),

    Activation('relu'),

    Dense(3),

    Activation('softmax')])

model.summary()
from keras.optimizers import RMSprop,Adam,Nadam

from keras.callbacks import ModelCheckpoint, EarlyStopping

#opt=RMSprop()

opt=Nadam()

checkpoint = ModelCheckpoint("best-RP.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')

es = EarlyStopping(monitor='val_acc', patience=10)

model.compile(optimizer=opt,

              loss='categorical_crossentropy',

              metrics=['acc'])
model.fit(X_train, y_train_oh, epochs=1000, batch_size=1024,validation_data=[X_val,y_val_oh],callbacks=[checkpoint, es])
model.load_weights("best-RP.hdf5")
X_test=tfidf.transform(test_df["text_proc"].tolist())
y_pred=model.predict(X_test).argmax(axis=-1)
y_pred
enc.categories_
submissions=list()

for pred in y_pred:

    submissions.append(enc.categories_[0][pred])
submissions[0:3]
test_df.head()
test_df["gold_label"]=submissions
test_df.head()
output_df=test_df.drop(columns=["text","text_proc"])

output_df.head()
output_df.to_csv("submission_NN.csv")