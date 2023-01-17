# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

from sklearn.model_selection import train_test_split

import keras

from keras import layers

from keras.callbacks import EarlyStopping

from keras.layers import Dense,Dropout, Activation

from keras.models import Model, Sequential

from keras.optimizers import Adam

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
closed= pd.read_csv("../input/archive/Closed.csv")

notclosed= pd.read_csv("../input/archive/NotClosed.csv")
frame = closed.append(notclosed, ignore_index=True)
frame.loc[frame['ClosedDate'].notnull(), 'ClosedDate'] =1

frame['ClosedDate'].fillna(0, inplace=True)



frame.loc[frame['AcceptedAnswerId'].notnull(), 'AcceptedAnswerId'] =1

frame['AcceptedAnswerId'].fillna(0, inplace=True)



frame.loc[frame['DeletionDate'].notnull(), 'DeletionDate'] =1

frame['DeletionDate'].fillna(0, inplace=True)



frame['ViewCount'].fillna(0, inplace=True)



frame.loc[frame['LastEditDate'].notnull(), 'LastEditDate'] =1

frame['LastEditDate'].fillna(0, inplace=True)



frame.loc[frame['Title'].notnull(), 'Title'] =1

frame['Title'].fillna(0, inplace=True)



frame.loc[frame['Tags'].notnull(), 'Tags'] =1

frame['Tags'].fillna(0, inplace=True)



frame['AnswerCount'].fillna(0, inplace=True)



frame.loc[frame['ProfileImageUrl'].notnull(), 'ProfileImageUrl'] =1

frame['ProfileImageUrl'].fillna(0, inplace=True)
frame.Score.fillna(0, inplace=True)



frame.CommentCount.fillna(0, inplace=True)



frame['PosterReputation'].fillna(0, inplace=True)

frame['DayDiff'].fillna(0, inplace=True)

frame['BodyLength'].fillna(0, inplace=True)



frame.UpVotes.fillna(0, inplace=True)

frame.DownVotes.fillna(0, inplace=True)

frame =frame.sample(frac=1).reset_index(drop=True)
y = frame.ClosedDate

frame.drop(['ClosedDate'], inplace=True, axis=1)
X = frame
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
model = Sequential()

model.add(Dense(512, input_shape=(15,), activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(1))

model.add(Activation('sigmoid'))



early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1)

model.compile(loss='binary_crossentropy',

              optimizer=Adam(lr=.001),

              metrics=['accuracy'])

history = model.fit(X_train, y_train,

                    batch_size=128,

                    epochs=30,

                    verbose=1,

                    validation_split=.2,

                    callbacks=[early_stopping])