import pandas as pd
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
train_df.info()
test_df.info()
train_df.describe()
label = train_df["label"]
train_df = train_df.drop(["label"], axis=1)
train_df.describe()
train_df = train_df.astype('float32') / 255

test_df = test_df.astype('float32') / 255
from sklearn import model_selection
from keras.utils import to_categorical

from keras import models

from keras import layers

from keras.layers.normalization import BatchNormalization
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(train_df, label,

                                                                    test_size=0.01,stratify=label,random_state=0)
train_labels = to_categorical(train_y)

test_labels = to_categorical(valid_y)
train_x.shape
network = models.Sequential()

network.add(layers.Dense(1024,activation='relu',input_shape=(784,)))

network.add(layers.Dropout(0.4))

network.add(layers.Dense(256,activation='relu'))

network.add(layers.Dropout(0.4))

network.add(layers.Dense(64,activation='relu'))

network.add(layers.Dropout(0.4))

network.add(layers.Dense(10,activation='softmax'))
network.summary()
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
network.fit(train_x,train_labels,epochs=50,batch_size=64,validation_data=(valid_x, test_labels))
valid_loss, valid_acc = network.evaluate(valid_x, test_labels)

valid_acc
pred_test = network.predict(test_df)

ytestpred = pred_test.argmax(axis=1)
df = pd.read_csv('../input/sample_submission.csv')

df['Label'] = ytestpred

df.head()
df.to_csv('submission.csv', index=False)