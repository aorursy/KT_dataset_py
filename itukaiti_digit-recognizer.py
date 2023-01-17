# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from keras.layers import Dense, Dropout, Flatten, Activation, Input, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential, load_model, Model
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#データの読み込み
train_df=pd.read_csv("../input/train.csv")
test_df=pd.read_csv("../input/test.csv")
train_df.shape,test_df.shape
#データの確認
train_df.isnull().any().describe()
test_df.isnull().any().describe()
#トレーニングデータとテストデータの定義
X_train=train_df.iloc[:,1:].values
y_train=train_df.iloc[:,0].values
X_test=test_df.values
X_train.shape,y_train.shape,X_test.shape
#データの正規化とカテゴライズ、reshape
X_train=X_train.reshape(-1,28,28,1)/255
X_test=X_test.reshape(-1,28,28,1)/255
y_train=to_categorical(y_train)
X_train.shape,X_test.shape
#画像例
plt.imshow(X_train[0][:,:,0])
plt.show();
#トレーニングデータとバリデーションデータの切り分け
X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.1,random_state=0)
#CNNを用いて実装
model=Sequential()

model.add(Conv2D(filters=64,kernel_size=(5,5),padding="same",kernel_initializer='he_normal',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation("relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same",kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Activation("relu"))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dense(10))
model.add(Activation("softmax"))

model.compile("adadelta",loss="categorical_crossentropy",metrics=["accuracy"])
model.summary()
history=model.fit(X_train,y_train,batch_size=200,epochs=25,verbose=1,validation_data=(X_val,y_val))

plt.plot(history.history["acc"],label="acc",ls="-",marker="o")
plt.plot(history.history["val_acc"],label="val_acc",ls="-",marker="x")
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='best')
plt.show()
#予測値の確認
pred = np.argmax(model.predict([X_test]),axis=1)
print(pred[:10])
#submit用のcsvファイルを作成
submit = pd.DataFrame()
submit["label"] = pred
imageid = []
for i in range(len(pred)):
    imageid.append(i+1)
submit["ImageId"] = imageid
submit.to_csv("result4.csv", index=False)