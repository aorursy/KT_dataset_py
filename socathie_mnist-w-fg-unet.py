# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Softmax, Activation, Lambda, Concatenate, Dense, Flatten

from keras.utils import to_categorical

from keras.regularizers import l2

from keras import Model, Input

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.models import load_model

import keras.backend as K
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

train
test
X = [train.iloc[i,1:].values for i in range(len(train))]

X = [x.reshape(28,28,1,1) for x in X]

X = np.array(X)

X.shape
plt.imshow(X[0,:,:,0,0])

plt.show()
X_test = [test.iloc[i,:].values for i in range(len(test))]

X_test = [x.reshape(28,28,1,1) for x in X_test]

X_test = np.array(X_test)

X_test.shape
plt.imshow(X_test[0,:,:,0,0])

plt.show()
n_classes = 10

y = [train.iloc[i,0] for i in range(len(train))]

y = np.array(y)

print(np.unique(y, return_counts=True))

y = to_categorical(y, num_classes=n_classes)

y.shape
def FG_UNET(input_shape, n_classes, l2_rate):

    inputs = Input(input_shape)

    X = [Lambda(lambda x: x[:,:,:,:,i])(inputs) for i in range(input_shape[-1])]

    

    # first path

    conv_1_1 = Conv2D(200, (1,1), padding="same",

                      kernel_regularizer=l2(l2_rate), kernel_initializer="he_normal")

    relu_1_1 = Activation("relu")

    conv_1_2 = Conv2D(100, (1,1), padding="same",

                      kernel_regularizer=l2(l2_rate), kernel_initializer="he_normal")

    relu_1_2 = Activation("relu")

    conv_1_3 = Conv2D(50, (1,1), padding="same",

                      kernel_regularizer=l2(l2_rate), kernel_initializer="he_normal")

    

    out_1 = [conv_1_3(relu_1_2(conv_1_2(relu_1_1(conv_1_1(X[i]))))) for i in range(input_shape[-1])]

    

    # second path first half

    conv_2_1 = Conv2D(32, (3,3), padding="same",

                      kernel_regularizer=l2(l2_rate), kernel_initializer="he_normal")

    relu_2_1 = Activation("relu")

    conv_2_2 = Conv2D(32, (3,3), padding="same",

                      kernel_regularizer=l2(l2_rate), kernel_initializer="he_normal")

    relu_2_2 = Activation("relu")

    

    out_2a = [relu_2_2(conv_2_2(relu_2_1(conv_2_1(X[i])))) for i in range(input_shape[-1])]   

    

    # third path first half

    pool_3 = MaxPooling2D()

    conv_3_1 = Conv2D(64, (3,3), padding="same",

                      kernel_regularizer=l2(l2_rate), kernel_initializer="he_normal")

    relu_3_1 = Activation("relu")

    conv_3_2 = Conv2D(64, (3,3), padding="same",

                      kernel_regularizer=l2(l2_rate), kernel_initializer="he_normal")

    relu_3_2 = Activation("relu")

    

    out_3a = [relu_3_2(conv_3_2(relu_3_1(conv_3_1(pool_3(out_2a[i]))))) 

              for i in range(input_shape[-1])]

    

    # fourth path

    pool_4 = MaxPooling2D()

    conv_4_1 = Conv2D(128, (3,3), padding="same",

                      kernel_regularizer=l2(l2_rate), kernel_initializer="he_normal")

    relu_4_1 = Activation("relu")

    conv_4_2 = Conv2D(128, (3,3), padding="same",

                      kernel_regularizer=l2(l2_rate), kernel_initializer="he_normal")

    relu_4_2 = Activation("relu")

    up_4 = UpSampling2D()

    

    out_4 = [up_4(relu_4_2(conv_4_2(relu_4_1(conv_4_1(pool_4(out_3a[i])))))) 

             for i in range(input_shape[-1])]

    

    # third path second half

    concat_3 = Concatenate()

    conv_3_3 = Conv2D(64, (3,3), padding="same",

                      kernel_regularizer=l2(l2_rate), kernel_initializer="he_normal")

    relu_3_3 = Activation("relu")

    conv_3_4 = Conv2D(64, (3,3), padding="same",

                      kernel_regularizer=l2(l2_rate), kernel_initializer="he_normal")

    relu_3_4 = Activation("relu")

    up_3 = UpSampling2D()

    

    out_3 = [up_3(relu_3_4(conv_3_4(relu_3_3(conv_3_3(concat_3([out_3a[i], out_4[i]]))))))

             for i in range(input_shape[-1])]

    

    # second path third half

    concat_2 = Concatenate()

    conv_2_3 = Conv2D(32, (3,3), padding="same",

                      kernel_regularizer=l2(l2_rate), kernel_initializer="he_normal")

    relu_2_3 = Activation("relu")

    conv_2_4 = Conv2D(32, (3,3), padding="same",

                      kernel_regularizer=l2(l2_rate), kernel_initializer="he_normal")

    relu_2_4 = Activation("relu")

    

    out_2 = [relu_2_4(conv_2_4(relu_2_3(conv_2_3(concat_2([out_2a[i], out_3[i]]))))) 

            for i in range(input_shape[-1])]

    

    out = [Concatenate()([out_1[i], out_2[i]]) for i in range(input_shape[-1])]

    if input_shape[-1]>1:

        out = Concatenate()(out)

    else:

        out = out[0]

    out = Conv2D(n_classes, (1,1), padding="same",

                kernel_regularizer=l2(l2_rate), kernel_initializer="he_normal")(out)

    out = Flatten()(out)

    out = Dense(n_classes, activation="softmax", kernel_regularizer=l2(l2_rate), 

              kernel_initializer="glorot_uniform")(out)

    

    model = Model(inputs,out)

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])

    return model
l2_rate = 1.e-6



# training parameters

epochs = 20 # maximum number of epochs

batch_size = 32

has_early_stopping = True # use early stopping or not

val_rate = 0.1 # proportion of validation set, 0.05 means 5%, will not be used if has_early_stopping is false
model = FG_UNET((28, 28, 1,1), n_classes, l2_rate)

model.summary()
early_stopping = EarlyStopping(patience=0, verbose=1)

model_checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True, verbose=1)



if has_early_stopping:

    print("Model training with early stopping...")

    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, 

              validation_split=val_rate, callbacks=[early_stopping, model_checkpoint])

else:

    print("Model training with no early stopping...")

    history = model.fit(X, y, epochs=epochs, batch_size=batch_size,

              validation_split=val_rate, callbacks=[model_checkpoint])



model.save("model.h5")
best_model = load_model("best_model.h5")
print("Evaluating performance on all training data...")

results = best_model.evaluate(X, y, batch_size = 128)

print("train loss, train acc:", results)
print("Predicting on all available data...")

y_pred_one_hot = best_model.predict(X_test, verbose=1, batch_size=128)
y_pred = np.argmax(y_pred_one_hot, axis=-1)

print(y_pred.shape)
sub = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")

sub
for i in range(len(y_pred)):

    sub.iloc[i].Label = y_pred[i]
sub.to_csv("submission.csv", index=False)
sub