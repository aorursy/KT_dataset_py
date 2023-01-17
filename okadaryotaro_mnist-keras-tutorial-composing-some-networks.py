import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import tensorflow as tf

import tensorflow.keras as keras

from tensorflow.keras.callbacks import EarlyStopping



import random

import inspect



import matplotlib.pyplot as plt

%matplotlib inline
train_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

x_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



x_train = np.array(train_df.drop(columns='label'))

y_train = np.array(train_df['label'])

x_train, x_test = x_train / 255.0, x_test / 255.0

print(x_train.shape)
#import random # 読み込み済み



def reset_seeds(seed_num=0):

    np.random.seed(seed_num)

    random.seed(seed_num)

    tf.random.set_seed(seed_num)

    print("RANDOM SEEDS RESET")
# 乱数のリセット

reset_seeds()



#モデルの定義

model = keras.models.Sequential([

  keras.layers.Dense(128, activation='relu'),

  keras.layers.Dropout(0.2),

  keras.layers.Dense(10, activation='softmax'),

])



#モデルのコンパイル

model.compile(optimizer='adam',

  loss='sparse_categorical_crossentropy',

  metrics=['accuracy'])



#学習

#model.fit(x_train, y_train, epochs=3) # こう書いても学習はされるが、下のようにして戻り値を変数に入れる

result = model.fit(x_train, y_train, epochs=3)
result.history.keys()
#import matplotlib.pyplot as plt # 読み込み済み

#%matplotlib inline # 読み込み済み



epochs = 3

plt.plot(range(1, epochs+1), result.history['accuracy'], label="training")

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
def model_1(epochs=5):

    # 乱数のリセット

    reset_seeds()



    #モデルの定義

    model = keras.models.Sequential([

      keras.layers.Dense(128, activation='relu'),

      keras.layers.Dropout(0.2),

      keras.layers.Dense(10, activation='softmax'),

    ])



    #モデルのコンパイル

    model.compile(optimizer='adam',

      loss='sparse_categorical_crossentropy',

      metrics=['accuracy'])



    #学習

    result = model.fit(x_train, y_train, epochs=epochs, validation_split=0.2) # ここ！

    

    return(result, model)
epochs = 3

result , model = model_1(epochs)

result.history.keys()
def plot_result(result,epochs):

    plt.plot(range(1, epochs+1), result.history['accuracy'], label="training")

    plt.plot(range(1, epochs+1), result.history['val_accuracy'], label="val")

    plt.xlabel('Epochs')

    plt.ylabel('Accuracy')

    plt.legend()

    plt.show()
plot_result(result,epochs)
#from tensorflow.keras.callbacks import EarlyStopping # 読み込み済み

 

def model_2(epochs=100):

    # 乱数のリセット

    reset_seeds()



    #モデルの定義

    model = keras.models.Sequential([

      keras.layers.Dense(128, activation='relu'),

      keras.layers.Dropout(0.2),

      keras.layers.Dense(10, activation='softmax'),

    ])



    #モデルのコンパイル

    model.compile(optimizer='adam',

      loss='sparse_categorical_crossentropy',

      metrics=['accuracy'])





    # Early-stopping 

    # デフォルトでは val_loss の値を監視する

    early_stopping = EarlyStopping(patience=0, verbose=1)

    #early_stopping = EarlyStopping(monitor='val_loss', patience=0, verbose=1) # これでも同じ。

    

    #学習

    result = model.fit(x_train, y_train,

                        epochs=epochs,

                        validation_split=0.2,

                        callbacks=[early_stopping]) # ここ！

    

    return(result, model)
max_epochs = 100

result , model = model_2(max_epochs)

epochs = len(result.history['val_loss'])



plot_result(result,epochs)
def model_3(epochs=100):

    # 乱数のリセット

    reset_seeds()



    #モデルの定義

    model = keras.models.Sequential([

      keras.layers.Dense(10, activation='softmax'),

    ])



    #モデルのコンパイル

    model.compile(optimizer='SGD',

      loss='sparse_categorical_crossentropy',

      metrics=['accuracy'])





    # Early-stopping 

    early_stopping = EarlyStopping(patience=0, verbose=1)

    

    #学習

    result = model.fit(x_train, y_train,

                        epochs=epochs,

                        validation_split=0.2,

                        callbacks=[early_stopping])

    

    return(result, model)
result , model = model_3()

epochs = len(result.history['val_loss'])



plot_result(result,epochs)
def model_4(epochs=100):

    # 乱数のリセット

    reset_seeds()



    #モデルの定義

    model = keras.models.Sequential([

      keras.layers.Dense(10, activation='relu'),

    ])



    #モデルのコンパイル

    model.compile(optimizer='SGD',

      loss='sparse_categorical_crossentropy',

      metrics=['accuracy'])





    # Early-stopping 

    early_stopping = EarlyStopping(patience=0, verbose=1)

    

    #学習

    result = model.fit(x_train, y_train,

                        epochs=epochs,

                        validation_split=0.2,

                        callbacks=[early_stopping])

    

    return(result, model)
result , model = model_4()

epochs = len(result.history['val_loss'])



plot_result(result,epochs)
def model_5(epochs=100):

    # 乱数のリセット

    reset_seeds()



    #モデルの定義

    model = keras.models.Sequential([

      keras.layers.Dense(100, activation='relu'),

      keras.layers.Dense(100, activation='relu'),

      keras.layers.Dense(10, activation='softmax'),

    ])



    #モデルのコンパイル

    model.compile(optimizer='SGD',

      loss='sparse_categorical_crossentropy',

      metrics=['accuracy'])





    # Early-stopping 

    early_stopping = EarlyStopping(patience=0, verbose=1)

    

    #学習

    result = model.fit(x_train, y_train,

                        epochs=epochs,

                        validation_split=0.2,

                        callbacks=[early_stopping])

    

    return(result, model)
result , model = model_5()

epochs = len(result.history['val_loss'])



plot_result(result,epochs)
def model_6(epochs=100):

    # 乱数のリセット

    reset_seeds()



    #モデルの定義

    model = keras.models.Sequential([

      keras.layers.Dense(100, activation='relu'),

      keras.layers.Dropout(0.2),

      keras.layers.Dense(100, activation='relu'),

      keras.layers.Dropout(0.2),

      keras.layers.Dense(10, activation='softmax'),

    ])



    #モデルのコンパイル

    model.compile(optimizer='SGD',

      loss='sparse_categorical_crossentropy',

      metrics=['accuracy'])





    # Early-stopping 

    early_stopping = EarlyStopping(patience=0, verbose=1)

    

    #学習

    result = model.fit(x_train, y_train,

                        epochs=epochs,

                        validation_split=0.2,

                        callbacks=[early_stopping])

    

    return(result, model)
result , model = model_6()

epochs = len(result.history['val_loss'])



plot_result(result,epochs)
def model_7(epochs=100):

    # 乱数のリセット

    reset_seeds()



    #モデルの定義

    model = keras.models.Sequential([

        keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)),

        keras.layers.MaxPool2D(pool_size=(2,2)),

        keras.layers.Flatten(),

        keras.layers.Dropout(0.2),

        keras.layers.Dense(100, activation='relu'),

        keras.layers.Dropout(0.2),

        keras.layers.Dense(10, activation='softmax'),

    ])



    #モデルのコンパイル

    model.compile(optimizer='adam',

      loss='sparse_categorical_crossentropy',

      metrics=['accuracy'])





    # Early-stopping 

    early_stopping = EarlyStopping(patience=0, verbose=1)

    

    # 学習データを2次元に変換

    x_train_2 = x_train.reshape(-1,28,28,1)

    

    #学習

    result = model.fit(x_train_2, y_train,

                        epochs=epochs,

                        validation_split=0.2,

                        callbacks=[early_stopping])

    

    return(result, model)
result , model = model_7()

epochs = len(result.history['val_loss'])



plot_result(result,epochs)
#import inspect # 読み込み済み

for a in inspect.getmembers(keras.layers):

    print(a[0])
#sgd = optimizers.SGD(lr=0.01) #オプティマイザーを定義しておく。 lrが学習率

#model.compile(optimizer=sgd) # コンパイル時に上で定めた変数を指定
#ここにコードを書く
# CNN以外の場合

#out = model.predict(x_test)



# CNNの場合

x_test_2 = x_test.values.reshape(-1,28,28,1)

out = model.predict(x_test_2)





# 以下、共通部

y_test = [x.argmax() for x in out]

submit_df = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

submit_df['Label'] = y_test

submit_df.to_csv('submission.csv', index=False)

print("done.")