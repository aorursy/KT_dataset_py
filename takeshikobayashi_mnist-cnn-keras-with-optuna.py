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
#データインポート
#import dataset
import pandas as pd
import numpy as np

import optuna
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
 
train = pd.read_csv('../input/digit-recognizer/train.csv') 
test = pd.read_csv('../input/digit-recognizer/test.csv')
print('The size of the train data:' + str(train.shape))
print('The size of the test data:' + str(test.shape))
#　教師データ、テストデータに分類して、float型に変換する。
# transform float type
X_train = (train.iloc[:,1:].values).astype('float32') 
y_train = train.iloc[:,0].values
X_test = test.values.astype('float32')
#精度確認のため、教師データをさらに教師データ、テストデータに分割
#train_test_split again
from sklearn.model_selection import train_test_split
 
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train,
                                                        y_train, 
                                                        test_size = 0.2,
                                                        train_size = 0.8,#教師データ少なくなるのが怖いので4:1で分割
                                                        stratify = y_train)
print(f'X_train2 の長さ: {len(X_train2)}')
print(f'X_test2 の長さ: {len(X_test2)}')
#reshape data 
img_rows, img_cols = 28, 28
num_classes = 10

#本番用データの前処理（28×28の行列に変換）
X_train = X_train2.reshape(X_train2.shape[0], img_rows, img_cols, 1)
X_test = X_test2.reshape(X_test2.shape[0], img_rows, img_cols, 1)

#y_trainのデータをto_categoricalで2値クラスの行列へ変換
y_train= keras.utils.to_categorical(y_train2, num_classes)

import optuna
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
img_rows, img_cols = 28, 28
num_classes = 10

#CNNモデルの定義
#define the CNN model
def create_model(num_layer, mid_units, num_filters,dropout_rate):
    
    model = Sequential()
    model.add(Conv2D(filters=num_filters[0], kernel_size=(3, 3),
                 activation="relu",
                 input_shape=(img_rows, img_cols, 1)))
    for i in range(1,num_layer):
        model.add(Conv2D(filters=num_filters[i], kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(mid_units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model
def objective(trial):
    print("Optimize Start")
    
    #セッションのクリア
    #clear_session
    keras.backend.clear_session()
    
    
    #畳込み層の数のパラメータ
    #number of the convolution layer
    num_layer = trial.suggest_int("num_layer", 2, 5)
    
    #FC層のユニット数
    #number of the unit
    mid_units = int(trial.suggest_discrete_uniform("mid_units", 100, 300, 100))
    
    #各畳込み層のフィルタ数
    #number of the each convolution layer filter
    num_filters = [int(trial.suggest_discrete_uniform("num_filter_"+str(i), 16, 128, 16)) for i in range(num_layer)]
    
    #活性化関数
    #activation = trial.suggest_categorical("activation", ["relu", "sigmoid"])
    
    #Dropout率
    #dropout_rate
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.5)
    
    #optimizer
    optimizer = trial.suggest_categorical("optimizer", ["sgd", "adam"])
    
    model = create_model(num_layer, mid_units, num_filters,dropout_rate)
    model.compile(optimizer=optimizer,
          loss="categorical_crossentropy",
          metrics=["accuracy"])
    
    history = model.fit(X_train, y_train, verbose=0, epochs=20, batch_size=128, validation_split=0.1)
    
    scores = model.evaluate(X_train, y_train)
    print('accuracy={}'.format(*scores))
    
    #検証用データに対する正答率が最大となるハイパーパラメータを求める
    return 1 - history.history["val_accuracy"][-1]
study = optuna.create_study()
study.optimize(objective, n_trials=30)
#最適化されたパラメータを確認する関数
#Function to check optimized parameters

study.best_params
#最適化後の評価値を確認する関数
#Function to check the evaluation value after optimization

study.best_value

study.trials
#再度データインポート
#Import data again

import pandas as pd
import numpy as np
 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()



 
train = pd.read_csv('../input/digit-recognizer/train.csv') #教師データ
test = pd.read_csv('../input/digit-recognizer/test.csv') #テストデータ
print('The size of the train data:' + str(train.shape))
print('The size of the test data:' + str(test.shape))

#　教師データ、テストデータに分類して、float型に変換する。
X_train_best = (train.iloc[:,1:].values).astype('float32') #ピクセルの値をfloatに変換
y_train_best = train.iloc[:,0].values
X_test_best = test.values.astype('float32') #ピクセルの値をfloatに変換

#データ前処理
img_rows, img_cols = 28, 28
num_classes = 10

#本番用データの前処理（28×28の行列に変換）
X_train_best = X_train_best.reshape(X_train_best.shape[0], img_rows, img_cols, 1)
X_test_best = X_test_best.reshape(X_test_best.shape[0], img_rows, img_cols, 1)

#y_trainのデータをto_categoricalで2値クラスの行列へ変換
y_train_best= keras.utils.to_categorical(y_train_best, num_classes)
#最適paramterのセット
#set the optimized parameter

num_filters = [48,64,96,96,128]
mid_units= 100
dropout_rate = 0.4916904519518987
optimizer = 'sgd'

#最適化後のモデル
#optimized model
model_best = Sequential()
model_best.add(Conv2D(filters=num_filters[0], kernel_size=(3, 3),activation="relu",input_shape=(img_rows, img_cols, 1)))
model_best.add(Conv2D(filters=num_filters[1], kernel_size=(3,3), padding="same", activation="relu"))
model_best.add(Conv2D(filters=num_filters[2], kernel_size=(3,3), padding="same", activation="relu"))
model_best.add(Conv2D(filters=num_filters[3], kernel_size=(3,3), padding="same", activation="relu"))
model_best.add(Conv2D(filters=num_filters[4], kernel_size=(3,3), padding="same", activation="relu"))
model_best.add(MaxPooling2D(pool_size=(2, 2)))
model_best.add(Dropout(dropout_rate))
model_best.add(Flatten())
model_best.add(Dense(mid_units))
model_best.add(Dropout(dropout_rate))
model_best.add(Dense(num_classes, activation='softmax'))
#最適化手法などを決定
#Determine optimization method
model_best.compile(loss=keras.losses.categorical_crossentropy,
              #optimizer=keras.optimizers.Adadelta(),
              optimizer=optimizer,
              metrics=['accuracy'])

#学習を開始
#start learning
hist2 = model_best.fit(X_train_best, y_train_best,
                 batch_size=128,
                 epochs=20,
                 validation_split=0.1,
                 verbose=1)

scores_best = model_best.evaluate(X_train_best, y_train_best)
#print('accuracy={}'.format(*scores_best))


#loss
plt.plot(hist2.history['loss'])
plt.plot(hist2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Accuracy
plt.figure()
plt.plot(hist2.history['accuracy'])
plt.plot(hist2.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#テストデータの予測結果を出力（predeict_classes)
#Output prediction results
y_pred = model_best.predict_classes(X_test_best)
 

my_cnn = pd.DataFrame()
imageid = []
 
for i in range(len(X_test_best)):
    imageid.append(i+1)
    
my_cnn['ImageId'] = imageid
my_cnn["label"] = y_pred
my_cnn.to_csv("./cnn_optuna.csv", index=False)

print('csv書き出し終了')