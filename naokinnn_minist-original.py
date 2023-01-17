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
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
tf.__version__
from tensorflow.keras import layers 
import tensorflow.keras.backend as K
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout, Reshape, Input
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax, Nadam, RMSprop, Ftrl 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
device = torch.device("cuda")
df_train = pd.read_csv("../input/digit-recognizer/train.csv")
df_test = pd.read_csv("../input/digit-recognizer/test.csv")
df_sub = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
df_Id = df_sub["ImageId"]
X = df_train.drop("label", axis=1).values
y = df_train["label"].values
X_test = df_test.values
X = X / 255
X_test = X_test / 255
X = X.reshape((-1, 1, 28, 28))
X_test = X_test.reshape((-1, 1, 28, 28))
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=123)
t_X_train = torch.from_numpy(X_train).float()
t_y_train = torch.from_numpy(y_train).long()
t_X_val = torch.from_numpy(X_val).float()
t_y_val = torch.from_numpy(y_val).long()

dataset_train = TensorDataset(t_X_train, t_y_train)
dataset_val = TensorDataset(t_X_val, t_y_val)

loader_train = DataLoader(dataset_train, batch_size=20, shuffle=True)
loader_val = DataLoader(dataset_val, batch_size=20) 
activation_relu = torch.nn.ReLU()

class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv7 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv8 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        
        self.drop1 = nn.Dropout(p=0.5)
        
        self.fc1 = nn.Linear(7*7*64, 1024)
        
        self.drop2 = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(1024, 10)
        
    
    def forward(self, x):
        x = activation_relu(self.conv1(x))
        x = activation_relu(self.conv2(x))
        x = activation_relu(self.conv3(x))
        x = activation_relu(self.conv4(x))
        
        x = activation_relu(self.maxpool1(x))
        
        x = activation_relu(self.conv5(x))
        x = activation_relu(self.conv6(x))
        x = activation_relu(self.conv7(x))
        x = activation_relu(self.conv8(x))
        
        x = activation_relu(self.maxpool2(x))
        
        x = self.flatten(x)
        
        x = self.drop1(x)
        
        x = activation_relu(self.fc1(x))
        
        x = self.drop2(x)
        
        x = self.fc2(x)
    
        return x
    

cnn = cnn()
device = torch.device("cuda")
cnn.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.0002)
train_history = []
val_history = []

for epoch in range(10):
    
    total_train_loss = 0
    total_val_loss = 0
    
    total_train_acc = 0
    total_val_acc = 0
    
    total_train = 0
    total_val = 0
    
    for inputs, labels in loader_train:
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        cnn.train()
        
        optimizer.zero_grad()
        
        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss
        
        with torch.no_grad():
            y_pred = torch.argmax(cnn(inputs), dim=1)
            acc_train = int((y_pred==labels).sum())
            
            total_train_acc += acc_train
            
        total_train += len(labels)
            
    for inputs, labels in loader_val:
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        cnn.eval()
        
        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        
        total_val_loss += loss
        
        with torch.no_grad():
            y_pred = torch.argmax(cnn(inputs), dim=1)
            total_acc_val = int((y_pred==labels).sum())
        
        total_val += len(labels)
    
    avg_train_acc = total_train_acc / total_train
    avg_val_acc = total_acc_val / total_val
    
    train_history.append([total_train_loss, total_train_acc])
    val_history.append([total_val_loss, total_val_acc])
    
    print(f"Epoch {epoch+1} ---> train_loss: {total_train_loss:.5f}  |  val_loss: {total_val_loss:.5f}   ||   "\
         f"train_acc {avg_train_acc:.5f}  |  val_acc {avg_val_acc:.5f}")
x = torch.tensor([[1, 2, 3, 4, 5],
                [2, 2, 2, 2, 2]])
y = torch.tensor([1, 3, 4, 4, 5])
type(len(x))
cnn1 = tf.keras.models.Sequential([Conv2D(input_shape=(28, 28, 1), filters=16, kernel_size=(3, 3), activation="relu", padding="same"),
                                   
                                   Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same"),
                                   
                                   Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same"),
                                   
                                   Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same"),
                                   
                                   MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                                   
                                   Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same"),
                                   
                                   Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same"),
                                   
                                   Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same"),
                                   
                                   Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same"),
                                   
                                   MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                                   
                                   Flatten(),
                                   
                                   Dense(units=1024, activation="relu"),
                                  
                                   Dropout(rate=0.5),
                                   
                                   Dense(units=10, activation="softmax")])
def seq_train_val(X, y, model, optimizer=Adam, validation_data=None, learning_rate=0.0001, loss="categorical_crossentropy", batch_size=10, epochs=40):
    
    model.summary()
    
    learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min", patiencs=5, verbose=1, factor=0.5, min_lr=0.000001)
    
    model.compile(optimizer=optimizer(learning_rate=learning_rate), loss=loss, metrics=["acc"])
    
    model.fit(X_train, y_train, validation_data=validation_data, batch_size=batch_size, verbose=2, epochs=epochs, callbacks=[learning_rate_reduction])
    
    return None
    

def predict(X, model):
    
    y_pred = model.predict(X)
    y_pred = np.argmax(y_pred, axis=1)
    
    return y_pred
seq_train_val(X=X_train_std, y=y_train, validation_data=(X_val_std, y_val), model=cnn1)
validation(X_val_std, y_val, model=cnn)
y_pred = predict(X_test_std, model=cnn)
y_pred = pd.DataFrame(y_pred, columns=["Label"])
sub = pd.concat([df_Id, y_pred], axis=1)
sub.to_csv("submission.csv", index=False)
ac_relu = layers.Activation("relu")
ac_softmax = layers.Activation("softmax")

class cnn2(tf.keras.Model):
    
    def __init__(self, *args, **kwargs):
        super(cnn2, self).__init__(*args, **kwargs)
              
        self.conv1 = Conv2D(filters=8, kernel_size=(3, 3), padding="same")
        self.conv2 = Conv2D(filters=8, kernel_size=(3, 3), padding="same")
        self.conv3 = Conv2D(filters=8, kernel_size=(3, 3), padding="same")
        
        self.maxpool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")
        
        self.conv4 = Conv2D(filters=16, kernel_size=(3, 3), padding="same")
        self.conv5 = Conv2D(filters=16, kernel_size=(3, 3), padding="same")
        self.conv6 = Conv2D(filters=16, kernel_size=(3, 3), padding="same")

        self.maxpool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")
        
        self.conv7 = Conv2D(filters=32, kernel_size=(3, 3), padding="same")
        self.conv8 = Conv2D(filters=32, kernel_size=(3, 3), padding="same")
        self.conv9 = Conv2D(filters=32, kernel_size=(3, 3), padding="same")
        self.conv10 = Conv2D(filters=32, kernel_size=(3, 3), padding="same")
        
        self.maxpool3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        
        self.conv11 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")
        self.conv12 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")
        self.conv13 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")
        self.conv14 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")
        
        self.maxpool4 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        
        self.flatten = Flatten()
        
        self.dense1 = Dense(units=1024)
        
        self.drop = Dropout(rate=0.5)
        
        self.dense2 = Dense(units=10)
        
        
    def call(self, inputs, training=None):

        x2 = ac_relu(self.conv1(inputs))
        x3 = ac_relu(self.conv2(x2))
        x4 = ac_relu(self.conv3(x3))
        
        x5 = ac_relu(self.maxpool1(x4))
        
        x6 = ac_relu(self.conv4(x5))
        x7 = ac_relu(self.conv5(x6))
        x8 = ac_relu(self.conv6(x7))
        
        x9 = ac_relu(self.maxpool2(x8))
        
        x10 = ac_relu(self.conv7(x9))
        x11 = ac_relu(self.conv8(x10))
        x12 = ac_relu(self.conv9(x11))
        x13 = ac_relu(self.conv10(x12))
        
        x14 = ac_relu(self.maxpool3(x13))
        
        x15 = ac_relu(self.conv11(x14))
        x16 = ac_relu(self.conv12(x15))
        x17 = ac_relu(self.conv13(x16))
        x18 = ac_relu(self.conv14(x17))
        
        x19 = ac_relu(self.maxpool4(x18))
        
        x20 = self.flatten(x19)
        
        x21 = ac_relu(self.dense1(x20))
        
        x22 = self.drop(x21)
        
        outputs = ac_softmax(self.dense2(x22))
        
        return outputs 
    
cnn2 = cnn2()
def sub_train_val(X_train, y_train, X_val, y_val, model, optimizer=Adam, learning_rate=0.0001, loss_object=CategoricalCrossentropy(),
             batch_size=20, y_label=10, epochs=5):
    
    X_trainn = X_train.astype("float32")
    if y_label:
        y_trainn = tf.one_hot(y_train, 10, dtype="float32")
    else:
        y_trainn = y_train.astype("float32")
    train_sliced = tf.data.Dataset.from_tensor_slices((X_trainn, y_trainn))
    train_dataset = train_sliced.shuffle(100000).batch(batch_size)

    X_vall = X_val.astype("float32")
    if y_label:
        y_vall = tf.one_hot(y_val, 10, dtype="float32")
    else:
        y_vall = y_vall.astype("float32")
    val_sliced = tf.data.Dataset.from_tensor_slices((X_vall, y_vall))
    val_dataset = val_sliced.batch(batch_size)

    optimizer = optimizer(learning_rate=learning_rate)
    loss_object = loss_object
    
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    val_loss = tf.keras.metrics.Mean(name="val_loss")
    
    train_loss_list = []
    val_loss_list = []

    @tf.function
    def train_step(X_train, y_train, model, optimizer, loss_object, train_loss):
    
        training = True
        K.set_learning_phase(training)
    
        with tf.GradientTape() as tape:
        
            y_pred_tra = model(X_train, training=training)
        
            loss = loss_object(y_train, y_pred_tra)
        
            gradient = tape.gradient(loss, model.trainable_weights)
        
            optimizer.apply_gradients(zip(gradient, model.trainable_weights))
        
            train_loss(loss)
        

    @tf.function
    def val_step(X_val, y_val, model, optimizer, loss_object, val_loss):
    
        training = False
    
        y_pred_val = model(X_val, training=training)
    
        loss = loss_object(y_val, y_pred_val)
    
        val_loss(loss)
    
    
    for epoch in range(epochs):
        
        print(f"Epoch is {str(epoch+1).zfill(3)}  ----->", end="  ")
        
        for X, y in train_dataset:
            train_step(X, y, model=model, optimizer=optimizer, loss_object=loss_object, train_loss=train_loss)
        
        print("train_loss: {:.6f}".format(float(train_loss.result())), end="   |  ")
        
        train_loss_list.append(float(train_loss.result()))
        
        train_loss.reset_states()
    
    
        for X, y in val_dataset:
            val_step(X, y, model=model, optimizer=optimizer, loss_object=loss_object, val_loss=val_loss)
        
        print("val_loss: {:.6f}".format(float(val_loss.result())))
        
        val_loss_list.append(float(val_loss.result()))
    
        val_loss.reset_states()
    
    print()
    print("-"*100)
    print()
    print("Best val_loss is {:.6f}, and when the epoch is {}".format(min(val_loss_list), epoch+1))

    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, epochs+1), train_loss_list, marker="o", c="b", label="train_loss")
    plt.plot(range(1, epochs+1), val_loss_list, marker="v", c="r", label="validation_loss")
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("train_and_validation_loss")
train_val(X_train=X_train_std, y_train=y_train, X_val=X_val_std, y_val=y_val, model=cnn2, learning_rate=0.0001, epochs=40, optimizer=Adam)
def validate(X, y, model):
    X_tensor = tf.convert_to_tensor(X)
    y_pred = model(X_tensor)
    y_pred = y_pred.numpy()
    y_pred = np.argmax(y_pred, axis=1)
    acc = accuracy_score(y, y_pred)
    
    return acc


def predict(X, model):
    
    X_tensor = tf.convert_to_tensor(X)
    y_pred = model(X_tensor)
    y_pred = y_pred.numpy()
    y_pred = np.argmax(y_pred, axis=1)
    
    return y_pred
validate(X_train_std, y_train, cnn2)
import torch
import numpy as np
x = torch.empty(2, 3)
print(x)
x = torch.rand(2, 3)
print(x)
x = torch.zeros(2, 3, dtype=torch.float)
print(x)
x = torch.ones(2, 3, dtype=torch.int)
print(x)
x = torch.tensor([[0.0, 0.1, 0.2],
                 [1.0, 1.1, 1.2]])
print(x)
y = x.new_ones(1, 2)
print(y)
y = torch.ones_like(x, dtype=torch.int)
print(y)
x.size()
x[0, 1]
x[:2, 1:]
x[0, 1].item()
# PyTorchテンソルを、NumPy多次元配列に変換
b = x.numpy()    # 「numpy()」を呼び出すだけ。以下は注意点（メモリ位置の共有）

# ※PyTorchテンソル側の値を変えると、NumPy多次元配列値「b」も変化する（トラックされる）
print ('PyTorch計算→NumPy反映：')
print(b); x.add_(y); print(b)           # PyTorch側の計算はNumPy側に反映
print ('NumPy計算→PyTorch反映：')
print(x); np.add(b, b, out=b); print(x) # NumPy側の計算はPyTorch側に反映

# -----------------------------------------
# NumPy多次元配列を、PyTorchテンソルに変換
c = np.ones((2, 3), dtype=np.float64) # 2行3列の多次元配列値（1で初期化）を生成
d = torch.from_numpy(c)  # 「torch.from_numpy()」を呼び出すだけ

# ※NumPy多次元配列値を変えると、PyTorchテンソル「d」も変化する（トラックされる）
print ('NumPy計算→PyTorch反映：')
print(d); np.add(c, c, out=c); print(d)  # NumPy側の計算はPyTorch側に反映
print ('PyTorch計算→NumPy反映：')
print(c); d.add_(y); print(c)            # PyTorch側の計算はNumPy側に反映
if torch.cuda.is_available():              # CUDA（GPU）が利用可能な場合
    print('CUDA（GPU）が利用できる環境')
    print(f'CUDAデバイス数： {torch.cuda.device_count()}')
    print(f'現在のCUDAデバイス番号： {torch.cuda.current_device()}')  # ※0スタート
    print(f'1番目のCUDAデバイス名： {torch.cuda.get_device_name(0)}') # 例「Tesla T4」  

    device = torch.device("cuda")          # デフォルトのCUDAデバイスオブジェクトを取得
    device0 = torch.device("cuda:0")       # 1番目（※0スタート）のCUDAデバイスを取得

    # テンソル計算でのGPUの使い方は主に3つ：
    g = torch.ones(2, 3, device=device)    # （1）テンソル生成時のパラメーター指定
    g = x.to(device)                       # （2）既存テンソルのデバイス変更
    g = x.cuda(device)                     # （3）既存テンソルの「CUDA（GPU）」利用
    f = x.cpu()                            # （3'）既存テンソルの「CPU」利用

    # ※（2）の使い方で、GPUは「.to("cuda")」、CPUは「.to("cpu")」と書いてもよい
    g = x.to("cuda")
    f = x.to("cpu")

    # ※（3）の引数は省略することも可能
    g = x.cuda()

    # 「torch.nn.Module」オブジェクト（model）全体でのGPU／CPUの切り替え
    model.cuda()  # モデルの全パラメーターとバッファーを「CUDA（GPU）」に移行する
    model.cpu()   # モデルの全パラメーターとバッファーを「CPU」に移行する
else:
    print('CUDA（GPU）が利用できない環境')
!pip install playground-data

# playground-dataライブラリのplygdataパッケージを「pg」という別名でインポート
import plygdata as pg

# 設定値を定数として定義
PROBLEM_DATA_TYPE = pg.DatasetType.ClassifyCircleData # 問題種別：「分類（Classification）」、データ種別：「円（CircleData）」を選択
TRAINING_DATA_RATIO = 0.5  # データの何％を訓練【Training】用に？ (残りは精度検証【Validation】用) ： 50％
DATA_NOISE = 0.0           # ノイズ： 0％

# 定義済みの定数を引数に指定して、データを生成する
data_list = pg.generate_data(PROBLEM_DATA_TYPE, DATA_NOISE)

# データを「訓練用」と「精度検証用」を指定の比率で分割し、さらにそれぞれを「データ（X）」と「教師ラベル（y）」に分ける
X_train, y_train, X_valid, y_valid = pg.split_data(data_list, training_size=TRAINING_DATA_RATIO)

# データ分割後の各変数の内容例として、それぞれ5件ずつ出力
print('X_train:'); print(X_train[:5])
print('y_train:'); print(y_train[:5])
print('X_valid:'); print(X_valid[:5])
print('y_valid:'); print(y_valid[:5])