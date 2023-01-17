import random
import numpy as np
import pandas as pd
import tensorflow as tf

# Parameters
learning_rate = 0.01 # 学習率 高いとcostの収束が早まる
training_epochs = 100 # 学習全体をこのエポック数で区切り、区切りごとにcostを表示する
batch_size = 50     # 学習1回ごと( sess.run()ごと )に訓練データをいくつ利用するか
display_step = 1     # 1なら毎エポックごとにcostを表示
#train_size = 8000     # 全データの中でいくつ訓練データに回すか
step_size = 1000     # 何ステップ学習するか

# Network Parameters
n_hidden_1 = 64      # 隠れ層1のユニットの数
n_hidden_2 = 64      # 隠れ層2のユニットの数
n_input =  14         # 与える変数の数
n_classes = 4        # 分類するクラスの数 今回は最大18頭


df_train = pd.read_csv("train.csv")
df_simu = pd.read_csv("test.csv")

x_train = np.array(df_train[[]].fillna(0))
x_test = np.array(df_simu[['Kyori', 'BabaDiv', 'BabaCD', 'SyussoTosu', 'SayuDiv', 'Wakuban', 'Umaban', 'Futan',
                   'MinaraiCD', 'TanOdds', 'TanNinki', 'FukuOddsLow', 'FukuOddsHigh', 'FukuNinki']].fillna(0)) 

dtr = df_train["ResKakuteiJyuni"]
dte = df_simu["ResKakuteiJyuni"]

#1位だったら[1,0,0 ...,0]、2位だったら[0,1,0...,0]、18位だったら[0,0,0...,1]の行列を作る
#後に使う最尤推定法のため
a = np.zeros([len(dtr),4])
for i in range(len(dtr)):
    if dtr[i] >= 4:
        a[i][3] = 1
        continue
    for j in range(1,4):
        if dtr[i] == j:
            a[i][j-1] = 1
            break

y_train = a
b = np.zeros([len(dte),4])
for i in range(len(dtr)):
    if dtr[i] >= 4:
        a[i][3] = 1
        continue
    for j in range(1,4):
        if dtr[i] == j:
            a[i][j-1] = 1
            break
y_test = b

x = tf.placeholder("float", [None, n_input])#入力する変数は15種
y = tf.placeholder("float", [None, n_classes])#出力は1－18位の18種

#重みとバイアスを指定
weights = {
    'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'ww': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'bb': tf.Variable(tf.random_normal([n_classes]))
}

#2層のニューラルネットワークをつくる
def multilayer_neuron(x,weghts,biases):
    layer_1 = tf.add(tf.matmul(x,weghts["w1"]),biases["b1"])
    layer_1 = tf.nn.tanh(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1,weghts["w2"]),biases["b2"])
    layer_2 = tf.nn.tanh(layer_2) 
    out_layer = tf.add(tf.matmul(layer_2,weghts["ww"]),biases["bb"])
    return tf.nn.softmax(out_layer)#出力を0-1の範囲（確率）に変換

prediction = multilayer_neuron(x,weights,biases)
#正解する確率全ての積を最大化したい＝最尤推定法
P = tf.log(prediction)*y
loss = -tf.reduce_sum(P)
train_step = tf.train.AdamOptimizer().minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_loss = 0.

        # Loop over step_size
        for i in range(step_size):
            # 訓練データから batch_size で指定した数をランダムに取得
            ind = np.random.choice(batch_size, batch_size)
            x_train_batch = x_train[ind]
            y_train_batch = y_train[ind]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_step, loss], feed_dict={x: x_train_batch,
                                                          y: y_train_batch})
            # Compute average loss
            avg_loss += c / step_size
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "loss=", \
                "{:.9f}".format(avg_loss))
    print("Optimization Finished!")
    #print(prediction)

      # Test model
    correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: x_test, y: y_test}))