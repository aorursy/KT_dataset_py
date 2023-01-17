import numpy as np

import tensorflow as tf



y0 = np.zeros(100) ## model0のターゲット

y1 = np.ones(100) ## model1のターゲット

X0 = np.random.randn(100, 10)

X1 = np.random.randn(100, 10)



def get_model(X, y):

    inp = tf.keras.Input(10)

    x = tf.keras.layers.Dense(2048)(inp)

    x = tf.keras.layers.Dense(2048)(x)

    x = tf.keras.layers.Dense(2048)(x)

    x = tf.keras.layers.Dense(2048)(x)

    out = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs = inp, outputs = out)

    model.compile(loss = tf.keras.losses.mse)

    model.fit(X, y, epochs = 1, verbose = False)

    return model



## model0の訓練

model0 = get_model(X0, y0)



## model1の訓練

model1 = get_model(X1, y1)



## くっつけたモデルの作成

inp = tf.keras.Input(11)

# out = tf.keras.layers.Lambda(lambda inp : tf.keras.backend.switch(inp[:, -1:] == 0, model0(inp[:,:-1]), model1(inp[:,:-1])))(inp)

out = tf.where(inp[:,-1:] == 0, model0(inp[:,:-1]), model1(inp[:,:-1]))

model_concat = tf.keras.Model(inputs = inp, outputs = out)

model_concat.compile()



## 評価

X_test = np.random.randn(100, 10)

pred_type = np.random.randint(0, 2, 100)

p0 = model0.predict(X_test)

p1 = model1.predict(X_test)



X_concat = np.hstack([X_test, pred_type.reshape(-1, 1)])

p_concat = model_concat.predict(X_concat)

idx0 = pred_type == 0

print("loss with model0 ", np.max(np.abs(p0[idx0] - p_concat[idx0])))

idx1 = pred_type == 1

print("loss with model1 ", np.max(np.abs(p1[idx1] - p_concat[idx1])))

## だいたいあってる
%%time

## concatしたモデルは100回推論して4sぐらい

for k in range(100):

    p_concat = model_concat.predict(X_concat)
%%time

## 元々のモデルを100回ずつ推論して8sぐらい（呼び出し回数が，上のconcatモデルよりも多くなるから遅いだけ？）

for k in range(100):

    p_concat = model0.predict(X_test)

    p_concat = model1.predict(X_test)
## くっつけたモデルの作成

inp = tf.keras.Input(11)

# out = tf.keras.layers.Lambda(lambda inp : tf.keras.backend.switch(inp[:, -1:] == 0, model0(inp[:,:-1]), model1(inp[:,:-1])))(inp)

out1 = model0(inp[:,:-1])

out2 = model1(inp[:,:-1])

model_concat = tf.keras.Model(inputs = inp, outputs = [out1, out2])

model_concat.compile()
%%time

## concatしたモデルは100回推論して4sぐらい

for k in range(100):

    p_concat = model_concat.predict(X_concat)