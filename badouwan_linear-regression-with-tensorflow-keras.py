import numpy as np # linear algebra

# 设置随机数种子
np.random.seed(1984)
# 在 X轴 [-1,1] 范围内等距离生产200个点
X = np.linspace(-1, 1, 200)

# 把上述生产的200个点打散
np.random.shuffle(X)

# 构造的线性函数为 Y = Wx + b
# 按 Y = 2X + 3 构造数据，加入随机数作为噪音 bias
Y = 2 * X + np.random.normal(3, 0.01, (200, ))
# 数组中前160组作为训练数据，后40组作为测试数据
X_train, Y_train = X[:160], Y[:160]

X_test, Y_test = X[160:], Y[160:]
from tensorflow import keras

print(keras.__version__)
# 构造模型
model = keras.Sequential()

model.add(keras.layers.Dense(units=1, input_dim=1))

# 编译模型，优化函数为 随机梯度下降，损失函数为 均方差
model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])

# 训练模型
# model.fit(X_train, Y_train, epochs=100, batch_size=40)
for i in range(1001):
    cost = model.train_on_batch(X_train, Y_train)

    if i % 200 == 0:
        print('cost: ', cost)

# 测试模型
model.evaluate(X_test, Y_test, batch_size=40)

# 显示预测的 W 和 b
[[[W]], [b]] = model.layers[0].get_weights()

print('拟合所得的线性函数为 Y = %.2f * x + %.2f' % (W, b))
# 用图表形式展现出实际值与预测值，黑色为实际值，红色为预测值
import matplotlib.pyplot as plt

# 预测值
Y_predict = model.predict(X_test)

plt.scatter(X_test, Y_test, s=20)
plt.plot(X_test, Y_predict, color='red')
plt.figtext(0.25, 0.5, r'$Y=2x+3$')
plt.show()