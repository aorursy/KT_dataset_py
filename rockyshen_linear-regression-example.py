import numpy as np

from sklearn import preprocessing



# 加载数据

lines=np.loadtxt('../input/USA_Housing.csv', delimiter=',', dtype='str')

print("输入")

for i in range(lines.shape[1]-1):

    print(lines[0, i])

print("标签")

print(lines[0,-1])



x_total = lines[1:, :5].astype('float')

y_total = lines[1:, 5:].astype('float').flatten()



# 数据预处理和切分

x_total = preprocessing.scale(x_total)

y_total = preprocessing.scale(y_total)

x_train = x_total[:4000]

x_test = x_total[4000:]

y_train = y_total[:4000]

y_test = y_total[4000:]

print('训练集大小: ', x_train.shape[0])

print('测试集大小: ', x_test.shape[0])

X_train = np.hstack([x_train, np.ones((x_train.shape[0], 1))])

NE_solution = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X_train), X_train)), np.transpose(X_train)), y_train.reshape([-1, 1]))

print(NE_solution)



X_test = np.hstack([x_test, np.ones((x_test.shape[0], 1))])

y_pred_test = np.dot(X_test, NE_solution).flatten()



rmse_loss = np.sqrt(np.square(y_test - y_pred_test).mean())

print('rmse_loss:', rmse_loss)
from sklearn import linear_model



linreg = linear_model.LinearRegression()

linreg.fit(x_train, y_train)

print(linreg.coef_)

print(linreg.intercept_)

y_pred_test = linreg.predict(x_test)



rmse_loss = np.sqrt(np.square(y_test - y_pred_test).mean())

print('rmse_loss:', rmse_loss)
def shuffle_aligned_list(data):

    num = data[0].shape[0]

    shuffle_index = np.random.permutation(num)

    return [d[shuffle_index] for d in data]



def batch_generator(data, batch_size, shuffle=True):

    batch_count = 0

    while True:

        if batch_count * batch_size + batch_size >= data[0].shape[0]:

            batch_count = 0

            if shuffle:

                data = shuffle_aligned_list(data)

        start = batch_count * batch_size

        end = start + batch_size

        batch_count += 1

        yield [d[start:end] for d in data]
import matplotlib.pyplot as plt

num_steps = 600

learning_rate = 0.01

batch_size = 40



weight = np.zeros(6)

np.random.seed(0)

batch_g = batch_generator([x_train, y_train], batch_size, shuffle=True)

x_test_concat = np.hstack([x_test, np.ones([x_test.shape[0], 1])])



loss_list = []

for i in range(num_steps):

    rmse_loss = np.sqrt(np.square(np.dot(x_test_concat, weight) - y_test).mean())

    loss_list.append(rmse_loss)

    

    x_batch, y_batch = batch_g.__next__()

    x_batch = np.hstack([x_batch, np.ones([batch_size, 1])])

    y_pred = np.dot(x_batch, weight)

    w_gradient = (x_batch * np.tile((y_pred - y_batch).reshape([-1, 1]), 6)).mean(axis=0)

    weight = weight - learning_rate * w_gradient 



print('weight:', weight)

print('rmse_loss:', rmse_loss)

    

loss_array = np.array(loss_list)

plt.plot(np.arange(num_steps), loss_array)

plt.show()
