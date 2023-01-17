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

x_total_scaled = preprocessing.scale(x_total)

x_train_scaled = x_total_scaled[:4000]

x_test_scaled = x_total_scaled[4000:]

y_train = y_total[:4000]

y_test = y_total[4000:]

print('训练集大小: ', x_train_scaled.shape[0])

print('测试集大小: ', x_test_scaled.shape[0])
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



num_steps = 1000

learning_rate = 0.01

batch_size = 40

np.random.seed(0)



weight = np.zeros(6)

batch_g = batch_generator([x_train_scaled, y_train], batch_size, shuffle=True)

x_test_scaled_concat = np.hstack([x_test_scaled, np.ones([x_test_scaled.shape[0], 1])])



for i in range(num_steps):

    rmse_loss = np.sqrt(np.square(np.dot(x_test_scaled_concat, weight) - y_test).mean())

    if i % 20 == 0:

        print('current num_step:', i)

        print('rmse loss:', rmse_loss)

    

    x_batch, y_batch = batch_g.__next__()

    x_batch = np.hstack([x_batch, np.ones([batch_size, 1])])

    y_pred = np.dot(x_batch, weight)

    w_gradient = (x_batch * np.tile((y_pred - y_batch).reshape([-1, 1]), 6)).mean(axis=0)

    weight = weight - learning_rate * w_gradient 



rmse_loss = np.sqrt(np.square(np.dot(x_test_scaled_concat, weight) - y_test).mean())

print('final rmse loss:', rmse_loss)