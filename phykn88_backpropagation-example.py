import numpy as np

from keras import backend as K

from keras import models

from keras import layers
def make_data(batch_size, input_size, index):

    '''

    batch_size : Total batch size

    input_size : Total input size, values are random number except at the attention_index

    index : value is 0 or 1 at the index

    '''

    

    train_x = np.random.randn(batch_size, input_size)

    train_y = np.random.randint(2, size=(batch_size, 1))

    train_x[:, index] = train_y[:, 0]

    return train_x, train_y



# Data Set

train_x, train_y = make_data(10000, 10, 4)
# Keras

input_layer = layers.Input(shape=(train_x.shape[1],))

x = layers.Dense(10, activation='relu')(input_layer)

x = layers.Dense(1)(x)



model = models.Model(input_layer, x)

model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])

model.summary()

model.fit(train_x, train_y, epochs=2, batch_size=1, verbose=1)

    

# Evaluation

test_x, test_y = make_data(10000, 10, 4)

pred = np.around(model.predict(test_x)).flatten()

true = test_y.flatten()

score = 100*len(np.where(pred-true == 0)[0]) / len(true)

print('score: %f %%'%(score))
# Numpy

def relu(array):

    output = (array + np.abs(array))/2

    return output

def relu_diff(array):

    output = array > 0

    return 1*output



# Initialize

W = np.random.normal(size=(train_x.shape[1], 10))

b = np.zeros(shape=(10, ))

V = np.random.normal(size=(10, ))

c = np.zeros(shape=(1, ))



# Hyperparameter

epochs = 2

lr = 0.01



# Training

for i in range(epochs):

    for j in range(len(train_x)):

        x = train_x[j]

        y_t = train_y[j]



        y_1 = np.matmul(x, W) + b

        y_2 = relu(y_1)

        y_3 = np.matmul(y_2, V) + c

        m = 0.5*(y_3 - y_t)**2



        dm_dc = (y_3 - y_t)

        dm_dV = dm_dc * y_2

        dm_db = (y_3 - y_t) * V * relu_diff(y_1)

        dm_dW = np.einsum('j,i->ij', dm_db, x)



        W -= lr * dm_dW

        b -= lr * dm_db

        V -= lr * dm_dV

        c -= lr * dm_dc

    print('epochs: %i, loss: %f'%(i+1, m))

    

# Evaluation    

test_x, test_y = make_data(10000, 10, 4)

pred = []

for x in test_x:

    y_1 = np.matmul(x, W) + b

    y_2 = relu(y_1)

    y_3 = np.matmul(y_2, V) + c

    pred.append(y_3)

pred = np.array(pred)



pred = np.around(pred).flatten()

true = test_y.flatten()

score = 100*len(np.where(pred-true == 0)[0]) / len(true)

print('score: %f %%'%(score))