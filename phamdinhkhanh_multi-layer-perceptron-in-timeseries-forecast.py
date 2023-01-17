import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.layers.merge import concatenate

%matplotlib inline 

n_steps = 3
start_date = '2018-01-01'
end_date = '2018-09-30'
SPY = web.DataReader('SPY', 'yahoo', start_date, end_date)

SPY = SPY.pop('Close')
SPY[-5:]
SPY.plot()
def split_sequence(seq, n_step):
    targets = []
    features = []
    for i in range(len(seq)):
        eidx = i + n_step
        if eidx > len(seq)-1:
            break
        target = seq[i:eidx]
        feature = seq[eidx]
        targets.append(target)
        features.append(feature)
    return np.array(targets), np.array(features)

X, y = split_sequence(SPY, n_steps)
print('X shape: %s'%str(X.shape))
print('y shape: %s'%str(y.shape))
#Summary data
for i in range(5):
    print('{} {:.2f}'.format(X[i, :], y[i]))
# MLP model
model = Sequential()
model.add(Dense(100, activation = 'relu', input_dim = n_steps))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mse')
# fitting model
model.fit(X, y, epochs = 2000, verbose = 0)
X_input = np.array(X[-5:]).reshape(-1, n_steps)
print(X_input)
print('Actual value: %s'%str(y[-5:]))
print('Predict value:')
yhat = model.predict(X_input)
print(yhat)
#Hàm RMSE
def RMSE(yhat, y):
    return np.sqrt(np.mean((yhat - y)**2))

yhat = model.predict(X)
RMSE(yhat, y)
def plt_graph(y, yhat):
    plt.figure(figsize = (16, 8))
    plt.plot(y, label = 'Actual')
    plt.plot(yhat, label = 'Predict')
    plt.xlabel('No transaction')
    plt.ylabel('Price')
    legend = plt.legend(loc = 'upper center', shadow = True, fontsize = 'x-large')
    legend.get_frame().set_facecolor('C')
    
plt_graph(y, yhat)
# Hàm load dữ liệu
def load_data(symbol):
    return np.array(web.DataReader(symbol, 'yahoo', start_date, end_date).pop('Close')).reshape(-1, 1)

AAPL = load_data('AAPL')
GOOGL = load_data('GOOGL')
SPY = load_data('SPY')
# Chồng bảng theo chiều ngang
dataset = np.hstack((AAPL, GOOGL, SPY))
print(dataset.shape)
dataset[-5:,...]
# Tạo hàm split_sequences với 3 bước thời gian
def split_sequences(seq, n_steps = 3):
    X, y = [], []
    for i in range(seq.shape[0]):
        eidx = i + n_steps
        if eidx + 1 > seq.shape[0]:
            break
        X_i, y_i = seq[i:eidx, :2], seq[eidx, -1]
        X.append(X_i)
        y.append(y_i)
    return np.array(X), np.array(y)

X, y = split_sequences(dataset)
print(X.shape)
print(y.shape)
# In 3 giá trị đầu tiên
for i in range(3):
    print('Row {}; X:{} ; Y:{}'.format(i, X[i,:], y[i]))
# flatten dữ liệu
n_shape = X.shape[1]*X.shape[2]
X = X.reshape(-1, n_shape)
print(X.shape)
model = Sequential()
model.add(Dense(100, activation = 'relu', input_dim = n_shape))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mse')
model.fit(X, y, epochs = 2000, verbose = 0)
X_input = X[-1:, :]
print('X input: %s'%X_input)
model.predict(X_input)
yhat = model.predict(X)
RMSE(y, yhat)
plt_graph(y, yhat)
# Xác định đầu vào của model đầu tiên
input1 = Input(shape = (n_steps,))
dense1 = Dense(100, activation = 'relu')(input1)

# Xác định đầu vào của model thứ 2
input2 = Input(shape = (n_steps,))
dense2 = Dense(100, activation = 'relu')(input2)

# Kết hợp kết quả 2 model
model_merge = concatenate([dense1, dense2])
output = Dense(1)(model_merge)

# Nối input và output
model = Model(inputs=[input1, input2], outputs = output)

# Compile model
model.compile(optimizer = 'adam', loss = 'mse')
# # Khởi tạo dữ liệu đầu vào
X1 = X[:, :3]
X2 = X[:, -3:]

print('X1 shape: \n %s'%str(X1.shape))
print('X2 shape: \n %s'%str(X2.shape))
model.fit([X1, X2], y, epochs = 2000, verbose = 0)
# Dự báo mô hình
yhat = model.predict([X1, X2])
# Đánh giá mô hình
print('RMSE: %f'%RMSE(y, yhat))
# Vẽ biểu đồ
plt_graph(y, yhat)
# Tạo hàm split_sequences
def split_sequences(seq, n_steps = 3):
    X, y = [], []
    for i in range(seq.shape[0]):
        eidx = i + n_steps
        if eidx + 1 > seq.shape[0]:
            break
        X_i, y_i = seq[i:eidx, :], seq[eidx, :]
        X.append(X_i)
        y.append(y_i)
    return np.array(X), np.array(y)

X, y = split_sequences(dataset)
print('X shape: %s'%str(X.shape))
print('y shape: %s'%str(y.shape))
print('X: \n %s'%str(X[-1:]))
print('y: \n %s'%str(y[-1:]))
n_input = X.shape[1] * X.shape[2]
n_output = y.shape[1]
#Reshape X sao cho các khối bước thời gian dành thành hàng ngang
X = X.reshape((X.shape[0], n_input))

#Thiết kế mạng nơ ron
model = Sequential()
model.add(Dense(100, activation = 'relu', input_dim = n_input))
model.add(Dense(n_output))
model.compile(optimizer = 'adam', loss = 'mse')
#Fitting model
model.fit(X, y, epochs = 2000, verbose = 0)
# Dự báo mô hình
yhat = model.predict(X)
# Đánh giá mô hình
print('RMSE: %f'%RMSE(y, yhat))
# Vẽ biểu đồ
plt_graph(y, yhat)
# define models
visible = Input(shape = (n_input,))
dense = Dense(100, activation = 'relu')(visible)

# define outputs
output1 = Dense(1)(dense)
output2 = Dense(1)(dense)
output3 = Dense(1)(dense)

# tie together
model = Model(inputs = visible, outputs = [output1, output2, output3])
model.compile(optimizer = 'adam', loss = 'mse')

# separate output
y1 = y[:, 0].reshape((y.shape[0], 1))
y2 = y[:, 1].reshape((y.shape[0], 1))
y3 = y[:, 2].reshape((y.shape[0], 1))

# model fit
model.fit(X, [y1, y2, y3], epochs = 2000, verbose = 0)
# Dự báo mô hình
yhat = model.predict(X)
# Đánh giá mô hình
print('RMSE: %f'%RMSE(y, yhat))
# Vẽ biểu đồ
# plt_graph(y, yhat)