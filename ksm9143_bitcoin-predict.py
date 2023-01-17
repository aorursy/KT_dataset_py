from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM



import numpy as np 

import pandas as pd 

from matplotlib import pyplot as plt



from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()
df = pd.read_csv("../input/bitcoinksm9413/bitcoin.csv")

print(len(df))

df
df_norm = df.drop(['2018-08-14 00:00:00'], 1, inplace=True)
days = 3



df_train= df[:len(df)-days]

df_test= df[len(df)-days:]
training_set = df_train.values

training_set = min_max_scaler.fit_transform(training_set)



x_train = training_set[0:len(training_set)-1]

y_train = training_set[1:len(training_set)]

x_train = np.reshape(x_train, (len(x_train), 1, 1))

print(x_train)
activation_function = 'sigmoid'

optimizer = 'adam'

loss_function = 'mean_squared_error'

num_units = 4

batch_size = 5

epoch = 100





model = Sequential()



#입력층 및 은닉층

model.add(LSTM(units = num_units, activation = activation_function, input_shape=(None, 1)))



# 출력층 추가

model.add(Dense(1))



# 컴파일

model.compile(optimizer = optimizer, loss = loss_function)



# 학습 시작

model.fit(x_train, y_train, batch_size = batch_size, epochs = epoch)
test_set = df_test.values



inputs = np.reshape(test_set, (len(test_set), 1))

inputs = min_max_scaler.transform(inputs)

inputs = np.reshape(inputs, (len(inputs), 1, 1))

print(inputs)



predicted_price = model.predict(inputs)

predicted_price = min_max_scaler.inverse_transform(predicted_price)

print(predicted_price)
plt.plot(test_set[:, 0], color='red', label='Real')

plt.plot(predicted_price[:, 0], color = 'blue', label = 'Predicted')

plt.plot(test_set[:, 0],'or')

plt.plot(predicted_price[:, 0],'or')



plt.title('bitcoin Prediction', fontsize = 15)

plt.xlabel('Time', fontsize=15)

plt.ylabel('price', fontsize = 15)

plt.legend(loc = 'best')

plt.show()
