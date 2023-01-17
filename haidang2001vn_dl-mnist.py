# 1. Thêm các thư viện cần thiết

import numpy as np

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import np_utils

# from keras.datasets import mnist
# 2. Load dữ liệu MNIST

def load_data(path):

    with np.load(path) as f:

        x_train, y_train = f['x_train'], f['y_train']

        x_test, y_test = f['x_test'], f['y_test']

        return (x_train, y_train), (x_test, y_test)



(X_train, y_train), (X_test, y_test) = load_data('../input/mnist.npz')



# (X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_train /= 255

X_test /= 255



X_val, y_val = X_train[50000:60000,:], y_train[50000:60000]

X_train, y_train = X_train[:50000,:], y_train[:50000]

print(X_train.shape)
# 3. Reshape lại dữ liệu cho đúng kích thước mà keras yêu cầu

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

print(X_train.shape)
# 4. One hot encoding label (Y)

Y_train = np_utils.to_categorical(y_train, 10)

Y_val = np_utils.to_categorical(y_val, 10)

Y_test = np_utils.to_categorical(y_test, 10)

print('Dữ liệu y ban đầu ', y_train[0])

print('Dữ liệu y sau one-hot encoding ',Y_train[0])
# 5. Định nghĩa model

model = Sequential()

 

# Thêm Convolutional layer với 32 kernel, kích thước kernel 3*3

# dùng hàm sigmoid làm activation và chỉ rõ input_shape cho layer đầu tiên

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))



# Thêm Convolutional layer

model.add(Conv2D(64, (3, 3), activation='relu'))



# Thêm Max pooling layer

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))



# Flatten layer chuyển từ tensor sang vector

model.add(Flatten())



# Thêm Fully Connected layer với 128 nodes và dùng hàm sigmoid

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))



# Output layer với 10 node và dùng softmax function để chuyển sang xác xuất.

model.add(Dense(10, activation='softmax'))
# 6. Compile model, chỉ rõ hàm loss_function nào được sử dụng, phương thức 

# đùng để tối ưu hàm loss function.

model.compile(loss='categorical_crossentropy',

              optimizer='adadelta',

              metrics=['accuracy'])
# 7. Thực hiện train model với data

numOfEpoch = 15

H = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),

          batch_size=128, epochs=numOfEpoch, verbose=1)
# 8. Vẽ đồ thị loss, accuracy của traning set và validation set

fig = plt.figure()

plt.plot(np.arange(0, numOfEpoch), H.history['loss'], label='training loss')

plt.plot(np.arange(0, numOfEpoch), H.history['val_loss'], label='validation loss')

plt.plot(np.arange(0, numOfEpoch), H.history['acc'], label='accuracy')

plt.plot(np.arange(0, numOfEpoch), H.history['val_acc'], label='validation accuracy')

plt.title('Accuracy and Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss|Accuracy')

plt.legend()
# 9. Đánh giá model với dữ liệu test set

score = model.evaluate(X_test, Y_test, verbose=0)

print(score)
# 10. Dự đoán ảnh

index_test = 321

plt.imshow(X_test[index_test].reshape(28,28), cmap='gray')



y_predict = model.predict(X_test[index_test].reshape(1,28,28,1))

print('Giá trị dự đoán: ', np.argmax(y_predict))
output = model.predict(X_test)

Y = []

for i in range(output.shape[0]):

    if np.argmax(output[i]) != np.argmax(Y_test[i]): Y.append(i)



print(Y)
NumIm = 20

fig = plt.figure(figsize=(15, 30))

columns = 5

rows = 10



for index in range(0, NumIm):

    fig.add_subplot(rows, columns, index + 1)

    

    index_test = Y[index]

    plt.imshow(X_test[index_test].reshape(28,28), cmap='gray')

    y_predict = model.predict(X_test[index_test].reshape(1,28,28,1))

    print(index + 1, ': hình ảnh thứ ', index_test, ' có giá trị dự đoán và thực tế: ', np.argmax(y_predict), ' và ', np.argmax(Y_test[index_test]))



plt.show()