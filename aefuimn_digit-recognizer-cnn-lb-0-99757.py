# coding:utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras import utils
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Dense, Flatten, Dropout, MaxPool2D
from keras.callbacks import LearningRateScheduler
from keras.utils import plot_model


batch_size = 64
num_classes = 10
epochs = 45
input_shape = (28, 28, 1)
# Input data files are available in the "../input/" directory. 
# 输入数据文件在",,/input/"目录下。
data = pd.read_csv('../input/train.csv', sep=',')

# Reordering, this may not be necessary.
# 重新排序，这可能不是那么必要。
data = data.reindex(np.random.permutation(data.index))

labels = data['label']
labels = np.array(labels)
features = data.loc[:, 'pixel0':'pixel783']
features = np.array(features)

# Change dimension
# 更改维度
features = np.reshape(features, (-1, 28, 28, 1))

# labels to one-hot
# 使用独热编码
one_hot_labels = utils.to_categorical(labels, num_classes=num_classes)

# cutting data
# 分割数据
X_train = features[:38000]
X_val = features[38000:]
Y_train = one_hot_labels[:38000]
Y_val = one_hot_labels[38000:]

# Test data processing
# 测试数据处理
test_features = pd.read_csv('../input/test.csv', sep=',')
test_features = np.array(test_features)
X_test = np.reshape(test_features, (-1, 28, 28, 1))
plt.figure(figsize=(10, 10))
 
# figure interval setting.
# 设置figsize间隔。
plt.subplots_adjust(top=0.5)

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.title('label:{}'.format(labels[i]))
    plt.imshow(X_train[i].reshape(28, 28))
plt.figure(figsize=(10,10))
plt.subplots_adjust(top=0.5)
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
# keras Sequential
# keras顺序模型
model = Sequential()

model.add(Conv2D(32, kernel_size=2, activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=4, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=2, activation='relu'))
model.add(MaxPool2D(pool_size=2))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(32, kernel_size=4, activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# model compile
# 编译模型
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
datagen = ImageDataGenerator(
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180) 随机旋转度数范围
        zoom_range=0.1,  # Randomly zoom image 随机缩放范围
        width_shift_range=0.1, 
        height_shift_range=0.1)  
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
history = model.fit_generator(datagen.flow(X_train, 
                                           Y_train, 
                                           batch_size=batch_size),      
                              epochs=epochs, 
                              steps_per_epoch=X_train.shape[0] // batch_size,
                              validation_data=(X_val, Y_val), 
                              callbacks=[annealer], verbose=1)
max(history.history['acc'])
model.save('CNN_model')
plot_model(model, to_file='CNN_model.png', show_shapes='true')
results = model.predict(X_test)
results = np.argmax(results, axis=1)
results = pd.Series(results, name="Label")
submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), results], axis=1)
submission.to_csv("Digit_predict.csv", index=False)
plt.figure(figsize=(10,40))
plt.subplots_adjust(top=0.5)
for i in range(40):
    plt.subplot(8, 5, i + 1)
    plt.title('label:{}'.format(results[i]))
    plt.imshow(X_test[i].reshape(28, 28))