import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# 设置基本参数
batch_size=32
epochs=20
num_classes=10
input_shape=(28,28,1)
# 导入测试集的数据
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# train 为 42000 * 785矩阵，X_train 取后784列，Y_train 取第一列（即标签）
# test 为 28000 * 784矩阵
X_train = train.iloc[:, 1:].values
Y_train = train.iloc[:, 0].values
X_test = test.values
# 节约少许内存开销（约400M）
del train, test
# 将二维矩阵 42000 * 784
# 转换成四维矩阵，42000 * 28 * 28 * 1
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
# 将 0-255 的数据，归化为 0-1 的数据
X_train = np.divide(X_train, 255.)
X_test = np.divide(X_test, 255.)
import tensorflow as tf
from tensorflow import keras

# 打印版本号
print(tf.__version__)
print(keras.__version__)
from sklearn.model_selection import train_test_split

# 将 Y_train 也即2维的 42000 * 1 矩阵，按 one-shot 编码为 42000 * 10的矩阵
# 原值 0-9 的数值转换为 1 * 10 的列对应为1，其余为0
Y_train = keras.utils.to_categorical(Y_train, num_classes)

# 将样本 42000 切分为训练组 train 和校验组 val 两组数据
# 参数 train_size / test_size 既可按比例切分，也可以直接指定数量
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 2000)
# define a model with tensorflow.keras
model = keras.Sequential()

# 填充方式 same 或 valid，大小写敏感！
model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same',
                              input_shape=input_shape))
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(rate=0.25))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=128, activation='relu'))
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(units=10, activation='softmax'))
# 用以生成一个batch的图像数据，支持实时数据提升
datagen = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
# 修改学习率
learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=2,factor=0.5,min_lr=0.00001)
# 编译模型
optimizer=keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
# 训练模型
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                              epochs =epochs, validation_data = (X_val,Y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size+1,
                              callbacks=[learning_rate_reduction])
# 测试模型
loss, accuracy = model.evaluate(X_val, Y_val, steps=batch_size)
loss, accuracy
# 作出预测
predicted = model.predict(X_test)

# 将预测结果 28000 * 10 处理成 28000 * 1
# 算法是取 10 个值中的最大值
results = np.argmax(predicted, axis=1)
# 将预测值保存成竞赛要求的格式
submission = pd.read_csv('../input/sample_submission.csv')
submission["Label"] = results
submission.to_csv('tf_keras.csv', index=False)