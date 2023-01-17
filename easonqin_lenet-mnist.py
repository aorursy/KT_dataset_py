from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
import numpy as np
import cv2

numClasses = 10
model = Sequential()
inputShape = (28,28,1)
# 第一层卷积+激活+池化
model.add(Conv2D(20, 5, padding="same",input_shape=inputShape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# 第二层卷积+激活+池化
model.add(Conv2D(50, 5, padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# 第三层全连接层
model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))

# 第四层全连接层 用softmax函数做输出层
model.add(Dense(numClasses))
model.add(Activation("softmax"))



# 加载并重塑数据集中图像
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
testData = testData.reshape((testData.shape[0], 28, 28, 1))

# 归一化处理feed
trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0

#标签OneHot化
trainLabels = np_utils.to_categorical(trainLabels, 10)
testLabels = np_utils.to_categorical(testLabels, 10)

# optimizer 用Stocken Gradient Descent
opt = SGD(lr=0.01)
# 编译一手
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
# 学习一手
print("[INFO] training...")
model.fit(trainData, trainLabels, batch_size=128, epochs=20,verbose=1)

# 显示学习过程
print("[INFO] evaluating...")
(loss, accuracy) = model.evaluate(testData, testLabels,
batch_size=128, verbose=1)
print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

# 在testdata里随机挑一张
for i in np.random.choice(np.arange(0, len(testLabels)), size=(10,)):
    # 进行预测，输出一个10元概率组
    probs = model.predict(testData[np.newaxis, i])
    # 找到最大的那个index
    prediction = probs.argmax(axis=1)

image = (testData[i] * 255).astype("uint8")
# 合并通道
image = cv2.merge([image] * 3)

# 扩大图片进行可视化
image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)

#展示预测
cv2.putText(image, str(prediction[0]), (5, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

print("[INFO] Predicted: {}, Actual: {}".format(prediction[0],np.argmax(testLabels[i])))
img=cv2.imshow("Digit", image)
cv2.waitKey(0)