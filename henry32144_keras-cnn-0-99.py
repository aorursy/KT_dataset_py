import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import gc
import seaborn as sns

from keras.preprocessing import image
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

%matplotlib inline
# from csv file
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
# preview train data
display(train.head())
# seperate the label column as target y.
y_train = train['label']
X_train = train.drop("label", axis=1)
# reshape the data into image like, the first argument -1 means not specify the numbers of image, 
# numpy will automatically calculate it for us.

# 將訓練資料轉成圖片的形式，第一個參數 -1 表示的是不指定數量，讓numpy幫我們計算。

X_train = X_train.values.reshape(-1, 28, 28, 1)
X_train.shape
# turn the 10 classes(0 - 9) into one hot encoding.
# example: 1 -->  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# 將原本數字形式的類別轉換成one hot encoding的形式
y_train = to_categorical(y_train, num_classes=10)
# Normalize the image into the range in 0 to 1 will help the convergence of the model
# 將圖片標準化，縮放成0~1的範圍有助於模型的收斂
datagen = image.ImageDataGenerator(rescale=1./255)
# split the data into train and validation set
# 將資料分成訓練及和驗證集
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
# This callback function reduce the learning rate while the specificed indicator is not improved, 
# model can typically get the better result by this.

# 這個回調函數可以在指定的指標沒有在改善的時候調低學習率，通常可以讓模型取得更好的結果

#https://keras.io/callbacks/

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0001)
# Build model

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'same', 
                 activation ='relu'))
model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])
# Train the model, data generator will produce the data for us, 
# the "stpes_per_epoch" is usually set as the length of the training data divide by batch_size

# 訓練模型，圖片生成器會幫助我們批次產生圖片，steps_per_epoch這個參數通常是設定成訓練資料的總數量除以批次大小

history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=128),
          steps_per_epoch=len(X_train) / 128, epochs=15, validation_data=(X_valid, y_valid),
                   callbacks=[learning_rate_reduction])
print(history.history.keys())
# plot the accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
# plot the loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
# plot confusion matrix, this can help us check the error of the answer predicted by model
# 畫混淆矩陣，可以讓我們看到模型在哪個類別上最常出錯
"""
Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
By shaypal https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823
"""
def print_confusion_matrix(confusion_matrix, class_names, figsize = (12,7), fontsize=14):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# the original output is one hot encoding like, we use argmax to transfer it back to number
# 模型的輸出是one hot encoding，所以我們把它轉回成原本數字的樣子

# For example:
# before: [2.1901782e-12, 1.5896548e-10, 1.0000000e+00, 1.1254729e-09,
#         1.4391949e-12, 1.4586092e-15, 8.8379358e-13, 2.0718469e-09,
#         1.9908906e-10, 1.6548666e-13]
# after: 2

y_pred = model.predict(X_valid)

y_pred = np.argmax(y_pred,axis = 1) 

y_true = np.argmax(y_valid,axis = 1) 

confusion_mtx = confusion_matrix(y_true, y_pred) 
# plot the confusion matrix
print_confusion_matrix(confusion_mtx, class_names = range(10))
# reshape and scale it
X_test = (test.values.reshape(-1, 28, 28, 1)) / 255.0
pred = model.predict(X_test)
pred[0]
results = np.argmax(pred,axis = 1)

results = pd.Series(results,name="Label")
results[0]
# follow the required format
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("submission.csv",index=False)