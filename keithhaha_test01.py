import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
train = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')
train
train_label = train.pop('label')
train_label
train
train = np.array(train)
train = np.reshape(train,(60000,28,28,1))
train = tf.cast(train,tf.float32)
train = train/255.
train[0].shape
test = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')
test_label = test.pop('label')
test_data = np.array(test)
test_data = np.reshape(test_data,(10000,28,28,1))
test_data = tf.cast(test_data,tf.float32)
test_data = test_data/255.
tf.random.set_seed(2020)
np.random.seed(2020)
class simpleNet(tf.keras.Model):
    def __init__(self):
        super(simpleNet,self).__init__()
        self.conv1 = layers.Conv2D(32,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu')
        self.maxpool = layers.MaxPool2D(pool_size=(2,2))
        self.dp = layers.Dropout(0.5)
        self.avg = layers.AveragePooling2D()
        self.fc1 = layers.Dense(32,activation='relu')
        self.fc2 = layers.Dense(10,activation='softmax')
    def call(self,inputs,training=None):
        x = self.conv1(input1)
        x = self.maxpool(x)
        x = self.dp(x)
        x = self.avg(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
model = tf.keras.Sequential()
model.add(layers.Conv2D(input_shape=(28, 28, 1),
                        filters=32, kernel_size=(3,3), strides=(1,1), padding='valid',
                       activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
train_data = tf.data.Dataset.from_tensor_slices((train,train_label))
train_data = train_data.shuffle(len(train)).batch(256).repeat()
val_data = tf.data.Dataset.from_tensor_slices((test_data,test_label))
val_data = val_data.batch(256)
cloth_dict = {0:'T-shirt/top',1:'Trouser',2:'Pullover',3:'Dress',4:'Coat',5:'Sandal',6:'Shirt',7:'Sneaker',8:'Bag',9:'Ankle boot'}
for img,label in train_data.take(1):
    img1 = tf.reshape(img[15],(28,28))
    label = label[15].numpy()
    plt.imshow(img1)
    plt.title(cloth_dict[label])
    plt.show()
model.compile(optimizer=keras.optimizers.Adam(),
             # loss=keras.losses.CategoricalCrossentropy(),  # 需要使用to_categorical
             loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
model.summary()
history = model.fit(train,np.array(train_label),shuffle=True,epochs=30,validation_split=0.2,validation_steps=30)
plt.figure(figsize=(14,7))
plt.subplot(121)
plt.plot(history.history['loss'],c='r',label='loss')
plt.plot(history.history['val_loss'],c='g',label='val_loss')
plt.legend()
plt.subplot(122)
plt.plot(history.history['accuracy'],c='r',label='acc')
plt.plot(history.history['val_accuracy'],c='g',label='val_acc')
plt.legend()
plt.show()
res = model.evaluate(test_data,test_label)
pred = model.predict(test_data)
pred = [pred[i,:].argmax() for i in range(10000)]
from sklearn.metrics import accuracy_score
score = accuracy_score(pred,test_label)
score
from sklearn.metrics import confusion_matrix
import seaborn as sns
con_ma = confusion_matrix(test_label,pred)
sns.heatmap(con_ma,annot=True,cmap='Blues')
plt.show()
data = pd.DataFrame({'scores':[res[1]]})
data.to_csv('/kaggle/working/fashion_mnist.csv')
