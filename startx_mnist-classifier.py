import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
print(os.listdir("../input"))
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
train_data['label'].value_counts()
ones_index = train_data['label'] == 1
ones_value = train_data.loc[ones_index]
sevens_index = train_data['label'] == 7
sevens_value = train_data.loc[sevens_index]
plt.imshow(ones_value.values[26,1:].reshape(28,28), cmap='gray')
plt.show()
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
train_data = shuffle(train_data)
train_data = train_data.values
test_x  = test_data.values.reshape(-1,28,28,1)
train_x = train_data[:,1:].reshape(-1,28,28,1)
train_y = train_data[:,0]
lb = LabelBinarizer().fit(train_y)
train_y = lb.transform(train_y)
train_x.shape
plt.imshow(train_x[6,1:27, 1:27, ...].squeeze(), cmap='gray')
plt.show()
m, h, w, c = train_x.shape
train_x_ex = np.zeros((m*9, 26, 26, c))
train_y_ex = np.zeros((m*9, 10))

for i in range(m):
    for x in range(3):
        for y in range(3):
            train_x_ex[i*9+x*3+y,...] = train_x[i, x:x+26, y:y+26, ...]
            train_y_ex[i*9+x*3+y,...] = train_y[i,...]
plt.imshow(train_x_ex[5,...].squeeze(), cmap='gray')
plt.show()
train_x, val_x, train_y, val_y = train_test_split(train_x_ex, train_y_ex, test_size=0.1, shuffle=True)
train_x.shape
plt.imshow(train_x[5,...].squeeze(), cmap='gray')
plt.show()
from keras.models import Model, load_model
from keras.layers import Conv2D, Dense, Input, Activation, Flatten, Reshape
from keras.layers import Dropout, Lambda, MaxPooling2D, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.initializers import he_uniform
def innerLayer(img_in):
    pre = Lambda(lambda x: (x - 128) / 255)(img_in)
    Conv1 = Conv2D(32, (3,3), kernel_initializer = he_uniform(), name='conv1')(pre)
    # Conv1 = BatchNormalization(axis=-1)(Conv1)
#     Conv1 = LeakyReLU(0.2)(Conv1)
    Conv1 = ELU()(Conv1)
#     Conv1 = MaxPooling2D(pool_size=(2, 2))(Conv1)
    
    Conv2 = Conv2D(32, (3,3), kernel_initializer = he_uniform(), name='conv2')(Conv1)
    # Conv2 = BatchNormalization(axis=-1)(Conv2)
#     Conv2 = LeakyReLU(0.2)(Conv2)
    Conv2 = ELU()(Conv2)
    Conv2 = MaxPooling2D(pool_size=(2, 2))(Conv2)
    
    Conv3 = Conv2D(64, (3,3), kernel_initializer = he_uniform(), name='conv3')(Conv2)
    # Conv3 = BatchNormalization(axis=-1)(Conv3)
#     Conv3 = LeakyReLU(0.2)(Conv3)
    Conv3 = ELU()(Conv3)
#     Conv3 = MaxPooling2D(pool_size=(2, 2))(Conv3)
    
    Conv4 = Conv2D(128, (3,3), kernel_initializer = he_uniform(), name='conv4')(Conv3)
    # Conv4 = BatchNormalization(axis=-1)(Conv4)
#     Conv4 = LeakyReLU(0.2)(Conv4)
    Conv4 = ELU()(Conv4)
    Conv4 = MaxPooling2D(pool_size=(2, 2))(Conv4)
    
    Conv4 = Dropout(0.5)(Conv4)
    
    Conv5 = Conv2D(256, (3,3), kernel_initializer = he_uniform(), name='conv5')(Conv4)
#     Conv5 = LeakyReLU(0.2)(Conv5)
    Conv5 = ELU()(Conv5)
    
#     Conv5 = Dropout(0.5)(Conv5)
    
    Conv6 = Conv2D(256, (1,1), kernel_initializer = he_uniform(), name='conv6')(Conv5)
#     Conv6 = LeakyReLU(0.2)(Conv6)
    Conv6 = ELU()(Conv6)
    
#     Conv6 = Dropout(0.5)(Conv6)
    
    Conv7 = Conv2D(10, (1,1), name='output')(Conv6)
    Conv7 = Activation('softmax')(Conv7)

    return Conv7
img_in1 = Input(shape=(26,26,1))
out_layer1 = innerLayer(img_in1)
out_layer1 = Reshape((10,))(out_layer1)
model1 = Model(inputs=img_in1, outputs=out_layer1)
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model1.summary()
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
rl = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1)
mc = ModelCheckpoint('model.h5', monitor='val_acc', verbose=1, save_best_only=True)
callback = [es, rl, mc]
from keras.preprocessing.image import ImageDataGenerator
generator = ImageDataGenerator(rotation_range = 10,zoom_range = 0.1,width_shift_range = 0.1,
                               height_shift_range = 0.1,horizontal_flip = False,vertical_flip = False)
generator.fit(train_x)
model1.fit_generator(generator.flow(train_x, train_y, batch_size=64), int(train_x.shape[0]/64),
                    epochs=30, verbose=1, callbacks=[rl, mc], 
                    validation_data=(val_x, val_y))
del model1
model1 = load_model('model.h5')
print(model1.evaluate(train_x, train_y))
print(model1.evaluate(val_x, val_y))
model1.save_weights('model1_weights.h5')
from sklearn.metrics import confusion_matrix
def polt_confusion_matrix(data_x, data_y, model):
    data_y_hat = model.predict(data_x)
    data_y_hat = lb.inverse_transform(data_y_hat)
    cm_data = confusion_matrix(lb.inverse_transform(data_y), data_y_hat)
    
    row_sum = cm_data.sum(axis=1, keepdims=True)
    norm_cm_data = cm_data / row_sum
    np.fill_diagonal(norm_cm_data, 0)
    plt.matshow(norm_cm_data, cmap=plt.cm.gray, )
    return cm_data
cm_train = polt_confusion_matrix(train_x, train_y, model1)
cm_train
cm_val = polt_confusion_matrix(val_x, val_y, model1)
cm_val
img_in2 = Input(shape=(28,28,1))
out_layer2 = innerLayer(img_in2)
out_layer2 = Reshape((-1, 10))(out_layer2)
model2 = Model(inputs=img_in2, outputs=out_layer2)
model2.load_weights('model1_weights.h5', by_name=True)
model2.summary()
test_yhat = model2.predict(test_x)
test_yhat = np.sum(test_yhat, axis=1)
labels = lb.inverse_transform(test_yhat)
result = {'ImageId':np.arange(labels.size)+1,'Label':labels}
result = pd.DataFrame(result)
result.to_csv("Prediction.csv",index=False)
result
