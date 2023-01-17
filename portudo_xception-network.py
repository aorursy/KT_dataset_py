import pandas as pd
data_train = pd.read_csv('data/mnist/train.csv')
data_test = pd.read_csv('data/mnist/test.csv')
from keras import layers as ly
from keras import Model
from keras import Input
x_train = data_train.iloc[:, 1:].values.astype(float).reshape(-1, 28, 28, 1) / 255.0
y_train = data_train.iloc[:, 0].values
x_test = data_test.values.reshape(-1, 28, 28, 1) / 255.0
x_test.shape
def x_module(x, f1, f2, f3, fm, mp, stride):
    
    branch_1 = ly.Conv2D(f1, 1, strides=stride, padding='same', activation='relu')(x)
    
    branch_2 = ly.SeparableConv2D(f2[0], 2, strides=stride, padding='same', activation='relu')(x)
    branch_2 = ly.Conv2D(f2[1], 1, activation='relu')(branch_2)
    
    branch_3 = ly.SeparableConv2D(f3[0], 3, strides=stride, padding='same', activation='relu')(x)
    branch_3 = ly.SeparableConv2D(f3[1], 1, activation='relu')(branch_3)
    
    branch_4 = ly.MaxPooling2D(mp, strides=stride, padding='same')(x)
    branch_4 = ly.Conv2D(fm, 1, activation='relu')(branch_4)
    
    return ly.Concatenate(axis=-1)([branch_1, branch_2, branch_3, branch_4])
def model():
    i = Input(shape=(28, 28, 1))
    
    x = ly.Conv2D(32, 5, padding='same', activation='relu')(i)
    x = ly.BatchNormalization()(x)
    x = ly.MaxPooling2D(strides=2, padding='same')(x)
    x = ly.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = ly.BatchNormalization()(x)
    x = ly.MaxPooling2D(strides=2, padding='same')(x)
    
    f1 = 128
    f2 = (64, 128)
    f3 = (64, 128)
    fm = 128
    mp = 3
    y = x_module(x, f1, f2, f3, fm, mp, stride=1)
    y = ly.BatchNormalization()(y)
    f1 = 256
    f2 = (64, 256)
    f3 = (64, 256)
    fm = 256
    mp = 2
    y = x_module(y, f1, f2, f3, fm, mp, stride=1)
    y = ly.BatchNormalization()(y)
    
    x = ly.Conv2D(f1 + f2[1] + f3[1] + fm, 1)(x)
    x = ly.Add()([x, y])
    x = ly.MaxPooling2D(strides=2, padding='same')(x)

    f1 = 512
    f2 = (128, 512)
    f3 = (128, 512)
    fm = 512
    mp = 2
    y = x_module(x, f1, f2, f3, fm, mp, stride=1)
    y = ly.BatchNormalization()(y)
    
    f1 = 1024
    f2 = (256, 1024)
    f3 = (256, 1024)
    fm = 1024
    mp = 2
    y = x_module(y, f1, f2, f3, fm, mp, stride=1)
    y = ly.BatchNormalization()(y)
    
    x = ly.Conv2D(f1 + f2[1] + f3[1] + fm, 1)(x)
    x = ly.Add()([x, y])
    x = ly.MaxPooling2D(strides=2, padding='same')(x)
    
    x = ly.Conv2D(512, 1, activation='relu')(x)
    x = ly.BatchNormalization()(x)
    x = ly.Conv2D(10, 1, activation='softmax')(x)
    x = ly.GlobalMaxPooling2D()(x)
    
    return Model(inputs=i, outputs=x)
    
    
m = model()
m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
m.summary()
from keras.preprocessing.image import ImageDataGenerator
train_gen = ImageDataGenerator(rotation_range=10,
                               width_shift_range=0.25, 
                               height_shift_range=0.25, 
                               shear_range=0.2,
                               zoom_range=0.2)
train_gen.fit(x_train)
x_train.shape
y_train.shape
from keras.utils import to_categorical
batch_size = 64
m.fit(x_train, to_categorical(y_train), batch_size=batch_size,
                   epochs=10)
res = m.predict(x_test)
import numpy as np
res_max = np.argmax(res, axis=1)
res_max
df = pd.DataFrame(data={'ImageId': range(1, len(res_max)+1), 'Label': res_max})
df.to_csv('data/mnist/result1.csv')
## got 99.4% accuracy after data augmentation with multiple transformations
