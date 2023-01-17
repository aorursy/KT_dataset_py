import tensorflow.keras as k
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def load_data(train = True):
    if train:
        df = pd.read_csv('../input/train.csv')
    else:
        df = pd.read_csv('../input/test.csv')
    return df

def split_data(df):
    y = df['label']
    X = df.iloc[:, 1:]
    X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.2, random_state=20)
    X_train = X_train/255
    X_train = X_train.reshape([-1,28,28,1])
    X_test = X_test/255
    X_test = X_test.reshape([-1, 28, 28, 1])

    return X_train, X_test, y_train, y_test

def basic_CNN():
    net = k.Sequential([
        k.layers.Conv2D(64, kernel_size =3, strides = 1, padding='same', activation=tf.nn.relu),
#         k.layers.Dropout(0.5),
        k.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1)),  
        
        k.layers.Conv2D(128, kernel_size=3,strides = 1, padding='same', activation=tf.nn.relu),
#         tf.keras.layers.Dropout(0.5),
        k.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1)),
        
        k.layers.Conv2D(256, kernel_size=3, strides = 1,padding='same', activation=tf.nn.relu),
        k.layers.Conv2D(256, kernel_size=3, strides = 1,padding='same', activation=tf.nn.relu),
#         tf.keras.layers.Dropout(0.5),
        k.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1)),
        
        k.layers.Conv2D(256, kernel_size=3, strides = 1,padding='same', activation=tf.nn.relu),
        k.layers.Conv2D(256, kernel_size=3, strides = 1,padding='same', activation=tf.nn.relu),
#         tf.keras.layers.Dropout(0.5),
        k.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1)),       
        
        k.layers.Flatten(),
        
        k.layers.Dense(512, activation=tf.nn.relu),
        k.layers.BatchNormalization(),
        k.layers.Dropout(0.5),
        
        k.layers.Dense(512, activation=tf.nn.relu),
        k.layers.BatchNormalization(),
        k.layers.Dropout(0.5),
        
        k.layers.Dense(10, activation=tf.nn.softmax)
    ])
    return net



def resnet(input_shape):
    x = k.layers.Input(shape = input_shape)
    y = k.layers.Conv2D(64, kernel_size = (7,7), strides = (2,2), padding = 'same')(x)
    y = k.layers.BatchNormalization()(y)
    y = k.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(y)
    y = k.layers.Dropout(0.5)(y)
    
    filters = 64
    for i in range(3):
        for j in range(2):
            y = _residual_block(y, filters)
        filters *= 2
    y = k.layers.AveragePooling2D(strides = (2,2))(y)
    y = k.layers.Flatten()(y)
    y = k.layers.Dense(1024)(y)
    y = k.layers.BatchNormalization()(y)
    y = k.layers.LeakyReLU()(y)
    y = k.layers.Dropout(0.5)(y)
    y = k.layers.Dense(1024)(y)
    y = k.layers.BatchNormalization()(y)
    y = k.layers.LeakyReLU()(y)
    y = k.layers.Dropout(0.5)(y)
    y = k.layers.Dense(10, activation=tf.nn.softmax)(y)
    model = k.models.Model(inputs=x,outputs=y)
    return model

def _residual_block(y, nb_filters, _strides = (1,1)):
    shortcut = y
    y = k.layers.Conv2D(nb_filters*4, kernel_size = (1,1), strides = _strides, padding ='same')(y)
    y = k.layers.BatchNormalization()(y)
    y = k.layers.LeakyReLU()(y)
    y = k.layers.Dropout(0.5)(y)
    y = k.layers.Conv2D(nb_filters*4, kernel_size = (3,3), strides = (1,1), padding ='same')(y)
    
    shortcut = k.layers.Conv2D(nb_filters*4, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
    
    
    y = k.layers.add([shortcut, y])
    y = k.layers.BatchNormalization()(y)
    y = k.layers.LeakyReLU()(y)   
    return y

def train(net):
    net.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

df = load_data()
X_train, X_test, y_train, y_test = split_data(df)


#net = basic_CNN()
net = resnet((28,28,1))
train(net)
for i in range(10):
    net.fit(X_train, y_train, epochs = 5, validation_split = 0.2)
    scores = net.evaluate(X_test, y_test)
    print("\n%s: %.2f%%" % (net.metrics_names[1], scores[1]*100))
    print("Saving network.....")
    net.save('resnet_v9.h5')
    print("Network saved")
new_df = load_data(False)
X = np.array(new_df).reshape([-1,28,28,1])/255
pred = net.predict(X)
out  = np.argmax(pred, axis = 1)
out_df = pd.DataFrame(data = out)
out_df.columns = ['Label']
out_df.index = range(1,28001)
out_df['Imageld'] = range(1,28001)
out_df.to_csv('pred_v9.csv')
print(out, pred.shape)
out_df = pd.DataFrame(data = out)
out_df.columns = ['Label']
out_df.index = range(1,28001)
out_df['Imageld'] = range(1,28001)
out_df
