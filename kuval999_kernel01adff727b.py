import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras import Sequential
from keras.models import model_from_json
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam


def f_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32,kernel_size=(3,3), activation='relu', padding='same',input_shape=input_shape))
    #model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Conv2D(64,kernel_size=(3,3), activation='relu', padding='same'))
    #model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Conv2D(128,kernel_size=(3,3), activation='relu', padding='same'))
    #model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(),
                  metrics=['accuracy'])
    
    model.summary()

    return model


def train(model,train_images,train_labels,batch_size,epochs,model_weights_file):
    train_model = model.fit(train_images, train_labels,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1)
    train_loss = train_model.history['loss']
    train_accuracy = train_model.history['acc']
    
    train_model_json = model.to_json()
    with open('model.json', 'w') as file:
        file.write(train_model_json)
    model.save_weights(model_weights_file)
    
    return train_loss, train_accuracy


def evaluate(model_weights_file, test_images, test_labels):
    with open('model.json','r') as file:
        loaded_model_json = file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_weights_file)
    
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(),
                  metrics=['accuracy'])
    
    eval_model = model.evaluate(test_images, test_labels,
                                verbose=0)
    test_loss = eval_model[0]
    test_accuracy = eval_model[1]
    
    predicted_classes = model.predict(test_images)
    predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
    
    #accuracy_score2 = accuracy_score(y_true=test_labels, y_pred=predicted_classes)
    
    print("Test Loss {}".format(test_loss))
    print("Test Accuracy {}".format(test_accuracy))
    #print("Accuracy Score (Metrics) {}".format(accuracy_score2))
    
    return test_loss, test_accuracy, predicted_classes


def get_data():
    data_train = pd.read_csv('../input/train.csv')
    data_test = pd.read_csv('../input/test.csv')
    
    print('Train data shape: {}'.format(data_train.shape))
    print('Test data shape: {}'.format(data_test.shape))
    
    X = np.array(data_train.iloc[:,1:])
    y = to_categorical(np.array(data_train.iloc[:,0]))
    
    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=42)
    
    X_test = np.array(data_test.iloc[:,:])
    #y_test = to_categorical(np.array(data_test.iloc[:,0]))
    
    X_train = X_train.reshape(X_train.shape[0], 28,28,1)
    X_test = X_test.reshape(X_test.shape[0], 28,28,1)
    X_val = X_val.reshape(X_val.shape[0], 28,28,1)
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_val = X_val.astype('float32')
    X_train /= 255.
    X_test /= 255.
    X_val /= 255.
    
    print('X_train shape {}, y_train shape {}'.format(X_train.shape, y_train.shape))
    print('X_val shape {}, y_val shape {}'.format(X_val.shape, y_val.shape))
    print('X_test shape {}'.format(X_test.shape))
    
    return X_train, X_val, X_test, y_train, y_val, (28,28,1)





X_train, X_val, X_test, y_train, y_val, input_shape = get_data()

model = f_model(input_shape)

train_loss, train_accuracy = train(model, X_train, y_train, 64, 5, 'model_weights.h5')

test_loss, test_accuracy, accuracy, y_pred = evaluate('model_weights.h5',X_val,y_val)

