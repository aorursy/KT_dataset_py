import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten, LeakyReLU



from tensorflow.keras.datasets import cifar10



import matplotlib.pyplot as plt
import requests

def send_message(text):

    # Send message to telegram

    # script from https://www.codementor.io/garethdwyer/building-a-telegram-bot-using-python-part-1-goi5fncay

    # list of proxie servers http://spys.one/proxys/DE/

    TOKEN = "" # your bot token

    URL = "https://api.telegram.org/bot{}/".format(TOKEN)

    chat_id = '' # your chat_id, get from URL + getUpdate



    ip = '192.169.156.211:7157'

    proxies = {'http':'socks5://{}'.format(ip),

           'https':'socks5://{}'.format(ip)}

    

    url = URL + "sendMessage?text={}&chat_id={}".format(text, chat_id)

    response = requests.get(url, proxies=proxies)

    content = response.json()

    return  print('Send message: {}.'.format(content['ok'])) 

send_message('test')
# load and preporcessing 

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, x_test = x_train / 255, x_test / 255



# break training set into training and validation sets

(x_train, x_valid) = x_train[10000:], x_train[:10000]

(y_train, y_valid) = y_train[10000:], y_train[:10000]
from tensorflow.keras.preprocessing.image import ImageDataGenerator



# create and configure augmented image generator

datagen_train = ImageDataGenerator(

    width_shift_range=0.2,  # randomly shift images horizontally (10% of total width)

    height_shift_range=0.2,  # randomly shift images vertically (10% of total height)

    horizontal_flip=True) # randomly flip images horizontally



# fit augmented image generator on data

datagen_train.fit(x_train)
# build model

model = Sequential([

    Conv2D(filters=128, kernel_size=2, strides=1, padding='same', input_shape=(32, 32, 3), activation=LeakyReLU(alpha=0.3)),

    MaxPool2D(pool_size=2, strides=2, padding='same'),

    Dropout(0.15),

    Conv2D(filters=384, kernel_size=2, strides=1, padding='same', activation=LeakyReLU(alpha=0.3)),

    MaxPool2D(pool_size=2, strides=2, padding='same'),

    Dropout(0.2),

    Conv2D(filters=512, kernel_size=2, strides=1, padding='same', activation=LeakyReLU(alpha=0.3)),

    MaxPool2D(pool_size=2, strides=2, padding='same'),

    Dropout(0.3),

    Conv2D(filters=1024, kernel_size=2, strides=1, padding='same', activation=LeakyReLU(alpha=0.3)),

    MaxPool2D(pool_size=2, strides=2, padding='same'),

    Dropout(0.3),

    Conv2D(filters=2048, kernel_size=2, strides=1, padding='same', activation=LeakyReLU(alpha=0.3)),

    MaxPool2D(pool_size=2, strides=2, padding='same'),

    Flatten(),

#     Dense(5070, activation='relu'),

#     Dropout(0.3),

    Dense(2048, activation='relu'),

    Dropout(0.3),

    Dense(1024, activation='relu'),

    Dropout(0.4),

    Dense(500, activation='relu'),

    Dropout(0.4),

    Dense(10, activation='softmax')

])

model.compile(loss='sparse_categorical_crossentropy',

             optimizer='adam',

             metrics=['accuracy'])

# model.summary()
# run without augmentation



# hist = model.fit(x_train, y_train, batch_size=512, epochs=40, verbose=0, validation_split=0.2)

# send_message('Done! val_acc: {:.3f}'.format(hist.history['val_acc'][-1]))

# plt.plot(hist.history['acc']);

# plt.plot(hist.history['val_acc']);

# text = ' max acc: {:.3f}\n max val_acc {:.3f}'.format(max(hist.history['acc']), max(hist.history['val_acc']))

# send_message(text)

# print(text)
%%time

batch_size = 512

hist = model.fit_generator(datagen_train.flow(x_train, y_train, batch_size=batch_size),

                          steps_per_epoch=x_train.shape[0] // batch_size, epochs=40, verbose=0,

                          validation_data=(x_valid, y_valid),

                          validation_steps=x_valid.shape[0] // batch_size)

# send_message('Done! val_acc: {:.3f}'.format(hist.history['val_acc'][-1]))

plt.plot(hist.history['acc']);

plt.plot(hist.history['val_acc']);

text = ' max acc: {:.3f}\n max val_acc {:.3f}'.format(max(hist.history['acc']), max(hist.history['val_acc']))

# send_message(text)

print(text)
score = model.evaluate(x_test, y_test, verbose=0)

print('Test accuracy: {:.3f}'.format(score[1]))