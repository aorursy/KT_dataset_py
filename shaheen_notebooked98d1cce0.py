## CNN

img_size = 128

    

def cnn_model():

    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(1, img_size, img_size), activation='relu'))

    model.add(Convolution2D(32, 3, 3, activation='relu',border_mode='same'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))

    model.add(Convolution2D(64, 3, 3, activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))



    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))

    model.add(Convolution2D(128, 3, 3, activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))



    model.add(Flatten())

    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(1))

    model.add(Activation('sigmoid'))



    return model





model1 = cnn_model()





model1.compile(loss='binary_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])