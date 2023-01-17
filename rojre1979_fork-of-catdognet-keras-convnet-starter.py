from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Convolution2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

TRAIN_DIR = '../input/train/'

TEST_DIR = '../input/test/'



# dimensions of our images.

img_width, img_height = 150, 150



train_data_dir = TRAIN_DIR

validation_data_dir = TEST_DIR

nb_train_samples = 2000

nb_validation_samples = 800

nb_epoch = 50

model = Sequential()

model.add(Convolution2D(32, 3, 3, input_shape=(3, img_width, img_height)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))



model.add(Convolution2D(32, 3, 3))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))



model.add(Convolution2D(64, 3, 3))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))



model.add(Flatten())

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(1))

model.add(Activation('sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])



# this is the augmentation configuration we will use for training

train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)



# this is the augmentation configuration we will use for testing:

# only rescaling

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        train_data_dir,

        target_size=(img_width, img_height),

        batch_size=32,

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        validation_data_dir,

        target_size=(img_width, img_height),

        batch_size=32,

        class_mode='binary')



model.fit_generator(

        train_generator,

        samples_per_epoch=nb_train_samples,

        nb_epoch=nb_epoch,

        validation_data=validation_generator,

        nb_val_samples=nb_validation_samples)



model.load_weights('first_try.h5')
def show_cats_and_dogs(idx):

    cat = read_image(train_cats[idx])

    dog = read_image(train_dogs[idx])

    pair = np.concatenate((cat, dog), axis=1)

    plt.figure(figsize=(10,5))

    plt.imshow(pair)

    plt.show()

    

for idx in range(0,5):

    show_cats_and_dogs(idx)
dog_avg = np.array([dog[0].T for i, dog in enumerate(train) if labels[i]==1]).mean(axis=0)

plt.imshow(dog_avg)

plt.title('Your Average Dog')
cat_avg = np.array([cat[0].T for i, cat in enumerate(train) if labels[i]==0]).mean(axis=0)

plt.imshow(cat_avg)

plt.title('Your Average Cat')
optimizer = RMSprop(lr=1e-4)

objective = 'binary_crossentropy'





def catdog():

    

    model = Sequential()



    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, ROWS, COLS), activation='relu'))

    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))



    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

    

    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))

    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

    

    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))

    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))

    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))



    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))

    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))

    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))



    model.add(Flatten())

    model.add(Dense(256, activation='relu'))

    model.add(Dropout(0.5))

    

    model.add(Dense(256, activation='relu'))

    model.add(Dropout(0.5))



    model.add(Dense(1))

    model.add(Activation('sigmoid'))



    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])

    return model





model = catdog()
nb_epoch = 10

batch_size = 16



## Callback for loss logging per epoch

class LossHistory(Callback):

    def on_train_begin(self, logs={}):

        self.losses = []

        self.val_losses = []

        

    def on_epoch_end(self, batch, logs={}):

        self.losses.append(logs.get('loss'))

        self.val_losses.append(logs.get('val_loss'))



early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')        

        

def run_catdog():

    

    history = LossHistory()

    model.fit(train, labels, batch_size=batch_size, nb_epoch=nb_epoch,

              validation_split=0.25, verbose=0, shuffle=True, callbacks=[history, early_stopping])

    



    predictions = model.predict(test, verbose=0)

    return predictions, history



predictions, history = run_catdog()
loss = history.losses

val_loss = history.val_losses



plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.title('VGG-16 Loss Trend')

plt.plot(loss, 'blue', label='Training Loss')

plt.plot(val_loss, 'green', label='Validation Loss')

plt.xticks(range(0,nb_epoch)[0::2])

plt.legend()

plt.show()
for i in range(0,10):

    if predictions[i, 0] >= 0.5: 

        print('I am {:.2%} sure this is a Dog'.format(predictions[i][0]))

    else: 

        print('I am {:.2%} sure this is a Cat'.format(1-predictions[i][0]))

        

    plt.imshow(test[i].T)

    plt.show()