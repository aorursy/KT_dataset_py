import os, cv2, random

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

from matplotlib import ticker

import seaborn as sns

%matplotlib inline 



from keras.models import Sequential

from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation

from keras.optimizers import RMSprop

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

from keras.utils import np_utils
TRAIN_DIR = '../input/train/'

TEST_DIR = '../input/test/'



ROWS = 64

COLS = 64

CHANNELS = 3



train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset

train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]

train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]



#test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]

test_images =  [TEST_DIR+str(i)+".jpg" for i in range(1, 12500+1)]





# slice datasets for memory efficiency on Kaggle Kernels, delete if using full dataset

#train_images = train_dogs[:1000] + train_cats[:1000]

train_images = train_dogs[:2000] + train_cats[:2000]

random.shuffle(train_images)

#test_images =  test_images[:25]



def read_image(file_path):

    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE

    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)





def prep_data(images):

    count = len(images)

    data = np.ndarray((count, ROWS, COLS, CHANNELS), dtype=np.uint8)



    for i, image_file in enumerate(images):

        data[i] = read_image(image_file)

        if i%250 == 0: print('Processed {} of {}'.format(i, count))

    

    return data



train = prep_data(train_images)

test = prep_data(test_images)



print("Train shape: {}".format(train.shape))

print("Test shape: {}".format(test.shape))
labels = []

for i in train_images:

    if 'dog' in i:

        labels.append(1)

    else:

        labels.append(0)



sns.countplot(labels)

sns.plt.title('Cats and Dogs')
def show_cats_and_dogs(idx):

    cat = read_image(train_cats[idx])

    dog = read_image(train_dogs[idx])

    pair = np.concatenate((cat, dog), axis=1)

    plt.figure(figsize=(10,5))

    plt.imshow(pair)

    plt.show()

    

for idx in range(0,5):

    show_cats_and_dogs(idx)
dog_avg = np.array([dog for i, dog in enumerate(train) if labels[i]==1]).mean(axis=0).mean(axis=2)

plt.imshow(dog_avg)

plt.title('Your Average Dog')
cat_avg = np.array([cat for i, cat in enumerate(train) if labels[i]==0]).mean(axis=0).mean(axis=2)

plt.imshow(cat_avg)

plt.title('Your Average Cat')
optimizer = RMSprop(lr=1e-4)

objective = 'binary_crossentropy'





def catdog():

    

    model = Sequential()



    model.add(Conv2D(32, 3, padding='same', activation='relu', input_shape=(ROWS, COLS, 3)))

    #model.add(Conv2D(32, 3, padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Conv2D(64, 3, padding='same', activation='relu'))

    #model.add(Conv2D(64, 3, padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    

    model.add(Conv2D(128, 3, padding='same', activation='relu'))

    #model.add(Conv2D(128, 3, padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    

    model.add(Conv2D(256, 3, padding='same', activation='relu'))

    #model.add(Conv2D(256, 3, padding='same', activation='relu'))

#     model.add(Conv2D(256, 3, padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Conv2D(256, 3, padding='same', activation='relu'))

#     model.add(Conv2D(256, 3, padding='same', activation='relu'))

#     model.add(Conv2D(256, 3, padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))



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
epochs = 10

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

    model.fit(train, labels, batch_size=batch_size, epochs=epochs,

              validation_split=0.25, verbose=1, shuffle=True, callbacks=[history, early_stopping])

    



    predictions = model.predict(test, verbose=0)

    return predictions, history



predictions, history = run_catdog()

predictions = predictions[:,0]
loss = history.losses

val_loss = history.val_losses



plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.title('VGG-16 Loss Trend')

plt.plot(loss, 'blue', label='Training Loss')

plt.plot(val_loss, 'green', label='Validation Loss')

plt.xticks(range(0, epochs)[0::2])

plt.legend()

plt.show()
submission = pd.DataFrame({"id": range(1, predictions.shape[0] + 1), "label": predictions})

submission.to_csv("submission.csv", index = False)
for i in range(0,10):

    if predictions[i] >= 0.5: 

        print('I am {:.2%} sure this is a Dog'.format(predictions[i]))

    else: 

        print('I am {:.2%} sure this is a Cat'.format(1-predictions[i]))

        

    plt.imshow(cv2.cvtColor(test[i], cv2.COLOR_BGR2RGB))

    plt.show()