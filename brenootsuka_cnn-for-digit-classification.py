import pandas as pd



from keras.utils import to_categorical
train_fname = '/kaggle/input/digit-recognizer/train.csv'



train_db = pd.read_csv(train_fname)



train_label = train_db['label'].to_numpy()

train_label = to_categorical(train_label, num_classes=10)



train_input = train_db.drop('label', axis=1).to_numpy()

train_input = train_input.reshape((train_input.shape[0], 28,28,1))

train_input = train_input.astype('float32')/ 255
from keras.regularizers import l2

from keras.models import Sequential

from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
model = Sequential()



model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(MaxPooling2D((2,2)))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D((2,2)))



model.add(Flatten())



model.add(Dropout(0.5))

model.add(Dense(256, activation='relu', kernel_regularizer=l2(1e-3)))

model.add(Dense( 10, activation='softmax'))



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()
perf = model.fit(train_input,

                 train_label,

                 epochs=30,

                 batch_size=128,

                 validation_split=0.2)
import matplotlib.pyplot as plt





plt.plot(perf.history['acc'], label='training accuracy')

plt.plot(perf.history['val_acc'], label='validation accuracy')

plt.title('training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(perf.history['loss'], label='training loss')

plt.plot(perf.history['val_loss'], label='validation loss')

plt.title('training and validation loss')

plt.legend()



plt.show()
test_fname = '/kaggle/input/digit-recognizer/test.csv'



test_db = pd.read_csv(test_fname)



test_input = test_db.to_numpy()

test_input = test_input.reshape((test_input.shape[0], 28,28,1))

test_input = test_input.astype('float32')/ 255
predictions = model.predict_classes(test_input)

imageid = list(range(1,len(predictions)+1))



submission = pd.DataFrame({"ImageId": imageid, "Label": predictions})

submission.to_csv('submission.csv', index=False, header=True)