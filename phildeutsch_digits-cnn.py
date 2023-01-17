from matplotlib import pyplot as plt

import numpy as np

from PIL import Image

import pandas as pd

import itertools



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation

from keras.utils import np_utils

from keras.optimizers import RMSprop, SGD



%matplotlib inline
def read_train(filename):

    with open(filename) as f:

        lines = f.readlines()

    labels = [line.split(',')[0] for line in lines[1:]]

    digits = [[int(d) for d in line.split(',')[1:]] for line in lines[1:]]

    return labels, digits
[labels, digits] = read_train('../input/train.csv')

X_train, X_test, y_train, y_test = train_test_split(digits, labels, test_size=0.25, random_state=17)
print(X_train[3])

print(y_train[3])
def plot_digit(digit, label):

    picture = np.array(digit).reshape((28, 28))

    img = Image.fromarray(picture.astype('uint8'))

    plt.title(label)

    plt.imshow(picture)

    

    return None
plot_digit(X_train[3], y_train[3])
plot_digit(X_test[3], '???')
# convert class vectors to binary class matrices

y_train_1hot = np_utils.to_categorical(y_train, 10)

y_test_1hot = np_utils.to_categorical(y_test, 10)



model = Sequential()



model.add(Dense(32, input_shape=(784,)))

model.add(Activation('relu'))

model.add(Dropout(0.2))



model.add(Dense(32))

model.add(Activation('relu'))

model.add(Dropout(0.2))



model.add(Dense(10))

model.add(Activation('softmax'))



model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])



history = model.fit(X_train, y_train_1hot, batch_size=64, nb_epoch=10, verbose=1, validation_data=(X_test, y_test_1hot))

score = model.evaluate(X_test, y_test_1hot, verbose=0)

print('Test score:', score[0])

print('Test accuracy:', score[1])
def predict_labels(input_data):

    labels_predicted = model.predict(input_data, batch_size=32, verbose=0)

    predicted = pd.DataFrame(labels_predicted)

    predicted = predicted.idxmax(1)

    return(predicted.values)
y_test_predicted = [str(p) for p in predict_labels(X_test)]
cm = confusion_matrix(y_test, y_test_predicted)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    (This function is copied from the scikit docs.)

    """

    plt.figure()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(cm)

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

plot_confusion_matrix(cm, range(10))
def read_test(filename):

    with open(filename) as f:

        lines = f.readlines()

    digits = [[int(d) for d in line.split(',')] for line in lines[1:]]

    return digits

digits_test = read_test('../input/test.csv')



submission = pd.DataFrame({'ImageId':range(1,28001), 'Label':predict_labels(digits_test)})

submission.to_csv('submission.csv', index=False)

submission.head()