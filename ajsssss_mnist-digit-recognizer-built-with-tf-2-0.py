import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
training_dataset = pd.read_csv('/kaggle/input/digit-recognizer/train.csv', header=0)

training_dataset.head()
bincount = np.bincount(training_dataset.label)

print(dict(zip(np.nonzero(bincount)[0], bincount)))
y = training_dataset['label'].copy()

X = training_dataset.drop(['label'], axis='columns')

del training_dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
print('X train shape: {}'.format(X_train.shape))

print('y train shape: {}'.format(y_train.shape))

print('X test shape: {}'.format(X_test.shape))

print('y test shape: {}'.format(y_test.shape))
train_class_counter = dict(zip(np.nonzero(np.bincount(y_train))[0], np.bincount(y_train)))

print(train_class_counter)
test_class_counter = dict(zip(np.nonzero(np.bincount(y_test))[0], np.bincount(y_test)))

print(test_class_counter)
X_train = np.array(X_train)

y_train = np.array(y_train)

X_test = np.array(X_test)

y_test = np.array(y_test)



X_train = X_train.reshape(-1, 28, 28)

X_test = X_test.reshape(-1, 28, 28)



print('X train shape: {}'.format(X_train.shape))

print('y train shape: {}'.format(y_train.shape))

print('X test shape: {}'.format(X_test.shape))

print('y test shape: {}'.format(y_test.shape))
def show_digits(X=X_train, y=y_train, n=10, figsize=(20,3)):

    

    fig, axes = plt.subplots(1, n, figsize=figsize)

    for i in range(n):

        ax = axes[i]

        ax.imshow(X[i], cmap='gray_r')

        ax.set_title(str(y[i]), fontsize=12, color='white')

    plt.tight_layout()

    plt.show()
show_digits(n=13, figsize=(26, 2))
X_train = X_train / 255.

X_test = X_test / 255.
show_digits(n=13, figsize=(26, 2))
from tensorflow.keras.layers import Flatten, Dense, Dropout

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam
model = Sequential([

    Flatten(input_shape=(28, 28)),

    Dense(units=128, activation='relu'),

    Dropout(0.2),

    Dense(units=64, activation='relu'),

    Dropout(0.15),

    Dense(units=10, activation='softmax')

])



optim = Adam(learning_rate=0.0001, epsilon=1e-8)



model.compile(optimizer=optim, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()
history = model.fit(X_train, y_train, epochs=200, batch_size=1024)
model.evaluate(X_test, y_test, verbose=2)
metrics = pd.DataFrame(history.history)

metrics.head()
epochs = range(1, 201)

f, axes = plt.subplots(1, 2, figsize=(16, 8))

sns.set()

sns.lineplot(x=epochs, y=metrics.loss, ax=axes[0]).set(title='Training loss', xlabel='Epochs', ylabel='Loss value')

sns.lineplot(x=epochs, y=metrics.accuracy, ax=axes[1], color='orange').set(title='Training accuracy', xlabel='Epochs', ylabel='Accuracy value')

plt.show()
submission_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

submission_data.head()
submission_data = np.array(submission_data)

submission_data = submission_data.reshape(-1, 28, 28)

print(submission_data[0])
submission_data = submission_data / 255.

submission_classes = model.predict_classes(submission_data)

print(submission_classes[0])
submission_classes.shape
sub = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

sub.head()
sample_sub = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

sample_sub.head()
ids = [i for i in range(1, sub.shape[0] + 1)]

submission = pd.DataFrame({'ImageId': ids, 'Label': submission_classes})

submission.head()
filename = 'Mnist_submission.csv'

submission.to_csv(filename, index=False)