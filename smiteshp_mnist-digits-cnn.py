# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

print(tf.__version__)
import csv



sample_submission = '/kaggle/input/digit-recognizer/sample_submission.csv'

test_file = '/kaggle/input/digit-recognizer/test.csv'

train_file = '/kaggle/input/digit-recognizer/train.csv'



with open(sample_submission, newline='') as csvfile:

    reader = csv.reader(csvfile, delimiter=',', quotechar='|')

    i = 0

    for row in reader:

        if i > 10:

            break

        i += 1

        print(', '.join(row))
import csv



sample_submission = '/kaggle/input/digit-recognizer/sample_submission.csv'

test_file = '/kaggle/input/digit-recognizer/test.csv'

train_file = '/kaggle/input/digit-recognizer/train.csv'



train_labels = list()

train_images = list()



skip_header = True



with open(train_file, newline='') as csvfile:

    reader = csv.reader(csvfile, delimiter=',', quotechar='|')

    for row in reader:

        if skip_header:

            skip_header = False

            continue

        else:

            train_labels.append(np.array(row[0], dtype = np.float32))

            temp_img = np.array(row[1:], dtype = np.float32)

            temp_img = np.reshape(temp_img, (28, 28))

            train_images.append(temp_img)

            

train_labels = np.array(train_labels)

train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)

train_images = np.array(train_images)

train_images = np.expand_dims(train_images, axis = 3)

train_images /= 255
print(train_images.shape)

print(train_labels.shape)
import matplotlib.pyplot as plt

%matplotlib inline
sample_index = 10



print('digit is a {}'.format(np.argmax(train_labels[sample_index])))

image = train_images[sample_index]

image.resize(28, 28)



fig = plt.figure

plt.imshow(image, cmap='gray')

plt.show()
num_epochs = 5

batch_sz = 32

learn_rate = 0.001
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=[28, 28, 1]),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.Dense(10, activation='softmax')

])



model.summary()
model.compile(

    optimizer = tf.keras.optimizers.Adam(lr=learn_rate),

    loss = 'categorical_crossentropy',

    metrics = ['accuracy']

)
history = model.fit(

                train_images,

                train_labels,

                epochs = num_epochs,

                batch_size = batch_sz

            )
plt.plot(history.history['accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.show()
plt.plot(history.history['loss'])

plt.title('model loss')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.show()
import csv



sample_submission = '/kaggle/input/digit-recognizer/sample_submission.csv'

test_file = '/kaggle/input/digit-recognizer/test.csv'

train_file = '/kaggle/input/digit-recognizer/train.csv'



test_images = list()



skip_header = True



with open(test_file, newline='') as csvfile:

    reader = csv.reader(csvfile, delimiter=',', quotechar='|')

    for row in reader:

        if skip_header:

            skip_header = False

            continue

        else:

            temp_img = np.array(row, dtype = np.float32)

            temp_img = np.reshape(temp_img, (28, 28))

            test_images.append(temp_img)

            

test_images = np.array(test_images)

test_images = np.expand_dims(test_images, axis = 3)

test_images /= 255
print(test_images.shape)
predict = model.predict(test_images)

predict = np.argmax(predict, axis=1)
print(predict)

print(predict.shape)
import csv



submission_file = '/kaggle/working/submission.csv'

open(submission_file, 'a').close()



skip_header = True



with open(submission_file, 'w') as csvfile:

    filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    filewriter.writerow(['ImageId', 'Label'])

    for img_id, label in enumerate(predict):

        filewriter.writerow([img_id+1, label])