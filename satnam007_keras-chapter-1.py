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
from random import randint

from sklearn.utils import shuffle

from sklearn.preprocessing import MinMaxScaler
train_labels = []

train_samples = []
for i in range(50):

    # The ~5% of younger individuals who did experience side effects

    random_younger = randint(13,64)

    train_samples.append(random_younger)

    train_labels.append(1)

#     print(train_samples)



    # The ~5% of older individuals who did not experience side effects

    random_older = randint(65,100)

    train_samples.append(random_older)

    train_labels.append(0)



for i in range(1000):

    # The ~95% of younger individuals who did not experience side effects

    random_younger = randint(13,64)

    train_samples.append(random_younger)

    train_labels.append(0)



    # The ~95% of older individuals who did experience side effects

    random_older = randint(65,100)

    train_samples.append(random_older)

    train_labels.append(1)
#creating dataframe and saving the data

df = pd.DataFrame(list(zip(train_samples, train_labels)), 

               columns =['Sample', 'label']) 

df 
df.to_csv (r'dataframe.csv', index = False, header=True)

train_labels = np.array(train_labels)

train_samples = np.array(train_samples)

train_labels, train_samples = shuffle(train_labels, train_samples)
scaler = MinMaxScaler(feature_range=(0,1))

scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))

print(df.head())

print(scaled_train_samples)
type(scaled_train_samples)
scaled_train_samples.shape
train_labels.shape
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Activation, Dense

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.optimizers import Adadelta

from tensorflow.keras.metrics import categorical_crossentropy
# or you can do like this -->  model.add(l4)

# so i prefer to do it like this --> model = sequential([l1,l2,l3])
model = Sequential([

    Dense(units=16, input_shape=(1,), activation='relu'),

    Dense(units=32, activation='relu'),

    #Dense(units=2, activation='sigmoid')

    Dense(units=2, activation='softmax')

])
model.summary()
model.compile(optimizer=Adadelta(learning_rate=0.1), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x=scaled_train_samples, y=train_labels, batch_size=10, epochs=30, verbose=2)
model.fit(x=scaled_train_samples, y=train_labels,validation_split = 0.1, batch_size=10, epochs=30, verbose=2)
test_labels =  []

test_samples = []



for i in range(10):

    # The 5% of younger individuals who did experience side effects

    random_younger = randint(13,64)

    test_samples.append(random_younger)

    test_labels.append(1)

    

    # The 5% of older individuals who did not experience side effects

    random_older = randint(65,100)

    test_samples.append(random_older)

    test_labels.append(0)



for i in range(200):

    # The 95% of younger individuals who did not experience side effects

    random_younger = randint(13,64)

    test_samples.append(random_younger)

    test_labels.append(0)

    

    # The 95% of older individuals who did experience side effects

    random_older = randint(65,100)

    test_samples.append(random_older)

    test_labels.append(1)



test_labels = np.array(test_labels)

test_samples = np.array(test_samples)



test_labels, test_samples = shuffle(test_labels, test_samples)



scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1,1))

print(scaled_test_samples[0:5,:])
predictions = model.predict(

      x=scaled_test_samples

    , batch_size=10

    , verbose=0

)
type(predictions)
for i in predictions[:10]:

    print(i)
predictions[:10]
rounded_predictions = np.argmax(predictions, axis=-1)

rounded_predictions[:10]
%matplotlib inline

from sklearn.metrics import confusion_matrix

import itertools

import matplotlib.pyplot as plt
cm = confusion_matrix(y_true=test_labels, y_pred=rounded_predictions)
def plot_confusion_matrix(cm, classes,

                        normalize=False,

                        title='Confusion matrix',

                        cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

            horizontalalignment="center",

            color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
cm_plot_labels = ['no_side_effects','had_side_effects']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
model.save('medical_trial_model.h5')
from tensorflow.keras.models import load_model

new_model = load_model('medical_trial_model.h5')
new_model.summary()
new_model.optimizer
new_model.loss
json_string = model.to_json()

json_string
from tensorflow.keras.models import model_from_json

model_architecture = model_from_json(json_string)
model_architecture.summary()
model.save_weights('my_model_weights.h5')
model2 = Sequential([

    Dense(units=16, input_shape=(1,), activation='relu'),

    Dense(units=32, activation='relu'),

    Dense(units=2, activation='softmax')

])



model2.load_weights('my_model_weights.h5')