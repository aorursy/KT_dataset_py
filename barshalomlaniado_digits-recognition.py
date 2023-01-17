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
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

import tensorflow as tf

from tensorflow import keras

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
if data.isnull().any().sum() == 0:

    print("There are no missing values")

else:

    print("There are","data.isnull().any().sum()","missing values")
min_match = data.min().min() == 0

max_match = data.max().max() == 255

print("The value of the minimum pixel in the data is: ", data.min().min())

print("Does it match what we expected?", min_match)

print("-------------")

print("The value of the maximum pixel we expected is: ", data.max().max())

print("Does it match what we expected?", max_match)

print("-------------")

print("Is the value range of the pixels as we expected?: ", (min_match == True & max_match == True))

data.info()
sns.countplot(x = 'label', data= data)

plt.title("Distribution of labels - full dataset")

plt.show()
X = data.iloc[:,1:]

y = data['label']

print("There are", X.shape[0], "examples in the full data and", X.shape[1], "features")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print("There are", X_train.shape[0], "examples in the training data set") 
sns.countplot(x = y_train, data= y_train)

plt.title("Distribution of labels - Train")

plt.show()
X_train = np.array(X_train) / 255.0
num_digits = len(y_train.unique())

y_train = keras.utils.to_categorical(y_train, num_digits)

model = keras.Sequential()

model.add(tf.keras.layers.Dense(784, activation='relu', input_shape = (784,)))

model.add(tf.keras.layers.Dense(100, activation='relu'))

model.add(tf.keras.layers.Dense(784, activation='relu'))

model.add(tf.keras.layers.Dense(100, activation='relu'))

model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

fit_model = model.fit(X_train, y_train, epochs= 60)



X_test_array = np.array(X_test) / 255.0
nns_predict = model.predict_classes(X_test_array)
(nns_predict == y_test).sum() / len(y_test)
test_result = pd.DataFrame(y_test)

test_result.reset_index(inplace = True)

test_result.drop('index', axis = 1,inplace = True)

test_result['Predict_label'] = nns_predict

test_result['Correct_prediction'] = test_result['label'] == test_result['Predict_label']
test_result
test_result1 = test_result[test_result['Correct_prediction'] == False]
sns.countplot(x = "Predict_label", data= test_result1)

plt.show()
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    import itertools

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()
cm = confusion_matrix(y_test, test_result['Predict_label'],np.sort(y_test.unique()))


plot_confusion_matrix(cm,np.sort(y_test.unique()),title='Confusion matrix')
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
test['Label'] = model.predict_classes(test) 

test['ImageId'] = np.array(range(1,test.shape[0] + 1))

test = test[['ImageId', 'Label']]
test.to_csv('submission2.csv', index=False)

print('Your submission was successfully saved!')