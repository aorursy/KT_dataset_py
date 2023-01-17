# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_file = "../input/train.csv"

#data = np.loadtxt(train_file, skiprows=1, delimiter=',')

data = np.genfromtxt(train_file, dtype= None, delimiter=',|\"', skip_header=1, encoding='ascii')

data.shape
import pandas as pd  

import csv

df = pd.read_csv(train_file,sep=',',skipinitialspace=True,quotechar='"',engine='python')

df
from sklearn.feature_extraction import DictVectorizer

vec = DictVectorizer()

#df[['Ticket', 'Fare']]

dict_values = df[['Sex']].T.to_dict().values()

gender_one_hot_arr = vec.fit_transform(dict_values).toarray()

gender_one_hot_df = pd.DataFrame(gender_one_hot_arr, columns=vec.get_feature_names())

gender_one_hot_df
df = pd.concat((df, gender_one_hot_df), axis=1)

df
from sklearn.feature_extraction import DictVectorizer

vec = DictVectorizer()

#df[['Ticket', 'Fare']]

dict_values = df[['Embarked']].T.to_dict().values()

embarked_one_hot_arr = vec.fit_transform(dict_values).toarray()

embarked_one_hot_df = pd.DataFrame(embarked_one_hot_arr, columns=vec.get_feature_names())

embarked_one_hot_df
df = pd.concat((df, embarked_one_hot_df), axis=1)

df
df['Cabin'] = df['Cabin'].astype(str).apply(lambda x : x[0])

df
from sklearn.feature_extraction import DictVectorizer

vec = DictVectorizer()

#df[['Ticket', 'Fare']]

dict_values = df[['Cabin']].T.to_dict().values()

cabin_one_hot_arr = vec.fit_transform(dict_values).toarray()

cabin_one_hot_df = pd.DataFrame(cabin_one_hot_arr, columns=vec.get_feature_names())

cabin_one_hot_df
df = pd.concat((df, cabin_one_hot_df), axis=1)

df
features = ['Pclass', 'SibSp', 'Fare', 'Sex=male', 'Sex=female', 'Embarked=C', 'Embarked=Q', 'Embarked=S']

features.extend(vec.get_feature_names())

features
x = df[features].values

x.shape
x = x.reshape(x.shape[0], 17, 1)

x.shape
y = df[['Survived']].values[:,0]

y.shape
from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv1D



mnist_model = Sequential()



num_classes = 2
mnist_model.add(Conv1D(12, kernel_size=1, activation="relu", input_shape=(x.shape[1], 1)))

mnist_model.add(Conv1D(20, activation='relu', kernel_size=1, strides=2))

mnist_model.add(Conv1D(20, activation='relu', kernel_size=1))

mnist_model.add(Flatten())

mnist_model.add(Dense(100, activation='relu'))

mnist_model.add(Dense(num_classes, activation='softmax'))

mnist_model.summary()
mnist_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
callbacks_list = [keras.callbacks.EarlyStopping(monitor='acc', patience=30)]

mnist_model.fit(x, y, batch_size=100, epochs = 30, validation_split = 0.2, callbacks = callbacks_list, verbose=1)
result = mnist_model.predict(x, verbose=1)
import numpy as np

import matplotlib.pyplot as plt



from sklearn import svm, datasets

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.utils.multiclass import unique_labels





def plot_confusion_matrix(y_true, y_pred, classes,

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

    classes = classes[unique_labels(y_true, y_pred)]

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

           xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax
predict = result.argmax(axis=1)

predict
actual = y

actual
classes = np.unique(predict)

classes
# Plot non-normalized confusion matrix

plot_confusion_matrix(actual, predict, classes=classes,

                      title='Confusion matrix, without normalization')



# Plot normalized confusion matrix

plot_confusion_matrix(actual, predict, classes=classes, normalize=True,

                      title='Normalized confusion matrix')
test_file = "../input/test.csv"

#data = np.loadtxt(train_file, skiprows=1, delimiter=',')

data = np.genfromtxt(test_file, dtype= None, delimiter=',|\"', skip_header=1, encoding='ascii')

data.shape
import pandas as pd  

import csv

df = pd.read_csv(test_file,sep=',',skipinitialspace=True,quotechar='"',engine='python')

df
from sklearn.feature_extraction import DictVectorizer

vec = DictVectorizer()

#df[['Ticket', 'Fare']]

dict_values = df[['Sex']].T.to_dict().values()

gender_one_hot_arr = vec.fit_transform(dict_values).toarray()

gender_one_hot_df = pd.DataFrame(gender_one_hot_arr, columns=vec.get_feature_names())

gender_one_hot_df



df = pd.concat((df, gender_one_hot_df), axis=1)



vec = DictVectorizer()

#df[['Ticket', 'Fare']]

dict_values = df[['Embarked']].T.to_dict().values()

embarked_one_hot_arr = vec.fit_transform(dict_values).toarray()

embarked_one_hot_df = pd.DataFrame(embarked_one_hot_arr, columns=vec.get_feature_names())

embarked_one_hot_df



df = pd.concat((df, embarked_one_hot_df), axis=1)



features = ['Pclass', 'SibSp', 'Fare', 'Sex=male', 'Sex=female', 'Embarked=C', 'Embarked=Q', 'Embarked=S']

features.extend(vec.get_feature_names())

x = x.reshape(x.shape[0], 17, 1)

x.shape

result = mnist_model.predict(x, verbose=1)
output_df = df[['PassengerId']]

output_df['Survived'] = pd.DataFrame(result.argmax(axis=1))

output_df
output_df.to_csv('output.csv', index=False)