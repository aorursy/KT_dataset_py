import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

print(os.listdir("../input"))

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix

from sklearn.utils.multiclass import unique_labels

from sklearn.metrics import accuracy_score

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/creditcard.csv")

data.head(10)
#Pandas Series

data['Amount']
#Numpy Array

data['Amount'].values
data['Amount'].values.shape
data['Amount'].values.shape[-1]
data['Amount'].values.reshape(-1,1).shape
#Data Preprocessing

data['normalized_amt'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))

data.head()
data = data.drop(['Amount'], axis=1)

data.head()
data = data.drop(['Time'], axis=1)

data.head()
#Split Data in features and labels

x = data.iloc[:, data.columns!= 'Class']

y = data.iloc[:, data.columns== 'Class']
x.head()
y.head()
#Split data in train, val and test

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=0)
#No. of entries in train and test data

for i in [x_train, x_test, y_train, y_test]:

    print(i.shape)
x_train.head() #dataframe
np.array(x_train) #dataframe to numpy array
x_train = np.array(x_train)

x_test = np.array(x_test)

y_train = np.array(y_train)

y_test = np.array(y_test)
x_train
#Nueral Network

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout
#input_dim -> no. of columns we are inputing(29 columns)

#we will hv a  0.5 probability of dropping each node



model = Sequential([Dense(units = 16, input_dim = 29, activation = 'relu'),

                    Dense(units = 24,activation = 'relu'),

                    Dropout(0.5),

                    Dense(20,activation = 'relu'),

                    Dense(24,activation = 'relu'),

                    Dense(1, activation = 'sigmoid')])



model.summary()
#Training 

model.compile(optimizer='adam', loss = 'binary_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train, batch_size=15, epochs=5)
score = model.evaluate(x_test,y_test)

print(score)
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='None',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

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

    plt.xticks(tick_marks, classes, rotation = 45)

    plt.yticks(tick_marks, classes)

    

    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            plt.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()
y_pred = model.predict(x_test)

y_pred.shape
y_test.shape
y_test = pd.DataFrame(y_test)

y_test.head()
cnf_matrix = confusion_matrix(y_test, y_pred.round())

print(cnf_matrix)
plot_confusion_matrix(cnf_matrix, classes = [0,1])
#confusion matrix on entire dataset



y_pred1 = model.predict(x)

y_actual = pd.DataFrame(y)

cnf_matrix1 = confusion_matrix(y_actual, y_pred1.round())

plot_confusion_matrix(cnf_matrix1, classes = [0,1])