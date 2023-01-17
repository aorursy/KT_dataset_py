# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from scipy.io import loadmat

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn import svm

from sklearn.model_selection import train_test_split



from keras.models import Sequential

from keras.layers import Conv1D, Flatten, Dense, Activation



import matplotlib.pyplot as plt

import pickle



# read data

raw_data = loadmat('../input/Subject1_2D.mat')

print("Raw Data Type : ",type(raw_data))
raw_data.keys()
raw_data['readme']
electrode_names = "FP1 FP2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 FZ CZ PZ".split()

electrode_names_dictionary = {}

for i in range(len(electrode_names)):

    electrode_names_dictionary[i] = electrode_names[i]
# Create data.

left_forward1 = pandas.DataFrame(raw_data['LeftForward1']).rename(columns=electrode_names_dictionary)

left_forward2 = pandas.DataFrame(raw_data['LeftForward2']).rename(columns=electrode_names_dictionary)

left_forward3 = pandas.DataFrame(raw_data['LeftForward3']).rename(columns=electrode_names_dictionary)

left_forward_imagined = pandas.DataFrame(raw_data['LeftForwardImagined']).rename(columns=electrode_names_dictionary)



left_backward1 = pandas.DataFrame(raw_data['LeftBackward1']).rename(columns=electrode_names_dictionary)

left_backward2 = pandas.DataFrame(raw_data['LeftBackward2']).rename(columns=electrode_names_dictionary)

left_backward3 = pandas.DataFrame(raw_data['LeftBackward3']).rename(columns=electrode_names_dictionary)

left_backward_imagined = pandas.DataFrame(raw_data['LeftBackwardImagined']).rename(columns=electrode_names_dictionary)



right_forward1= pandas.DataFrame(raw_data['RightForward1']).rename(columns=electrode_names_dictionary)

right_forward2 = pandas.DataFrame(raw_data['RightForward2']).rename(columns=electrode_names_dictionary)

right_forward3 = pandas.DataFrame(raw_data['RightForward3']).rename(columns=electrode_names_dictionary)

right_forward_imagined = pandas.DataFrame(raw_data['RightForwardImagined']).rename(columns=electrode_names_dictionary)



right_backward1 = pandas.DataFrame(raw_data['RightBackward1']).rename(columns=electrode_names_dictionary)

right_backward2 =pandas.DataFrame(raw_data['RightBackward2']).rename(columns=electrode_names_dictionary)

right_backward3 = pandas.DataFrame(raw_data['RightBackward3']).rename(columns=electrode_names_dictionary)

right_backward_imagined = pandas.DataFrame(raw_data['RightBackwardImagined']).rename(columns=electrode_names_dictionary)



right_leg = pandas.DataFrame(raw_data['RightLeg']).rename(columns=electrode_names_dictionary)

left_leg = pandas.DataFrame(raw_data['LeftLeg']).rename(columns=electrode_names_dictionary)
print("LeftBackward1 Data Shape : ",left_backward1.shape)

print("LeftBackward2 Data Shape : ",left_backward2.shape)



# To find the real time of EEG recordings

print("LeftBackward1 Real Time (seconds) : ",left_backward1.shape[0]/500)

print("LeftBackward2 Real Time (seconds) : ",left_backward2.shape[0]/500)
left_backward1.sample(4)
left_backward2.sample(4)
left_backward1.describe()
left_backward2.describe()
plt.figure(figsize=(40,10))

plt.plot(left_backward1)

plt.title("Left Backward Data 1",size=25)

plt.legend(electrode_names)

plt.show()
plt.figure(figsize=(40,10))

plt.plot(left_backward2)

plt.title("Left Backward Data 2",size=25)

plt.legend(electrode_names)

plt.show()
fig, a = plt.subplots(3,2,figsize=(30,30))



a[0][0].plot(left_backward1[:500])

a[0][0].set_title("Sol Kol Arka",size=20)

a[0][0].legend(electrode_names)



a[0][1].plot(left_forward1[:500])

a[0][1].set_title("Sol Kol Ön",size=20)

a[0][1].legend(electrode_names)



a[1][0].plot(right_backward1[:500])

a[1][0].set_title("Sağ Kol Arka",size=20)

a[1][0].legend(electrode_names)



a[1][1].plot(right_forward1[:500])

a[1][1].set_title("Sağ Kol Ön",size=20)

a[1][1].legend(electrode_names)



a[2][0].plot(left_leg[:500])

a[2][0].set_title("Sol Ayak",size=20)

a[2][0].legend(electrode_names)



a[2][1].plot(right_leg[:500])

a[2][1].set_title("Sağ Ayak",size=20)

a[2][1].legend(electrode_names)



plt.show()
fig, a = plt.subplots(2,2,figsize=(30,20))



a[0][0].plot(left_backward1[:500])

a[0][0].set_title("Sol Kol Arka",size=20)

a[0][0].legend(electrode_names)



a[0][1].plot(left_backward_imagined[:500])

a[0][1].set_title("Sol Kol Arka Hayali",size=20)

a[0][1].legend(electrode_names)



a[1][0].plot(right_backward1[:500])

a[1][0].set_title("Sağ Kol Arka",size=20)

a[1][0].legend(electrode_names)



a[1][1].plot(right_backward_imagined[:500])

a[1][1].set_title("Sağ Kol Arka Hayali",size=20)

a[1][1].legend(electrode_names)





plt.show()
left_backward_labels_count = left_backward1.shape[0] + left_backward2.shape[0] + left_backward3.shape[0] + left_backward_imagined.shape[0]

left_forward_labels_count = left_forward1.shape[0] + left_forward2.shape[0] + left_forward3.shape[0] + left_forward_imagined.shape[0]

right_backward_labels_count = right_backward1.shape[0] + right_backward2.shape[0] + right_backward3.shape[0] + right_backward_imagined.shape[0]

right_forward_labels_count = right_forward1.shape[0] + right_forward2.shape[0]+ right_forward3.shape[0] + right_forward_imagined.shape[0]

left_leg_labels_count = left_leg.shape[0]

right_leg_labels_count = right_leg.shape[0]



left_forward_labels = pandas.DataFrame(['LeftForward' for _ in range(left_forward_labels_count)])

left_backward_labels = pandas.DataFrame(['LeftBackward' for _ in range(left_backward_labels_count)])

right_forward_labels = pandas.DataFrame(['RightForward' for _ in range(right_forward_labels_count)])

right_backward_labels = pandas.DataFrame(['RightBackward' for _ in range(right_backward_labels_count)])

right_leg_labels = pandas.DataFrame(['RightLeg' for _ in range(right_leg_labels_count)])

left_leg_labels = pandas.DataFrame(['LeftLeg' for _ in range(left_leg_labels_count)])
# Concat Data

data = pandas.concat([left_forward1,

                      left_forward2,

                      left_forward3,

                      left_forward_imagined,

                      left_backward1,

                      left_backward2,

                      left_backward3,

                      left_backward_imagined,

                      right_forward1,

                      right_forward2,

                      right_forward3,

                      right_forward_imagined,

                      right_backward1,

                      right_backward2,

                      right_backward3,

                      right_backward_imagined,

                      right_leg,

                      left_leg])

labels =pandas.concat([left_forward_labels,

                       left_backward_labels,

                       right_forward_labels,

                       right_backward_labels,

                       right_leg_labels,

                       left_leg_labels])
print("Data Shape : ",data.shape)

print("Labels Shape : ",labels.shape)



train_X, test_X, train_y, test_y = train_test_split(data,labels,test_size=0.2)



print("Train Data Shape : ",train_X.shape)

print("Train Labels Shape : ",train_y.shape)

print("Test Data Shape :",test_X.shape)

print("Test Labels Shape : ",test_y.shape)
# Create base machine learning models

machine_learning_models = {'Logistic Regression':LogisticRegression(),

                          'K-Neighbors Classifier':KNeighborsClassifier(),

                          'Decision Tree Classifier':DecisionTreeClassifier(),

                          'Random Forest Classifier':RandomForestClassifier()}

print("Models created !")
print("Training Models...")

for i in machine_learning_models.values():

    i.fit(train_X,train_y)

print("Models training completed !")
models_success_rates = {}

print("Models Simulating...")

print("Models Success Rate Calculating...")

for i,j in zip(machine_learning_models.keys(),machine_learning_models.values()):

    rate = (j.score(test_X,test_y))*100

    models_success_rates[i]=rate

    print(str(i)+" models success rate : %",rate)
fig = plt.figure(figsize=(10,5))

ax = fig.add_axes([0,0,1,1])

plt.title("Machine Learning Models Success Rate",size=15)

keys = models_success_rates.keys()

values = models_success_rates.values()

ax.bar(keys,values)

plt.show()
# Save all models on different files

for i,j in zip(machine_learning_models.keys(),machine_learning_models.values()):

    pickle.dump(j,open(str(i)+".sav",'wb'))
# Create new train and test data for CNN Network

data_arr = np.array(data) 

label_arr  = np.array(pandas.get_dummies(labels))



X_train,X_test,y_train,y_test = train_test_split(data_arr,label_arr,test_size=0.2)

# Create CNN Network

model = Sequential()

model.add(Dense(64, activation='relu', input_dim=19))

model.add(Dense(128,activation='relu'))

model.add(Dense(6, activation='softmax'))

model.compile(optimizer='rmsprop',

              loss='categorical_crossentropy',

              metrics=['accuracy'])



# Train the model, iterating on the data in batches of 32 samples

model.fit(X_train,y_train,validation_data=(X_test,y_test), epochs=10, batch_size=250)
model.save("cnnmodel.h5")
history = model.history

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()