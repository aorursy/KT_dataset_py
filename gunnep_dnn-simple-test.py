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
import pandas as pd

import keras

import sklearn



data = pd.read_csv('../input/deep-learning-az-ann/Churn_Modelling.csv')

data.head(3)
data_to_analyze = data.drop(['CustomerId', 'Surname','Exited','RowNumber'], axis=1)

labels = data['Exited']



#substitute values

mapping_geo = {'France': 1, 'Germany': 2, 'Spain':3}

mapping_gender = {'Female': 1, 'Male': 2}



data_to_analyze = data_to_analyze.replace({'Geography': mapping_geo, 'Gender': mapping_gender})

#let's prepare the DNN by converting the tuples in numpy vectors

import numpy as np



labels_arr = np.asarray(labels)

data_to_analyze_arr = np.asarray(data_to_analyze)



print(data_to_analyze_arr.shape)

print(labels_arr.shape)
#let's split train and test arrays



print(data_to_analyze_arr.shape[0])



data_to_analyze_arr_train = data_to_analyze_arr[:int(data_to_analyze_arr.shape[0]/2),:]

data_to_analyze_arr_test = data_to_analyze_arr[int(data_to_analyze_arr.shape[0]/2):,:]



labels_arr_train = labels_arr[:int(data_to_analyze_arr.shape[0]/2)]

labels_arr_test = labels_arr[int(data_to_analyze_arr.shape[0]/2):]
#Keras model

from keras.models import Sequential

from keras.layers import Dense



model = Sequential()

model.add(Dense(260, input_dim=10, activation='tanh'))

model.add(Dense(100, activation='tanh'))

model.add(Dense(10, activation='tanh'))

model.add(Dense(1, activation='relu'))



print(model.summary())



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset

history = model.fit(data_to_analyze_arr_train, labels_arr_train, epochs=10, batch_size=64)

# evaluate the keras model

_, accuracy = model.evaluate(data_to_analyze_arr_train, labels_arr_train)

print('Train accuracy: %.2f' % (accuracy*100))



_, accuracy = model.evaluate(data_to_analyze_arr_test, labels_arr_test)

print('Test accuracy: %.2f' % (accuracy*100))
#plotting predictions and ROC curve

from sklearn.metrics import roc_curve, auc, roc_auc_score



from keras.utils import plot_model

plot_model(model)
#plot the loss as a function of the epoch

import matplotlib.pyplot as plt



# summarize history for accuracy

plt.plot(history.history['acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
predictions = model.predict(data_to_analyze_arr_test)



fpr, tpr, _ = roc_curve(labels_arr_test, predictions)

roc_auc = auc(fpr, tpr)

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.05])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic curve')

print('AUC: %f' % roc_auc)



plt.show()
predictions_Bkg=[]

predictions_Sig=[]



for f in range(labels_arr_test.shape[0]):

    if(labels_arr_test[f]==0):

        predictions_Bkg.append(predictions[f])

    elif(labels_arr_test[f]==1):

        predictions_Sig.append(predictions[f])



predictions_Bkg_array = np.asarray(predictions_Bkg)

predictions_Bkg_array = predictions_Bkg_array.astype(np.float)



predictions_Sig_array = np.asarray(predictions_Sig)

predictions_Sig_array = predictions_Sig_array.astype(np.float)



plt.hist(predictions_Sig_array,label='exited',normed=True,alpha = 0.5)

plt.hist(predictions_Bkg_array,label='non exited',normed=True,alpha = 0.5)

plt.legend(loc='upper right')

plt.title("DNN Output")

plt.xlabel("Value")

plt.ylabel("Number Of Events")

plt.show()
#investigate the predictions

#try generator

#improve performance

#tuning hyperparameters