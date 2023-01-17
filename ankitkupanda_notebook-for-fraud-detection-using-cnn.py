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
data = pd.read_csv('../input/creditcardfraud/creditcard.csv')
data.head()
count_classes = pd.value_counts(data['Class'], sort = True).sort_index()
print(count_classes)
from sklearn.preprocessing import StandardScaler



data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

print(data['normAmount'])
data = data.drop(['Time','Amount'],axis=1)

data.head()
X = data.loc[:, data.columns!= 'Class']

y = data.loc[:, data.columns == 'Class']
print(X)
print(y)
print(y)
number_records_fraud = len(data[data.Class == 1])

fraud_indices = np.array(data[data.Class == 1].index)
normal_indices = data[data.Class == 0].index



random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace = False)

random_normal_indices = np.array(random_normal_indices)

under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])



under_sample_data = data.iloc[under_sample_indices,:]



X_undersample = under_sample_data.loc[:, under_sample_data.columns != 'Class']

y_undersample = under_sample_data.loc[:, under_sample_data.columns == 'Class']



print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data.Class == 0])/len(under_sample_data))

print("Percentage of fraud transactions: ", len(under_sample_data[under_sample_data.Class == 1])/len(under_sample_data))

print("Total number of transactions in resampled data: ", len(under_sample_data))
from sklearn.model_selection import train_test_split



# Whole dataset

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)



print("Number transactions train dataset: ", len(X_train))

print("Number transactions test dataset: ", len(X_test))

print("Total number of transactions: ", len(X_train)+len(X_test))



# Undersampled dataset

X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample,y_undersample,test_size = 0.3

                                                                                                   ,random_state = 0)
print("Number transactions train dataset: ", len(X_train_undersample))

print("Number transactions test dataset: ", len(X_test_undersample))

print("Total number of transactions: ", len(X_train_undersample)+len(X_test_undersample))
X_train_undersample.shape
X_train_undersample.shape, y_train_undersample.shape, X_test_undersample.shape, y_test_undersample.shape
X_train_undersample
y_train_undersample = y_train_undersample.to_numpy()

y_test_undersample = y_test_undersample.to_numpy()
X_train_undersample = X_train_undersample.values.reshape(X_train_undersample.shape[0],X_train_undersample.shape[1],1)

X_test_undersample = X_test_undersample.values.reshape(X_test_undersample.shape[0],X_test_undersample.shape[1],1)
from tensorflow import keras

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization

from tensorflow.keras.layers import Conv1D,MaxPool1D

from tensorflow.keras.optimizers import Adam
model = Sequential()

model.add(Conv1D(64,2,activation='relu',input_shape = X_train_undersample[0].shape))

model.add(BatchNormalization())

model.add(Dropout(0.2))



model.add(Conv1D(128,2,activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Flatten())



model.add(Dense(100, activation='relu'))

model.add(Dense(50, activation='relu'))

model.add(Dense(25, activation='relu'))



model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer=Adam(lr=0.0001), loss = 'binary_crossentropy', metrics=['accuracy'])
model.summary()
from tensorflow.keras.callbacks import EarlyStopping



monitor = EarlyStopping(monitor='loss', min_delta=1e-2, patience=30, verbose=1, mode='auto',

        restore_best_weights=True)

model.fit(X_train_undersample, y_train_undersample, callbacks=[monitor], verbose=2, epochs=1000)
y_pred_undersample = model.predict(X_test_undersample)
y_pred_undersample.shape
for i in range(y_pred_undersample.shape[0]):

    if(y_pred_undersample[i]>0.5):

        y_pred_undersample[i] = 1

    else:

        y_pred_undersample[i] = 0
y_pred_undersample
import matplotlib.pyplot as plt
import itertools



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

    plt.xticks(tick_marks, classes, rotation=0)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        #print("Normalized confusion matrix")

    else:

        1#print('Confusion matrix, without normalization')



    #print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 
cnf_matrix = confusion_matrix(y_test_undersample,y_pred_undersample)
class_names = [0,1]

plt.figure()

plot_confusion_matrix(cnf_matrix

                      , classes=class_names

                      , title='Confusion matrix')

plt.show()
print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

fpr, tpr, thresholds = roc_curve(y_test_undersample,y_pred_undersample)

roc_auc = auc(fpr,tpr)



# Plot ROC

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.0])

plt.ylim([-0.1,1.01])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()



X_test = X_test.values.reshape(X_test.shape[0],X_test.shape[1],1)
y_pred = model.predict(X_test)
y_test.shape, y_pred.shape
for i in range(y_pred.shape[0]):

    if(y_pred[i]>0.5):

        y_pred[i] = 1

    else:

        y_pred[i] = 0

    
cnf_matrix = confusion_matrix(y_test,y_pred)
class_names = [0,1]

plt.figure()

plot_confusion_matrix(cnf_matrix

                      , classes=class_names

                      , title='Confusion matrix')

plt.show()
print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

fpr, tpr, thresholds = roc_curve(y_test,y_pred)

roc_auc = auc(fpr,tpr)



# Plot ROC

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.0])

plt.ylim([-0.1,1.01])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()

count_1 = 0

for i in range(y_test.shape[0]):

    if(y_test.iloc[i,0] == 1):

        count_1 = count_1 + 1
print(count_1)