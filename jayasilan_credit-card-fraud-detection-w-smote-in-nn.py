import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from imblearn.datasets import fetch_datasets
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler,RobustScaler

data = pd.read_csv('../input/creditcard.csv',header =0)
data.info()
data.head()
data.describe()
data.isna().sum()
print('Number of Fraud transactions',round(data['Class'].value_counts()[1]/len(data)*100,2),'% of total data')
print('Number of Normal transactions',round(data['Class'].value_counts()[0]/len(data)*100,2),'% of total data')
sns.set(style = 'darkgrid')
sns.countplot(x = 'Class',data = data, palette = ['b','r'])
plt.title('Class Distributions \n 0: Normal Transaction , 1: Fraud Trnasaction', fontsize=16)
print('Normal Transaction:',len(data[data['Class']==0]))
print('Fraud Transaction:',len(data[data['Class']==1]))
Normal_Transaction = data[data['Class']==0]
Fraud_Transaction = data[data['Class']==1]
plt.figure(figsize = (15,8))
plt.subplot(221)
Normal_Transaction.Amount.plot.hist(title = "Normal Transaction")
plt.subplot(222)
Fraud_Transaction.Amount.plot.hist(title = "Fraud Transaction")
plt.subplot(224)
Fraud_Transaction[Fraud_Transaction["Amount"]<= 3000].Amount.plot.hist(title="Fraud Transaction")
plt.subplot(223)
Normal_Transaction[Normal_Transaction["Amount"]<=3000].Amount.plot.hist(title="Normal Transaction")
fig, ax = plt.subplots(1, 2, figsize=(18,4))

amnt_val = data['Amount'].values
time_val = data['Time'].values

sns.distplot(amnt_val,ax = ax[0],color = 'g')
ax[0].set_title('Distribution of Transaction amount', fontsize = 14)
ax[0].set_xlim([min(amnt_val),max(amnt_val)])

sns.distplot(time_val,ax = ax[1],color = 'r')
ax[1].set_title('Distribution of Transaction amount', fontsize = 14)
ax[1].set_xlim([min(time_val),max(time_val)])

plt.show()
f, ax = plt.subplots(2, 1, sharex=True, figsize=(12,4))

bins = 50

ax[0].hist(data.Amount[data.Class == 1], bins = bins)
ax[0].set_title('Fraud Transactions')

ax[1].hist(data.Amount[data.Class == 0], bins = bins)
ax[1].set_title('Normal Transactions')

plt.xlabel('Amount in $')
plt.ylabel('Number of Transactions')
plt.yscale('log')
plt.show()
print ("Fraud")
print (data.Amount[data.Class == 1].describe())
print ( )
print ("Normal")
print (data.Amount[data.Class == 0].describe())
R_scaler = RobustScaler()        # RobustScaler is less effected by Outliers

data['Norm_amount'] = R_scaler.fit_transform(data['Amount'].values.reshape(-1,1))

data.drop(['Time','Amount'], axis=1, inplace=True)
data.head()
from sklearn.model_selection import StratifiedShuffleSplit

# Take out the Class label from the dataset and assign it to test set
X = data.drop('Class', axis=1)
y = data['Class']

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=12)    # for cross validation

for train_index, test_index in sss.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


# Check the Distribution of the labels


# Turn into an array
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

# See if both the train and test label distribution are similar
train_unique_label, train_counts_label = np.unique(y_train, return_counts=True)
test_unique_label, test_counts_label = np.unique(y_test, return_counts=True)
print('-' * 100)

print('Label Distributions: \n')
print(train_counts_label/ len(y_train))
print(test_counts_label/ len(y_test))
# Checking the sample length of splits
print('Length of X (train): {} | Length of y (train): {}'.format(len(X_train), len(y_train)))
print('Length of X (test): {} | Length of y (test): {}'.format(len(X_test), len(y_test)))

# SMOTE Technique (OverSampling) It provides a better Precision Recall value for highly unbalanced data
smt = SMOTE(ratio='minority', random_state=42)

# This will be the data were we are going to 
Xsm_train, ysm_train = smt.fit_sample(X_train, y_train)
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense,Dropout
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

n_inputs = Xsm_train.shape[1]

#Sequential model with 7 layer( 5 hidden layer)
#Output layer is softmax that chooses the most probable class 

SMOTE_model = Sequential()
SMOTE_model.add(Dense(n_inputs,input_shape = (n_inputs, ),activation = 'relu'))
SMOTE_model.add(Dropout(0.2))
SMOTE_model.add(Dense(64,activation = 'relu'))
SMOTE_model.add(Dropout(0.2))
SMOTE_model.add(Dense(64,activation = 'relu'))
SMOTE_model.add(Dropout(0.2))
SMOTE_model.add(Dense(64,activation = 'relu'))
SMOTE_model.add(Dropout(0.2))
SMOTE_model.add(Dense(64,activation = 'relu'))
SMOTE_model.add(Dropout(0.2))
SMOTE_model.add(Dense(64,activation = 'relu'))
SMOTE_model.add(Dropout(0.2))
SMOTE_model.add(Dense(2, activation = 'softmax'))   # class = 2 ( Fraud or Normal)

SMOTE_model.compile(Adam(lr = 0.001),loss = 'sparse_categorical_crossentropy',metrics =['accuracy'])
epochs = 20
batch_size = 400
SMOTE_model.fit(Xsm_train,ysm_train,validation_split = 0.2,batch_size = batch_size,epochs = epochs,shuffle = True, verbose = 1)
score,accuracy = SMOTE_model.evaluate(X_test,y_test)
print('Test score:',score)
print('accuracy:',accuracy)
history = SMOTE_model.fit(Xsm_train, ysm_train, batch_size = batch_size, epochs = epochs,validation_data = (X_test, y_test), verbose = 1)
history.history.keys()
f, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='g', label="Train loss")
ax[0].plot(history.history['val_loss'], color='r', label="Test loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='g', label="Train accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Test accuracy")
legend = ax[1].legend(loc='best', shadow=True)
plt.show()
SMOTE_predictions = SMOTE_model.predict(X_test, batch_size=300, verbose=1)
SMOTE_Fraud_predictions = SMOTE_model.predict_classes(X_test, batch_size=300, verbose=1)
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
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
    plt.title(title, fontsize=14)
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

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
SMOTE_Result = confusion_matrix(y_test, SMOTE_Fraud_predictions)
actual_cm = confusion_matrix(y_test,y_test)
labels = ['Normal', 'Fraud']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(SMOTE_Result, labels, title="SMOTE_Result \n Confusion Matrix", cmap=plt.cm.Oranges)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)
