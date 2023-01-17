import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import sklearn

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

%matplotlib inline

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import SGD

import tensorflow as tf

from imblearn.over_sampling import SMOTE

from sklearn.metrics import confusion_matrix

import itertools



#reading the dataset

df = pd.read_csv("../input/creditcard.csv")

df.head()
df.tail()
dataset2 = df.drop(columns = ['Class'])
dataset2.corrwith(df.Class).plot.bar(

        figsize = (20, 10), title = "Correlation with Class", fontsize = 20,

        rot = 45, grid = True)
#As all the features from V1 to V28 are already normalized, so only normalizing the Amount

df['normalized_amount']=StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))

# Dropping the actual Amount column from the dataset.

df=df.drop(['Amount'],axis=1)

#  the dataset for changed column

df.head()
#Assigning x and y

X = df.iloc[:,:-1]

y = df['Class']


# splitting the data into 70% of the data into training set and 30% of the data into test set.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



print("Size of training set: ", X_train.shape)

print("Size of training set: ", X_test.shape)
frauds = df.loc[df['Class'] == 1]

non_frauds = df.loc[df['Class'] == 0]

print( len(frauds), "fraud data points and", len(non_frauds), "non-fraud data points.")
print("original data")

print(pd.value_counts(pd.Series(y_train)))



sns.set(style="darkgrid")

sns.countplot(y_train,label = "Count",palette="Set2")
#Synthetic Minority oversampling technique 

#define resampling method

method=SMOTE(kind='regular')

#applying resampling to train data 

X_resampled,y_resampled=method.fit_sample(X_train,y_train)
#after resampling

print("after resampling")

print(pd.value_counts(pd.Series(y_resampled)))



sns.set(style="darkgrid")

sns.countplot(y_resampled,label = "Count",palette="Set2")
model = Sequential()

#First Layer

model.add(Dense(16, input_dim=30, activation='relu')) 

#second Layer

model.add(Dense(20,activation='relu'))

#third Layer

model.add(Dense(10,activation='relu'))

#fourth Layer

model.add(Dense(1, activation='sigmoid'))   

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train,y_train,batch_size=15, epochs=5)

model.summary()
print("Loss: ", model.evaluate(X_test, y_test, verbose=0))
y_predicted= model.predict(X_test)

y_expected=pd.DataFrame(y_test)
#Defining the confusion matrix





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
#Confusion matrix of our Test set

c_mat=confusion_matrix(y_expected,y_predicted.round())

plot_confusion_matrix(c_mat,classes=[0,1])


acurracy = 0

for i in range(2):

    acurracy += c_mat[i][i]

print(acurracy/len(y_test))
