# Importing usefull things 



from tensorflow.keras.models import load_model

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from pandas import DataFrame

import sklearn.feature_extraction.text as sk_text

from sklearn.feature_extraction.text import TfidfVectorizer 

import collections

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import shutil

from sklearn import metrics

import seaborn as sns

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras import optimizers

from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.utils import class_weight

from collections import Counter

from sklearn.metrics import roc_curve, auc



import time

import csv

import pandas as pd



%matplotlib inline

# import the data

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)



df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')

df.head(10)
df.shape

# check for any missing values if there is 

# convert all missing values in specified column to median

df.isnull().values.any()
# now drop all redundant record 

# all duplicates rows will be dropped

df.drop_duplicates(keep = False, inplace = True)

df.shape

# supposely 1854 duplicate records were dropped 
Counter(df['Class'])

# 282493 cases of non-fraud

# 460 cases of fraud
# check the distribution of Fraud and Non-fraud 

count = pd.value_counts(df['Class'], sort = True)

count.plot(kind='bar', rot=0)

plt.title('Transaction Distribution Class')

labels = ['Normal', 'Fraud']

plt.xticks(range(2), labels)

plt.xlabel('Class')

plt.ylabel('Transactions');
# Convert a Pandas dataframe to the x,y inputs that TensorFlow needs

def to_xy(df, target):

    result = []

    for x in df.columns:

        if x != target:

            result.append(x)

    # find out the type of the target column. 

    target_type = df[target].dtypes

    target_type = target_type[0] if isinstance(target_type, collections.Sequence) else target_type

    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.

    if target_type in (np.int64, np.int32):

        # Classification

        dummies = pd.get_dummies(df[target])

        return df[result].values.astype(np.float32), dummies.values.astype(np.float32)

    else:

        # Regression

        return df[result].values.astype(np.float32), df[target].values.astype(np.float32)

    

def encode_text_index(df, name):

    le = preprocessing.LabelEncoder()

    df[name] = le.fit_transform(df[name])

    return le.classes_



outcome = encode_text_index(df, 'Class')
x,y = to_xy(df, 'Class')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state = 42)
y.shape
model = Sequential()

model.add(Dense(50, input_dim = x.shape[1], activation='relu'))

model.add(Dense(25, activation='relu'))

model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



#class weight provides bias toward the minority which create a more balance dataset

class_weight = class_weight.compute_class_weight('balanced', np.unique(outcome), outcome)

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, verbose=1, mode='auto')

model.fit(x_train, y_train, validation_data=(x_test,y_test), callbacks=[monitor], epochs=100,verbose=1, class_weight=class_weight)

pred = model.predict(x)



predict_classes = np.argmax(pred,axis=1)



true_classes = np.argmax(y,axis=1)



print("Predictions: {}".format(predict_classes))

print("True: {}".format(true_classes))



print(outcome[predict_classes[0:1000]])
#For all of the class predictions, what percent were correct?  

y_true = np.argmax(y_test[:], axis=1)

pred = model.predict(x_test[:])

pred = np.argmax(pred, axis=1)



correct = metrics.f1_score(true_classes, predict_classes, average="weighted")

print("F1 Score: {}".format(correct))


# Plot a confusion matrix.

# cm is the confusion matrix, names are the names of the classes.

def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(names))

    plt.xticks(tick_marks, names, rotation=45)

    plt.yticks(tick_marks, names)

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

# Compute confusion matrix

cm = confusion_matrix(y_true, pred)

print(cm)

print('Plotting confusion matrix')



plt.figure()

plot_confusion_matrix(cm, outcome)

plt.show()



print(classification_report(y_true, pred))