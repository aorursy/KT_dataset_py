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
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn import preprocessing

from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression

from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import accuracy_score
data=pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
data.shape
data.head()
data.isnull().sum()
data.info()
var = data.columns.values



i = 0

t0 = data.loc[data['Class'] == 0]

t1 = data.loc[data['Class'] == 1]



sns.set_style('whitegrid')

plt.figure()

fig, ax = plt.subplots(8,4,figsize=(16,28))



for feature in var:

    i += 1

    plt.subplot(8,4,i)

    sns.kdeplot(t0[feature], bw=0.5,label="Class = 0")

    sns.kdeplot(t1[feature], bw=0.5,label="Class = 1")

    plt.xlabel(feature, fontsize=12)

    locs, labels = plt.xticks()

    plt.tick_params(axis='both', which='major', labelsize=12)

plt.show()
plt.figure(figsize = (16,10))

plt.title('Credit Card Transactions features correlation plot', size = 20)

corr = data.corr()

sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,cmap="Greens",fmt='.1f',annot=True)

plt.show()
data["Class"].value_counts().plot(kind="bar",color="red")

plt.title("Frequency of the target classes", size=20)

plt.xlabel("Target Labels", size = 18)
target = pd.DataFrame(data["Class"].value_counts())

target.style.background_gradient(cmap="Reds")
X=data.drop(columns=["Class"])

y=data["Class"]
names=X.columns

scaled_df = preprocessing.scale(X)

scaled_df = pd.DataFrame(scaled_df,columns=names)
scaled_df.head()
scaled_df[["Amount","Time"]].describe()
X_train, X_test, y_train, y_test = train_test_split(scaled_df, y, test_size = 0.30, random_state = 0, shuffle = True, stratify = y)
X_train.shape, X_test.shape
y_train.value_counts()
y_test.value_counts()
sm = SMOTE(random_state = 33)

X_train_new, y_train_new = sm.fit_sample(X_train, y_train.ravel())
pd.Series(y_train_new).value_counts().plot(kind="bar")
clf = LogisticRegression(solver = 'lbfgs')

clf.fit(X_train_new, y_train_new)

train_pred = clf.predict(X_train_new)

test_pred = clf.predict(X_test)
print('Accuracy score for Training Dataset = ', accuracy_score(train_pred, y_train_new))

print('Accuracy score for Testing Dataset = ', accuracy_score(test_pred, y_test))
cm=confusion_matrix(y_test, test_pred)

cm
plt.figure(figsize=(8,6))

sns.set(font_scale=1.2)

sns.heatmap(cm, annot=True, fmt = 'g', cmap="Reds", cbar = False)

plt.xlabel("Predicted Label", size = 18)

plt.ylabel("True Label", size = 18)

plt.title("Confusion Matrix Plotting for Logistic Regression model", size = 20)
print("Percentage for 'no fraud' cases wrong classification using Logistic Regression is:", (2018/85295)*100)

print("Percentage for 'Fraud' cases wrong prediction Logistic Regression is:", (13/148)*100)
model = Sequential()

model.add(Dense(X_train_new.shape[1], activation = 'relu', input_dim = X_train_new.shape[1]))

model.add(BatchNormalization())



model.add(Dense(64, activation = 'relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Dense(64, activation = 'relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Dense(1, activation = 'sigmoid'))
optimizer = keras.optimizers.Adam(lr=0.0001)

model.compile(optimizer = optimizer, loss = 'binary_crossentropy')
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 10)
history = model.fit(x=X_train_new, y=y_train_new, batch_size = 256, epochs=150,

          validation_data=(X_test, y_test), verbose=1,

          callbacks=[early_stop])
evaluation_metrics=pd.DataFrame(model.history.history)

evaluation_metrics.plot(figsize=(10,5))

plt.title("Loss for both Training and Validation", size = 20)
y_pred = model.predict_classes(X_test)
cm_nn=confusion_matrix(y_test, y_pred)

cm_nn
plt.figure(figsize=(8,6))

sns.set(font_scale=1.2)

sns.heatmap(cm_nn, annot=True, fmt = 'g', cmap="winter", cbar = False)

plt.xlabel("Predicted Label", size = 18)

plt.ylabel("True Label", size = 18)

plt.title("Confusion Matrix Plotting for Neural Network model", size = 20)