# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
data.head()
LABELS = ["Normal", "Fraud"]

count_classes = pd.value_counts(data['Class'], sort = True)

count_classes.plot(kind = 'bar', rot=0)

plt.title("Transaction class distribution")

plt.xticks(range(2), LABELS)

plt.xlabel("Class")

plt.ylabel("Frequency");
x = data.iloc[: , 1:30].values

y = data.iloc[:, 30].values
print("Input Shape : ", x.shape)

print("Output Shape : ", y.shape)
print ("Labels : \n", y)
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)

imputer = imputer.fit(x[:, 1:30])

x[:, 1:30] = imputer.fit_transform(x[:, 1:30])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state = 0)
print("xtrain.shape : ", xtrain.shape)

print("xtest.shape  : ", xtest.shape)

print("ytrain.shape : ", ytrain.shape)

print("xtest.shape  : ", xtest.shape)
scale_x = StandardScaler()

xtrain = scale_x.fit_transform(xtrain)

xtest = scale_x.transform(xtest)

print("Standardised Training Set : \n", xtrain[0])
from keras.models import Sequential

from keras.layers import Dense
model = Sequential()

model.add(Dense(12, input_dim=29, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
from keras import backend as K



def recall_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



def precision_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision



def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))



# compile the model

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])



# fit the model

history = model.fit(xtrain, ytrain, validation_split=0.3, epochs=10, verbose=0)



# evaluate the model

loss, accuracy, f1_score, precision, recall = model.evaluate(xtest, ytest, verbose=0)
model.summary()
y_pred = model.predict(xtest)

y_pred = (y_pred > 0.5)

score = model.evaluate(xtest, ytest)

score
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

print(classification_report(ytest,y_pred))
cf=confusion_matrix(ytest,y_pred)

df=pd.DataFrame(cf,index=(0,1),columns=(0,1))

plt.figure(figsize=(10,7))

sns.set(font_scale=1.4)

sns.heatmap(df,annot=True,fmt='g')

print("Test Data Accuracy:%0.4f"%accuracy_score(ytest,y_pred))