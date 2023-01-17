import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input"))

dataMath = pd.read_csv('../input/student_math.csv', sep=';')

dataLan = pd.read_csv('../input/student_language.csv', sep=';')

dataMath.info()
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()



for i in range(dataMath.shape[1]):    

    if dataMath.dtypes[i] == 'object':

        dataMath[dataMath.columns[i]] = enc.fit_transform(dataMath[dataMath.columns[i]])



dataMath.head(3)
corr_matrix = dataMath.corr()

corr_matrix["G3"].sort_values(ascending=False)
dataM = dataMath.copy()

low_corr = []

for index,c_val in enumerate(corr_matrix["G3"]):

    if abs(c_val) < 0.10:

        low_corr.append(dataM.columns[index])



for name in low_corr:

    dataM.pop(name)



corr_matrix = dataM.corr()

plt.figure(figsize=(20,10))

sns.heatmap(corr_matrix,annot=True,cmap="YlGnBu",fmt=".2f",linecolor='white', cbar=True,linewidths=1)

X = dataM.iloc[:, :-1].values

y = dataM.iloc[:, -1].values



for i in range(len(y)):

    if y[i]>=12:

        y[i] = 1

    else:

        y[i] = 0
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression

LR_classifier = LogisticRegression(random_state = 0)

LR_classifier.fit(X_train, y_train)



y_LR = LR_classifier.predict(X_test)



# make confusion matrix

from sklearn.metrics import confusion_matrix

cm_LR = confusion_matrix(y_test, y_LR)

cm_LR
from sklearn.svm import SVC

SVC_classifier = SVC(kernel = 'linear', random_state = 0)

SVC_classifier.fit(X_train, y_train)



y_SVC = SVC_classifier.predict(X_test)



# make confusion matrix

cm_SVC = confusion_matrix(y_test, y_SVC)

cm_SVC
from xgboost import XGBClassifier

XGB_classifier = XGBClassifier()

XGB_classifier.fit(X_train, y_train)



y_XGB = XGB_classifier.predict(X_test)



# Making the Confusion Matrix

cm_XGB = confusion_matrix(y_test, y_XGB)

cm_XGB
import keras

from keras.models import Sequential

from keras.layers import Dense



# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu', input_dim = 15))



# Adding the second hidden layer

classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu'))



# Adding the output layer

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set

classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)





y_ANN = classifier.predict(X_test)

y_ANN = (y_ANN > 0.5)



# confusion matrix

cm_ANN = confusion_matrix(y_test, y_ANN)

cm_ANN