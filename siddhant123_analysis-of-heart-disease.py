# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")

df.head()
df.info()
cp = pd.get_dummies(df["cp"],prefix = "cp",drop_first = True)

df = pd.concat([df,cp],axis = 1)

slp = pd.get_dummies(df["slope"],prefix = "slope",drop_first = True)

df = pd.concat([df,slp],axis = 1)

thl = pd.get_dummies(df["thal"],prefix = "thal",drop_first = True)

df = pd.concat([df,thl],axis = 1)

ecg = pd.get_dummies(df["restecg"],prefix = "restecg",drop_first = True)

df.drop(["cp","thal","slope","restecg"],axis = 1,inplace = True )
X = df.drop(["target"],axis = 1)

y = df["target"].values
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# Importing the Keras libraries and packages

import keras

from keras.models import Sequential

from keras.layers import Dense
# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 17))



# Adding the second hidden layer

classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))



# Adding the output layer

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
# Predicting the Test set results

y_pred = classifier.predict(X_test)
import seaborn as sns

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred.round())

sns.heatmap(cm,annot=True,cmap = 'YlOrRd',fmt="d",cbar=False)

#accuracy score

from sklearn.metrics import accuracy_score

ac=accuracy_score(y_test, y_pred.round())

print('accuracy of the model: ',ac)