# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.




dataset = pd.read_csv("../input/Churn_Modelling.csv")

X = dataset.iloc[:,3:-1].values
y= dataset.iloc[:,-1].values


#Encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le_1= LabelEncoder()
X[:,1]=le_1.fit_transform(X[:,1])
le_2 = LabelEncoder()
X[:,2]=le_2.fit_transform(X[:,2])
oh= OneHotEncoder(categorical_features=[1])
X=oh.fit_transform(X).toarray()

X = X[:,1:]


#splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)


#Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


#ANN

import keras
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()

#Add Sequential Layers

classifier.add(Dense(output_dim = 6, init='uniform' , activation = 'relu', input_dim = 11 ))
classifier.add(Dense(output_dim = 6, init='uniform' , activation = 'relu' ))
classifier.add(Dense(output_dim = 1, init='uniform' , activation = 'sigmoid'))

#Compile the ANN
classifier.compile(optimizer = 'adam', loss= 'binary_crossentropy', metrics = ['accuracy'] )

#fit 
classifier.fit(X_train, y_train, batch_size= 10, nb_epoch= 100)

#predict
y_pred= classifier.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred>0.5)

    




dataset.columns
classifier
y_pred

cm
(1511+207)/2000