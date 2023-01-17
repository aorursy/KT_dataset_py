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
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



dataset = pd.read_csv('../input/Churn_Modelling.csv')

x = dataset.iloc[:,3:-1].values

y = dataset.iloc[:,-1].values

x_eval = np.array(([[0,0,600,1,40,3,60000,2,1,1,50000]]))

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder_x_1 = LabelEncoder()

x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])

labelencoder_x_2 = LabelEncoder()

x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])

#since are categorical variables are not ordinal we need to one hot encode them

onehotencoder = OneHotEncoder(categorical_features = [1])

x = onehotencoder.fit_transform(x).toarray()

#removing of one dummy variable here to avoid fallling into the dummy variable trap

x = x[:,1:]



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)



from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)

x_test = sc_x.transform(x_test)

x_eval = sc_x.transform(x_eval)

import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

'''classifier = Sequential()

classifier.add(Dense(output_dim = 6,init = 'uniform',activation = 'relu',input_dim = 11))

classifier.add(Dropout(p = 0.1))

classifier.add(Dense(output_dim = 6,init = 'uniform',activation = 'relu'))

classifier.add(Dropout(p = 0.1))

classifier.add(Dense(output_dim = 1,init = 'uniform',activation = 'sigmoid'))

classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

classifier.fit(x_train,y_train,batch_size = 10,nb_epoch = 100)

y_pred = classifier.predict(x_test)

y_pred = (y_pred > 0.5)

new_pred = classifier.predict(x_eval)

new_pred = (new_pred > 0.5)

print("Predicted value is ",new_pred)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)

'''

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

def build_classifier():

    #we only need the part of code that builds the architecture and not the training part

    classifier = Sequential()

    classifier.add(Dense(output_dim = 6,init = 'uniform',activation = 'relu',

                         input_dim = 11))

    classifier.add(Dense(output_dim = 6,init = 'uniform',activation = 'relu'))

    classifier.add(Dense(output_dim = 1,init = 'uniform',activation = 'sigmoid'))

    classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

    return classifier

classifier = KerasClassifier(build_fn = build_classifier,batch_size = 10,epochs = 100)

accuracies = cross_val_score(estimator = classifier,X = x_train,y = y_train,

                             cv = 10,n_jobs = -1)

mean = accuracies.mean()

variance = accuracies.std()

print(mean)

print(variance)