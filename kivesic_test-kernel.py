# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Just testing some ANN functionality here, next step is to make a REAL model 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



dataset = pd.read_csv('../input/KaggleV2-May-2016.csv')



# independent vars

X = dataset.iloc[:,2:-1].values



# dependent var

y = dataset.iloc[:, -1].values



# sort the days out. Assumption: we need a date diff. between the appointment

# and the sceduled date

columns = np.arange(3, 11)

columns = np.append(0, columns) # indices of X that we need



X_no_dates = X[:, columns] # X without dates



# time difference

X_time_diff = pd.to_datetime(X[:,2]) - pd.to_datetime(X[:,1])

X_time_diff = X_time_diff.days # take days only



# now, for some reason (dataset error?), some diffs are negativ, set them to 0

X_time_diff = [ 0 if x < 0 else x for x in X_time_diff ]



# finally, full X with all relevant data, 1st col. is the time in days

# between the scheduling date and the appointment

X_full = np.c_[X_time_diff, X_no_dates]



# Next, encode categorical values in X - gender and neighbourhood

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder = LabelEncoder()

# encode gender as either 0 or 1

X_full[:, 1] = labelencoder.fit_transform(X_full[:, 1])



# encode place

X_full[:, 3] = labelencoder.fit_transform(X_full[:, 3])

onehotencoder = OneHotEncoder(categorical_features = [3])

X_full = onehotencoder.fit_transform(X_full).toarray()

# reduce the # of dummy vars for place

X_full = X_full[:, 1:]



# encode dependent var

y = labelencoder.fit_transform(y)



# Split the dataset into the training set and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size = 0.2, random_state = 0)



# feature scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)



# Making the ANN

# Importing the Keras libraries and packages 

import keras

from keras.models import Sequential # to initialize NN

from keras.layers import Dense # to create layers in NN



# Initializing the ANN: (defining as the seq. of layers (other way would be to define a graph)

classifier = Sequential()



# Adding the input layer and the 1st hidden layer

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 89)) 



# Adding the second hidden layer

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu')) 



# Adding the output layer

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid')) # in case of multi value output, replace 'sigmoid' with 'soft???'



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fit ANN to the training set

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)



# Predict the Test set results

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)



# Make the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
cm