# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# let's read data into numpy array

from numpy import genfromtxt

my_data = genfromtxt('../input/diabetes.csv', delimiter=',', skip_header=True, 

                     dtype="i8,i8,i8,i8,i8,f8,f8,i8,i1")



my_data.dtype.names=("Pregnancies","Glucose","BloodPressure","SkinThickness",

                    "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome")



features = my_data[["Pregnancies","Glucose","BloodPressure","SkinThickness",

                    "Insulin","BMI","DiabetesPedigreeFunction","Age"]]

labels=my_data[["Outcome"]]
#print(features[0])

#print(type(features[0]))

#print(my_data.shape)



#features.shape=(768,8)

#features_2=np.zeros((768,8))

#features_2.dtype.names=("Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome")





print(features_2)



import keras

from keras.models import Sequential



model = Sequential()

from keras.layers import Dense, Activation



model.add(Dense(output_dim=64, input_shape=(features.shape[0],)))

model.add(Activation("relu"))

model.add(Dense(output_dim=1, input_shape=(64,)))

model.add(Activation("softmax"))



model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

                    
model.fit(features, labels, nb_epoch=5, batch_size=32)