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
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import keras
# importing the dataset

dataset = pd.read_csv("../input/churn-modelling/Churn_Modelling.csv")
dataset
# our aim will be to predict whether the employee will leave the bank or not
# Y will be the exited column
#Seperating the dependent and independent variable sets

X = dataset.iloc[: , 3 : 13]
y = dataset.iloc[: , 13]
#as in the X dataset the "Gender" and "Geography" are categorical variables, so creating the dummy variables
#for the two columns

geography = pd.get_dummies(X['Geography'] , drop_first = True)
gender = pd.get_dummies(X['Gender'] , drop_first = True)

#now concatinating the new dummy columns with the X data frame

X = pd.concat([X , geography , gender] , axis = 1)
# now deleting the original categorical columns

X = X.drop(['Geography' , 'Gender'] , axis = 1)
X.describe()
# now splitting the X dataset into training set and testing set

from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 0)
X_test
# now we will do some feature scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense

# intializing the sequencer

classifier = Sequential()
#lets add the input layer in our ANN
classifier.add(Dense(units = 6 , activation = "relu" , kernel_initializer = "he_uniform" , input_dim = 11))
# adding the first hidden layer

classifier.add(Dense(units = 6 , activation = "relu" , kernel_initializer = "he_uniform"))
#Adding the output layer

classifier.add(Dense(units = 1 , activation = "sigmoid" , kernel_initializer = "glorot_uniform"))
# Now compiling our ANN

classifier.compile(optimizer = "adam" , loss = "binary_crossentropy" , metrics = ['accuracy'])
classifier.summary()
# fitting the ANN on our trainig set

model_history = classifier.fit(X_train , y_train , validation_split = 0.33 , batch_size = 10 , epochs = 100 , verbose = 0)
# now we have successfully trained our model, lets check its accuracy

print(model_history.history['val_accuracy'])
# Our model got the training accuracy of 85.3 percent
#Now testing our model

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test , y_pred)
print(cm)
# printing the accuracy score

from sklearn.metrics import accuracy_score

score = accuracy_score(y_pred , y_test)
print(score)
