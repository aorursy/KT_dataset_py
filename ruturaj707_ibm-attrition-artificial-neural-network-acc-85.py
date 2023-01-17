# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



# Importing the dataset

dataset = pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')

dataset['Attrition'] = dataset['Attrition'].map({'Yes': 1, 'No': 0})

X = dataset.drop(['Attrition','EducationField','Over18'], axis=1).values

y = dataset.Attrition.values







# Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.compose import ColumnTransformer



label_encoder_x_1 = LabelEncoder()

X[: , 1] = label_encoder_x_1.fit_transform(X[:,1])

label_encoder_x_2 = LabelEncoder()

X[: , 3] = label_encoder_x_2.fit_transform(X[:,3])

label_encoder_x_3 = LabelEncoder()

X[: , 9] = label_encoder_x_3.fit_transform(X[:,9])

label_encoder_x_4 = LabelEncoder()

X[: , 13] = label_encoder_x_4.fit_transform(X[:,13])

label_encoder_x_5 = LabelEncoder()

X[: , 15] = label_encoder_x_5.fit_transform(X[:,15])

label_encoder_x_6 = LabelEncoder()

X[: , 19] = label_encoder_x_6.fit_transform(X[:,19])









transformer = ColumnTransformer(

    transformers=[

        ("OneHot",        

         OneHotEncoder(), 

         [1]             

         )

    ],

    remainder='passthrough' 

)

X = np.array(transformer.fit_transform(X),dtype=np.int)

X = X[:, 1:]





transformer = ColumnTransformer(

    transformers=[

        ("OneHot3",        

         OneHotEncoder(), 

         [4]             

         )

    ],

    remainder='passthrough' 

)

X = np.array(transformer.fit_transform(X),dtype=np.int)

X = X[:, 1:]





transformer = ColumnTransformer(

    transformers=[

        ("OneHot9",        

         OneHotEncoder(),

         [17]            

         )

    ],

    remainder='passthrough' 

)

X = np.array(transformer.fit_transform(X),dtype=np.int)

X = X[:, 1:]





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



# first hidden layer

classifier.add(Dense(units = 18, kernel_initializer = 'uniform', activation = 'relu', input_dim = 35))



#second hidden layer

classifier.add(Dense(units = 18, kernel_initializer = 'uniform', activation = 'relu'))



# output layer

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)



# Predicting the Test set results

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)



#Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)