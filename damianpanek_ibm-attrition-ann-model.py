



import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import keras

from keras.models import Sequential

from keras.layers import Dense

#Importing the dataset

dataset = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')




class MultiColumnLabelEncoder:

    def __init__(self,columns = None):

        self.columns = columns # array of column names to encode



    def fit(self,X,y=None):

        return self # not relevant here



    def transform(self,X):

        '''

        Transforms columns of X specified in self.columns using

        LabelEncoder(). If no columns specified, transforms all

        columns in X.

        '''

        output = X.copy()

        if self.columns is not None:

            for col in self.columns:

                output[col] = LabelEncoder().fit_transform(output[col])

        else:

            for colname,col in output.iteritems():

                output[colname] = LabelEncoder().fit_transform(col)

        return output



    def fit_transform(self,X,y=None):

        return self.fit(X,y).transform(X)


dataset = MultiColumnLabelEncoder(columns = ['BusinessTravel','Department', 'EducationField', 

                                   'Gender', 'JobRole', 'OverTime', 'Over18', 

                                   'Attrition']).fit_transform(dataset)

dataset =  dataset.drop('MaritalStatus', 1)



X = dataset.iloc[:,  np.r_[0,  2:32]].values

y = dataset.iloc[:,1].values



y = [0 if x == "Yes" else 1 for x in y]

y =  np.array(y)









    







# Splitting the dataset into the Training set and Test set



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)







sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)


y_train = y_train.reshape((-1, 1))







# Initialising the ANN

classifier = Sequential()





classifier.add(Dense(output_dim = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 31))





classifier.add(Dense(output_dim = 6, kernel_initializer = 'uniform', activation = 'relu'))





classifier.add(Dense(output_dim = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))





classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])





classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)




# Predicting the Test set results

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm