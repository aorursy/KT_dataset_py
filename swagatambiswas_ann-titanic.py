# Importing the libraries



import numpy as np

import matplotlib.pyplot as plt

import pandas as pd





train_pd = pd.read_csv('/kaggle/input/titanic/train.csv')

test_pd = pd.read_csv('/kaggle/input/titanic/test.csv')



X_train = train_pd.iloc[:,[0,2,4,5,6,7,9,11]]

y_train = train_pd.iloc[:,1]

X_test = test_pd.iloc[:,[0,1,3,4,5,6,8,10]]





#Clean the training data.

i=0

for item in X_train["Sex"]:

    if item == 'female':

        X_train.at[i,"Sex"] = 1

    elif item == 'male':

        X_train.at[i,"Sex"] = 0

    else:

        print("Could not assign gender: {}".format(item))

    i+=1

i=0

for item in X_train["Embarked"]:

    if item == 'C':

        X_train.at[i,"Embarked"] = 0

    elif item == 'S':

        X_train.at[i,"Embarked"] = 1

    elif item == 'Q':

        X_train.at[i,"Embarked"] = 2

    else:

        print("Could not assign embarked label: {}. Assigning to 3 instead.".format(item))

        X_train.at[i,"Embarked"] = 3

    i+=1

i=0

for item in X_train["Age"]:

    if pd.isnull(item):

        X_train.at[i,"Age"] = -1

    i+=1



#Clean the test data.

i=0

for item in X_test["Sex"]:

    if item == 'female':

        X_test.at[i,"Sex"] = 1

    elif item == 'male':

        X_test.at[i,"Sex"] = 0

    else:

        print("Could not assign gender: {}".format(item))

    i+=1

i=0

for item in X_test["Embarked"]:

    if item == 'C':

        X_test.at[i,"Embarked"] = 0

    elif item == 'S':

        X_test.at[i,"Embarked"] = 1

    elif item == 'Q':

        X_test.at[i,"Embarked"] = 2

    else:

        print("Could not assign embarked label: {}. Assigning to 3 instead.".format(item))

        X_test.at[i,"Embarked"] = 3

    i+=1

i=0

for item in X_test["Age"]:

    if pd.isnull(item):

        X_test.at[i,"Age"] = -1

    i+=1



i=0

for item in X_test["Fare"]:

    if pd.isnull(item):

        X_test.at[i,"Fare"] =-1

    i+=1

    

# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)    



import keras

from keras.models import Sequential

from keras.layers import Dense



#initializing ann

classifier = Sequential()



#adding the input layer

classifier.add(Dense(output_dim = 6,init = 'uniform',activation = 'relu',input_dim = 8))

#adding 2nd hidden layer

classifier.add(Dense(output_dim = 6,init = 'uniform',activation = 'relu'))

#adding output layer

classifier.add(Dense(output_dim = 1,init = 'uniform',activation = 'sigmoid'))

#compiling the ann

classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

# Fitting classifier to the Training set

classifier.fit(X_train, y_train,batch_size = 10, nb_epoch = 100)



y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)







submission = pd.DataFrame()

submission['PassengerId'] = test_pd['PassengerId']

submission['Survived'] = y_pred



submission[['Survived']] = submission[['Survived']].astype(int)


