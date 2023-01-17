import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



# Neural Network

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

import keras

from keras.optimizers import SGD

import graphviz
# load the data

df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')

# create a new feature "Family"



df_train['Family'] = df_train['SibSp'] + df_train['Parch'] + 1



df_train.Family = df_train.Family.map(lambda x: 0 if x > 4 else x)



df_test['Family'] = df_test['SibSp'] + df_test['Parch'] + 1



df_test.Family = df_test.Family.map(lambda x: 0 if x > 4 else x)
# Filling missing values for the feature Embarked. 



df_train['Embarked'].fillna('S', inplace = True)

df_test['Embarked'].fillna('S', inplace = True)



df_train['Age'].fillna(df_train['Age'].mean(), inplace = True)

df_test['Age'].fillna(df_test['Age'].mean(), inplace = True)

df_test['Fare'].fillna(df_test['Fare'].mean(), inplace = True)



# From our earlier analysis, we choose the following features to train our model.



selected_features = ['Pclass', 'Sex', 'Age', 'Embarked', 'Family', 'Fare']

X_train = df_train[selected_features]

X_test = df_test[selected_features]

y_train = df_train['Survived']



X_train = pd.get_dummies(X_train).astype(np.float64, copy=False)

X_test = pd.get_dummies(X_test).astype(np.float64, copy=False)





X_test.info() 

# data normalization



from sklearn import preprocessing

scale = preprocessing.MinMaxScaler()

X_train = scale.fit_transform(X_train)

X_train = pd.DataFrame(X_train)



X_test = scale.transform(X_test)

X_test = pd.DataFrame(X_test)



X_train.head(20)
# Creating the model

model = Sequential()



# Inputing the first layer input dimensions

model.add(Dense(40, activation='relu', input_dim=9,

               kernel_initializer='uniform'))



# The argument being passed to each Dense layer(18) is the number of hidden units of the layer

# A hidden unit is a dimensiion in the representatiion space of fhe layer

# Stacks of Dense layers with relu activations can solve a wide range of problems

#(including sentiment classification), and you'll likely use tehm frequently



#model.add(Dropout(0.5))



#Adding second hidden layer

model.add(Dense(30, kernel_initializer='uniform',activation='relu'))



#Adding second hidden layer

#model.add(Dense(, kernel_initializer='uniform',activation='relu'))



#model.add(Dropout(0.5))



#model.add(Dense(25, kernel_initializer='uniform',activation='relu'))

#model.add(Dropout(0.5))



#model.add(Dense(25, kernel_initializer='uniform',activation='relu'))

#model.add(Dropout(0.5))



#model.add(Dense(25, kernel_initializer='uniform',activation='relu'))

#model.add(Dropout(0.5))



#model.add(Dense(10, kernel_initializer='uniform',activation='relu'))



#Adding the output layer that is binary [0,1]

model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))



# visualize the model

model.summary()
# Compiling the NN

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Train the NN

model.fit(X_train, y_train, batch_size = 30, epochs = 200, validation_split=0.1)

y_pred = model.predict(X_test)

y_final = (y_pred > 0.5).astype(int).reshape(X_test.shape[0])

print(y_final)



output = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': y_final})

output.to_csv('prediction.csv', index=False)




