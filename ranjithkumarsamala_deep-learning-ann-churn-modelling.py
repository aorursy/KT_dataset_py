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
#visulization 

import matplotlib.pyplot as plt

#splitting

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

#feature scaling

from sklearn.preprocessing import StandardScaler

#Keras libraries and packages

import keras

from keras.wrappers.scikit_learn import KerasClassifier

from keras.models import Sequential 

from keras.layers import Dense,Activation,Embedding,Flatten, LeakyReLU,PReLU,ELU,BatchNormalization, Dropout

#confustion matrix

from sklearn.metrics import confusion_matrix, accuracy_score
churn_data = pd.read_csv("../input/churn-modelling/Churn_Modelling.csv")

print(churn_data.shape)
churn_data.head(10)
#creating copy of raw data

churn = pd.DataFrame.copy(churn_data)
#removing unimportant features : first 3 columns

drop_column = ["RowNumber","CustomerId", "Surname"]

churn.drop(drop_column, axis=1, inplace=True)
churn.shape
# creating dummy variables for Geography and Gender

geography = pd.get_dummies(churn["Geography"],drop_first=True)

gender = pd.get_dummies(churn["Gender"],drop_first=True)
# combining to in 'x' dataframe

churn = pd.concat([churn,geography,gender], axis=1)

print(churn.info())
# now drop original geography & gender columns

drop_geo_gen = ["Geography","Gender"]

churn.drop(drop_geo_gen, axis=1, inplace=True)
# defining X and Y

x = pd.DataFrame.copy(churn)

x = x.drop(["Exited"],axis=1)

y = pd.DataFrame.copy(churn["Exited"])
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)
# feature scaling

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
# creating custom functon for iterations to tune hyper parameters

def create_model(layers, activation):

    model = Sequential()

    for i, nodes in enumerate(layers):

        if i==0:

            model.add(Dense(nodes,input_dim=x_train.shape[1]))

            model.add(Activation(activation))

            model.add(Dropout(0.3))

        else:

            model.add(Dense(nodes))

            model.add(Activation(activation))

            model.add(Dropout(0.3))

            

    model.add(Dense(units = 1, kernel_initializer= 'glorot_uniform', activation = 'sigmoid')) # Note: no activation beyond this point

    

    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

    return model

    

model = KerasClassifier(build_fn=create_model, verbose=0)





layers = [[20], [40, 20], [45, 30, 15]]

activations = ['sigmoid', 'relu']

param_grid = dict(layers=layers, activation=activations, batch_size = [128, 256], epochs=[30])

grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=5)



grid_result = grid.fit(x_train, y_train)



print(grid_result.best_score_,grid_result.best_params_)
#predicting test set

y_pred = grid.predict(x_test)

y_pred = (y_pred>0.5)
#confusion matrix

cm = confusion_matrix(y_test,y_pred)

cm
# accuracy score of test data

score = accuracy_score(y_pred,y_test) 

score
# Make a prediction using the Random Forest on the wanted columns

predictions = grid.predict(x_test)



# Our predictions array is comprised of 0's and 1's (Dead or Survived)

predictions[:20]