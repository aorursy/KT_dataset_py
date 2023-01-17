#Importing libraries

import numpy as np

import pandas as pd
#Importing dataset

dataset = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")

x= dataset.iloc[:,0:4].values

y= dataset.iloc[:,4].values

y=y.reshape(150,1)
#encoding data

from sklearn.preprocessing import LabelEncoder

labelencoder_x= LabelEncoder()

y[:,0] = labelencoder_x.fit_transform(y[:,0])

y=y.astype(float)

#Splitting dataset into training and test set

from sklearn.model_selection import  train_test_split

x_train, x_test,y_train,y_test= train_test_split(x,y,test_size=.1)

#Feature scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train= sc.fit_transform(x_train)

x_test=sc.fit_transform(x_test)
#Fitting classifier to model

from sklearn.svm import SVC

classifier = SVC(kernel = 'linear',C=1)

classifier.fit(x_train, y_train.ravel())
#Choosing hyperparameters

from sklearn.model_selection import GridSearchCV

parameters=[{'C':[1,10,100,.001],'kernel':['linear']},

             {'C':[1,10,100,.001], 'kernel':['rbf']}]

grid_search=GridSearchCV(estimator=classifier,

                         param_grid=parameters,

                         cv=10,

                         n_jobs=-1)

grid_search= grid_search.fit(x_train,y_train.ravel())

acc=grid_search.best_score_

param=grid_search.best_params_

#Predicting test set result

y_pred= classifier.predict(x_test)

#Making confusion matrix

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)
#Exporting result file

y_pred=pd.DataFrame(y_pred)

result= y_pred.to_csv('result.csv')
