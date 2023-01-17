#Generic Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#SK-Learn Libraries

from sklearn.model_selection import train_test_split

from sklearn.semi_supervised import LabelSpreading

from sklearn import datasets

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
#Load Data

iris = datasets.load_iris()
print("Total Records: ", len(iris.data))
# Random Number Generator

rng = np.random.RandomState(0)



# Label Spreading

label_prop_model = LabelSpreading()
#Define How Many Samples Should be Unlabeled

random_unlabeled_points = rng.rand(len(iris.target)) <= 0.5 #Almost 50% samples are unlabeled



#Seperate list for Unlabeled Samples

Unlabeled = np.copy(iris.target)

Unlabeled[random_unlabeled_points] = -1
#Inspect

print(Unlabeled.T)
#fit to Label Spreading 

label_prop_model.fit(iris.data, Unlabeled)



# Predict the Labels for Unlabeled Samples

pred_lb = label_prop_model.predict(iris.data)



#Accuracy of Prediction

print("Accuracy of Label Spreading: ",'{:.2%}'.format(label_prop_model.score(iris.data,pred_lb)))
# Feature& Target  Dataset

X = iris.data

y = pred_lb  # labels predicted by Label Spreading



#Dataset Split  [train = 90%, test = 10%]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0) 



#Define Model

model = RandomForestClassifier(verbose = 0, max_depth=2, random_state=0)



#Fit

model.fit(X_train,y_train)



#Prediction

rf_pred = model.predict(X_test)



#Accuracy Score

acc = accuracy_score(y_test, rf_pred)

print("Random Forest Model Accuracy (after Label Spreading): ",'{:.2%}'.format(acc))