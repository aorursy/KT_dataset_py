import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.neighbors import KNeighborsClassifier # the KNN from Scikit-lean

from sklearn.model_selection import train_test_split # this allows me to separate data into train and test 

from sklearn.metrics import mean_squared_error

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

#first we read the .csv file, here Iris.csv file, make sure to add the directory 

filename="../input/Iris.csv"

iris=pd.read_csv(filename)



#making sure we have the data, we can take a look at the first 5 rows, the two options before print

#allow us to see all columns and rows:

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

print(iris.head(n=5))

print("-"*80)

print(iris.tail(n=5))

print("-"*80)

print(iris.keys())
# we need to split the data randomly into test data set to show accuracy of model and train data set, 

#70% for train and 30% for test is a common ratio

# I am choosing 10 neighbors here

knn=KNeighborsClassifier(n_neighbors=10)

# we need to convert Species into dummy variables ( 0 and 1 here setosa is 0 and versicolor is 1)

iris_dum=pd.get_dummies(iris)

print(iris_dum.head())

print(iris_dum.tail())

y=iris_dum['Species_Iris-virginica'].values

X=iris_dum.drop('Species_Iris-virginica', axis=1).values

X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.7)

#fit the classifier to the train data

knn.fit(X_train,y_train)



# Predict the labels for training data

y_predict=knn.predict(X_test)



# using .score() method the accuracy of classifier's prediction can be computed and printed

print("score=",knn.score(X_test, y_test))



#Grid search cross-validation for hyperparameter tuning

#we import GridSearchCV to help us determine the best hyperparameter for our KNN classification

from sklearn.model_selection import GridSearchCV



#we make a dictionary of the KNN hyperparameter "n_neighbors" and assign to it the values of n are lists we want to search



gridparam={'n_neighbors':np.arange(1,100)}



# we then stantiate the classifier



knn2=KNeighborsClassifier()



knn_cv=GridSearchCV(knn2,gridparam, cv=5)



# cv is the number of folds we wish to use

knn_cv.fit(X,y)



y_pred=knn_cv.predict(X_test)

r2 = knn_cv.score(X_test,y_test)

mse = mean_squared_error(y_test, y_pred)



print("Tuned KNN parameter:{}".format(knn_cv.best_params_))

print("Best score is:{}".format(knn_cv.best_score_))

print("Tuned KNN n_neighbor R2: {}".format(r2))

print("Tuned KNN n_neighbor MSE: {}".format(mse))

# I am not sure why the score is now lower than when we did not tune the hyperparameter, maybe thesis two scores are not comparable.
#Grid serach CV can be computationally expensive (not here really), so I realized I can also 

#use RandomizedSearchCV



from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint



gridparam2={'n_neighbors':randint(1,100)}

# I felt I can generate random variables and then RandomizedSearchCV can choose from them



knn3=KNeighborsClassifier()

knn_cv2=RandomizedSearchCV(knn3,gridparam2, cv=5)

knn_cv2.fit(X,y)



print("Tuned KNN parameter:{}".format(knn_cv2.best_params_))

print("Best score is:{}".format(knn_cv2.best_score_))



y_pred=knn_cv2.predict(X_test)

r2 = knn_cv2.score(X_test,y_test)

mse = mean_squared_error(y_test, y_pred)





print("Tuned KNN n_neighbor R2: {}".format(r2))

print("Tuned KNN n_neighbor MSE: {}".format(mse))

#RandomizedSearchCV but wihtout randint

gridparam2={'n_neighbors':np.arange(1,100)}



knn3=KNeighborsClassifier()

knn_cv2=RandomizedSearchCV(knn3,gridparam2, cv=5)

knn_cv2.fit(X,y)



y_pred=knn_cv2.predict(X_test)

r2 = knn_cv2.score(X_test,y_test)

mse = mean_squared_error(y_test, y_pred)





print("Tuned KNN n_neighbor R2: {}".format(r2))

print("Tuned KNN n_neighbor MSE: {}".format(mse))



print("Tuned KNN parameter:{}".format(knn_cv2.best_params_))

print("Best score is:{}".format(knn_cv2.best_score_))