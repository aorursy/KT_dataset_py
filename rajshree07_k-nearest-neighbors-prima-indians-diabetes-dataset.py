import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import warnings

warnings.filterwarnings('ignore')

import os

print(os.listdir("../input"))

%matplotlib inline



#importing train_test_split

from sklearn.model_selection import train_test_split
# Import the campaign dataset from Excel (Sheet 0 = Non Responders, Sheet 1 = Responders)

diabetes_df = pd.read_csv("../input/diabetes.csv")

diabetes_df.head()
#Examine Shape of Dataset

diabetes_df.shape
#Examine Class Distribution

diabetes_df.Outcome.value_counts() / len(diabetes_df)
# Create array to store our features and target variable

X = diabetes_df.drop('Outcome',axis=1).values

y = diabetes_df['Outcome'].values
# Apply Standard Scaler to our X dataset

import sklearn.preprocessing as preproc

X_scaled = preproc.StandardScaler().fit_transform(X)

X_scaled
#Split our data into a train and test set

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=42, stratify=y)
# Import KNN Classifier

from sklearn.neighbors import KNeighborsClassifier



# Create K values (1-10) & Create Arrays to store train/test performance accuracy

k = np.arange(1,50)

train_accuracy = np.empty(len(k))

test_accuracy = np.empty(len(k))



for i,k in enumerate(k):

    # Instantiate NN Classifier with K Neighbors

    knn = KNeighborsClassifier(n_neighbors=k)

    

    # Fit KNN model

    knn.fit(X_train, y_train)

    

    # Evaluate train performance 

    train_accuracy[i] = knn.score(X_train, y_train)

    

    # Evaluate test performance

    test_accuracy[i] = knn.score(X_test, y_test)
# Visualize Train/Test Performance

k = np.arange(1,50)

plt.title('k-NN Varying number of neighbors')

plt.plot(k, test_accuracy, label='Testing Accuracy')

plt.plot(k, train_accuracy, label='Training accuracy')

plt.legend()

plt.xlabel('# K-Neighbors')

plt.ylabel('Accuracy')

plt.show()
#import GridSearchCV

from sklearn.model_selection import GridSearchCV



#In case of classifier like knn the parameter to be tuned is n_neighbors

param_grid = {'n_neighbors':np.arange(1,50)}



knn = KNeighborsClassifier()

knn_cv= GridSearchCV(knn,param_grid,cv=5)

knn_cv.fit(X_scaled,y)
knn_cv.best_score_
knn_cv.best_params_