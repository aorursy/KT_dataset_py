import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

data = pd.read_csv("../input/diabetes.csv")

data.head(5)



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))
# Total number of rows in the dataset

print(len(data)) 
#Separate features and labels

X = data.iloc[:,0:8].values

y = data.iloc[:,8].values



#Test train split

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)



#Standard scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)



#Applying PCA here

from sklearn.decomposition import PCA

pca = PCA(n_components= None) #We will set it none so that we can see the variance explained and then choose no of comp.

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)



explained_variance = pca.explained_variance_ratio_

explained_variance
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)



pca = PCA(n_components= 2) # here you can change this number to play around

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)



# Create the classifier and train using training data

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)

classifier.fit(X_train,y_train)



#Predict the test set values

y_pred = classifier.predict(X_test)



#Compute confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)

cm
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)



pca = PCA(n_components= 4) #I have tried different no of components here

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)



# Create the classifier and train using training data

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)

classifier.fit(X_train,y_train)



#Predict the test set values

y_pred = classifier.predict(X_test)



#Compute confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)

cm
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)



#Create classifier object

from sklearn.svm import SVC

classifier_svm_kernel = SVC(C=5.0,kernel='rbf', gamma=0.12,tol=0.00001)

classifier_svm_kernel.fit(X_train,y_train)



#Predict the result for test values

y_pred = classifier_svm_kernel.predict(X_test)



#Compute confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)

cm
#Comparing the predictions with the actual results

comparison = pd.DataFrame(y_test,columns=['y_test'])

comparison['y_predicted'] = y_pred

comparison.head(5)
#Apply k-fold validation here

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=classifier_svm_kernel,X=X_train,y=y_train,cv=10)

accuracies
plt.hist(accuracies)

plt.show()
#Applying grid search for optimal parameters and model after k-fold validation

from sklearn.model_selection import GridSearchCV



parameters = [{'C':[0.01,0.1,1,10,50,100,500,1000], 'kernel':['rbf'], 'gamma': [0.1,0.125,0.15,0.17,0.2]}]

grid_search = GridSearchCV(estimator=classifier_svm_kernel, param_grid=parameters, scoring ='accuracy',cv=10,n_jobs=-1)

grid_search = grid_search.fit(X_train,y_train)
best_accuracy = grid_search.best_score_

best_accuracy
opt_param = grid_search.best_params_

opt_param
#Reloading the features and labels and normalizing them

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)



#Choosing 2 principal components

pca = PCA(n_components= 2) # here you can change this number to play around

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)



#Create classifier object

classifier_svm_kernel = SVC(C=5.0,kernel='rbf', gamma=0.12,tol=0.00001)

classifier_svm_kernel.fit(X_train,y_train)



# Grid search and k fold validation libraries already imported. So start the grid search

grid_search = GridSearchCV(estimator=classifier_svm_kernel, param_grid=parameters, scoring ='accuracy',cv=10,n_jobs=-1)

grid_search = grid_search.fit(X_train,y_train)



best_accuracy = grid_search.best_score_

best_accuracy
#Reloading the features and labels and normalizing them

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)



#Choosing different principal components

pca = PCA(n_components= 7) # here you can change this number to play around

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)



#Create classifier object

classifier_svm_kernel = SVC(C=5.0,kernel='rbf', gamma=0.12,tol=0.00001)

classifier_svm_kernel.fit(X_train,y_train)



# Grid search and k fold validation libraries already imported. So start the grid search

grid_search = GridSearchCV(estimator=classifier_svm_kernel, param_grid=parameters, scoring ='accuracy',cv=10,n_jobs=-1)

grid_search = grid_search.fit(X_train,y_train)



best_accuracy = grid_search.best_score_

best_accuracy
