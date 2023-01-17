# See dataset description on kaggle
# Importing requrired packages

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt 

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC  

from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split

import numpy as np 

%matplotlib inline 
#Loading dataset

wine = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
# Plot the data type

print('Data type: ', type(wine))

# Plot the first five rows of the dataset

wine.head()
# Check the number of values and datatypes for each column

wine.info()
# Check the dataset structure

print(f'Number of rows: {wine.shape[0]}\nNumber of columns: {wine.shape[1]}')
# Plot the statistical values for each column

wine.describe()
#Detect missing values

wine.isnull().sum()
# Plot each feature against quality

fig = plt.figure(figsize = (10,6))

sns.barplot(x = 'quality', y = 'fixed acidity', data = wine)
fig = plt.figure(figsize = (10,6))

sns.barplot(x = 'quality', y = 'volatile acidity', data = wine)
fig = plt.figure(figsize = (10,6))

sns.barplot(x = 'quality', y = 'citric acid', data = wine)
fig = plt.figure(figsize = (10,6))

sns.barplot(x = 'quality', y = 'residual sugar', data = wine)
fig = plt.figure(figsize = (10,6))

sns.barplot(x = 'quality', y = 'chlorides', data = wine)
fig = plt.figure(figsize = (10,6))

sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = wine)
fig = plt.figure(figsize = (10,6))

sns.barplot(x = 'quality', y = 'total sulfur dioxide', data = wine)
fig = plt.figure(figsize = (10,6))

sns.barplot(x = 'quality', y = 'sulphates', data = wine)
fig = plt.figure(figsize = (10,6))

sns.barplot(x = 'quality', y = 'alcohol', data = wine)
# Histograms

wine.hist(bins=20, figsize=(15,15));
# Plot scatter matrix

pd.plotting.scatter_matrix(wine,  figsize=(20,20));
# Plot correlation heatmap with 

corr = wine.corr()

plt.figure(figsize=(15,15))

ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, annot=True, square=False)

ax.set_xticklabels(ax.get_xticklabels(),rotation=45, horizontalalignment='right')

ax.set_ylim(len(corr)-0.5, -0.5)



# fix for mpl bug that cuts off top/bottom of seaborn visualization

b, t = plt.ylim() # discover the values for bottom and top

b += 0.5 # Add 0.5 to the bottom

t -= 0.5 # Subtract 0.5 from the top

plt.ylim(b, t) # update the ylim(bottom, top) values

plt.show()
# Preprocessing Data

bins = (0, 6.5, 8)  # Define bins

group_names = ['bad', 'good'] # Define names of classes

wine['quality'] = pd.cut(wine['quality'], bins=bins, labels=group_names) # Bin quality values into the specified bins

# This function is also useful for going from a continuous variable to a categorical variable

wine['quality'].unique()
label_quality = LabelEncoder() # Create instance of LabelEncoder
wine['quality'] = label_quality.fit_transform(wine['quality']) # Transform data: bad --> 0, good --> 1
wine.head(10)
# Anzahl an Werte Gut & Schlecht

wine['quality'].value_counts()
sns.countplot(wine['quality'])
# Values() = von Pandas DataFrame zurück in ein Array

X = wine.drop('quality', axis=1).values

y = wine['quality'].values
type(X)
X.shape
y.shape
# 20% test 80% training data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Training data: X: {X_train.shape} y: {y_train.shape}')

print(f'Test data: X: {X_test.shape} y: {y_test.shape}')
#Applying Standard scaling

feature_scaler = StandardScaler()

X_train = feature_scaler.fit_transform(X_train)

X_test = feature_scaler.transform(X_test)
RFC_base_classifier = RandomForestClassifier(n_estimators=100)   # Define base estimator
RFC_base_classifier.get_params()  # Show all changeable parameters
# Define parameter grid



grid_param_RFC = {

    'n_estimators': [100, 120, 140, 160, 180, 200, 230, 260, 300, 500, 800, 1000],

    'criterion': ['gini', 'entropy'],

    'max_depth': [None, 3, 4, 5, 7, 9, 10, 15, 20],

    'bootstrap': [True, False]

}
# Grid Search Instanz definieren (cv=5)



from sklearn.model_selection import GridSearchCV



grid_search_RFC = GridSearchCV(estimator=RFC_base_classifier,

                           param_grid=grid_param_RFC,

                           scoring='accuracy',

                           cv=5,                     # CV=5 --> 5-fold cross-validation

                           n_jobs=-1)
grid_search_RFC.fit(X_train, y_train)
# Show best parameter outcome



best_parameters_RFC = grid_search_RFC.best_params_

print(f'Best parameters: {best_parameters_RFC}')
# Mean cross-validation score of best model



best_result_RFC = grid_search_RFC.best_score_

print(f'Mean cross-validation score of best model: {best_result_RFC}')
SVC_base_classifier = SVC()   # Define base estimator
SVC_base_classifier.get_params()  # Show all changeable parameters
# Due to the fact that the calculation and therefore the grid search with SVM is computationally very

# expensive, we get ourselves a first look at good parameter values with help of the RandomizedSearch

# Instead of specifiing discrete values for the parameters 'C', 'gamma' and 'degree', we have to specify a 

# distribution of values. The aim of this process is to find the best kernel for the SVM and to get a first look

# of a good parameter values for the other parameters. Therefore you can run the randomized_search_SVC.fit()

# command multiple times and observe the output of the best parameters.



dist_param_SVC = {

    'C': np.arange(0.1, 10, 0.2),

    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],

    'gamma': np.arange(0.01, 1, 0.01),

    'degree': np.arange(3, 10, 1)

}
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV



randomized_search_SVC = RandomizedSearchCV(estimator=SVC_base_classifier,

                           param_distributions=dist_param_SVC,

                           scoring='accuracy',

                           cv=5,

                           n_jobs=-1)
randomized_search_SVC.fit(X_train, y_train)
# Show best parameter outcome



best_parameters_SVC = randomized_search_SVC.best_params_

print(f'Best parameters: {best_parameters_SVC}')
# Mean cross-validation score



best_result_SVC = randomized_search_SVC.best_score_

print(f'Mean cross-validation score of best model: {best_result_SVC}')
# Now we want to get the exact best parameter values of the model with the normal GridSearch

# With RandomizedSearch we have seen that the best kernel is the rbf-kernel. With this knowledge we can now compute

# the other parameter values much faster, because we do not have to calculate every kernel.

# For the other parameters you have to test a few intervals to find the best value.



grid_param_SVC = {

    'C': [2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 6.5, 7],

    'kernel': ['rbf'],

    'gamma': ['scale', 'auto', 0.2, 0.5, 0.6, 0.7, 0.8],

    'degree': [4, 5, 6, 7, 8, 9]

}
grid_search_SVC = GridSearchCV(estimator=SVC_base_classifier,

                           param_grid=grid_param_SVC,

                           scoring='accuracy',

                           cv=5,

                           n_jobs=-1)
grid_search_SVC.fit(X_train, y_train)
# Show best parameter outcome



best_parameters_SVC = grid_search_SVC.best_params_

print(f'Best parameters: {best_parameters_SVC}')
# Mean cross-validation score of best model

# Now we found the best parameter values for the SVC-model



best_result_SVC = grid_search_SVC.best_score_

print(f'Mean cross-validation score of best model: {best_result_SVC}')
knn_base_classifier = KNeighborsClassifier(n_neighbors=5) # Define base estimator
knn_base_classifier.get_params() # Show all changeable parameters
# Define parameter grid



grid_param_knn = {

    'n_neighbors': [3, 5, 7, 9, 11, 13],

    'weights': ['uniform', 'distance'],

    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']

}
grid_search_knn = GridSearchCV(estimator=knn_base_classifier,

                           param_grid=grid_param_knn,

                           scoring='accuracy',

                           cv=5,

                           n_jobs=-1)
grid_search_knn.fit(X_train, y_train)
# Show best parameter outcome



best_parameters_knn = grid_search_knn.best_params_

print(f'Best parameters: {best_parameters_knn}')
# Mean cross-validation score of best model



best_result_knn = grid_search_knn.best_score_

print(f'Mean cross-validation score of best model: {best_result_knn}')
# Initialize RFC with the calculated best parameters

# Note that a max_depth value of None can easily lead to overfitting



RFC = RandomForestClassifier(n_estimators=200, max_depth=None, criterion='gini', bootstrap=True)

RFC.fit(X_train, y_train)

pred_RFC = RFC.predict(X_test)
# Performance of model

print(classification_report(y_test, pred_RFC))

confmat_RFC = confusion_matrix(y_test, pred_RFC)
# Plot confusion matrix



fig, ax = plt.subplots(figsize=(2.5, 2.5))

ax.matshow(confmat_RFC, cmap=plt.cm.Blues, alpha=0.3)

for i in range(confmat_RFC.shape[0]):

    for j in range(confmat_RFC.shape[1]):

        ax.text(x=j, y=i, 

               s=confmat_RFC[i, j],

               va='center', ha='center')

plt.xlabel('Predicted Label')

plt.ylabel('True Label')

plt.show()
# Initialize SVC with the calculated best parameters



SVC_clf = SVC(kernel='rbf', C=2, gamma=0.6, degree=4)

SVC_clf.fit(X_train, y_train)

pred_SVC = SVC_clf.predict(X_test)
# Performance of model

print(classification_report(y_test, pred_SVC))

confmat_SVC = confusion_matrix(y_test, pred_SVC)
# Plot confusion matrix

fig, ax = plt.subplots(figsize=(2.5, 2.5))

ax.matshow(confmat_SVC, cmap=plt.cm.Blues, alpha=0.3)

for i in range(confmat_SVC.shape[0]):

    for j in range(confmat_SVC.shape[1]):

        ax.text(x=j, y=i, 

               s=confmat_SVC[i, j],

               va='center', ha='center')

plt.xlabel('Predicted Label')

plt.ylabel('True Label')

plt.show()
# Initialize KNN with the calculated best parameters



knn_clf = KNeighborsClassifier(n_neighbors=5, algorithm='auto', weights='distance')

knn_clf.fit(X_train, y_train)

pred_knn = knn_clf.predict(X_test)
# Performance of model

print(classification_report(y_test, pred_knn))

confmat_knn = confusion_matrix(y_test, pred_knn)
# Plot confusion matrix

fig, ax = plt.subplots(figsize=(2.5, 2.5))

ax.matshow(confmat_knn, cmap=plt.cm.Blues, alpha=0.3)

for i in range(confmat_knn.shape[0]):

    for j in range(confmat_knn.shape[1]):

        ax.text(x=j, y=i, 

               s=confmat_knn[i, j],

               va='center', ha='center')

plt.xlabel('Predicted Label')

plt.ylabel('True Label')

plt.show()
# Now we want to use the model on new, unseen data. Therefore we create a new, randomly specified data point

# You can change the values of pH and alcohol to see different outcomes



pH = 3

alcohol = 12

Xnew = [[7.8, 0.22, 0.99, 2.0, 0.01, 9.0, 18.0, 0.9968, pH, 1.8, alcohol]]
Xnew = feature_scaler.transform(Xnew) #Use same transformer and model
ynew_RFC = RFC.predict(Xnew)

ynew_SVC = SVC_clf.predict(Xnew)

ynew_knn = knn_clf.predict(Xnew)



if ynew_RFC==0:

    label_RFC = 'bad'

else:

    label_RFC = 'good'

    

if ynew_SVC==0:

    label_SVC = 'bad'

else:

    label_SVC = 'good'

    

if ynew_knn==0:

    label_knn = 'bad'

else:

    label_knn = 'good'



print('Result of classification: ')

print(f'Random Forest Classifier: Label = {ynew_RFC} --> {label_RFC} wine')

print(f'Support Vector Classifier: Label = {ynew_SVC} --> {label_SVC} wine')

print(f'K-Nearest Neighbors Classifier: Label = {ynew_knn} --> {label_knn} wine')