# Importing the libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Importing the dataset
dataset = pd.read_csv("../input/indian_liver_patient.csv")
dataset_desc = dataset.describe(include = 'all')
print(dataset_desc)

# Identifying the dependent (x) and independent (y) variables
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 10].values
# Checking missing data
dataset_mis = dataset.isnull().sum()
print(dataset_mis)
# Taking care of missing data!
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
X[:, 2:10] = imputer.fit_transform(X[:, 2:10])
# Encoding the categorical values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
onehotencoder_X = OneHotEncoder(categorical_features = [1])
onehotencoder_X.fit_transform(X).toarray()
# Splitting the dataset into trainig and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# Dimesnionality Reduction using PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = None)
X_train  = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
print("Explained Variance:\n", explained_variance)
pca = PCA(n_components = 6)
X_train  = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
# Fitting Random Forest model into the training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)
# Making a prediction
from sklearn.metrics import classification_report
y_pred = classifier.predict(X_test)
test_accuracy = classification_report(y_test, y_pred)
print(test_accuracy)
# Grid Search to find the best tuning
# Params for Random Forest
parameters = [{'criterion' : ['gini', 'entropy'],
               'max_depth' : [5, 6, 7, 8, 9, 10, 11, 12],
               'max_features' : [1, 2, 3],
               'n_estimators' : [14, 15, 16, 17, 18, 19],
               'random_state' : [7, 8, 9, 10, 11, 12, 13]}]

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           n_jobs = -1, cv = 10)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
print("Best accuracy of the model for the training set is:", best_accuracy)
best_params = grid_search.best_params_
print("Best parameters of the model for the training set is:", best_params)
# Tune the hyperparameters of Random Forest Classifier based on best_params resulted from Grid Search method
classifier = RandomForestClassifier(criterion = 'gini',
                                    max_depth = 9,
                                    max_features = 1,
                                    n_estimators = 16,
                                    random_state = 10)
classifier.fit(X_train, y_train)
# Let's check the test accuracy after optimised
y_pred = classifier.predict(X_test)
test_accuracy_optimized = classification_report(y_test, y_pred)
print(test_accuracy_optimized)
# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
test_accuracy_optimized_cm = (cm[0,0]+cm[1,1])/146 #146 is the total number of testing data
print("\nTesting accuracy based on the Confusion Matrix:\n", test_accuracy_optimized_cm)
# K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier,
                            X = X_train, y = y_train,
                            cv = 10, n_jobs = -1)
print("Showing all 10 of K-Fold Cross Validation accuracies:\n", accuracies)
accuracies_mean = accuracies.mean()
print("\nMean of accuracies:\n", accuracies_mean)
accuracies_std = accuracies.std()
print("\nStandard Deviation:\n", accuracies_std)