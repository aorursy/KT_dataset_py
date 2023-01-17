# Importing the numpy and pandas libraries

import numpy as np

import pandas as pd
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Data Preprocessing - Importing the dataset

# Vector X contains Pclass, Sex, Age, Sibsp and Parch columns

# Vector Y contains Survived column

dataset = pd.read_csv('../input/train.csv')

X = dataset.iloc[:, [2,4,5,6,7,11]].values

y = dataset.iloc[:, 1:2].values



# Removing 2 rows with Nan Embarked

X = np.delete(X, (61, 829), axis=0)

y = np.delete(y, (61, 829), axis=0)

print(X[61:66])

# print(y)
# Data Preprocessing - Replacing the missing Age values with average age of the dataset

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputer = imputer.fit(X[:,2:3])

X[:, 2:3] = imputer.transform(X[:, 2:3])

# print(X)
# Data Preprocessing - Encoding the Categorical Data for 'PClass' and 'Sex'

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()

X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

X[:, 1] = labelencoder_X.fit_transform(X[:, 1])

X[:, 5] = labelencoder_X.fit_transform(X[:, 5])

onehotencoder = OneHotEncoder(categorical_features = [0,1,5])

X = onehotencoder.fit_transform(X).toarray()

print (X[0:6,0:8])

# print (y)



# Encoded Columns of X:

# Index 0 - Pclass value 1

# Index 1 - Pclass value 2

# Index 2 - Pclass value 3

# Index 3 - Sex value 'female'

# Index 4 - Sex value 'male'

# Index 5 - Embarked value 'C'

# Index 6 - Embarked value 'Q'

# Index 7 - Embarked value 'S'



# Avoiding the Dummy Variable Trap

# Removing the first column from each set of Dummy Variables for Pclass and Sex.

# This removes the column representing Pclass value 1 and and Sex value 'female' respectively.

X=X[:,[1,2,4,6,7,8,9,10]]

print(X[0:6,0:8])



# Final Indexes of X after removing one column each from both the Dummy Variable sets for Pclass and Sex:

# Index 0 - Pclass value 2

# Index 1 - Pclass value 3

# Index 2 - Sex value 'male'

# Index 3 - Embarked value 'Q'

# Index 4 - Embarked value 'S'

# Index 5 - Age

# Index 6 - SibSp

# Index 7 - Parch
# Data Preprocessing - Creating the Training and the Test Set

# There is a Test set provided with this exercise but that is to be used for submission purposes as it does not have Survived values

# So, Splitting the dataset into the Training set and Test set to evaluate the performance of my Machine Learning Model

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# print(X_train)

# print(y_train)

# print(X_test)

# print(y_test)
# Data Preprocessing - Applying Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

# print(X_train)

# print(X_test)

# print(y_train)

# print(y_test)
# Applying LDA

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# lda = LDA(n_components = 3)

# X_train = lda.fit_transform(X_train, y_train.ravel())

# X_test = lda.transform(X_test)



# Applying Kernel PCA

#from sklearn.decomposition import KernelPCA

#kpca = KernelPCA(n_components = 2, kernel = 'rbf')

#X_train = kpca.fit_transform(X_train)

#X_test = kpca.transform(X_test)
# Creating the Machine Learning Model using Random Forest Classification method

# The optimal hyper parameters for the classifier has been optimized using Grid Search later in this code

# from sklearn.ensemble import RandomForestClassifier

# classifier = RandomForestClassifier(n_estimators = 12, criterion = 'gini', max_features = 'auto', random_state = 34)

# classifier.fit(X_train, y_train.ravel())



# Creating the Machine Learning Model using SVM method

from sklearn.svm import SVC

classifier = SVC(C = 0.3, gamma = 0.2, kernel = 'rbf', random_state = 0)

classifier.fit(X_train, y_train.ravel())





# Predicting the Test set results

y_pred = classifier.predict(X_test)

# print(y_test)

# print(y_pred)
# Evaluating the Performance of the Machine Learning Model (classifier) - By creating a confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred.ravel())

print(cm)



# Evaluating the Performance of the Machine Learning Model (classifier) - By applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train.ravel(), cv = 10)

mean=accuracies.mean()

std=accuracies.std()

print(mean)

print(std)
# Applying Grid Search - For choosing the best model and the optimal values for the hyper parameters of the SVM classifier



from sklearn.model_selection import GridSearchCV

parameters = [{'C':[0.30, 0.325, 0.35, 0.375, 0.40],

               'kernel': ['rbf'],

               'gamma' : [0.16, 0.17, 0.18, 0.19, 0.20],

               'random_state':[0, 5, 10]}

             ]

grid_search = GridSearchCV(estimator = classifier,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 10,

                           n_jobs = -1)

print('started...')

grid_search = grid_search.fit(X_train, y_train.ravel())

print('ended...')

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_

print(best_accuracy)

print(best_parameters)



#Best Parameters - {'C': 0.75, 'gamma' : 0.25, 'kernel' : 'rbf', 'random_state': 0}

# These values are applied above in the Classifier
# Data Preprocessing - Importing the Test Set provided for submission

dataset = pd.read_csv('../input/test.csv')

X1 = dataset.iloc[:, [1,3,4,5,6,10]].values

# print(X1)
# Data Preprocessing - Replacing missing Age values with average age in the Test Set

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputer = imputer.fit(X1[:,2:3])

X1[:, 2:3] = imputer.transform(X1[:, 2:3])

# print(X1)
# Data Preprocessing - Encoding Categorical Data for 'PClass' and 'Sex' in the Test Set

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X1 = LabelEncoder()

X1[:, 0] = labelencoder_X1.fit_transform(X1[:, 0])

X1[:, 1] = labelencoder_X1.fit_transform(X1[:, 1])

X1[:, 5] = labelencoder_X1.fit_transform(X1[:, 5])

onehotencoder1 = OneHotEncoder(categorical_features = [0,1,5])

X1 = onehotencoder1.fit_transform(X1).toarray()

# print (X1[0:11,:])



#Encoded Columns:

# Index 0 - Pclass value 1

# Index 1 - Pclass value 2

# Index 2 - Pclass value 3

# Index 3 - Sex value 'female'

# Index 4 - Sex value 'male'



# Avoiding the Dummy Variable Trap

# Removing one dummy column each for Pclass and Sex i.e values 1 and 'female' respectively

X1=X1[:,[1,2,4,6,7,8,9,10]]

# print(X1)



#Final Indexes:

# Index 0 - Pclass value 2

# Index 1 - Pclass value 3

# Index 2 - Sex value 'male'

# Index 3 - Age

# Index 4 - SibSp

# Index 5 - Parch
# Data Preprocessing - Defining the new Test set with the sample data provided in the exercise to evaluate whether all the female passengers got saved.

X1_test = X1

# print (X1_test)
# Data Preprocessing -  Feature Scaling

from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()

X1_test = sc1.fit_transform(X1_test)

# print(X1_test)
# Predicting the Test set results

y1_pred = classifier.predict(X1_test)

# print(y1_pred)
# Preparing the output in the desired format of PassengerID in the first column and Survival Prediction in the second column with the correct headers

passengerID = dataset.iloc[:, 0:1].values

# print (passengerID)

result_set=np.column_stack((passengerID,y1_pred))

columnnames = [_ for _ in ['PassengerID','Survived']]

output = pd.DataFrame(result_set, columns=columnnames)

print(output)



# Dumping the predictions of the provided Test Set in the output.csv file

output.to_csv('rbf.csv', index=False, header=True, sep=',')