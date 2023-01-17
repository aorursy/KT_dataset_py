# Get the data

import pandas as pd

dataset = pd.read_csv('../input/train.csv') # read the csv file

X = dataset.iloc[:, :20].values # Independent variables(features): first 20 columns

y = dataset.iloc[:, 20].values # dependent variable price: last column
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42) # Split the data into training and testing data

# Preprocessing - Feature Scaling

# This step needs to be done as all the independent variables must be in the same scale

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# Training the model

from sklearn.svm import SVC

classifier_svm = SVC(C = 0.5 , kernel = 'linear', random_state = 42)

classifier_svm.fit(X_train, y_train)
# Predicting the Test set results

y_svm = classifier_svm.predict(X_test)
# Printing accuracy

from sklearn.metrics import accuracy_score

print('Accuracy using Kernel SVM : ',accuracy_score(y_test,y_svm)*100,'%')