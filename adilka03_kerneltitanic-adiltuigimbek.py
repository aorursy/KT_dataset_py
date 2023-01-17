# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

project_dir = "../input/"

# Any results you write to the current directory are saved as output.
df = pd.read_csv(project_dir+'train.csv')
def preprocessing(df):

    y = df[['Survived']]

    X = df.iloc[:,2:]

    X = X.iloc[:, 0:]

    X['Sex'] = X['Sex'].map({'male': 1, 'female': 0})

    X = X.drop(['Ticket', 'Cabin', 'Name'], axis=1)

    X['Age'] = X['Age'].fillna(X['Age'].mean())

    X = pd.get_dummies(X, drop_first=True)

    print(X.head())

    print(X.info())

    X = X.values

    y = y.values.ravel()

    return X, y
X, y = preprocessing(df)
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# Setup arrays to store train and test accuracies

neighbors = np.arange(1, 15)

train_accuracy = np.empty(len(neighbors))

test_accuracy = np.empty(len(neighbors))



# Loop over different values of k

for i, k in enumerate(neighbors):

    # Setup a k-NN Classifier with k neighbors: knn

    knn = KNeighborsClassifier(n_neighbors=k)



    # Fit the classifier to the training data

    knn.fit(X_train, y_train)

    

    #Compute accuracy on the training set

    train_accuracy[i] = knn.score(X_train, y_train)



    #Compute accuracy on the testing set

    test_accuracy[i] = knn.score(X_test, y_test)



# Generate plot

plt.title('k-NN: Varying Number of Neighbors')

plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')

plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')

plt.legend()

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')

plt.show()
# Setup a k-NN Classifier with k neighbors: knn

knn = KNeighborsClassifier(n_neighbors=9)



# Fit the classifier to the training data

knn.fit(X_train, y_train)



#Compute accuracy on the training set

print("Accurancy train: %s" % (knn.score(X_train, y_train)))



#Compute accuracy on the testing set

print("Accurancy test: %s" % (knn.score(X_test, y_test)))
# Import necessary modules

from sklearn.linear_model import LogisticRegression 

from sklearn.model_selection import GridSearchCV



# Setup the hyperparameter grid

c_space = np.logspace(-5, 8, 15)

param_grid = {'C': c_space}



# Instantiate a logistic regression classifier: logreg

logreg = LogisticRegression(solver='liblinear')



# Instantiate the GridSearchCV object: logreg_cv

logreg_cv = GridSearchCV(logreg,param_grid=param_grid, cv=5)



# Fit it to the data

logreg_cv.fit(X,y)



# Print the tuned parameters and score

print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 

print("Best score is {}".format(logreg_cv.best_score_))
# Create the hyperparameter grid

c_space = np.logspace(-5, 8, 15)

param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}



# Instantiate the logistic regression classifier: logreg

logreg = LogisticRegression(solver='liblinear')



# Create train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,  random_state=42)



# Instantiate the GridSearchCV object: logreg_cv

logreg_cv = GridSearchCV(logreg, param_grid, cv=5)



# Fit it to the training data

logreg_cv.fit(X_train,y_train)



# Print the optimal parameters and best score

print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))

print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))
y_pred = logreg_cv.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

# Generate the confusion matrix and classification report

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

logreg_cv.score(X_test,y_test)
test_data = pd.read_csv(project_dir + 'test.csv')
test_data.head()
testX = test_data.iloc[:, 1:]

pasangers = test_data[['PassengerId']]

testX['Sex'] = testX['Sex'].map({'male': 1, 'female': 0})

testX = testX.drop(['Ticket', 'Cabin', 'Name'], axis=1)

testX['Age'] = testX['Age'].fillna(testX['Age'].mean())

testX = pd.get_dummies(testX, drop_first=True)

print(testX.head())

print(testX.info())
print(testX.isnull().sum())
# Drop missing values and print shape of new DataFrame

testX = testX.fillna(testX.mean())
y_pred = logreg_cv.predict(testX)
#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not

submission = pd.DataFrame({'PassengerId':pasangers.values.ravel(),'Survived':y_pred})



#Visualize the first 5 rows

submission.head()
#Convert DataFrame to a csv file that can be uploaded

#This is saved in the same directory as your notebook

filename = 'Titanic_Predictions_AT.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)