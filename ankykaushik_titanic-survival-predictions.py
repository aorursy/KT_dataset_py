# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# importing the given titanic training and test datasets

titanic= pd.read_csv('../input/titanic/train.csv')

titanic_test= pd.read_csv('../input/titanic/test.csv')



pd.set_option('display.max_columns', 10)
print(titanic.info())

print(titanic_test.info())
# Changing the columns to categorical datatype in given training dataset

titanic[['Pclass', 'Sex','Survived']]= titanic[['Pclass','Sex', 'Survived']].astype('category')

titanic.info()
# Extracting matrix of features from training dataset

X = titanic.iloc[:,[2,4,5,6,7,9]].values

y = titanic.iloc[:,1].values

print (X)

print (y)
# Imputing values in columns containing numerical data in matrix of features

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer.fit(X[:,[2]])# in matrix of features X, age column is at 3rd position, hence column number 2

X[:,[2]]= imputer.transform(X[:,[2]])

print(X)
# Encoding categorical variables in matrix of features

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')

X = np.array(ct.fit_transform(X))

le_sur = LabelEncoder()

X[:,3] = le_sur.fit_transform(X[:,3])
# Removing dummy variable

X = X[:,1:]
# Splitting data into training and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=0)
# Scaling the values in matrix of features

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)
# Applying Logistic regression model

from sklearn.linear_model import LogisticRegression

classifier_LR = LogisticRegression(random_state=0)

classifier_LR.fit(X_train, y_train)



# predicting values from LogisticRegression model

y_pred_LR = classifier_LR.predict(X_test)



# confusion matrix

from sklearn.metrics import confusion_matrix, accuracy_score

cm_LR = confusion_matrix(y_test,y_pred_LR)

print(cm_LR)

# accuracy 

accuracy_score_LR= accuracy_score(y_test, y_pred_LR)

print("Accuracy of Logistic regression model on training dataset is ",round(accuracy_score_LR*100, 2),"%")
# Applying KNN model

from sklearn.neighbors import KNeighborsClassifier

classifier_knn = KNeighborsClassifier(n_neighbors=5)

classifier_knn.fit(X_train, y_train)



# predicting values from LogisticRegression model

y_pred_knn = classifier_knn.predict(X_test)



# confusion matrix

from sklearn.metrics import confusion_matrix, accuracy_score

cm_knn = confusion_matrix(y_test,y_pred_knn)

print(cm_knn)

# accuracy 

accuracy_score_knn= accuracy_score(y_test, y_pred_knn)

print("Accuracy of KNearestNeighbors model on training dataset is ",round(accuracy_score_knn*100, 2),"%")
# Applying naive bayes model

from sklearn.naive_bayes import GaussianNB

classifier_NB = GaussianNB()

classifier_NB.fit(X_train, y_train)



# predicting values from LogisticRegression model

y_pred_NB = classifier_NB.predict(X_test)



# confusion matrix

from sklearn.metrics import confusion_matrix, accuracy_score

cm_NB = confusion_matrix(y_test,y_pred_NB)

print(cm_NB)

# accuracy 

accuracy_score_NB= accuracy_score(y_test, y_pred_NB)

print("Accuracy of Naive Bayes model on training dataset is ",round(accuracy_score_NB*100, 2),"%")
# Applying SVM model

from sklearn.svm import SVC

classifier_SVC = SVC(random_state=0)

classifier_SVC.fit(X_train, y_train)



# predicting values from LogisticRegression model

y_pred_SVC = classifier_SVC.predict(X_test)



# confusion matrix

from sklearn.metrics import confusion_matrix, accuracy_score

cm_SVC = confusion_matrix(y_test,y_pred_SVC)

print(cm_SVC)

# accuracy 

accuracy_score_SVC= accuracy_score(y_test, y_pred_SVC)

print("Accuracy of SVC model on training dataset is ",round(accuracy_score_SVC*100, 2),"%")
# Applying Decison Tree model

from sklearn.tree import DecisionTreeClassifier

classifier_DT = DecisionTreeClassifier(criterion='entropy',random_state=0)

classifier_DT.fit(X_train, y_train)



# predicting values from LogisticRegression model

y_pred_DT = classifier_DT.predict(X_test)



# confusion matrix

from sklearn.metrics import confusion_matrix, accuracy_score

cm_DT = confusion_matrix(y_test,y_pred_DT)

print(cm_DT)

# accuracy 

accuracy_score_DT= accuracy_score(y_test, y_pred_DT)

print("Accuracy of DecisonTree model on training dataset is ",round(accuracy_score_DT*100, 2),"%")
# Applying Random Forest model

from sklearn.ensemble import RandomForestClassifier

classifier_RF = RandomForestClassifier(random_state=0)

classifier_RF.fit(X_train, y_train)



# predicting values from LogisticRegression model

y_pred_RF = classifier_RF.predict(X_test)



# confusion matrix

from sklearn.metrics import confusion_matrix, accuracy_score

cm_RF = confusion_matrix(y_test,y_pred_RF)

print(cm_RF)

# accuracy 

accuracy_score_RF= accuracy_score(y_test, y_pred_RF)

print("Accuracy of Random forest model on training dataset is ",round(accuracy_score_RF*100, 2),"%")
# Preparing test dataset for analysis



# converting respecive columns to categorical datatype

titanic_test[['Pclass', 'Sex']]= titanic_test[['Pclass', 'Sex']].astype('category')
# Extracting matrix of features from test dataset

X_test_2= titanic_test.iloc[:,[1,3,4,5,6,8]].values

print(X_test_2)
# Imputing missing values in matrix of features in test dataset

imputer_test= SimpleImputer(missing_values=np.nan, strategy='mean')

imputer_test.fit(X_test_2[:,[2,5]])

X_test_2[:,[2,5]]= imputer_test.transform(X_test_2[:,[2,5]])
# Encoding categorical varibales in matrix of features in test dataset

ct_test = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[0])], remainder= 'passthrough')

X_test_2 = np.array(ct.fit_transform(X_test_2))

le_test = LabelEncoder()

X_test_2[:,3] = le_test.fit_transform(X_test_2[:,3])

print(X_test_2)

# Avoidng dummy variable

X_test_2 = X_test_2[:,1:]

print(X_test_2)
# Scaling the features in test dataset

X_test_2 = sc_X.transform(X_test_2)
# Predicting values on test dataset

y_pred_test_RF = classifier_RF.predict(X_test_2)

print(y_pred_test_RF)
# Storing result in output file

output = pd.DataFrame({"PassengerId":titanic_test.PassengerId, "Survived": y_pred_test_RF})

print(output)

# result to csv file

output.to_csv('Submission1.csv', index= False)