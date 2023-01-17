# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import Libraries

#from mlxtend.plotting import plot_decision_regions

import numpy as np

import pandas as pd

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix

import seaborn as sns

sns.set()

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
#Loading the dataset

diabetesDataSet = pd.read_csv('/kaggle/input/diabetes/diabetes.csv')
# Diaplay dataframe.

diabetesDataSet
# Basic Info about data

# Information about the data types,columns, null value counts, memory usage etc

diabetesDataSet.info(verbose=True)
# Statistic details

diabetesDataSet.describe()
# Shape

diabetesDataSet.shape
# Above summary show some column have zero(0)

# So we replace zeros with nan since

diabetesDataCopy = diabetesDataSet.copy(deep = True)

diabetesDataCopy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetesDataCopy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
## showing the count of Nans

print(diabetesDataCopy.isnull().sum())
diabetesDataCopy.describe()
# Data Visualization

# Univariate Plots

# histograms

diabetesDataSet.hist(figsize = (20,20))

plt.show()
# calculate nan value and compare above.

diabetesDataCopy['Glucose'].fillna(diabetesDataCopy['Glucose'].mean(), inplace = True)

diabetesDataCopy['BloodPressure'].fillna(diabetesDataCopy['BloodPressure'].mean(), inplace = True)

diabetesDataCopy['SkinThickness'].fillna(diabetesDataCopy['SkinThickness'].median(), inplace = True)

diabetesDataCopy['Insulin'].fillna(diabetesDataCopy['Insulin'].median(), inplace = True)

diabetesDataCopy['BMI'].fillna(diabetesDataCopy['BMI'].median(), inplace = True)
# histograms

copyPlt = diabetesDataCopy.hist(figsize = (20,20))
# Multivariate Plots

# scatter plot matrix

scatter_matrix(diabetesDataSet, figsize=(20, 20))

plt.show()
# Data cleaning

p=sns.pairplot(diabetesDataCopy, hue = 'Outcome')
#split dataset in features and target variable

feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']

X = diabetesDataSet[feature_cols] # Features
X.head()
# Target variable

y = diabetesDataCopy.Outcome 
y.head()
# Split dataset into training set and test set

# 70% training and 30% test

validation_size = 0.3

seed = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_size, random_state=seed) 
# Create Model

# Create Decision Tree classifer object

# 1. Apply DecisionTreeClassifier Algo for get accuracy

clf = DecisionTreeClassifier()



# Train Decision Tree Classifer

clf = clf.fit(X_train,y_train)



#Predict the response for test dataset

predictions = clf.predict(X_test)



# Model Accuracy, how often is the classifier correct?

print("Training Score Accuracy for Decision Tree:",metrics.accuracy_score(y_test, predictions))

print('Test score: {}'.format(clf.score(X_test, y_test)))
# 2. Apply KNeighborsClassifier Algo for get accuracy 



# Train KNeighborsClassifier

knn = KNeighborsClassifier()



#Predict the response for test dataset

knn = knn.fit(X_train,y_train)



#Predict the response for test dataset

predictions = knn.predict(X_test)



# Model Accuracy, how often is the classifier correct?

print("Training Score Accuracy for Knn :",metrics.accuracy_score(y_test, predictions))

print('Test score: {}'.format(clf.score(X_test, y_test)))
# 3. Apply LogisticRegression Algo for get accuracy 



# Train LogisticRegression

lr = LogisticRegression(solver='liblinear', multi_class='ovr')



#Predict the response for test dataset

lr = lr.fit(X_train,y_train)



#Predict the response for test dataset

predictions = lr.predict(X_test)



# Model Accuracy, how often is the classifier correct?

print("Training Score Accuracy for LR :",metrics.accuracy_score(y_test, predictions))

print('Test score: {}'.format(clf.score(X_test, y_test)))
# 4. Apply LinearDiscriminantAnalysis Algo for get accuracy 



# Train LinearDiscriminantAnalysis

LDA = LinearDiscriminantAnalysis()



#Predict the response for test dataset

LDA = LDA.fit(X_train,y_train)



#Predict the response for test dataset

predictions = LDA.predict(X_test)



# Model Accuracy, how often is the classifier correct?

print("Training Score Accuracy for LDA :",metrics.accuracy_score(y_test, predictions))

print('Test score: {}'.format(clf.score(X_test, y_test)))
# 5. Apply GaussianNB Algo for get accuracy 



# Train GaussianNB

NB = GaussianNB()



#Predict the response for test dataset

NB = NB.fit(X_train,y_train)



#Predict the response for test dataset

predictions = NB.predict(X_test)



# Model Accuracy, how often is the classifier correct?

print("Training Score Accuracy for NB :",metrics.accuracy_score(y_test, predictions))

print('Test score: {}'.format(clf.score(X_test, y_test)))
# 6. Apply SVC Algo for get accuracy 



# Train SVC

SVM = SVC(gamma='auto')



#Predict the response for test dataset

SVM = SVM.fit(X_train,y_train)



#Predict the response for test dataset

predictions = SVM.predict(X_test)



# Model Accuracy, how often is the classifier correct?

print("Training Score Accuracy for SVM :",metrics.accuracy_score(y_test, predictions))

print('Test score: {}'.format(clf.score(X_test, y_test)))