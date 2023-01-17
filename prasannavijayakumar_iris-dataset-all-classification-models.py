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
#import neccessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Loading the dataset 
df = pd.read_csv('../input/iris-flower-dataset/IRIS.csv')
# see the first 5 rows of data 
df.head()
#lets see the shape of dataset
df.shape
#datatypes of features
df.dtypes
#lets understand target label
df['species'].value_counts()
#checking if missing values are present
df.isna().sum()
#understand statistical summary of numerical columns
df.describe()
#understand the correlation between features
df.corr()
#check variance of the dataset
df.var()
sns.swarmplot(x='species', y='petal_length', data=df)
plt.show()
sns.swarmplot(x='species', y='sepal_width', data=df)
plt.show()
sns.pairplot(df, hue='species', diag_kind='hist')
plt.show()
sns.scatterplot(x='petal_length', y='sepal_width', data=df, hue='species')
plt.show()
#dropping redudant feature
#as we know dropping petal_length or petal_width would help imporve model performance

df_1= df.drop('petal_width', axis=1)
df_1.head()
#Change Class label dtype to Category
df_1['species']= df_1['species'].astype('category')
# Seperating the data into dependent and independent variables
X = df_1.iloc[:, :-1].values
y = df_1.iloc[:,-1].cat.codes
#import necessary scikit learn modules
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
#split the training, test set
X_train, X_test, y_train, y_test= train_test_split(X, y, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

#note= statify is not used since the distribution of target labels are even
#instantiate KNN model
knn= KNeighborsClassifier(n_neighbors=6)

#fit KNeighbors Classifiers model to training set
knn.fit(X_train, y_train)
#predict labels for test set
y_pred= knn.predict(X_test)
print(y_pred)
#Checking the training and test set accuracy to understand if model is overfit or underfit

training_accuracy= knn.score(X_train, y_train)
test_accuracy= knn.score(X_test, y_test)

print('training set accuracy is',training_accuracy)
print('test set accuracy is',test_accuracy)
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))
from sklearn.linear_model import LogisticRegression

#instantiate Logistic Regression model
logreg= LogisticRegression(solver='lbfgs')

#Fit Logistic Regression model to training data
logreg.fit(X_train, y_train)
#predict labels for test set
y_pred2= logreg.predict(X_test)
print(y_pred2)
#Checking the training and test set accuracy to understand if model is overfit or underfit

logreg_training_accuracy= logreg.score(X_train, y_train)
logreg_test_accuracy= logreg.score(X_test, y_test)

print('training set accuracy is',logreg_training_accuracy)
print('test set accuracy is',logreg_test_accuracy)
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred2,y_test))
from sklearn.tree import DecisionTreeClassifier

dt= DecisionTreeClassifier(max_depth=2)

#fit Decision Tree Classifier to training set
dt.fit(X_train, y_train)
y_pred3= dt.predict(X_test)
y_pred3
#Checking the training and test set accuracy to understand if model is overfit or underfit

dt_training_accuracy= dt.score(X_train, y_train)
dt_test_accuracy= dt.score(X_test, y_test)

print('training set accuracy is',dt_training_accuracy)
print('test set accuracy is',dt_test_accuracy)
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred3,y_test))
from sklearn.ensemble import RandomForestClassifier

rf= RandomForestClassifier(n_estimators=25,)

#Fit the random forest model to the training data
rf.fit(X_train, y_train)

y_pred4= dt.predict(X_test)
y_pred4
#Checking the training and test set accuracy to understand if model is overfit or underfit

rf_training_accuracy= rf.score(X_train, y_train)
rf_test_accuracy= rf.score(X_test, y_test)

print('training set accuracy is',rf_training_accuracy)
print('test set accuracy is',rf_test_accuracy)