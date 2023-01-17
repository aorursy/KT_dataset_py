# Importing basic packages for data preprocessing
import numpy as np
import pandas as pd
import os
# Importing packages for plotting
import matplotlib.pyplot as plt
import seaborn as sns
print(os.listdir("../input"))
dataset=pd.read_csv("../input/indian_liver_patient.csv")
dataset.head()
dataset.describe()
dataset.isnull().sum()
dataset[dataset['Albumin_and_Globulin_Ratio'].isnull()].index.tolist()
# Using the above row indexes, removing rows with missing values in Albumin_and_Globulin_Ratio column
dataset.drop(dataset.index[[209,241,253,312]], inplace=True)
# Creating copy of the dataset
dataset_orig = dataset.copy()
# Transforming Gender column (indepedent variable) to numerics (0s and 1s)
# Importing required package
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
dataset['Gender'] = labelencoder_x.fit_transform(dataset['Gender'])
dataset['Gender'].head()
# Finding the correlation of independent data with the dependent data Income column 
corrmat = dataset.corr()
corrmat
import seaborn as sns
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=1, cmap="YlGnBu", square=True,linewidths=.5, annot=True)
plt.show()
# Obtaining top K columns which affects the Income the most
k= 10
corrmat.nlargest(k, 'Dataset')
# Replotting the heatmap with the above data
cols = corrmat.nlargest(k, 'Dataset')['Dataset'].index
cm = np.corrcoef(dataset[cols].values.T)
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(cm, cmap="YlGnBu", cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
sns.catplot(x="Gender", y="Age", hue="Dataset", data=dataset_orig)
plt.show()
f, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x="Albumin", y="Albumin_and_Globulin_Ratio", hue="Dataset", style="Dataset", data=dataset_orig);
plt.show()
# Splitting Independent (X) and dependent (y) variables from the dataset
X = dataset[['Albumin_and_Globulin_Ratio', 'Albumin','Total_Protiens']]
y = dataset [['Dataset']]
X[0:5]
y[0:5]
# Splitting the data into Training and Test set with 80-20 ratio
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=0)
print("X_train: " , X_train.shape)
print("X_test: ", X_test.shape)

# Import required package
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
X_train[0:5]
X_test[0:5]
from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression(random_state=0, solver='lbfgs')
classifier_lr.fit(X_train,y_train.values.reshape(-1,))
# predict the test set result
y_predLR = classifier_lr.predict(X_test)
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, y_predLR)
cm
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predLR)
from sklearn.neighbors import KNeighborsClassifier
classifierKNN = KNeighborsClassifier(n_neighbors=5,p=2, metric='minkowski')
classifierKNN.fit(X_train, y_train.values.reshape(-1,))

# predict the test set result
y_predKNN = classifierKNN.predict(X_test)
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, y_predKNN)
cm
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predKNN)
# Importing the required package 
from sklearn.svm import SVC
classifier_svm = SVC(kernel='linear', random_state=0)
classifier_svm.fit(X_train, y_train.values.reshape(-1,))
# predict the test set result
y_predSVM = classifier_svm.predict(X_test)
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, y_predSVM)
cm
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predSVM)
from sklearn.naive_bayes import GaussianNB
classifierNB = GaussianNB()
classifierNB.fit(X_train, y_train.values.reshape(-1,))

# predict the test set result
y_predNB = classifierNB.predict(X_test)
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, y_predNB)
cm
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predNB)
