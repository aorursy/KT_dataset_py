# Importing the libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import metrics
# Importing the dataset

data = pd.read_csv('../input/Iris.csv')
# Printing the 1st 5 columns

data.head()
# Printing the dimenions of data

data.shape
# Viewing the column heading

data.columns
# Inspecting the target variable

data.Species.value_counts()
data.dtypes
# Identifying the unique number of values in the dataset

data.nunique()
# Checking if any NULL values or any inconsistancies are present in the dataset

data.isnull().sum()
# See rows with missing values

data[data.isnull().any(axis=1)]
# Viewing the data statistics

data.describe()
data.info()
# Dropping the Id column as it is unnecessary for our model

data.drop('Id',axis=1,inplace=True)
fig = data[data.Species=='Iris-setosa'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', color='blue', label='Setosa')

data[data.Species=='Iris-versicolor'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', color='orange', label='Versicolor', ax=fig)

data[data.Species=='Iris-virginica'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', color='green', label='Virginica', ax=fig)

fig.set_xlabel("Sepal Length")

fig.set_ylabel("Sepal Width")

fig.set_title("Sepal Length vs Sepal Width")

fig=plt.gcf()

# fig.set_size_inches(20,10)

plt.show()
fig = data[data.Species=='Iris-setosa'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='blue', label='Setosa')

data[data.Species=='Iris-versicolor'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='orange', label='Versicolor', ax=fig)

data[data.Species=='Iris-virginica'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='green', label='Virginica', ax=fig)

fig.set_xlabel("Petal Length")

fig.set_ylabel("Petal Width")

fig.set_title("Petal Length vs Petal Width")

fig=plt.gcf()

# fig.set_size_inches(20,10)

plt.show()
data.hist(edgecolor='black', linewidth=2)

fig=plt.gcf()

fig.set_size_inches(12,10)

plt.show()
sns.boxplot(x="Species", y="PetalLengthCm", data=data)
sns.boxplot(x="Species", y="PetalWidthCm", data=data)
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.violinplot(x='Species',y='PetalLengthCm',data=data)

plt.subplot(2,2,2)

sns.violinplot(x='Species',y='PetalWidthCm',data=data)

plt.subplot(2,2,3)

sns.violinplot(x='Species',y='SepalLengthCm',data=data)

plt.subplot(2,2,4)

sns.violinplot(x='Species',y='SepalWidthCm',data=data)
sns.stripplot(x="Species", y="PetalLengthCm", data=data, jitter=True, edgecolor="gray")
# Distribution density plot KDE (kernel density estimate)

sns.FacetGrid(data, hue="Species", size=6).map(sns.kdeplot, "PetalLengthCm").add_legend()
# Plotting bivariate relations between each pair of features (4 features x4 so 16 graphs) with hue = "Species"

sns.pairplot(data, hue="Species", size=4)
# Finding out the correlation between the features

corr = data.corr()

corr.shape
# Plotting the heatmap of correlation between features

plt.figure()

sns.heatmap(corr, cbar=True, square= True, fmt='.2f', annot=True, annot_kws={'size':15}, cmap = 'Greens')
# Spliting target variable and independent variables

X = data.drop(['Species'], axis = 1)

y = data['Species']
# Splitting the data into training set and testset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)

print("Size of training set:", X_train.shape)

print("Size of training set:", X_test.shape)
# Logistic Regression



# Import library for LogisticRegression

from sklearn.linear_model import LogisticRegression



# Create a Logistic regression classifier

logreg = LogisticRegression()



# Train the model using the training sets 

logreg.fit(X_train, y_train)
# Prediction on test data

y_pred = logreg.predict(X_test)
# Calculating the accuracy

acc_logreg = round( metrics.accuracy_score(y_test, y_pred) * 100, 2 )

print( 'Accuracy of Logistic Regression model : ', acc_logreg )
# Gaussian Naive Bayes



# Import library of Gaussian Naive Bayes model

from sklearn.naive_bayes import GaussianNB



# Create a Gaussian Classifier

model = GaussianNB()



# Train the model using the training sets 

model.fit(X_train,y_train)
# Prediction on test set

y_pred = model.predict(X_test)
# Calculating the accuracy

acc_nb = round( metrics.accuracy_score(y_test, y_pred) * 100, 2 )

print( 'Accuracy of Gaussian Naive Bayes model : ', acc_nb )
# Decision Tree Classifier



# Import Decision tree classifier

from sklearn.tree import DecisionTreeClassifier



# Create a Decision tree classifier model

clf = DecisionTreeClassifier(criterion = "gini" , min_samples_split = 100, min_samples_leaf = 10, max_depth = 50)



# Train the model using the training sets 

clf.fit(X_train, y_train)
# Prediction on test set

y_pred = clf.predict(X_test)
# Calculating the accuracy

acc_dt = round( metrics.accuracy_score(y_test, y_pred) * 100, 2 )

print( 'Accuracy of Decision Tree model : ', acc_dt )
# Random Forest Classifier



# Import library of RandomForestClassifier model

from sklearn.ensemble import RandomForestClassifier



# Create a Random Forest Classifier

rf = RandomForestClassifier()



# Train the model using the training sets 

rf.fit(X_train,y_train)
# Prediction on test data

y_pred = rf.predict(X_test)
# Calculating the accuracy

acc_rf = round( metrics.accuracy_score(y_test, y_pred) * 100 , 2 )

print( 'Accuracy of Random Forest model : ', acc_rf )
# SVM Classifier



# Creating scaled set to be used in model to improve the results

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# Import Library of Support Vector Machine model

from sklearn import svm



# Create a Support Vector Classifier

svc = svm.SVC()



# Train the model using the training sets 

svc.fit(X_train,y_train)
# Prediction on test data

y_pred = svc.predict(X_test)
# Calculating the accuracy

acc_svm = round( metrics.accuracy_score(y_test, y_pred) * 100, 2 )

print( 'Accuracy of SVM model : ', acc_svm )
# Random Forest Classifier



# Import library of RandomForestClassifier model

from sklearn.neighbors import KNeighborsClassifier



# Create a Random Forest Classifier

knn = KNeighborsClassifier()



# Train the model using the training sets 

knn.fit(X_train,y_train)
# Prediction on test data

y_pred = knn.predict(X_test)
# Calculating the accuracy

acc_knn = round( metrics.accuracy_score(y_test, y_pred) * 100, 2 )

print( 'Accuracy of KNN model : ', acc_knn )
models = pd.DataFrame({

    'Model': ['Logistic Regression', 'Naive Bayes', 'Decision Tree', 'Random Forest', 'Support Vector Machines', 

              'K - Nearest Neighbors'],

    'Score': [acc_logreg, acc_nb, acc_dt, acc_rf, acc_svm, acc_knn]})

models.sort_values(by='Score', ascending=False)