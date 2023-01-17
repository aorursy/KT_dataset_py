# Importing the libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# Importing the dataset

dataset = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv", delimiter=",")
# Viewing the top 5 rows in the dataset

dataset.head()
# To find out if there are any missing values 

dataset.isna().any()
# Building HeatMap to identify which features are relevant for determining quality of red wine

plt.figure(figsize=(20,10))

g = sns.heatmap(data      = dataset.corr(),  

            square    = True, 

            cbar_kws  = {'shrink': .3}, 

            annot     = True, 

            annot_kws = {'fontsize': 12},

           )

g.set(ylim=(0,12))

g.set(xlim=(0,12))

plt.show()
# Plotting the distribution of dependent variable (quality)

sns.distplot(dataset['quality'])

plt.show()
# Plotting the distribution of independent variable (alcohol)

sns.distplot(dataset['alcohol'])

plt.show()
# Plotting the distribution of independent variable (sulphates)

sns.distplot(dataset['sulphates'])

plt.show()
# Plotting the distribution of independent variable (citric acid)

sns.distplot(dataset['citric acid'])

plt.show()
# some more visualisation

# Alcohol vs Quality

sns.barplot("quality", y="alcohol", data=dataset, saturation=.5, palette = 'inferno')

plt.show()
# Citric Acid vs Quality

sns.barplot("quality", y="citric acid", data=dataset,saturation=.5, palette = 'inferno')

plt.show()
# Suplhates vs Quality

sns.barplot("quality", y="sulphates", data=dataset, saturation=.5, palette = 'inferno')

plt.show()
Features = ['citric acid','sulphates','alcohol']
# Creating features(x) and dependent variable(y)

x = dataset.iloc[:,[2,9,10]].values

y = dataset.iloc[:,-1].values
print(x)
print(y)
# Splitting the dataset into training set and test set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print(x_train[0])
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train[:,[2]] = sc.fit_transform(x_train[:,[2]])

x_test[:,[2]] = sc.transform(x_test[:,[2]])
print(x_train[0])
# Running for loop to determine the number of neighbors for best accuracy

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, accuracy_score

list1 = []

for neighbors in range(3,20,1):

    classifier = KNeighborsClassifier(n_neighbors=neighbors, metric='minkowski')

    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)

    list1.append(accuracy_score(y_test,y_pred))

plt.plot(list(range(3,20,1)), list1)

plt.show()
# Training the K Nearest Neighbor Classifier on the Training set with n_neighbors = 10

classifier = KNeighborsClassifier(n_neighbors=10, metric='minkowski')

classifier.fit(x_train, y_train)
# Predicting the Test Set results

y_pred = classifier.predict(x_test)

print(y_pred)
# Printing the predicted test set results and actual test results

np.set_printoptions()

print(np.concatenate( (y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)) 
# Making the confusion matrix and accuracy

mylist = []

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

ac = accuracy_score(y_test, y_pred)

mylist.append(ac)

print(cm)

print(ac)
# Distplot between Quality of Red Wine of test set results and predicted results

plt.rcParams['figure.figsize']=8,4 

sns.set_style("darkgrid")

sns.distplot(y_test, color = "blue", kde = False, label = "Test set results", hist_kws = {"align": "right"})

sns.distplot(y_pred, color = "magenta", kde = False, label = "Actual results", hist_kws = {"align": "left"})

plt.xlabel("Quality")

plt.ylabel("Number of People")

plt.legend()

plt.show()
# Training the Naive Bayes Classifier on the Training set

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(x_train, y_train)
# Predicting the Test Set result

y_pred = classifier.predict(x_test)

print(y_pred)
np.set_printoptions()

print(np.concatenate( (y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)) 
# Making the confusion matrix and accuracy

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

ac = accuracy_score(y_test, y_pred)

print(cm)

print(ac)

mylist.append(ac)
# Distplot between Quality of Red Wine of test set results and predicted results

plt.rcParams['figure.figsize']=8,4 

sns.set_style("darkgrid")

sns.distplot(y_test, color = "blue", kde = False, label = "Test set results",  hist_kws = {"align": "left"})

sns.distplot(y_pred, color = "magenta", kde = False, label = "Actual results",  hist_kws = {"align": "right"})

plt.xlabel("Quality")

plt.ylabel("Number of People")

plt.legend()

plt.show()
# Training the Support Vector Classifier on the Training set

from sklearn.svm import SVC

classifier = SVC(random_state=0, kernel = 'rbf')

classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

print(y_pred)
np.set_printoptions()

print(np.concatenate( (y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)) 
# Making the confusion matrix and accuracy

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

ac = accuracy_score(y_test, y_pred)

print(cm)

print(ac)

mylist.append(ac)
# Distplot between Quality of Red Wine of test set results and predicted results

plt.rcParams['figure.figsize']=8,4 

sns.set_style("darkgrid")

sns.distplot(y_test, color = "blue", kde = False, label = "Test set results",  hist_kws = {"align": "left"})

sns.distplot(y_pred, color = "magenta", kde = False, label = "Actual results", hist_kws = {"align": "right"})

plt.xlabel("Quality")

plt.ylabel("Number of People")

plt.legend()

plt.show()
# Determining the max_leaf_nodes using for loop for leaves 2-25

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix, accuracy_score

list1 = []

for leaves in range(2,25):

    classifier = DecisionTreeClassifier(max_leaf_nodes = leaves, random_state=0, criterion='entropy')

    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)

    list1.append(accuracy_score(y_test,y_pred))

#print(mylist)

plt.plot(list(range(2,25)), list1)

plt.show()
# we can see the optimum number of max_leaf_nodes = 5
# Training the Decision Tree Classifier on the Training set

classifier = DecisionTreeClassifier(max_leaf_nodes = 5, random_state=0, criterion='entropy')

classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

print(y_pred)
np.set_printoptions()

print(np.concatenate( (y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)) 
# Making the confusion matrix and accuracy

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

ac = accuracy_score(y_test, y_pred)

print(cm)

print(ac)

mylist.append(ac)
# Distplot between Quality of Red Wine of test set results and predicted results

plt.rcParams['figure.figsize']=8,4 

sns.set_style("darkgrid")

sns.distplot(y_test, color = "blue", kde = False, label = "Test set results", hist_kws = {"align": "left"})

sns.distplot(y_pred, color = "magenta", kde = False, label = "Actual results", hist_kws = {"align": "right"})

plt.xlabel("Quality")

plt.ylabel("Number of People")

plt.legend()

plt.show()
# Determining the optimum number of n_estimators

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, accuracy_score

list1 = []

for estimators in range(100,200):

    classifier = RandomForestClassifier(n_estimators = estimators, random_state=0, criterion='entropy')

    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)

    list1.append(accuracy_score(y_test,y_pred))

#print(mylist)

plt.plot(list(range(100,200)), list1)

plt.show()
# Training the Random Forest Classifier on the Training set

classifier = RandomForestClassifier(n_estimators=180, random_state=0, criterion='entropy')

classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

print(y_pred)
np.set_printoptions()

print(np.concatenate( (y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)) 
# Making the confusion matrix and accuracy

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

ac = accuracy_score(y_test, y_pred)

print(cm)

print(ac)

mylist.append(ac)
# Distplot between Quality of Red Wine of test set results and predicted results

plt.rcParams['figure.figsize']=8,4 

sns.set_style("darkgrid")

sns.distplot(y_test, color = "blue", kde = False, label = "Test set results", hist_kws = {"align": "left"} )

sns.distplot(y_pred, color = "magenta", kde = False, label = "Actual results", hist_kws = {"align": "right"})

plt.xlabel("Quality")

plt.ylabel("Number of People")

plt.legend()

plt.show()
# Training the XGBoost Classifier on the Training set

from xgboost import XGBClassifier

classifier = XGBClassifier()

classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
print(y_pred)
np.set_printoptions()

print(np.concatenate( (y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)) 
# Making the confusion matrix and accuracy

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

ac = accuracy_score(y_test, y_pred)

print(cm)

print(ac)

mylist.append(ac)
# Accuracy score of different models

mylist
mylist2 = ["KNearestNeighbours","NaiveBayes","SupportVector","DecisionTree","RandomForest","XGBoost"]
# Plotting the accuracy score for different models

plt.rcParams['figure.figsize']=10,6 

sns.set_style("darkgrid")

ax = sns.barplot(x=mylist2, y=mylist, palette = "rocket", saturation =1.5)

plt.xlabel("Classifier Models", fontsize = 20 )

plt.ylabel("Accuracy", fontsize = 20)

plt.title("Accuracy of different Classifier Models", fontsize = 20)

plt.xticks(fontsize = 11, horizontalalignment = 'center', rotation = 8)

plt.yticks(fontsize = 13)

for p in ax.patches:

    width, height = p.get_width(), p.get_height()

    x, y = p.get_xy() 

    ax.annotate(f'{height:.2%}', (x + width/2, y + height*1.02), ha='center', fontsize = 'x-large')

plt.show()