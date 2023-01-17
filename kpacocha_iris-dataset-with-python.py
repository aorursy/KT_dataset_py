#Importing the packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib

import seaborn as sns

from sklearn.metrics import confusion_matrix

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import mean_absolute_error
dataset = pd.read_csv('../input/irisdataset/iris.csv')
dataset.head() #first five rows
dataset.shape #dimension of dataset
dataset.dtypes #type of every variable
dataset['variety']=dataset['variety'].astype('category') 
dataset.isnull().sum() #how many misssing values we have
dataset.info()
dataset.variety.value_counts() #frequency by category of dependent variable
dataset.describe() #basic statistics
#Boxplots for each independent variable

dataset.plot(kind='box')
#Box plots by variety category

dataset.boxplot(by="variety",figsize=(10,10))
#Histograms for every numerical variable:

dataset.hist(figsize=(10,5))

plt.show()
#Plots by category

sns.pairplot(dataset,hue="variety")
#Preparing data to the split

X = dataset.iloc[:,:4]

y = dataset.variety



#Splitting the dataset into the train and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)

print("X_train:",X_train.shape,

      '\n',"X_test:",X_test.shape,

      '\n',"y_train:",y_train.shape,

      '\n',"y_test:",y_test.shape)
#Decision Tree

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

print('Accuracy is:',accuracy_score(y_pred,y_test))

pd.concat([X_test, y_test, pd.Series(y_pred, name='predicted', index=X_test.index)], 

          ignore_index=False, axis=1)

pd.crosstab(y_test, y_pred, rownames=['variety'], colnames=['predicted'])
#Random Forest

model = RandomForestClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

print('Accuracy is:',accuracy_score(y_pred,y_test))

pd.concat([X_test, y_test, pd.Series(y_pred, name='predicted', index=X_test.index)], 

          ignore_index=False, axis=1)
pd.crosstab(y_test, y_pred, rownames=['variety'], colnames=['predicted'])
#K-Nearest Neighbours

model2 = KNeighborsClassifier(n_neighbors=2)

model2.fit(X_train, y_train)

y_pred = model2.predict(X_test)

print('Accuracy is:',accuracy_score(y_pred,y_test))

pd.concat([X_test, y_test, pd.Series(y_pred, name='predicted', index=X_test.index)], 

          ignore_index=False, axis=1)
pd.crosstab(y_test, y_pred, rownames=['variety'], colnames=['predicted'])
#How many neighbors we need?

scores = []

for n in range(1,15):

    model = KNeighborsClassifier(n_neighbors=n)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    scores.append(accuracy_score(y_pred,y_test))

    

plt.plot(range(1,15), scores)

plt.xlabel('Number of neighbors')

plt.ylabel('Accuracy')

plt.show()
#Let's try one more time with 8 neighbors

model3 = KNeighborsClassifier(n_neighbors=8)

model3.fit(X_train, y_train)

y_pred = model3.predict(X_test)

print('Accuracy is:',accuracy_score(y_pred,y_test))

pd.concat([X_test, y_test, pd.Series(y_pred, name='predicted', index=X_test.index)], 

          ignore_index=False, axis=1)
pd.crosstab(y_test, y_pred, rownames=['variety'], colnames=['predicted'])
#Support Vector Machine

from sklearn.svm import SVC

model5=SVC()

model5.fit(X_train, y_train)

y_pred = model5.predict(X_test)

print('Accuracy is:',accuracy_score(y_pred,y_test))

pd.concat([X_test, y_test, pd.Series(y_pred, name='predicted', index=X_test.index)], 

          ignore_index=False, axis=1)
pd.crosstab(y_test, y_pred, rownames=['variety'], colnames=['predicted'])
#Naive Bayes

from sklearn.naive_bayes import GaussianNB

model6 = GaussianNB()

model6.fit(X_train, y_train)

y_pred = model6.predict(X_test)

print('Accuracy is:',accuracy_score(y_pred,y_test))

pd.concat([X_test, y_test, pd.Series(y_pred, name='predicted', index=X_test.index)], 

          ignore_index=False, axis=1)

pd.crosstab(y_test, y_pred, rownames=['variety'], colnames=['predicted'])