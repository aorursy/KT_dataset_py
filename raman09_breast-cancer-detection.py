# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# # load the breast-cancer-wisconsin-data

import pandas as pd

df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')

df.head()
# M - malignant menas harmful

# B - benign means not harmful



df['diagnosis'].unique()
# count the number of rows and columns in the dataset

df.shape
# count the number of empty (Nan, NAN, na) values in each columns

df.isna().sum()
# drop the column with all missing values

df = df.dropna(axis =1)

df.shape
# Get the count of diagnosis categorical data which have malignant(M) and benign(B)

df['diagnosis'].value_counts()
# visualize the count

sns.countplot(df['diagnosis'], label='count')
# look at the data types

df.dtypes
# encode the categorical data values

from sklearn.preprocessing import LabelEncoder

labelencoder_Y = LabelEncoder()

df.iloc[:,1] = labelencoder_Y.fit_transform(df.iloc[:,1].values)
df.iloc[:,1].value_counts()
# create a pair plot

sns.pairplot(df.iloc[:,1:6], hue='diagnosis')
# print the first five rows of data

df.head()
#  get the correlation of the columns

df.iloc[:, 1:12].corr()
# correlation with the %

plt.figure(figsize=(10,10))

sns.heatmap(df.iloc[:, 1:12].corr(), annot=True, fmt='.0%')
# split the data set into independent (X) and dependent  (Y) data sets

X = df.iloc[:, 2:32].values

Y = df.iloc[:,1].values
# split the data set into 75% training and 25% testing

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=0)



# check the shape of all the train test split

print("X_train, X_test, Y_train, Y_test", X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
# scale the data (Feature scaling)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)
X_train[10:]
# create a function for models



def models(X_train, Y_train):

    

    # Logistic Regression

    from sklearn.linear_model import LogisticRegression

    log = LogisticRegression(random_state=0)

    log.fit(X_train, Y_train)

    

    # Decision Tree

    from sklearn.tree import DecisionTreeClassifier

    tree = DecisionTreeClassifier(criterion='entropy', random_state=0)

    tree.fit(X_train, Y_train)

    

    # Random Forest Classifier

    from sklearn.ensemble import RandomForestClassifier

    forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state=0)

    forest.fit(X_train, Y_train)

    

    # print the model accuracy on the training data 

    print('[0]Logistic Regression Training Accuracy:', log.score(X_train, Y_train))

    print('[5]Decision Tree Classifier Training Accuracy:', tree.score(X_train, Y_train))

    print('[6]Random Forest Classifier Training Accuracy:', forest.score(X_train, Y_train))

  

    return log, tree, forest
# Train the models

model  = models(X_train, Y_train)
# test the mode accuracy on the test data on the confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, model[0].predict(X_test))

print(cm)
# get the testing accuracy of all the model



for i in range(len(model)):

    print("Model ", i)

    cm = confusion_matrix(Y_test, model[i].predict(X_test))

    

    TP = cm[0][0]

    TN = cm[1][1]

    FN = cm[1][0]

    FP = cm[0][1]

    print(cm)

    print("Testing Accuracy = ", (TP+TN)/(TP+TN+FP+FN))

    print()
# another way to get metrics of the models

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score



for i in range(len(model)):

    print("Model ", i)

    print(classification_report(Y_test, model[i].predict(X_test)))

    print(accuracy_score(Y_test, model[i].predict(X_test)))

    print()
# Print the prediction of the Random Forest Classifier model

pred = model[2].predict(X_test)

print(pred)

print()

print(Y_test)