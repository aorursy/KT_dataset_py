import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))
# load the iris dataset

iris_data = pd.read_csv('../input/Iris.csv')
# my personal reusable function for detecting missing data

def missing_value_describe(data):

    # check missing values in training data

    missing_value_stats = (data.isnull().sum() / len(data)*100)

    missing_value_col_count = sum(missing_value_stats > 0)

    missing_value_stats = missing_value_stats.sort_values(ascending=False)[:missing_value_col_count]

    print("Number of columns with missing values:", missing_value_col_count)

    if missing_value_col_count != 0:

        # print out column names with missing value percentage

        print("\nMissing percentage (desceding):")

        print(missing_value_stats)

    else:

        print("No misisng data!!!")

missing_value_describe(iris_data)
# take a peek

iris_data.head()
iris_data = iris_data.drop(['Id'], axis=1)

iris_data.columns
# dimension

print("the dimension:", iris_data.shape)
print(iris_data.describe())
# class distribution

print(iris_data.groupby('Species').size())
# import ploting tool

import matplotlib.pyplot as plt
# iris flower dataset class distribution

nameplot = iris_data['Species'].value_counts().plot.bar(title='Flower class distribution')

nameplot.set_xlabel('class',size=20)

nameplot.set_ylabel('count',size=20)
# box and whisker plots

iris_data.plot(kind='box', subplots=True, layout=(2,2), 

               sharex=False, sharey=False, title="Box and Whisker plot for each attribute")

plt.show()
# plot histogram

iris_data.hist()

plt.show()
import seaborn as sns

sns.set(style="ticks")

sns.pairplot(iris_data, hue="Species")
from sklearn.model_selection import train_test_split
# we will split data to 80% training data and 20% testing data with random seed of 10

X = iris_data.drop(['Species'], axis=1)

Y = iris_data['Species']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=7)
print("X_train.shape:", X_train.shape)

print("X_test.shape:", X_test.shape)

print("Y_train.shape:", X_train.shape)

print("Y_test.shape:", Y_test.shape)
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
# models

models = []



# linear models

models.append(('LR', LogisticRegression(solver='liblinear', multi_class="auto")))

models.append(('LDA', LinearDiscriminantAnalysis()))



# nonlinear models

models.append(('CART', DecisionTreeClassifier()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('GNB', GaussianNB()))

models.append(('SVC', SVC(gamma="auto")))



# evaluate each model in turn

print("Model Accuracy:")

names = []

accuracy = []

for name, model in models:

    # 10 fold cross validation to evalue model

    kfold = KFold(n_splits=10, random_state=7)

    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')

    

    # display the cross validation results of the current model

    names.append(name)

    accuracy.append(cv_results)

    msg = "%s: accuracy=%f std=(%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
ax = sns.boxplot(x=names, y=accuracy)

ax.set_title('Model Accuracy Comparison')
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
# models

models = []

models.append(('KNN', KNeighborsClassifier()))

models.append(('GNB', GaussianNB()))

models.append(('SVC', SVC(gamma="auto")))
# reusable function to test our model

def test_model(model):

    model.fit(X_train, Y_train) # train the whole training set

    predictions = model.predict(X_test) # predict on test set

    

    # output model testing results

    print("Accuracy:", accuracy_score(Y_test, predictions))

    print("Confusion Matrix:")

    print(confusion_matrix(Y_test, predictions))

    print("Classification Report:")

    print(classification_report(Y_test, predictions))
# predict values with our test set

for name, model in models:

    print("----------------")

    print("Testing", name)

    test_model(model)