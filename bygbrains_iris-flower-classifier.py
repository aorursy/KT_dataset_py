import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))


dataset = pd.read_csv("../input/iris/Iris.csv") 

dataset.head(5)
dataset['Species'].value_counts()
X=dataset.iloc[:,:5]

X.head(2)
Y=dataset['Species']

Y.head(2)
import matplotlib.pyplot as plt 

dataset['Species'].value_counts().plot.pie(explode=[0.1,0.1,0.1],autopct='%1.1f%%',shadow=True,figsize=(10,8))

plt.show()



from  sklearn import  datasets

iris=datasets.load_iris()

x=iris.data

y=iris.target

iris
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=.5)

x_train.shape,x_test.shape,y_train.shape,y_test.shape
from sklearn import tree

classifier=tree.DecisionTreeClassifier()

classifier
# Finding  the accuracy score



from sklearn.metrics import accuracy_score

classifier.fit(x_train, y_train)

train_predictions = classifier.predict(x_test)

acc = accuracy_score(y_test, train_predictions)

acc
from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC, NuSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



classifiers = [

    KNeighborsClassifier(3),

    SVC(kernel="rbf", C=0.025, probability=True),

    NuSVC(probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis()]



# Logging for Visual Comparison

log_cols=["Classifier", "Accuracy", "Log Loss"]

log = pd.DataFrame(columns=log_cols)



for clf in classifiers:

    clf.fit(x_train, y_train)

    name = clf.__class__.__name__

    train_predictions = clf.predict(x_test)

    acc = accuracy_score(y_test, train_predictions)

        

    train_predictions = clf.predict_proba(x_test)

    ll = log_loss(y_test, train_predictions)

       

    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)

    log = log.append(log_entry)    

import seaborn as sns

sns.set_style('whitegrid')

sns.set_color_codes("muted")

sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")



plt.xlabel('Accuracy %')

plt.title('Classifier Accuracy')

plt.show()



sns.set_color_codes("muted")

sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")



plt.xlabel('Log Loss')

plt.title('Classifier Log Loss')

plt.show()
# Testing the output

dataset.iloc[:1,:]
testData=dataset.iloc[:1,:5]

testData
# Prdicting the output

train_prediction = classifier.predict(testData)

print ("The flower is "+ train_prediction[0])