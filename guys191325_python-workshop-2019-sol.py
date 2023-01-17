import random as r

import math as m

inside = 0

total = 100000

# Iterate for the number of darts.

for i in range(0, total):

    # Generate random x, y in [0, 1].

    x2 = r.random()**2

    y2 = r.random()**2

    # Increment if inside unit circle.

    if m.sqrt(x2 + y2) < 1.0:

        inside += 1

mystery = (float(inside) / total) * 4

print(mystery)
text_file = open("Output.txt", "w")

text_file.write("Number: %s" % mystery)

text_file.close()



#Lets download the file we have just created.

#add a new markdown cell with the following content:

#<a href="Output.txt"> Download File </a>

#Run the cell.


##Don't forget to turn the internet on. Right menu->settings->Internet.

!pip install git+https://github.com/goolig/dsClass.git
from dsClass.path_helper import get_file_path

%matplotlib inline

import pandas as pd

import matplotlib.pyplot as plt



#very important, let talk about it

f_name = get_file_path('fruit_data_with_colors.csv')



fruits = pd.read_csv(f_name)

fruits.head()

#How many samples and features?

fruits.shape
#Which fruits?

fruits['fruit_name'].unique()
fruits.describe()
#Is the data balanced?

fruits.groupby('fruit_name').size()

import seaborn as sns

#We can also visualize it:

sns.countplot(fruits['fruit_name'],label="Count")

plt.show()
fruits.drop('fruit_label', axis=1).plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False, figsize=(9,9), 

                                        title='Box Plot for each input variable')

plt.savefig('fruits_box')

plt.show()

#Discuss box and whiskers plots
import pylab as pl

fruits.drop('fruit_label' ,axis=1).hist(bins=30, figsize=(9,9))

pl.suptitle("Histogram for each numeric input variable")

plt.savefig('fruits_hist')

plt.show()



import matplotlib.pyplot as plt

corr = fruits.corr()

corr.style.background_gradient()

from sklearn.model_selection import train_test_split

feature_names = ['mass', 'width', 'height', 'color_score']



X = fruits[feature_names]

X['volume'] = X['height']*X['width']

y = fruits['fruit_name']



X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train,y_train) #TODO: complete the fit. Try Shift + Tab. Try pressing it twice, 3 and 4 times.

print('Accuracy of Logistic regression classifier on training set: {:.2f}'

     .format(logreg.score(X_train, y_train)))

print('Accuracy of Logistic regression classifier on test set: {:.2f}'

     .format(logreg.score(X_test, y_test)))

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier().fit(X_train, y_train)

print('Accuracy of Decision Tree classifier on training set: {:.2f}'

     .format(clf.score(X_train, y_train)))

print('Accuracy of Decision Tree classifier on test set: {:.2f}'

     .format(clf.score(X_test, y_test)))

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

print('Accuracy of K-NN classifier on training set: {:.2f}'

     .format(knn.score(X_train, y_train)))

print('Accuracy of K-NN classifier on test set: {:.2f}'

     .format(knn.score(X_test, y_test)))

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train, y_train)

print('Accuracy of GNB classifier on training set: {:.2f}'

     .format(gnb.score(X_train, y_train)))

print('Accuracy of GNB classifier on test set: {:.2f}'

     .format(gnb.score(X_test, y_test)))

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()

clf.fit(X_train, y_train)  

print('Accuracy of rf classifier on training set: {:.2f}'

     .format(clf.score(X_train, y_train)))

print('Accuracy of rf classifier on test set: {:.2f}'

     .format(clf.score(X_test, y_test)))
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

pred = knn.predict(X_test)

print(confusion_matrix(y_test, pred))

print(classification_report(y_test, pred))

ans = pd.DataFrame(X_test)

ans['prediction'] = list(knn.predict(X_test))

ans