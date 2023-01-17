#importing libraries

import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
data = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
print("The feature attributes present in the data set are \n{} \nTotal of which are {} features ".format(data.columns, len(data.columns)))
data.info()
data.head(3)
X = data.iloc[:, :-1]

Y = data.iloc[:, -1:]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=42)
from sklearn import tree

clf = tree.DecisionTreeClassifier()

clf = clf.fit(x_train, y_train)
clf.score(x_test, y_test)
plt.barh(data=X, y = data['quality'], width=data['fixed acidity'],

        color='black', hatch='+')

plt.xlabel('Fixed Acidity')

plt.ylabel('Quality')

plt.title("Quality vs Fixed Acidity")
fig = plt.figure(figsize=(4,6))

plt.bar(height=data['volatile acidity'], x=data['quality'],

       color='green', hatch='/')

plt.xlabel('Quality')

plt.ylabel('Volatile Acidity')

plt.title('Volatile Acidity vs Quality')
fig = plt.figure(figsize=(3,4))

plt.bar(height=data['citric acid'], x=data['quality'],

       color='orange', hatch='.')

plt.xlabel('Quality')

plt.ylabel('Citric Acid')

plt.title('Citric Acid vs Quality')
fig = plt.figure(figsize=(3,4))

plt.bar(height=data['residual sugar'], x=data['quality'],

       color='blue', hatch='.')

plt.xlabel('Quality')

plt.ylabel('Residual Sugar')

plt.title('Residual Sugar vs Quality')
fig = plt.figure(figsize=(3,4))

plt.bar(height=data['chlorides'], x=data['quality'],

       color='#2980b9', hatch='.')

plt.xlabel('Quality')

plt.ylabel('Chlorides')

plt.title('Chlorides vs Quality')
fig = plt.figure(figsize=(3,4))

plt.bar(height=data['free sulfur dioxide'], x=data['quality'],

       color='#16a085', hatch='.')

plt.xlabel('Quality')

plt.ylabel('Free Sulfur Dioxide')

plt.title('Free Sulfur Dioxide vs Quality')
fig = plt.figure(figsize=(3,4))

plt.bar(height=data['total sulfur dioxide'], x=data['quality'],

       color='#34495e', hatch='.')

plt.xlabel('Quality')

plt.ylabel('Totla Sulfur Dioxide')

plt.title('Total Sulfur Dioxide vs Quality')
fig = plt.figure(figsize=(3,4))

plt.bar(height=data['density'], x=data['quality'],

       color='#c0392b', hatch='.')

plt.xlabel('Quality')

plt.ylabel('Density')

plt.title('Density vs Quality')
fig = plt.figure(figsize=(3,4))

plt.bar(height=data['pH'], x=data['quality'],

       color='#8e44ad', hatch='.')

plt.xlabel('Quality')

plt.ylabel('pH')

plt.title('pH vs Quality')
fig = plt.figure(figsize=(3,4))

plt.bar(height=data['sulphates'], x=data['quality'],

       color='#bdc3c7', hatch='.')

plt.xlabel('Quality')

plt.ylabel('Sulphates')

plt.title('Sulphates vs Quality')
fig = plt.figure(figsize=(3,4))

plt.bar(height=data['alcohol'], x=data['quality'],

       color='#7f8c8d', hatch='.')

plt.xlabel('Quality')

plt.ylabel('Alcohol')

plt.title('Alcohol vs Quality')
a = pd.cut(data['quality'], bins= [2,6.5,8], labels=['bad', 'good'])

good = 0

bad = 0

for i in a:

    if(i=='good'):

        good=good+1

    else:

        bad=bad+1

        

print("Values with 'Good' labels are {}  \nAnd with 'Bad' labels are {}"

      .format(good, bad))



data['quality'] = a
#Encoding caterogical variables

from sklearn.preprocessing import LabelEncoder, StandardScaler



label_quality = LabelEncoder()

data['quality'] = label_quality.fit_transform(data['quality'])
data['quality'].value_counts()
Y = data['quality']

X = data

X = X.drop('quality', axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15,

                                                    random_state=45)

sc=StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.fit_transform(x_test)
from sklearn import tree

clf = tree.DecisionTreeClassifier()

clf = clf.fit(x_train, y_train)
from sklearn.metrics import classification_report

print(classification_report(y_test, clf.predict(x_test)))
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10)

clf = clf.fit(x_train, y_train)
print(classification_report(y_test, clf.predict(x_test)))