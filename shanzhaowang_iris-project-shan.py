# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib.pyplot as plt 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
IrisData = pd.read_csv("../input/Iris.csv")
IrisData.info() # To check if there is any NaN column 
IrisData.head()
Setosa = IrisData.loc[IrisData['Species'] == 'Iris-setosa'] #Select all Iris-setosa 

Versicolor = IrisData.loc[IrisData['Species'] == 'Iris-versicolor']

Virginica = IrisData.loc[IrisData['Species'] == 'Iris-virginica']
#Use pandas to directly draw a graph 

Scatter = Setosa.plot(x= 'SepalLengthCm', y= 'SepalWidthCm', kind='scatter', color = 'orange', title = 'Sepal Information', label = 'Setosa')

Versicolor.plot(x= 'SepalLengthCm', y= 'SepalWidthCm', kind='scatter', color = 'green', ax = Scatter, label = 'Versicolor')

Virginica.plot(x= 'SepalLengthCm', y= 'SepalWidthCm', kind='scatter', color = 'blue', ax = Scatter, label = 'Virginica')
#Do the same for petal 

Scatter1 = Setosa.plot(x= 'PetalLengthCm', y= 'PetalWidthCm', kind='scatter', color = 'orange', title = 'Petal Information', label = 'Setosa')

Versicolor.plot(x= 'PetalLengthCm', y= 'PetalWidthCm', kind='scatter', color = 'green', ax = Scatter1, label = 'Versicolor')

Virginica.plot(x= 'PetalLengthCm', y= 'PetalWidthCm', kind='scatter', color = 'blue', ax = Scatter1, label = 'Virginica')
#I am wondering is the area of sepal or petal gonna provide me with any useful information?

#First, create the column

Setosa['Sepal_area'] = Setosa['SepalLengthCm'] * Setosa['SepalWidthCm']

Versicolor['Sepal_area'] = Versicolor['SepalLengthCm'] * Versicolor['SepalWidthCm']

Virginica['Sepal_area'] = Virginica['SepalLengthCm'] * Virginica['SepalWidthCm']
ScatterSepal = Setosa.plot(x= 'Id', y= 'Sepal_area', kind='scatter', color = 'orange', title = 'Sepal Area', label = 'Setosa')

Versicolor.plot(x= 'Id', y= 'Sepal_area', kind='scatter', color = 'green', ax = ScatterSepal, label = 'Versicolor')

Virginica.plot(x= 'Id', y= 'Sepal_area', kind='scatter', color = 'blue', ax = ScatterSepal, label = 'Virginica')
#The Sepal area is not very useful, and I will check the Petal area instead 

Setosa['Petal_area'] = Setosa['PetalLengthCm'] * Setosa['PetalWidthCm']

Versicolor['Petal_area'] = Versicolor['PetalLengthCm'] * Versicolor['PetalWidthCm']

Virginica['Petal_area'] = Virginica['PetalLengthCm'] * Virginica['PetalWidthCm']

ScatterPetal = Setosa.plot(x= 'Id', y= 'Petal_area', kind='scatter', color = 'orange', title = 'Petal Area', label = 'Setosa')

Versicolor.plot(x= 'Id', y= 'Petal_area', kind='scatter', color = 'green', ax = ScatterPetal, label = 'Versicolor')

Virginica.plot(x= 'Id', y= 'Petal_area', kind='scatter', color = 'blue', ax = ScatterPetal, label = 'Virginica')
#It turns out the Pedal area is a useful one, and my next idea is to amplify its effect by adding 

# a "Petal_area" term. 

Setosa['Petal_area_square'] = Setosa['Petal_area']* Setosa['Petal_area']

Versicolor['Petal_area_square'] = Versicolor['Petal_area']*Versicolor['Petal_area']

Virginica['Petal_area_square'] = Virginica['Petal_area']*Virginica['Petal_area']

ScatterPetal = Setosa.plot(x= 'Id', y= 'Petal_area_square', kind='scatter', color = 'orange', title = 'Square of Petal Area', label = 'Setosa')

Versicolor.plot(x= 'Id', y= 'Petal_area_square', kind='scatter', color = 'green', ax = ScatterPetal, label = 'Versicolor')

Virginica.plot(x= 'Id', y= 'Petal_area_square', kind='scatter', color = 'blue', ax = ScatterPetal, label = 'Virginica')
IrisData['Petal_area'] = IrisData['PetalLengthCm'] * IrisData['PetalWidthCm']

IrisData['Petal_area_square'] = IrisData['Petal_area'] * IrisData['Petal_area']
plt.subplot(1,2,1)

sns.violinplot(x= 'Species', y='Petal_area', data=IrisData)

plt.subplot(1,2,2)

sns.violinplot(x= 'Species', y='Petal_area_square',color = 'm', data=IrisData)
#The last step of analysis is to see correlation factors of each variable. 

sns.heatmap(IrisData.corr(), annot=True, fmt=".2f")
# Start the ML part and we will split all data into training and testing parts.

from sklearn.cross_validation import train_test_split

train, test = train_test_split(IrisData, test_size = 0.3)
from sklearn import metrics #for checking the model accuracy
# Generate training and test datasets

Train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm', 'Petal_area','Petal_area_square']]

Train_Y = train[['Species']]

Test_X = test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm', 'Petal_area','Petal_area_square']]

Test_Y = test[['Species']]
Train_X.describe() #Take a look at the training dataset 
# Use LogisticRegression as the first ML method 

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(Train_X,Train_Y)

Prediction = logreg.predict(Test_X)

print('The accuracy of the Logistic Regression is',metrics.accuracy_score(Prediction,Test_Y))

# The next method is RandomForest

from sklearn.ensemble import RandomForestClassifier 

RandomForest = RandomForestClassifier()

RandomForest.fit(Train_X,Train_Y)

Prediction1 = RandomForest.predict(Test_X)

print('The accuracy of the Random Forest is', metrics.accuracy_score(Prediction1,Test_Y))

# The next method is DecisionTree

from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm

DT = DecisionTreeClassifier()

DT.fit(Train_X,Train_Y)

Prediction2 = DT.predict(Test_X)

print('The accuracy of the Desicion Tree is',metrics.accuracy_score(Prediction2,Test_Y))
# The next method is KNN

from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours

KNN = KNeighborsClassifier(n_neighbors= 5)

KNN.fit(Train_X,Train_Y)

Prediction3 = KNN.predict(Test_X)

print('The accuracy of the KNN is',metrics.accuracy_score(Prediction3,Test_Y))
# write a function to check different results of using different numbers of neighbours 

index = list(range(1,10,2))

list_of_prediction = pd.Series() 

for i in index:

    KNN = KNeighborsClassifier(n_neighbors= i)

    KNN.fit(Train_X,Train_Y)

    Prediction = KNN.predict(Test_X)

    list_of_prediction = list_of_prediction.append(pd.Series(metrics.accuracy_score(Prediction,Test_Y)))

plt.plot(index, list_of_prediction)        