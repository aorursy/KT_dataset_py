#Import the relevant libraries to complete the stated goals.

import pandas as pd

import seaborn as sns

sns.set()
#Load the iris dataset using pandas.

iris = pd.read_csv('../input/iris/Iris.csv')
#View the first 5 entries to understand format of the data set.

iris.head()
#View data types of the data set.

iris.dtypes
#Plot designated variables.

sns.pairplot(iris, vars=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'], hue='Species')
#Read the iris species and returns a integer.

def iris_species(x):

    if x == 'Iris-versicolor':

        return 0

    elif x == 'Iris-virginica':

        return 1

    else:

        return 2
#Create a new column labeled 'Species_Target' and apply the function 'iris_species'.

iris['Species_Target'] = iris['Species'].apply(iris_species)
#Check to ensure our formula properly coded the species into integers. Appears correct.

iris['Species_Target'].value_counts()
x_iris = iris.drop(['Id','Species','Species_Target'], axis=1)

y_iris = iris['Species_Target']
#Check to view correct dimensions of independent and dependent variables.

print(x_iris.shape)

print(y_iris.shape)
#Randomely split data observations into training and testing sets.

from sklearn.model_selection import train_test_split 

Xtrain, Xtest, ytrain, ytest = train_test_split(x_iris,y_iris,test_size = 0.25,random_state=1)
#Import model for normally distributed data (AKA Gaussian distributed data), train our model, and test the accuracy of our model.

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(Xtrain, ytrain)

y_model = model.predict(Xtest)
#To read the accuracy of how our model performed on our test data.

from sklearn.metrics import accuracy_score

print('Model accuracy: ',accuracy_score(ytest, y_model))
#Create a heatmap of the standard correlation coefficients.

sns.heatmap(iris.drop(['Species_Target'], axis=1).corr().round(decimals=2),annot=True,cmap='coolwarm')
x_iris_s = iris.drop(['Id','Species','PetalLengthCm','PetalWidthCm','Species_Target'], axis=1)

x_iris_p = iris.drop(['Id','Species','SepalLengthCm','SepalWidthCm','Species_Target'], axis=1)
#Randomly split data observations into training and testing sets.

from sklearn.model_selection import train_test_split 

Xtrain_s, Xtest_s, ytrain_s, ytest_s = train_test_split(x_iris_s,y_iris,test_size = 0.25,random_state=1)

Xtrain_p, Xtest_p, ytrain_p, ytest_p = train_test_split(x_iris_p,y_iris,test_size = 0.25,random_state=1)
#Import model for normally distributed data (AKA Gaussian distributed data), train our model, and test the accuracy of our model.

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(Xtrain_s, ytrain_s)

y_model_s = model.predict(Xtest_s)



model.fit(Xtrain_p, ytrain_p)

y_model_p = model.predict(Xtest_p)
#To read the accuracy of how our model performed on our test data.

from sklearn.metrics import accuracy_score

print('Model accuracy using sepal measurements: ',accuracy_score(ytest_s, y_model_s))

print('Model accuracy using petal measurements: ',accuracy_score(ytest_p, y_model_p))