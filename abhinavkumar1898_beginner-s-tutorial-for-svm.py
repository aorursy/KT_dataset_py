import pandas as pd     #load data and data manipulation

import seaborn as slt  #high level inteface visualization

import numpy as np      #for mathematical computational

from sklearn import svm  #svm (Support Vector Machine) a classification machine learning algorithm

from sklearn.model_selection import train_test_split  #splits the dataset into training and testing data

from mlxtend.plotting import plot_decision_regions    #for plotting SVM classes

from matplotlib import pyplot as plt                  #basic for visualization
#reading the csv of iris-data

df = pd.read_csv('../input/apndcts/apndcts.csv')

df.head()
#it has no columns 

#by default pandas takes my first row as column names

#storing the first rows i.e. Columns in variable

col = df.columns

col
#setting column names

#slen- sepal length

#swid- sepal width

#plen- petal length

#pwid- petal width



df.columns=['At1','At2','At3','At4','At5','At6','At7','class']

#adding our first row to the data

df.loc[150]=col
#dimensions of our dataset

print (df.shape)

#few top rows of iris data

df.head()
#checking for number of nan values in any column

df.isna().sum()
#this code is for mapping different categories of class to integer value

#Iris-setosa = 0

#Iris-versicolor = 1

#Iris-virginica =2

m = pd.Series(df['class']).astype('category')

df['class']=m.cat.codes
#dividing data into our dependent and independent attributes

#classes we are going to predict

Y=df['class']

#By using other attributes  

X = df.drop(columns=['class'])
#split the data randomly into train and test set

#where test data is 30% of original data

xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.3)

#re-indexing the ytrain data to linear numerical values between 0-105

#it is used for plotting SVM classes

ytrain.index=np.arange(105)
#select the classifier i.e. Linear SVC (Support Vector Classifier)

clf = svm.SVC(gamma='auto')
#fit the train data or training to our model

pre = clf.fit(xtrain,ytrain)
#check score of our model on test data

clf.score(xtest,ytest)