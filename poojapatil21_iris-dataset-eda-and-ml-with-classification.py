#Practice of Iris Dataset for Machine Learning and Exploring Data
# Exploratory Data Analysis of Iris Dataset
# sepal length,sepal width,petal length,petal width are features/independent variable
# 
# iris will be label/dependent variable
# 
# Problem Description : for new sepal length,width,petal length,width we have to identify iris species
# 
# This is classification problem

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#os.getcwd()
Iris = pd.read_csv("../input/iris_dataset.csv")
Iris.head()


Iris.shape
Iris.columns
Iris.describe()
Iris.isnull().sum()
Iris['child']='Nan'
Iris.head()
Iris.isnull().sum()
Iris['sepal length'].values[0]
#How many data points are present for each species?
#or How many flowers of each species?

#This helps to identify dataset is balanced or imbalanced
#no. of value counts for each class should be balanced in all classes

Iris['iris'].value_counts()
#2-D scatter plot

Iris.plot(kind='scatter',x='sepal length',y='sepal width');
plt.show()
sns.set_style('whitegrid');
sns.FacetGrid(Iris,hue='iris',size=4)   .map(plt.scatter,'sepal length','sepal width')    .add_legend();
plt.show()

#notice that we can easily separate blue points from red,green points
#by using sepal length and sepal width, setosa flower can identified
#separating versicolor and virginca si quire difficult as they have overlap
#pairplot
#to choose which two features are good to predict label

sns.set_style('whitegrid')
sns.pairplot(Iris,hue='iris',size=3)
plt.show()

#Draw histogram on petal length
#Try Univariate Analysis to choose which variable helps to predict labels
# Here, Sepal length is good option because setosa is mroe separated from versicolor and virginca
#for other features, more overlap is seen


sns.FacetGrid(Iris,hue='iris',size=5)    .map(sns.distplot,"petal length")   .add_legend()
    
plt.show()

#Draw histogram on petal length
sns.FacetGrid(Iris,hue='iris',size=5)    .map(sns.distplot,"petal width")   .add_legend()
    
plt.show()


#Draw histogram on petal length
sns.FacetGrid(Iris,hue='iris',size=5)    .map(sns.distplot,"sepal length")   .add_legend()
    
plt.show()
#Draw histogram on petal length
sns.FacetGrid(Iris,hue='iris',size=5)    .map(sns.distplot,"sepal width")   .add_legend()
    
plt.show()



print('Mean')
#print(np.mean(Iris-setosa['petal length']))
#Iris.mean('petal length')
Iris.describe()


Iris['petal length'].mean()

Iris['iris'].unique()
#find out mean for all species per feature
#means helps to idenfy central tendecy

#
Iris.groupby('iris').mean()


#variance is average of squared disatnce of all points from mean
#standard deviation is sq rt of variance
#find out std for all species per feature
#find out spread of distribution for all species means wideness in graphs

Iris.groupby('iris').std()


#Median,Percentile,Quantile,IQR,MAD


Iris.groupby('iris').median()


Iris.groupby('iris').quantile()

#25,50,75,100 th percentile values are called quantiles
#means 25th percentile value shows that 
#50th percentile values called median


np.percentile(Iris['petal length'],np.arange(0,100,25))


#boxplot and whisker
#boxplot
Iris.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
plt.show()

#boxplot shows 25,50,75 percentile values of all categories
sns.boxplot(x='iris',y='petal length',data=Iris)
plt.show()


#violin plot
sns.violinplot(x='iris',y='petal length',data=Iris,size=8)
plt.show()


#Apply Machine Learning methods to Iris dataframe
#create X=features and Y=Label from dataset

X = Iris.iloc[:, 1:4]
y = Iris.iloc[:, 4]


#split dataset into training and test datasets 

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

#test_size: if integer, number of examples into test dataset; if between 0.0 and 1.0, means proportion
print('There are {} samples in the training set and {} samples in the test set'.format(X_train.shape[0], X_test.shape[0]))


#scale dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

sc=StandardScaler()

sc.fit(X_train)


#X_train_std and X_test_std are the scaled datasets to be used in algorithms
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


#Apply Support Vector Classification
from sklearn.svm import SVC

svm = SVC(kernel = 'rbf',random_state=0,gamma = .10,C=1.0)
svm.fit(X_train_std,y_train)

print('The accuracy of the SVM classifier on training data is {:.2f}'.format(svm.score(X_train_std, y_train)))
print('The accuracy of the SVM classifier on test data is {:.2f}'.format(svm.score(X_test_std, y_test)))



#Apply KNN 
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7,p=2,metric='minkowski')
knn.fit(X_train_std,y_train)



print('The accuracy of the Knn classifier on training data is {:.2f}'.format(knn.score(X_train_std, y_train)))
print('The accuracy of the Knn classifier on test data is {:.2f}'.format(knn.score(X_test_std, y_test)))




#Applying Decision Tree
from sklearn import tree

#Create tree object
decision_tree = tree.DecisionTreeClassifier(criterion='gini')

#Train DT based on scaled training set
decision_tree.fit(X_train_std, y_train)

#Print performance
print('The accuracy of the Decision Tree classifier on training data is {:.2f}'.format(decision_tree.score(X_train_std, y_train)))
print('The accuracy of the Decision Tree classifier on test data is {:.2f}'.format(decision_tree.score(X_test_std, y_test)))


#Applying RandomForest
from sklearn.ensemble import RandomForestClassifier

#Create Random Forest object
random_forest = RandomForestClassifier()

#Train model
random_forest.fit(X_train_std, y_train)

#Print performance
print('The accuracy of the Random Forest classifier on training data is {:.2f}'.format(random_forest.score(X_train_std, y_train)))
print('The accuracy of the Random Forest classifier on test data is {:.2f}'.format(random_forest.score(X_test_std, y_test)))


