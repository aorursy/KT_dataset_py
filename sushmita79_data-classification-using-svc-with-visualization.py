# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn import metrics

from sklearn.model_selection import train_test_split #to split the dataset for training and testing





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

#read the dataset into pandas dataframe

df=pd.read_csv("../input/Iris.csv")
#observe first two rows of dataframe

df.head(2)
#we check that the samples for 3 classes to classify are equally distributed

df.Species.value_counts().plot(kind='pie')
#Id column is not useful so drop it

df.drop('Id',axis=1,inplace=True)
#pairplot helps to analyse the relation between different features

#we observe that setosa class in blue is lineraly separable and model can well predict these samples whereas other two have some overlapping

sns.pairplot(hue='Species',data=df)
#heatmap can help us find out all the important features which can help model achieve better accuracy

#we can select only important features for training our model

#we observe that PetalWidth and PetalLength have high relation whereas SepalWidth and SepalLength have lower relation  

sns.heatmap(df.corr(),annot=True)
#Sepal Properties

figure=df[df.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa')

df[df.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue',label='versicolor',ax=figure)

df[df.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica',ax=figure)
#Petal Properties

figure = df[df.Species=='Iris-setosa'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='orange', label='Setosa')

df[df.Species=='Iris-versicolor'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='blue', label='versicolor',ax=figure)

df[df.Species=='Iris-virginica'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='green', label='virginica', ax=figure)

#boc plot can help us view the different values for a particular feature with respect to different species

figure_box=plt.figure(figsize=(9,9))

figure_box.add_subplot(2,2,1)

sns.boxplot(x='Species',y='PetalLengthCm',data=df)

figure_box.add_subplot(2,2,2)

sns.boxplot(x='Species',y='PetalWidthCm',data=df)

figure_box.add_subplot(2,2,3)

sns.boxplot(x='Species',y='SepalLengthCm',data=df)

figure_box.add_subplot(2,2,4)

sns.boxplot(x='Species',y='SepalWidthCm',data=df)
#The violinplot shows density of the length and width in the species. The thinner part denotes that there is less density whereas the fatter part conveys higher density

plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.violinplot(x='Species',y='PetalLengthCm',data=df)

plt.subplot(2,2,2)

sns.violinplot(x='Species',y='PetalWidthCm',data=df)

plt.subplot(2,2,3)

sns.violinplot(x='Species',y='SepalLengthCm',data=df)

plt.subplot(2,2,4)

sns.violinplot(x='Species',y='SepalWidthCm',data=df)



df.head()
#Considering all features

x=df.iloc[:,0:4]   #independent features 

y=df.iloc[:,4]     #dependent (target value)





#our model needs train data so that we can train the model and later test data to check our classifications

x_train,x_test,y_train,y_test=train_test_split(x,y, test_size = 0.3,random_state=42)
#It's better to scale your data so that models work better with scaled values



from sklearn.preprocessing import StandardScaler





sc = StandardScaler()  # Load the standard scaler

x_train=sc.fit_transform(x_train)  

x_test = sc.transform(x_test)  # Scale the feature data to be of mean 0 and variance 1

# importing all the necessary packages to use the various classification algorithms

from sklearn.linear_model import LogisticRegression # for Logistic Regression algorithm

from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours

from sklearn import svm  #for Support Vector Machine (SVM) Algorithm

from sklearn.tree import DecisionTreeClassifier#for using Decision Tree Algoithm
models=[LogisticRegression(),

       KNeighborsClassifier(n_neighbors=5),

       svm.SVC(),

       DecisionTreeClassifier(max_leaf_nodes=3)]
#check all model performance

for m in models:

    model=m.fit(x_train,y_train)

    y_pred=model.predict(x_test) 

    print('The accuracy of the is:',metrics.accuracy_score(y_test,y_pred))#now we check the 

    
x=df.iloc[:,2:4]   #Petal width and length

y=df.iloc[:,4]

x_train,x_test,y_train,y_test=train_test_split(x,y, test_size = 0.3,random_state=42)
from sklearn.preprocessing import StandardScaler





sc = StandardScaler()  # Load the standard scaler

x_train=sc.fit_transform(x_train)  # Compute the mean and standard deviation of the feature data

x_test = sc.transform(x_test)  # Scale the feature data to be of mean 0 and variance 1
for m in models:

    model=m.fit(x_train,y_train)

    y_pred=model.predict(x_test) 

    print('The accuracy of the is:',metrics.accuracy_score(y_test,y_pred))









