

#import required libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt
#Reading csv(comma separated values) file.......

iris=pd.read_csv("../input/Iris.csv")


#Check dataset.......

iris.head()



#Checking the dimensions....

iris.shape



#checking whether a dataset contain a missing value or not/.....

iris.isnull().sum()


'''checking if there is any inconsistency in the dataset as we see there 

are no null values in the dataset, so the data can be processed...../'''

iris.info()
#Checkig the unique values in species column which is our target variable..

iris["Species"].unique()
'''dropping the Id column as it is unnecessary, axis=1 specifies that 

it should be column wise, inplace =1 means 

the changes should be reflected into the dataframe'''

iris.drop('Id',axis=1,inplace=True)



#checking data after droping "ID column".....

iris.head()


'''Some Exploratory Data Analysis With Iris'''



fig = iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa')

iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='versicolor',ax=fig)

iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)

fig.set_xlabel("Sepal Length")

fig.set_ylabel("Sepal Width")

fig.set_title("Sepal Length VS Width")

fig=plt.gcf()

fig.set_size_inches(10,6)

plt.show()

fig.savefig("Sepal Length VS Width.png")


'''Some Exploratory Data Analysis With Iris'''

fig = iris[iris.Species=='Iris-setosa'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='orange', label='Setosa')

iris[iris.Species=='Iris-versicolor'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='blue', label='versicolor',ax=fig)

iris[iris.Species=='Iris-virginica'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='green', label='virginica', ax=fig)

fig.set_xlabel("Petal Length")

fig.set_ylabel("Petal Width")

fig.set_title(" Petal Length VS Width")

fig=plt.gcf()

fig.set_size_inches(10,6)

plt.show()

fig.savefig("Petal Length VS Width.png")



'''let us see how are the length and width are distributed'''

iris.hist(edgecolor='Yellow', linewidth=1.2)

fig=plt.gcf()

fig.set_size_inches(12,6)

plt.show()


'''Let us see how the length and width vary according to the species'''

plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.violinplot(x='Species',y='PetalLengthCm',data=iris)

plt.subplot(2,2,2)

sns.violinplot(x='Species',y='PetalWidthCm',data=iris)

plt.subplot(2,2,3)

sns.violinplot(x='Species',y='SepalLengthCm',data=iris)

plt.subplot(2,2,4)

sns.violinplot(x='Species',y='SepalWidthCm',data=iris)

fig.savefig("variable with species.png")

plt.figure(figsize=(7,4)) #7 is the size of the width and 4 is parts.... 

sns.heatmap(iris.corr(),annot=True,cmap='cubehelix_r') #draws  heatmap with input as the correlation matrix calculted by(iris.corr())

plt.show()


#Separating dependent and independent values..

X=iris.iloc[:, :-1].values

X

 

y=iris.iloc[:, -1].values

y

#splitting into training set and test.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 123)

# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# Fitting Random Forest Classification to the Training set

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)

# Predicting the Test set results

y_pred = classifier.predict(X_test)

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm



from sklearn import metrics #for checking the model accuracy

print('The accuracy of the Random forest is:',metrics.accuracy_score(y_pred,y_test))