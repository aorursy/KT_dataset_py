import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Load iris.csv in to a pandas dataframe.
iris = pd.read_csv("../input/Iris.csv")
iris.head()
#(Q) how many data points and features?
iris.shape
#(Q) what are the column names in our dataset?
iris.columns
#(Q) how many data points for each class are present?
#(or) how many flowers for each species are present>
iris["Species"].value_counts()

#Histogram of petal length
sns.FacetGrid(iris,hue="Species",size=5) \
    .map(sns.distplot,"PetalLengthCm") \
    .add_legend()
plt.show()
#Histogram of petal width
sns.FacetGrid(iris,hue="Species",size=5) \
    .map(sns.distplot,"PetalWidthCm") \
    .add_legend()
plt.show()
#Histogram of sepal length
sns.FacetGrid(iris,hue="Species",size=5) \
    .map(sns.distplot,"SepalLengthCm") \
    .add_legend()
plt.show()

#Histogram of sepal width
sns.FacetGrid(iris,hue="Species",size=5) \
    .map(sns.distplot,"SepalWidthCm") \
    .add_legend()
plt.show()
#2D scatter plot:
iris.plot(kind='scatter', x='SepalLengthCm',y='SepalWidthCm')
plt.show()
#2D scatter plot with color-coding for each flower type/class.
#here "sns" corresponds to seaborn.
sns.set_style("whitegrid")
sns.FacetGrid(iris, hue="Species", size=4) \
    .map(plt.scatter,"SepalLengthCm","SepalWidthCm") \
    .add_legend()
plt.show()

# Notice that the blue points can be easly seperated.
# from red and green by drawing a line.
# but red and green data points cannot be easly seperated.
#can we draw multiple 2-D scatter plots for each combination of features?
# How many combinations exist? 4C2 = 6.
sns.set_style("whitegrid")
sns.pairplot(iris,hue="Species",size=3,aspect=1)
plt.show()
#the diagonal elements are PDF for each feature.
iris.head(2) #show the first 2 rows from the dataset

#(Q) any null value is present or not?
iris.isnull().sum()
#(Q) what is the mean, varience and standard dieviation of the each feature?
iris.describe()
X = iris.drop(["Species"], axis=1)
y = iris["Species"]
#heatmap is to identify the highly correlated features
plt.figure(figsize=(7,4)) 
sns.heatmap(iris.corr(),annot=True,cmap='cubehelix_r')
plt.show()
from sklearn import metrics
from sklearn.model_selection import train_test_split
#Splitting The Data into Training And Testing Dataset
# the attribute test_size=0.3 splits the data into 70% and 30% ratio. train=70% and test=33%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
#select the algorithm
from sklearn import neighbors
model = neighbors.KNeighborsClassifier(n_neighbors=3)
# we train the algorithm with the training data and the training output
model.fit(X_train,y_train)
#now we pass the testing data to the trained algorithm
predict = model.predict(X_test)
#now we check the accuracy of the algorithm. 
#we pass the predicted output by the model and the actual output
from sklearn.metrics import accuracy_score #for checking the model accuracy
print('The accuracy of the KNN is',metrics.accuracy_score(predict,y_test))
#importing svm from sklearn library

from sklearn import svm
svc = svm.SVC(C=1.0, kernel='rbf') #select the algorithm
# we train the algorithm with the training data and the training output
svc.fit(X_train,y_train)
#now we pass the testing data to the trained algorithm
pred = svc.predict(X_test)
#now we check the accuracy of the algorithm. 
#we pass the predicted output by the model and the actual output
print('The accuracy of the SVM is:',metrics.accuracy_score(pred,y_test))
model=DecisionTreeClassifier()
model.fit(X_train,y_train) 
prediction=model.predict(X_test) 
print('The accuracy of the Decision Tree using Petals is:',metrics.accuracy_score(prediction,y_test))

from sklearn.linear_model import LogisticRegression # for Logistic Regression algorithm
model = LogisticRegression()
model.fit(X_train,y_train) 
prediction=model.predict(X_test) 
print('The accuracy of the Logistic Regression using Petals is:',metrics.accuracy_score(prediction,y_test))



