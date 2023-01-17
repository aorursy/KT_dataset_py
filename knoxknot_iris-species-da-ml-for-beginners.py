'''

Import the libraries for Data Analysis,

You may also import libraries in the middle of your analysis. 

But you must import the library before using any of its method.

'''



import numpy as np                       # for scientific calculation

import pandas as pd                      # for loading and exploring the data 

import matplotlib.pyplot as plt          # for visualizing your analysis

import seaborn as sns                    # built upon matplotlib also for visualizing your analysis

import os                                # for interfacing with the operating system

import sqlite3                           # for connecting to the sqlite dataset

!ls ../input



'''

 quality assurance measures to ensure that dataset is available before the analysis.

 an error message is flagged if neither - .csv or .sqlite - of the dataset are in the data folder

'''

data_folder_content = os.listdir("../input/")

sqlite_error_message = "Error: sqlite file not available, check instructions above to download it"

csv_error_message = "Error: csv file not available, check instructions above to download it"

assert "database.sqlite" in data_folder_content, sqlite_error_message

assert "Iris.csv" in data_folder_content, csv_error_message
iris = pd.read_csv("../input/Iris.csv")       #load the iris.csv data

iris[::10]                                #display every 10th row of the data
'''

ashwin used the code

iris.drop('Id',axis=1, inplace=True) to remove the Id column(represented as axis=1) within the data itself (inplace=True)

we could also accomplish this with del iris['Id']

'''

del iris['Id']      #remove the Id column we don't need it.
print(iris.isnull().any().any())       #check if there is any missing cell

print(iris.shape)                      #what is the number of rows, the number of columns
iris.info()      # see a detailed description of the dataset 
iris.describe().transpose()       #let's do some elementary statistic on the iris data
cnx = sqlite3.connect('../input/database.sqlite')           # Create a connection to the sql database

iris2 = pd.read_sql("SELECT * FROM Iris", cnx)          # Read the data with a pandas method
iris2.info()                    # see a detailed description of the dataset 
'''

fig variable to gather all the three species we plot, choose the kind of plot we want, in this case scatter,

give each specie a color and set a corresponding label for the species.

the plt.gcf() collates the scatter plot.

we made the choice of our axis limits from the dimension of our statistics min and max information

Also note that the order of the code below is important

'''



fig = iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='setosa')

iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='versicolor',ax=fig)

iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)

fig.set_xlabel("Sepal Length")

fig.set_ylabel("Sepal Width")

fig.set_title("Sepal Length VS Sepal Width")

fig = plt.gcf()

fig.set_size_inches(10,6)

plt.show()
fig = iris[iris.Species=='Iris-setosa'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='orange', label='Setosa')

iris[iris.Species=='Iris-versicolor'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='blue', label='versicolor',ax=fig)

iris[iris.Species=='Iris-virginica'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='green', label='virginica', ax=fig)

fig.set_xlabel("Petal Length")

fig.set_ylabel("Petal Width")

fig.set_title(" Petal Length VS Petal Width")

fig=plt.gcf()

fig.set_size_inches(10,6)

plt.show()
iris.hist(edgecolor='black', linewidth=1.2, bins=20)

fig=plt.gcf()

fig.set_size_inches(12,6)

plt.show()
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.violinplot(x='Species',y='PetalLengthCm',data=iris)

plt.subplot(2,2,2)

sns.violinplot(x='Species',y='PetalWidthCm',data=iris)

plt.subplot(2,2,3)

sns.violinplot(x='Species',y='SepalLengthCm',data=iris)

plt.subplot(2,2,4)

sns.violinplot(x='Species',y='SepalWidthCm',data=iris)

plt.show()
'''

Now we will import specific modules from SciKit-Learn

For Supervised Learning

'''



from sklearn.linear_model import LogisticRegression               # Logistic Regression algorithm

from sklearn.tree import DecisionTreeClassifier                   # Decision Tree Classifier algorithm

from sklearn.neighbors import KNeighborsClassifier                # K-nearest neighbours algorithm

from sklearn.svm import SVC                                       # Support Vector Classifier algorithm

from sklearn.model_selection import train_test_split              # for spiltting the dataset

from sklearn.metrics import accuracy_score                        # for checking the model's accuracy

plt.figure(figsize=(7,4)) 

sns.heatmap(iris.corr(),annot=True,cmap='cubehelix_r')  #heatmap with input as the correlation matrix calculted by(iris.corr())

plt.show()
'''

From our earlier explaination features are the attributes measured - petal length, petal width, setal length, sepal width

while targets are the variables the considered sample i.e Iris-setosa, Iris-virginica, and Iris-versicolor.

So we will create a list of these. 

'''

features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

target = ['Species']

X = iris[features]

y = iris['Species']           #to keep the column in 1d I didnt call target variable else a warning "not a 1d" will be raised
print(X[:5])

print(type(X))
print(y[:5])

print(type(y))
'''

The whole iris dataset is split into training and testing data.

The attribute test_size=0.3 cuts out 30% of the 150 sample data for testing.

You could equally assign the attribute train_size = 0.7 and the spilting will work just fine.

The random_state attribute randomizes the dataset division process instead of approaching it linearly.

'''

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state=324)

print(train_X.shape)              #look at the input data for the training phase

print(test_X.shape)               #look at the input data for the testing phase
train_X[:2] , train_y[:2]   # lets see what target some rows of the train data feature maps to.
test_X[:2] , test_y[:2]
model = SVC()                                       # the algorithm

model.fit(train_X,train_y)                          # we train the algorithm with the training input and the training output

prediction = model.predict(test_X)                  # now we pass the testing data to the trained algorithm

accuracy = accuracy_score(test_y,prediction)        #now we check the accuracy algorithm's prediction to the reserved test output.

print('The accuracy of the SVM is:',accuracy) 
model = LogisticRegression()                            # the algorithm

model.fit(train_X,train_y)                              # we train the algorithm with the training input and the training output

prediction = model.predict(test_X)                      # now we pass the testing data to the trained algorithm

accuracy = accuracy_score(prediction,test_y)            #now we check the accuracy algorithm's prediction to the reserved test output.

print('The accuracy of the Logistic Regression is:',accuracy) 
model = DecisionTreeClassifier(max_leaf_nodes=4, random_state=0) # the algorithm

model.fit(train_X,train_y)                              # we train the algorithm with the training input and the training output

prediction = model.predict(test_X)                      # now we pass the testing data to the trained algorithm

accuracy = accuracy_score(prediction,test_y)            #now we check the accuracy of the algorithm.

print('The accuracy of the Decision Tree Classifier is:',accuracy) 
model=KNeighborsClassifier(n_neighbors=3)               # this examines 3 neighbours for putting the new data into a class

model.fit(train_X,train_y)                              # we train the algorithm with the training input and the training output

prediction = model.predict(test_X)                      # now we pass the testing data to the trained algorithm

accuracy = accuracy_score(prediction,test_y)            # now we check the accuracy of the algorithm.

print('The accuracy of the KNN is:',accuracy) 
n= range(1,11)                        # the number of neighbours we would like to check

a=pd.Series()                         # the accuracy values store in pandas series data format

for i in range(1,11):

    model=KNeighborsClassifier(n_neighbors=i) 

    model.fit(train_X,train_y)

    prediction=model.predict(test_X)

    a=a.append(pd.Series(accuracy_score(prediction,test_y)))

plt.plot(n, a)

plt.xticks(n)

plt.xlabel("n_value")

plt.ylabel("accuracy")

plt.show()
'''

Preparing the feature and target for curated petal and sepal dataset

the p and s subscript to X and y indicate the feature and target for petals and sepals.

'''

Xp = iris[['PetalLengthCm','PetalWidthCm']]

yp = iris['Species']



Xs = iris[['SepalLengthCm','SepalWidthCm']]

ys = iris['Species']
train_Xp,test_Xp,train_yp,test_yp = train_test_split(Xp,yp,test_size=0.3,random_state=342)  #petals



train_Xs,test_Xs,train_ys,test_ys = train_test_split(Xs,ys,test_size=0.3,random_state=342)  #sepals

model=SVC()

model.fit(train_Xp,train_yp) 

prediction=model.predict(test_Xp)

accuracy = accuracy_score(prediction,test_yp)

print('The accuracy of the SVC using Petals is:',accuracy)



model=SVC()

model.fit(train_Xs,train_ys) 

prediction=model.predict(test_Xs) 

accuracy = accuracy_score(prediction,test_yp)

print('The accuracy of the SVC using Sepals is:',accuracy)
model = LogisticRegression()

model.fit(train_Xp,train_yp) 

prediction=model.predict(test_Xp)

accuracy = accuracy_score(prediction,test_yp)

print('The accuracy of the Logistic Regression using Petals is:',accuracy)



model = LogisticRegression()

model.fit(train_Xs,train_ys) 

prediction=model.predict(test_Xs) 

accuracy = accuracy_score(prediction,test_yp)

print('The accuracy of the Logistic Regression using Sepals is:',accuracy)
model=DecisionTreeClassifier()

model.fit(train_Xp,train_yp) 

prediction=model.predict(test_Xp)

accuracy = accuracy_score(prediction,test_yp)

print('The accuracy of the Decision Tree using Petals is:',accuracy)



model=DecisionTreeClassifier()

model.fit(train_Xs,train_ys) 

prediction=model.predict(test_Xs) 

accuracy = accuracy_score(prediction,test_yp)

print('The accuracy of the Decision Tree using Sepals is:',accuracy)
model=KNeighborsClassifier(n_neighbors=3) 

model.fit(train_Xp,train_yp) 

prediction=model.predict(test_Xp)

accuracy = accuracy_score(prediction,test_yp)

print('The accuracy of the KNN using Petals is:',accuracy)



model=DecisionTreeClassifier()

model.fit(train_Xs,train_ys) 

prediction=model.predict(test_Xs) 

accuracy = accuracy_score(prediction,test_yp)

print('The accuracy of the KNN using Sepals is:',accuracy)
'''

Now we will import specific modules from SciKit-Learn

for Unsupervised Learning

'''



from sklearn.cluster import KMeans

from itertools import cycle, islice

from pandas.plotting import parallel_coordinates

# Choose the features 

features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

X = iris[features]                          # X is the input data

X[:10]
# Use k-Means Clustering 

model = KMeans(n_clusters=3)

model.fit(X)                            #kmeans algorithm takes only one argument for fitting since we do not have a target

prediction = model.predict(X)           #now the model is classifying the input data based on the number of cluster we passed

print(prediction)

print(prediction.shape)

#What are the centers of 3 clusters we formed ? 

centers = model.cluster_centers_

centers
# Function that creates a DataFrame with a column for cluster group



def cluster_grp(features, centers):

    colNames = list(features)

    colNames.append('Prediction')



# zip each center with corresponding (predict) to form the cluster_data  

    Z = [np.append(center, predict) for predict, center in enumerate(centers)]



    # Convert to pandas data frame for plotting

    P = pd.DataFrame(Z,columns=colNames)

    P['Prediction'] = P['Prediction'].astype(int)

    return P
cluster_data = cluster_grp(features, centers)     #the cluster group and assigned prediction

cluster_data
colName = ['Prediction']

pred = pd.DataFrame(prediction,columns=colName)

data = X.join(pred,how='right')

data
fig = data[data.Prediction== 0].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='orange', label='Setosa')

data[data.Prediction== 1].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='blue', label='versicolor',ax=fig)

data[data.Prediction== 2].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='green', label='virginica', ax=fig)

fig.set_xlabel("Petal Length")

fig.set_ylabel("Petal Width")

fig.set_title(" Petal Length VS Petal Width")

fig=plt.gcf()

fig.set_size_inches(10,6)

plt.show()
fig = data[data.Prediction== 0].plot.scatter(x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa')

data[data.Prediction== 1].plot.scatter(x='SepalLengthCm',y='SepalWidthCm',color='blue', label='versicolor',ax=fig)

data[data.Prediction== 2].plot.scatter(x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)

fig.set_xlabel("Petal Length")

fig.set_ylabel("Petal Width")

fig.set_title(" Petal Length VS Petal Width")

fig=plt.gcf()

fig.set_size_inches(10,6)

plt.show()