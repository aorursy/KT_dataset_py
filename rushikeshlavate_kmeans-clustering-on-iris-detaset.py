import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data=pd.read_csv('../input/iris/Iris.csv')
data.head()
#find columns of dataframe
data.columns
#Display the data types of each column using the attribute dtype
data.dtypes
#the method value_counts to count the number of species with unique species values.
#to_frame() to convert it to a dataframe.
data.Species.value_counts().to_frame()
#find the summary of dataframe
data.info()
#The describe() function computes a summary of statistics for numeric entries such as count,mean,std,min,etc 
#The column which contain onject dtype then can not be include in it .

data.describe()
#above four column are numeric they displayed here but fifth column which is 'species' which has dtype object is not displayed
#to include this column write 
data.describe(include='all')
#Id Column is not required so we can drop it
data.drop(['Id'],axis=1,inplace=True)
#find unique species in 'Species' column
data['Species'].unique()
#groupby() is used to split data into groups
data.groupby('Species').size()
sns.countplot('Species',data=data)
#it show the count of each categprical data
sns.pairplot(data,hue='Species')
#A pairplot plot a pairwise relationships with other columns in datafeamw and also plot pairplot with itself .
plt.figure(figsize=(12,6))
sns.scatterplot(x=data['PetalLengthCm'],y=data['PetalWidthCm'],hue=data['Species'],marker='^',s = 100)
#it shows
#if petal_length is less that ~2 and petal_width is less than ~0.6 then species ia 'setosa'
#if petal_length is between ~3 and ~5.2 and petal_width is between ~1.0 and ~1.7 then species ia 'versicolor'
#if petal_length is greater that ~5.2 and petal_width is greater than ~1.7 then species ia 'virginica'
plt.figure(figsize=(14,10))
plt.subplot(2,2,1)
sns.boxplot(y=data['SepalLengthCm'],x=data['Species'],data=data)
plt.subplot(2,2,2)
sns.boxplot(y=data['PetalLengthCm'],x=data['Species'],data=data)
plt.subplot(2,2,3)
sns.boxplot(y=data['SepalWidthCm'],x=data['Species'],data=data)
plt.subplot(2,2,4)
sns.boxplot(y=data['PetalWidthCm'],x=data['Species'],data=data)
# spliting data for Training and Testing where 'x' training data and 'y' is testing data
x =data.iloc[ : , :4].values
y =data['Species']

#import required labrary and model in it
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=30,random_state=3)
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
KM = KMeans(n_clusters=3)
KM.fit(x)
print(KM.labels_)
#it shows the X and Y coordinates of centroid
KM.cluster_centers_
SSE= []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 0)
    kmeans.fit(x_train,y_train)
    SSE.append(kmeans.inertia_)
plt.figure(figsize=(10,6))
plt.plot(range(1,11),SSE,'*--')
plt.xlabel('K')
plt.ylabel('Sum of Squared error')
Y_predict = kmeans.fit_predict(x)
plt.figure(figsize=(12,6))
sns.scatterplot(x[Y_predict == 0, 0], x[Y_predict == 0, 1],color = 'red', label = 'Iris-setosa')
sns.scatterplot(x[Y_predict == 1, 0], x[Y_predict == 1, 1],color = 'blue', label = 'Iris-versicolour')
sns.scatterplot(x[Y_predict == 2, 0], x[Y_predict == 2, 1],color = 'green', label = 'Iris-virginica')         

# Plotting the centroids of the clusters
plt.scatter(KM.cluster_centers_[:, 0],KM.cluster_centers_[:,1],marker='s' ,color = 'black', label = 'Centroids')
plt.legend()
