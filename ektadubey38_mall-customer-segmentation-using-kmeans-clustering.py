# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for visualization
import seaborn as sns#machine learning algorithm
from sklearn.cluster import KMeans
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset=pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
df=dataset.copy()
df.drop(['CustomerID'],axis=1,inplace=True)
plt.figure(figsize=(10,6))
plt.title("Ages Frequency")
sns.axes_style('dark')
sns.violinplot(y=df["Age"])
plt.show()

#Here we are visualizing the number of buyers on the basis of age group

#plotting graphs between age,Annual Income(k$) and Spending Score(1-100)
sns.pairplot(df)

#Here we are determining the number of male and female buyers by counting and also visualizing 
genders=df.Gender.value_counts()
print(genders)
sns.set_style("darkgrid")
plt.figure(figsize=(10,4))
sns.barplot(x=genders.index,y=genders.values)
plt.show()
plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.boxplot(y=df["Spending Score (1-100)"],color='red')
plt.subplot(1,2,2)
sns.boxplot(y=df["Annual Income (k$)"])
plt.show()
#we can see that the average of both Spending Score and Annual Income is around 50
age18_25=df.Age[(df.Age<=25)&(df.Age>=18)]
age25_35=df.Age[(df.Age<=35)&(df.Age>=26)]
age35_45=df.Age[(df.Age<=45)&(df.Age>=36)]
age45_55=df.Age[(df.Age<=55)&(df.Age>=46)]
age55=df.Age[(df.Age>=56)]

x=['18-25','26-35','36-45','46-55','55above']
y=[len(age18_25),len(age25_35),len(age35_45),len(age45_55),len(age55)]

plt.figure(figsize=(15,6))
sns.barplot(x=x,y=y,palette='rocket')
plt.title('Number of Customers and Ages')
plt.xlabel('Age')
plt.ylabel('Number of Customers')
plt.show()
#plotting a graph between number of customers and Age
X=dataset.iloc[:,[3,4]].values
tmpDF=pd.DataFrame(X)
kmeans=KMeans(n_clusters=5,init='k-means++',random_state=42)
y_kmeans=kmeans.fit_predict(X)
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label='Careless')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='blue',label='Standard')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='green',label='Target')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='cyan',label='Careful')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='magenta',label='Sensible')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Incom(k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

#Here we are making 5 clusters with the following properties
#1. Red= These are the careless customers who have less income but high spending score
#2. Blue= These are the standard customers who have moderate income and moderate spending score 
#3. Green= These are the Target customers who have high income and high spending score
#4. Cyan= These are the Sensible customers who have less income and less spending score
#5. Magenta= These are the careful customers who have high income but low spending score
dataset['clusters']=kmeans.labels_
dataset.head()
dataset.sample(5)
centers=pd.DataFrame(kmeans.cluster_centers_)
centers
#Here we get the following insights:
#1.That people around $25k annual income have spending score of 79/100-->Careless
#2.That people around $55k annual income have spending score of 49/100-->Standard
#3.That people around $86k annual income have spending score of 82/100-->Target
#4.That people around $26k annual income have spending score of 20/100-->Sensible
#5.That people around $88k annual income have spending score of 88/100-->Careful
centers["clusters"]=range(5)
centers
#Assigning cluster numbers to the groups
dataset['ind']=dataset.index
dataset.head()
dataset=dataset.merge(centers)
dataset
#merging the riginal dataset with centers dataset
dataset.sample(20)
dataset=dataset.sort_values("ind")
dataset.head()
#sorting the values on the basis of index values
dataset=dataset.drop("ind",1)
dataset.head()
#deleting the index column as it is not needed
dataset=pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
d=dataset.iloc[:,[2,3,4]]
X=dataset.iloc[:,[2,3,4]].values
#Converting the Age,Annual Income(k$) and Spending Score(1-100) in array form
kmeans=KMeans(n_clusters=5,init='k-means++',random_state=42)
y_kmeans=kmeans.fit_predict(X)
dataset["clusters"]=kmeans.labels_
centers=pd.DataFrame(kmeans.cluster_centers_)
centers
#Here we get the following insights:
#1.People with Age around 40 have $87k annual income and a spending score of 17/100-->Careful
#2.People with Age around 43 have $55k annual income and a spending score of 49/100-->Standard
#3.People with Age around 25 have $26k annual income and a spending score of 78/100-->Careless
#4.People with Age around 32 have $86k annual income and a spending score of 82/100-->Target
#5.People with Age around 45 have $26k annual income and a spending score of 20/100-->Sensible
centers["clusters"]=range(5)
dataset["ind"]=dataset.index
dataset=dataset.merge(centers)
dataset.sample(5)
#Again merging dataset and centers dataframes
dataset=dataset.sort_values('ind')
dataset=dataset.drop("ind",1)
dataset 
#dropping indexcolumn as it is not needed