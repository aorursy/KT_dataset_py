# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
%matplotlib inline
sns.set(style="darkgrid")
plt.figure(figsize=(100,100))


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Importing dataset for analysis
dataset = pd.read_csv('../input/2015.csv')
# Let's describe the data to get a summary statistics of the dataset
dataset.head()
dataset.info()
dataset.describe()
dataset.columns = ["Country","Region","Rank","Score","StandardError","GDP","Family","Health","Freedom","Trust","Generosity","Dystopia"]
dataset.head()
sns.lmplot(x='Health' , y='Score',hue='Region',data=dataset,fit_reg=False)
sns.lmplot(x = "Health",y ="Score",data = dataset,aspect =1)
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
X = pd.DataFrame(dataset['Health'])
y = pd.DataFrame(dataset['Score'])
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0, test_size = 0.20 )
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)
y_pred = pd.DataFrame(y_pred,columns = ["Score"])

#Visualising the training set result
plt.scatter(X_train,y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="Blue")
#Visualing the Test set
plt.scatter(X_test,y_test,color="red")
plt.plot(X_train,regressor.predict(X_train),color="Blue")
rss = (y_pred - y_test)**2
sum = 0

for i in range(0,len(rss)):
    sum = sum + rss.iloc[i,]
print(sum)
rss_train = (pd.DataFrame(regressor.predict(X_train)) - X_train)**2
sum_train = 0

for i in range(0,len(rss_train)):
    sum_train = sum_train + rss_train.iloc[i,]
print(sum_train)
sum
plt.hist(x= 'Health',bins = 20,data = dataset,color='indianred',alpha=0.5,stacked=True)
plt.hist(x= 'GDP',bins = 20,data = dataset,color='blue',alpha=0.5,stacked=True)
plt.hist(x="Score",data=dataset,bins=20)
sns.jointplot(x='Health',y='Score',data=dataset,kind='hex')
sns.kdeplot(dataset.Score,dataset.Health,shade=True,bw='silverman',cbar=True)
#sns.kdeplot(data=dataset['Health'],color='Blue')
#sns.kdeplot(data=dataset['Dystopia'],color='Black')
#sns.kdeplot(data=dataset['Freedom'],color='Yellow')
sns.distplot(dataset['Score'],bins=25,rug=True,hist=False)
fig, ax = plt.subplots()
ax = sns.boxplot(x='Region',y='Dystopia',data=dataset)
plt.xticks(rotation=90)
# the size of A4 paper
fig.set_size_inches(11.7, 8.27)
sns.violinplot(x="Region",y="Generosity",data=dataset,inner='stick')
plt.xticks(rotation=90)
dframe1 = dataset.ix[1:50,['Score','GDP','Health','Dystopia']]
corr = dframe1.corr()
sns.heatmap(data=corr)
dataset.head()
sns.lmplot(x="GDP",y="Dystopia",data=dataset,fit_reg=False,hue="Region")
sns.lmplot(x="GDP",y="Trust",data=dataset,fit_reg=False,hue="Region")
#Cluster Analysis using K-Means Clustering using GDP and Trust
X1 = dataset.loc[:,["GDP","Trust"]].values

#Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss =[]
for i in range (1,11):
    kmeans = KMeans(n_clusters=i , init = "k-means++", max_iter=300, n_init=10,random_state=0)
    kmeans.fit(X1)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.show()
#Applying k-means to the dataset with the 4 number of clusters
kmeans = KMeans(n_clusters=3,init="k-means++",random_state=0)
y_kmeans = kmeans.fit_predict(X1)
print(y_kmeans)
#Visualizing the clusters
plt.scatter(X1[y_kmeans == 0,0],X1[y_kmeans == 0,1],s = 10, c ='blue',label = 'Medium Happy')
plt.scatter(X1[y_kmeans == 1,0],X1[y_kmeans == 1,1],s = 10, c ='red',label = 'Very Unhappy')
plt.scatter(X1[y_kmeans == 2,0],X1[y_kmeans == 2,1],s = 10, c ='green',label = 'Happy')
#plt.scatter(X1[y_kmeans == 3,0],X1[y_kmeans == 3,1],s = 10, c ='cyan',label = 'Cluster4')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s = 300, c ='Yellow',label = 'Centroids')
plt.title('Cluster of citizens')
plt.xlabel('GDP')	
plt.ylabel('Trust')
plt.legend()
plt.show()
dataset.head()
Health_Region = pd.DataFrame(dataset.groupby("Region")["Health"].sum())
Health_Region.plot(kind="bar")


