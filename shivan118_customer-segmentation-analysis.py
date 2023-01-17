from IPython.display import Image

Image(filename='../input/ffffff/1111.jpg', width="800", height='50')


from IPython.display import Image

Image(filename='../input/dddddd/1.png', width="800", height='50')
Image(filename='../input/dddddd/x2.jpg', width="800", height='50')
Image(filename='../input/x12345/x3.jpg', width="800", height='50')
Image(filename='../input/x12345/x33.jpg', width="800", height='50')
Image(filename='../input/x12345/x4.png', width="800", height='50')
Image(filename='../input/x12345/x5.png', width="800", height='50')
Image(filename='../input/finalx/01.png', width="800", height='50')
Image(filename='../input/finalx/02.png', width="800", height='50')
Image(filename='../input/finalx/03.png', width="800", height='50')
Image(filename='../input/finalx/05.png', width="800", height='50')
Image(filename='../input/finalx/06.png', width="800", height='50')
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



sns.set_style("whitegrid")

plt.style.use("fivethirtyeight")



# for interactive visualizations

import plotly.offline as py

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected = True)

from plotly import tools



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
### Loading the Datasets

mall_data = pd.read_csv("/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")
#### Read the top 5 rows using plotly library

data = ff.create_table(mall_data.head())

py.iplot(data)
##### Check the Descriptions of the datasets using plotly library. 

desc = ff.create_table(mall_data.describe())

py.iplot(desc)
#### Visualizing the null values using missingo function



import missingno as msno

msno.matrix(mall_data)
plt.figure(figsize = (16,5))



plt.subplot(1, 3, 1)

sns.distplot(mall_data['Age'])



plt.subplot(1, 3, 2)

sns.distplot(mall_data['Annual Income (k$)'])



plt.subplot(1, 3, 3)

sns.distplot(mall_data['Spending Score (1-100)'])



plt.show()
# Prepare Data

df = mall_data.groupby('Gender').size()



# Make the plot with pandas

df.plot(kind='pie', subplots=True, figsize=(15, 8))

plt.title("Pie Chart of Vehicle Class - Bad")

plt.ylabel("")

plt.show()
### hist plot

mall_data.hist(figsize = (15, 12))

plt.show()
### Pair plot

sns.pairplot(mall_data)

plt.title('Pairplot for the Data')

plt.show()
#### Pairplot only three columns beteen Gender columns

sns.pairplot(mall_data,vars = ['Spending Score (1-100)', 'Annual Income (k$)', 'Age'], hue="Gender")
plt.figure(figsize = (16,8))

sns.countplot(mall_data['Age'], palette = 'hsv')

plt.title('Distribution of Age', fontsize = 20)
plt.figure(figsize = (20,8))

sns.countplot(mall_data['Annual Income (k$)'], palette = 'rainbow')

plt.title('Distribution of Annual Income', fontsize = 20)
plt.figure(figsize = (20,8))

sns.countplot(x ='Spending Score (1-100)', data = mall_data,palette = 'rainbow' ) 

plt.title('Distribution of Annual Income')
#visualize the correlation

plt.figure(figsize = (20,8))

sns.heatmap(mall_data.corr(), cmap = 'Wistia', annot = True)

plt.title('Heatmap for the Data')

plt.show()
plt.figure(1 , figsize = (15 , 6))

for gender in ['Male' , 'Female']:

    plt.scatter(x = 'Age' , y = 'Annual Income (k$)' , data = mall_data[mall_data['Gender'] == gender] ,

                s = 200 , alpha = 0.5 , label = gender)

plt.xlabel('Age')

plt.ylabel('Annual Income') 

plt.title('Age vs Annual Income w.r.t Gender')

plt.legend()

plt.show()
plt.figure(1 , figsize = (15 , 6))

for gender in ['Male' , 'Female']:

    plt.scatter(x = 'Annual Income (k$)',y = 'Spending Score (1-100)' ,

                data = mall_data[mall_data['Gender'] == gender] ,s = 200 , alpha = 0.5 , label = gender)

plt.xlabel('Annual Income (k$)'), 

plt.ylabel('Spending Score (1-100)') 

plt.title('Annual Income vs Spending Score w.r.t Gender')

plt.legend()

plt.show()
#Considering only 2 features (Annual income and Spending Score) and no Label available

X= mall_data.iloc[:, [3,4]].values
#Building the Model

#KMeans Algorithm to decide the optimum cluster number , KMeans++ using Elbow Mmethod



from sklearn.cluster import KMeans

k=[]



for i in range(1,11):

    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)

    kmeans.fit(X)

    k.append(kmeans.inertia_)
#Visualizing the ELBOW method to get the optimal value of K 



plt.figure(1 , figsize = (15 , 6))

plt.plot(range(1,11), k)

plt.title('The Elbow Method')

plt.xlabel('no of clusters')

plt.ylabel('wcss')

plt.show()
#Model Build

model = KMeans(n_clusters= 5, init='k-means++', random_state=0)

y_kmeans= model.fit_predict(X)



#For unsupervised learning we use "fit_predict()" where in for supervised learning we use "fit_tranform()"

#y_kmeans is the final model . Now how and where we will deploy this model in production is depends on what tool we are using.

#This use case is very common and it is used in BFS industry(credit card) and retail for customer segmenattion.
#Visualizing all the clusters 

plt.figure(1 , figsize = (15 , 8))

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'magenta', label = 'Cluster 1') ### Cluster 1

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')  ## Cluster 2

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'cyan', label = 'Cluster 3')  ## Cluster 3

plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'green', label = 'Cluster 4')  ## Cluster 4

plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'red', label = 'Cluster 5')   ## Cluster 5

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 200, c = 'black', label = 'Centroids')

plt.title('K Means Clustering Algorithm')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()
#Awesome! Here, we can clearly visualize five clusters. The black dots represent the centroid of each cluster.





#Cluster 1 (Red Color) -> earning high but spending less

#cluster 2 (Blue Colr) -> average in terms of earning and spending 

#cluster 3 (Green Color) -> earning high and also spending high [TARGET SET]

#cluster 4 (cyan Color) -> earning less but spending more

#Cluster 5 (magenta Color) -> Earning less , spending less
### Refrence:  https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/